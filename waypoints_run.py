"""
waypoints_run.py — overnight training script for QuadX-Waypoints-v4.

Both SAC and PPO use Stable-Baselines3.
Both run for ~500,000 steps so comparisons are fair.

Run:
    python waypoints_run.py
    python waypoints_run.py > logs/waypoints.txt 2>&1
"""

import os
import numpy as np
import gymnasium as gym
import PyFlyt.gym_envs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque
import wandb
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback
from wandb.integration.sb3 import WandbCallback

from scripts.wrappers import FlattenWaypointEnv

# ── reproducibility ───────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── flight modes to train ─────────────────────────────────────
MODES = [7, 6, 4, 0]   # ← edit this list

# ── environment config ────────────────────────────────────────
ENV_ID           = "PyFlyt/QuadX-Waypoints-v4"
MAX_WAYPOINTS    = 4
GOAL_DISTANCE    = 4.0
DOME_SIZE        = 150.0
MAX_DURATION_SEC = 120.0
PROXIMITY_SCALE  = 0.3

# ── curriculum config ─────────────────────────────────────────
CURRICULUM_WINDOW    = 50
CURRICULUM_THRESHOLD = 0.70

# ── eval config ───────────────────────────────────────────────
MAX_STEPS  = 1000
EVAL_EPS   = 10
EVAL_SEEDS = list(range(SEED, SEED + EVAL_EPS))

# ── SAC hyperparameters  (~500k steps) ────────────────────────
SAC_TOTAL_TIMESTEPS = 500_000
SAC_BATCH_SIZE      = 256
SAC_GAMMA           = 0.99
SAC_TAU             = 0.005
SAC_ALPHA           = 0.2      # ent_coef
SAC_LR              = 3e-4
SAC_BUFFER          = 500_000
SAC_LEARNING_STARTS = 1_000

# ── PPO hyperparameters  (~500k steps) ────────────────────────
# 4 stages × 125k = 500k — matches SAC budget exactly
PPO_TIMESTEPS_PER_STAGE   = 125_000
PPO_MAX_RETRIES_PER_STAGE = 1    # no retries → strict 500k cap
PPO_N_STEPS  = 2048
PPO_BATCH    = 64
PPO_EPOCHS   = 10
PPO_GAMMA    = 0.99
PPO_GAE      = 0.95
PPO_CLIP     = 0.2
PPO_LR       = 3e-4

# ── WandB ─────────────────────────────────────────────────────
WANDB_PROJECT = "waypoints"

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# Wrappers & helpers
# ═══════════════════════════════════════════════════════════════

class ProximityRewardWrapper(gym.Wrapper):
    def __init__(self, env, proximity_scale=PROXIMITY_SCALE):
        super().__init__(env)
        self.proximity_scale = proximity_scale

    @staticmethod
    def _min_dist(obs):
        deltas = obs["target_deltas"]
        dists  = np.linalg.norm(deltas, axis=-1)
        dists  = np.where(dists < 1e-6, np.inf, dists)
        return float(np.min(dists)) if not np.all(np.isinf(dists)) else 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward += self.proximity_scale / (self._min_dist(obs) + 1e-6)
        return obs, reward, terminated, truncated, info


class CurriculumManager:
    def __init__(self, start_targets=1, max_targets=MAX_WAYPOINTS,
                 window=CURRICULUM_WINDOW, threshold=CURRICULUM_THRESHOLD):
        self.num_targets   = start_targets
        self.max_targets   = max_targets
        self.window        = window
        self.threshold     = threshold
        self._successes    = deque(maxlen=window)
        self.stage_history = []
        self._ep_count     = 0

    def record(self, episode_return):
        self._ep_count += 1
        self._successes.append(float(episode_return > 0.0))
        advanced = False
        if (len(self._successes) == self.window
                and np.mean(self._successes) >= self.threshold
                and self.num_targets < self.max_targets):
            self.num_targets += 1
            self._successes.clear()
            self.stage_history.append((self._ep_count, self.num_targets))
            advanced = True
        return advanced

    @property
    def success_rate(self):
        return float(np.mean(self._successes)) if self._successes else 0.0


def make_env(flight_mode, num_targets=MAX_WAYPOINTS):
    env = gym.make(
        ENV_ID,
        flight_mode          = flight_mode,
        goal_reach_distance  = GOAL_DISTANCE,
        flight_dome_size     = DOME_SIZE,
        max_duration_seconds = MAX_DURATION_SEC,
        num_targets          = num_targets,
    )
    env = ProximityRewardWrapper(env)
    env = FlattenWaypointEnv(env, max_waypoints=MAX_WAYPOINTS)
    return env


def eval_agent(policy_fn, label, flight_mode, num_targets=MAX_WAYPOINTS):
    env     = make_env(flight_mode, num_targets)
    returns = []
    for i, seed in enumerate(EVAL_SEEDS):
        obs, _ = env.reset(seed=seed)
        total  = 0.0
        for _ in range(MAX_STEPS):
            action = policy_fn(obs)
            obs, r, terminated, truncated, _ = env.step(action)
            total += r
            if terminated or truncated:
                break
        returns.append(total)
        print(f"  [{label}] ep {i+1}/{EVAL_EPS}  return={total:.2f}")
    env.close()
    mean, std = float(np.mean(returns)), float(np.std(returns))
    print(f"  {label:<20}  mean={mean:8.2f}  std={std:7.2f}")
    return mean, std, returns


def smooth(x, w=20):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def save_training_curve(returns, stage_at_ep, curriculum, flight_mode, agent):
    fig, ax = plt.subplots(figsize=(12, 4))
    stage_colors = {1: "#fff3e0", 2: "#e8f5e9", 3: "#e3f2fd", 4: "#f3e5f5"}
    stage_labels = {1: "1 wp", 2: "2 wp", 3: "3 wp", 4: "4 wp"}
    arr = np.array(stage_at_ep)
    if len(arr) > 0:
        prev_ep, prev_stage = 0, arr[0]
        labeled = set()
        for i, stage in enumerate(arr):
            if stage != prev_stage or i == len(arr) - 1:
                end = i if stage != prev_stage else i + 1
                lbl = stage_labels[prev_stage] if prev_stage not in labeled else None
                ax.axvspan(prev_ep, end, color=stage_colors[prev_stage], alpha=0.5, label=lbl)
                if lbl:
                    labeled.add(prev_stage)
                prev_ep, prev_stage = i, stage
    ax.plot(returns, alpha=0.2, color="steelblue")
    if len(returns) >= 20:
        ax.plot(range(len(smooth(returns))), smooth(returns),
                color="steelblue", lw=2, label="smoothed")
    ax.axhline(0, color="black", lw=0.8, linestyle=":")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode return (shaped)")
    ax.set_title(f"{agent} curriculum — {ENV_ID}  flight_mode={flight_mode}")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    path = f"logs/{agent.lower()}_curve_mode{flight_mode}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════

class CurriculumSACCallback(BaseCallback):
    """
    Tracks episode returns, drives the curriculum, and logs to WandB.
    When the curriculum advances, calls model.set_env() with the new env.
    """

    def __init__(self, curriculum, flight_mode):
        super().__init__()
        self.curriculum      = curriculum
        self.flight_mode     = flight_mode
        self.episode_returns = []
        self.stage_at_ep     = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_r = float(info["episode"]["r"])
                self.episode_returns.append(ep_r)
                self.stage_at_ep.append(self.curriculum.num_targets)
                ep_idx = len(self.episode_returns) - 1

                wandb.log({
                    "episode_return"  : ep_r,
                    "curriculum_stage": self.curriculum.num_targets,
                    "success_rate"    : self.curriculum.success_rate,
                    "episode"         : ep_idx,
                })

                advanced = self.curriculum.record(ep_r)
                if advanced:
                    new_env = make_env(self.flight_mode, self.curriculum.num_targets)
                    self.model.set_env(new_env)
                    print(f"  [ep {ep_idx+1}] Curriculum → {self.curriculum.num_targets} waypoints")
                    wandb.log({
                        "curriculum_advance": self.curriculum.num_targets,
                        "episode"           : ep_idx,
                    })

                if (ep_idx + 1) % 50 == 0:
                    print(f"  [SAC] ep {ep_idx+1}  "
                          f"stage={self.curriculum.num_targets}wp  "
                          f"last-50 mean={np.mean(self.episode_returns[-50:]):.2f}  "
                          f"success={self.curriculum.success_rate:.2f}")
        return True


# ═══════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════

for FLIGHT_MODE in MODES:
    print(f"\n{'='*60}")
    print(f"  FLIGHT MODE {FLIGHT_MODE}  ({MODES.index(FLIGHT_MODE)+1}/{len(MODES)})")
    print(f"{'='*60}\n")

    shared_config = {
        "flight_mode"          : FLIGHT_MODE,
        "max_waypoints"        : MAX_WAYPOINTS,
        "proximity_scale"      : PROXIMITY_SCALE,
        "goal_distance"        : GOAL_DISTANCE,
        "dome_size"            : DOME_SIZE,
        "max_duration_sec"     : MAX_DURATION_SEC,
        "curriculum_window"    : CURRICULUM_WINDOW,
        "curriculum_threshold" : CURRICULUM_THRESHOLD,
    }

    # ══════════════════════════════════════════════════════════
    # SAC
    # ══════════════════════════════════════════════════════════
    print(f"\n── SAC  flight_mode={FLIGHT_MODE} ──────────────────────────\n")

    wandb.init(
        project = WANDB_PROJECT,
        name    = f"SAC-curriculum-mode{FLIGHT_MODE}",
        config  = {
            **shared_config,
            "agent"               : "SAC",
            "sac_total_timesteps" : SAC_TOTAL_TIMESTEPS,
            "sac_batch_size"      : SAC_BATCH_SIZE,
            "sac_gamma"           : SAC_GAMMA,
            "sac_tau"             : SAC_TAU,
            "sac_alpha"           : SAC_ALPHA,
            "sac_lr"              : SAC_LR,
            "sac_buffer"          : SAC_BUFFER,
            "sac_learning_starts" : SAC_LEARNING_STARTS,
        }
    )

    sac_curriculum = CurriculumManager(start_targets=1)
    curriculum_cb  = CurriculumSACCallback(sac_curriculum, FLIGHT_MODE)
    wandb_cb       = WandbCallback(gradient_save_freq=0, verbose=0)

    env_sac = make_env(FLIGHT_MODE, sac_curriculum.num_targets)
    sac = SAC(
        policy          = "MlpPolicy",
        env             = env_sac,
        learning_rate   = SAC_LR,
        buffer_size     = SAC_BUFFER,
        batch_size      = SAC_BATCH_SIZE,
        gamma           = SAC_GAMMA,
        tau             = SAC_TAU,
        ent_coef        = SAC_ALPHA,
        learning_starts = SAC_LEARNING_STARTS,
        verbose         = 0,
        seed            = SEED,
    )

    sac.learn(
        total_timesteps = SAC_TOTAL_TIMESTEPS,
        callback        = [curriculum_cb, wandb_cb],
    )
    env_sac.close()

    sac_train_returns = curriculum_cb.episode_returns
    sac_stage_at_ep   = curriculum_cb.stage_at_ep
    print(f"\nSAC done. Stage history: {sac_curriculum.stage_history}")
    print(f"Episodes: {len(sac_train_returns)}")

    curve_path = save_training_curve(
        sac_train_returns, sac_stage_at_ep, sac_curriculum, FLIGHT_MODE, "SAC"
    )

    sac_mean, sac_std, sac_returns = eval_agent(
        lambda obs: sac.predict(obs, deterministic=True)[0],
        "SAC (curriculum)", FLIGHT_MODE
    )
    wandb.log({
        "eval_mean_return": sac_mean,
        "eval_std_return" : sac_std,
        "training_curve"  : wandb.Image(curve_path),
    })

    sac_ckpt = f"checkpoints/sac_waypoints_mode{FLIGHT_MODE}_curriculum"
    sac.save(sac_ckpt)
    wandb.save(sac_ckpt + ".zip")
    wandb.finish()
    print("SAC run closed.")

    # ══════════════════════════════════════════════════════════
    # PPO
    # ══════════════════════════════════════════════════════════
    print(f"\n── PPO  flight_mode={FLIGHT_MODE} ──────────────────────────\n")

    wandb.init(
        project = WANDB_PROJECT,
        name    = f"PPO-curriculum-mode{FLIGHT_MODE}",
        config  = {
            **shared_config,
            "agent"                   : "PPO",
            "ppo_timesteps_per_stage" : PPO_TIMESTEPS_PER_STAGE,
            "ppo_max_retries"         : PPO_MAX_RETRIES_PER_STAGE,
            "ppo_n_steps"             : PPO_N_STEPS,
            "ppo_batch"               : PPO_BATCH,
            "ppo_epochs"              : PPO_EPOCHS,
            "ppo_gamma"               : PPO_GAMMA,
            "ppo_gae"                 : PPO_GAE,
            "ppo_clip"                : PPO_CLIP,
            "ppo_lr"                  : PPO_LR,
            "total_steps"             : PPO_TIMESTEPS_PER_STAGE * MAX_WAYPOINTS,
        }
    )

    ppo_curriculum = CurriculumManager(start_targets=1)
    ppo_total_ts   = 0

    def train_ppo_stage(ppo_model, num_targets, timesteps):
        env_stage = make_env(FLIGHT_MODE, num_targets)
        ppo_model.set_env(env_stage)
        ppo_model.learn(
            total_timesteps     = timesteps,
            reset_num_timesteps = False,
            callback            = WandbCallback(gradient_save_freq=0, verbose=0),
        )
        env_stage.close()
        return ppo_model

    def probe_success_rate(ppo_model, num_targets, n_eps=CURRICULUM_WINDOW):
        env_probe = make_env(FLIGHT_MODE, num_targets)
        successes = []
        for seed_i in range(SEED, SEED + n_eps):
            obs, _ = env_probe.reset(seed=seed_i)
            ep_r   = 0.0
            for _ in range(MAX_STEPS):
                action, _ = ppo_model.predict(obs, deterministic=False)
                obs, r, term, trunc, _ = env_probe.step(action)
                ep_r += r
                if term or trunc:
                    break
            successes.append(float(ep_r > 0))
        env_probe.close()
        return float(np.mean(successes))

    env_ppo_init = make_env(FLIGHT_MODE, ppo_curriculum.num_targets)
    ppo = PPO(
        policy        = "MlpPolicy",
        env           = env_ppo_init,
        learning_rate = PPO_LR,
        n_steps       = PPO_N_STEPS,
        batch_size    = PPO_BATCH,
        n_epochs      = PPO_EPOCHS,
        gamma         = PPO_GAMMA,
        gae_lambda    = PPO_GAE,
        clip_range    = PPO_CLIP,
        verbose       = 0,
        seed          = SEED,
    )
    env_ppo_init.close()

    while ppo_curriculum.num_targets <= MAX_WAYPOINTS:
        current_stage = ppo_curriculum.num_targets

        for attempt in range(1, PPO_MAX_RETRIES_PER_STAGE + 1):
            print(f"\nStage {current_stage} — attempt {attempt}/{PPO_MAX_RETRIES_PER_STAGE} "
                  f"({PPO_TIMESTEPS_PER_STAGE} timesteps) ...")
            ppo           = train_ppo_stage(ppo, current_stage, PPO_TIMESTEPS_PER_STAGE)
            ppo_total_ts += PPO_TIMESTEPS_PER_STAGE

            success_rate = probe_success_rate(ppo, current_stage)
            print(f"  success_rate = {success_rate:.2f}  (threshold = {CURRICULUM_THRESHOLD})")

            wandb.log({
                "stage"       : current_stage,
                "attempt"     : attempt,
                "success_rate": success_rate,
                "total_ts"    : ppo_total_ts,
            })

            if success_rate >= CURRICULUM_THRESHOLD:
                break

        if current_stage < MAX_WAYPOINTS:
            ppo_curriculum.num_targets += 1
            ppo_curriculum.stage_history.append((ppo_total_ts, ppo_curriculum.num_targets))
            print(f"  → Advancing to {ppo_curriculum.num_targets} waypoints")
            wandb.log({"curriculum_advance": ppo_curriculum.num_targets, "total_ts": ppo_total_ts})
        else:
            break

    print(f"\nPPO done. Total timesteps: {ppo_total_ts:,}")

    ppo_mean, ppo_std, ppo_returns = eval_agent(
        lambda obs: ppo.predict(obs, deterministic=True)[0],
        "PPO (curriculum)", FLIGHT_MODE
    )
    wandb.log({"eval_mean_return": ppo_mean, "eval_std_return": ppo_std})

    ppo_ckpt = f"checkpoints/ppo_waypoints_mode{FLIGHT_MODE}_curriculum"
    ppo.save(ppo_ckpt)
    wandb.save(ppo_ckpt + ".zip")
    wandb.finish()
    print("PPO run closed.")

    print(f"\n── Mode {FLIGHT_MODE} summary ───────────────────────────────")
    print(f"  SAC  mean={sac_mean:.2f}  std={sac_std:.2f}")
    print(f"  PPO  mean={ppo_mean:.2f}  std={ppo_std:.2f}")

print("\n" + "="*60)
print("  All flight modes done.")
print("="*60)
