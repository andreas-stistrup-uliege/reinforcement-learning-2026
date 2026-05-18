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
 
# -- reproducibility -------------------------------------------
SEED = 42
np.random.seed(SEED)
 
# -- flight modes to train -------------------------------------
MODES = [7, 6, 4, 0, -1]
 
# -- environment config ----------------------------------------
ENV_ID           = "PyFlyt/QuadX-Waypoints-v4"
MAX_WAYPOINTS    = 4
GOAL_DISTANCE    = 4.0
DOME_SIZE        = 150.0
MAX_DURATION_SEC = 120.0
PROXIMITY_SCALE  = 0.3
 
# -- curriculum config -----------------------------------------
CURRICULUM_THRESHOLD = 0.70
PROBE_EPS            = 10
 
# -- eval config -----------------------------------------------
MAX_STEPS  = 1000
EVAL_EPS   = 10
EVAL_SEEDS = list(range(SEED, SEED + EVAL_EPS))
 
# -- shared budget----------------------------------------------
TIMESTEPS_PER_STAGE = 250_000
 
# -- SAC hyperparameters ---------------------------------------
SAC_BATCH_SIZE      = 256
SAC_GAMMA           = 0.99
SAC_TAU             = 0.005
SAC_ALPHA           = 0.2
SAC_LR              = 3e-4
SAC_BUFFER          = 500_000
SAC_LEARNING_STARTS = 1_000
 
# -- PPO hyperparameters ---------------------------------------
PPO_N_STEPS  = 2048
PPO_BATCH    = 64
PPO_EPOCHS   = 10
PPO_GAMMA    = 0.99
PPO_GAE      = 0.95
PPO_CLIP     = 0.2
PPO_LR       = 3e-4
 
# -- WandB -----------------------------------------------------
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
 
 
def make_env(flight_mode, num_targets=MAX_WAYPOINTS):
    env = gym.make(
        ENV_ID,
        flight_mode          = flight_mode,
        goal_reach_distance  = GOAL_DISTANCE,
        flight_dome_size     = DOME_SIZE,
        max_duration_seconds = MAX_DURATION_SEC,
        num_targets          = num_targets,
        render_mode          = None,
    )
    env = ProximityRewardWrapper(env)
    env = FlattenWaypointEnv(env, max_waypoints=MAX_WAYPOINTS)
    return env
 
 
def probe_success_rate(model, flight_mode, num_targets, n_eps=PROBE_EPS):
    """Run n_eps episodes and return fraction with return > 0."""
    env_probe = make_env(flight_mode, num_targets)
    successes = []
    for seed_i in range(SEED, SEED + n_eps):
        obs, _ = env_probe.reset(seed=seed_i)
        ep_r   = 0.0
        for _ in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=False)
            obs, r, term, trunc, _ = env_probe.step(action)
            ep_r += r
            if term or trunc:
                break
        successes.append(float(ep_r > 0))
    env_probe.close()
    return float(np.mean(successes))
 
 
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
 
 
def save_training_curve(returns, stage_at_ep, stage_history, flight_mode, agent):
    fig, ax = plt.subplots(figsize=(12, 4))
    stage_colors = {1: "#fff3e0", 2: "#e8f5e9", 3: "#e3f2fd", 4: "#f3e5f5"}
    stage_labels = {1: "1 wp", 2: "2 wp", 3: "3 wp", 4: "4 wp"}
    arr = np.array(stage_at_ep) if stage_at_ep else np.array([])
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
# Episode logging callback
# ═══════════════════════════════════════════════════════════════
 
class EpisodeLogCallback(BaseCallback):
    """Logs episode returns and curriculum stage to WandB each episode."""
 
    def __init__(self, stage_ref, label=""):
        super().__init__()
        self.stage_ref       = stage_ref
        self.label           = label
        self.episode_returns = []
        self.stage_at_ep     = []
 
    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_r   = float(info["episode"]["r"])
                ep_idx = len(self.episode_returns)
                self.episode_returns.append(ep_r)
                self.stage_at_ep.append(self.stage_ref[0])
                wandb.log({
                    "episode_return"  : ep_r,
                    "curriculum_stage": self.stage_ref[0],
                    "episode"         : ep_idx,
                })
                if (ep_idx + 1) % 50 == 0:
                    print(f"  [{self.label}] ep {ep_idx+1}  "
                          f"stage={self.stage_ref[0]}wp  "
                          f"last-50 mean={np.mean(self.episode_returns[-50:]):.2f}")
        return True
 
 
# ═══════════════════════════════════════════════════════════════
# Shared curriculum training function
# ═══════════════════════════════════════════════════════════════
 
def train_curriculum(model, flight_mode, label):
    """
    Train model through the waypoint curriculum.
    Stops early at each stage if success rate >= CURRICULUM_THRESHOLD.
    Returns (all_episode_returns, stage_at_ep, stage_history, total_ts).
    """
    current_stage = 1
    total_ts      = 0
    stage_history = []
    stage_ref     = [current_stage]
 
    all_returns  = []
    all_stage_at = []
 
    while current_stage <= MAX_WAYPOINTS:
        print(f"\n  [{label}] Stage {current_stage}/{MAX_WAYPOINTS} "
              f"({TIMESTEPS_PER_STAGE} timesteps) ...")
 
        env_stage  = make_env(flight_mode, current_stage)
        log_cb     = EpisodeLogCallback(stage_ref, label)
        wandb_cb   = WandbCallback(gradient_save_freq=0, verbose=0)
 
        model.set_env(env_stage)
        model.learn(
            total_timesteps     = TIMESTEPS_PER_STAGE,
            reset_num_timesteps = False,
            callback            = [log_cb, wandb_cb],
        )
        env_stage.close()
        total_ts += TIMESTEPS_PER_STAGE
 
        all_returns  += log_cb.episode_returns
        all_stage_at += log_cb.stage_at_ep
 
        # probe success rate
        success_rate = probe_success_rate(model, flight_mode, current_stage)
        print(f"  [{label}] Stage {current_stage} success_rate={success_rate:.2f} "
              f"(threshold={CURRICULUM_THRESHOLD})")
 
        wandb.log({
            "stage"        : current_stage,
            "success_rate" : success_rate,
            "total_ts"     : total_ts,
        })
 
        if success_rate >= CURRICULUM_THRESHOLD and current_stage < MAX_WAYPOINTS:
            current_stage += 1
            stage_ref[0]   = current_stage
            stage_history.append((total_ts, current_stage))
            print(f"  [{label}] → Advancing to {current_stage} waypoints")
            wandb.log({"curriculum_advance": current_stage, "total_ts": total_ts})
        else:
            # threshold not met or already at max stage -> stop
            break
 
    print(f"  [{label}] Curriculum done. Total ts: {total_ts:,}  "
          f"Stage history: {stage_history}")
    return all_returns, all_stage_at, stage_history, total_ts
 
 
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
        "curriculum_threshold" : CURRICULUM_THRESHOLD,
        "probe_eps"            : PROBE_EPS,
        "timesteps_per_stage"  : TIMESTEPS_PER_STAGE,
    }
 
    # ══════════════════════════════════════════════════════════
    # SAC
    # ══════════════════════════════════════════════════════════
    print(f"\n-- SAC  flight_mode={FLIGHT_MODE} --------------------------\n")
 
    wandb.init(
        project = WANDB_PROJECT,
        name    = f"SAC-curriculum-mode{FLIGHT_MODE}-{TIMESTEPS_PER_STAGE}",
        config  = {
            **shared_config,
            "agent"               : "SAC",
            "sac_batch_size"      : SAC_BATCH_SIZE,
            "sac_gamma"           : SAC_GAMMA,
            "sac_tau"             : SAC_TAU,
            "sac_alpha"           : SAC_ALPHA,
            "sac_lr"              : SAC_LR,
            "sac_buffer"          : SAC_BUFFER,
            "sac_learning_starts" : SAC_LEARNING_STARTS,
        }
    )
 
    env_sac_init = make_env(FLIGHT_MODE, 1)
    sac = SAC(
        policy          = "MlpPolicy",
        env             = env_sac_init,
        device          = "cuda",
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
    env_sac_init.close()
 
    sac_returns, sac_stage_at, sac_stage_hist, sac_ts = train_curriculum(
        sac, FLIGHT_MODE, "SAC"
    )
 
    curve_path = save_training_curve(
        sac_returns, sac_stage_at, sac_stage_hist, FLIGHT_MODE, "SAC"
    )
    sac_mean, sac_std, sac_eval = eval_agent(
        lambda obs: sac.predict(obs, deterministic=True)[0],
        "SAC (curriculum)", FLIGHT_MODE
    )
    wandb.log({
        "eval_mean_return": sac_mean,
        "eval_std_return" : sac_std,
        "total_ts"        : sac_ts,
        "training_curve"  : wandb.Image(curve_path),
    })
    sac_ckpt = f"checkpoints/sac_waypoints_mode{FLIGHT_MODE}_curriculum-{TIMESTEPS_PER_STAGE}"
    sac.save(sac_ckpt)
    wandb.save(sac_ckpt + ".zip")
    wandb.finish()
    print("SAC run closed.")
 
    # ══════════════════════════════════════════════════════════
    # PPO
    # ══════════════════════════════════════════════════════════
    print(f"\n-- PPO  flight_mode={FLIGHT_MODE} --------------------------\n")
 
    wandb.init(
        project = WANDB_PROJECT,
        name    = f"PPO-curriculum-mode{FLIGHT_MODE}-{TIMESTEPS_PER_STAGE}",
        config  = {
            **shared_config,
            "agent"       : "PPO",
            "ppo_n_steps" : PPO_N_STEPS,
            "ppo_batch"   : PPO_BATCH,
            "ppo_epochs"  : PPO_EPOCHS,
            "ppo_gamma"   : PPO_GAMMA,
            "ppo_gae"     : PPO_GAE,
            "ppo_clip"    : PPO_CLIP,
            "ppo_lr"      : PPO_LR,
        }
    )
 
    env_ppo_init = make_env(FLIGHT_MODE, 1)
    ppo = PPO(
        policy        = "MlpPolicy",
        env           = env_ppo_init,
        device        = "cuda",
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
 
    ppo_returns, ppo_stage_at, ppo_stage_hist, ppo_ts = train_curriculum(
        ppo, FLIGHT_MODE, "PPO"
    )
 
    curve_path = save_training_curve(
        ppo_returns, ppo_stage_at, ppo_stage_hist, FLIGHT_MODE, "PPO"
    )
    ppo_mean, ppo_std, ppo_eval = eval_agent(
        lambda obs: ppo.predict(obs, deterministic=True)[0],
        "PPO (curriculum)", FLIGHT_MODE
    )
    wandb.log({
        "eval_mean_return": ppo_mean,
        "eval_std_return" : ppo_std,
        "total_ts"        : ppo_ts,
        "training_curve"  : wandb.Image(curve_path),
    })
    ppo_ckpt = f"checkpoints/ppo_waypoints_mode{FLIGHT_MODE}_curriculum-{TIMESTEPS_PER_STAGE}"
    ppo.save(ppo_ckpt)
    wandb.save(ppo_ckpt + ".zip")
    wandb.finish()
    print("PPO run closed.")
 
    print(f"\n-- Mode {FLIGHT_MODE} summary -------------------------------")
    print(f"  SAC  mean={sac_mean:.2f}  std={sac_std:.2f}  ts={sac_ts:,}")
    print(f"  PPO  mean={ppo_mean:.2f}  std={ppo_std:.2f}  ts={ppo_ts:,}")
 
print("\n" + "="*60)
print("  All flight modes done.")
print("="*60)
 