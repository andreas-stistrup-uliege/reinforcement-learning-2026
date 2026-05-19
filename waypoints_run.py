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
# Agarwal et al. (2021) statistical tools
# ═══════════════════════════════════════════════════════════════

def bootstrap_ci(returns, n_bootstrap=2000, confidence=0.95):
    returns = np.array(returns)
    boot_means = [
        np.mean(np.random.choice(returns, size=len(returns), replace=True))
        for _ in range(n_bootstrap)
    ]
    alpha = (1 - confidence) / 2
    return (
        float(np.mean(returns)),
        float(np.percentile(boot_means, 100 * alpha)),
        float(np.percentile(boot_means, 100 * (1 - alpha))),
    )


def iqm(returns):
    returns = np.sort(np.array(returns, dtype=float))
    n  = len(returns)
    lo = int(np.floor(0.25 * n))
    hi = int(np.ceil(0.75 * n))
    return float(np.mean(returns[lo:hi]))


def probability_of_improvement(returns_a, returns_b, n_bootstrap=2000):
    a = np.array(returns_a)
    b = np.array(returns_b)
    point = float(np.mean(a[:, None] > b[None, :]))
    boot = []
    for _ in range(n_bootstrap):
        sa = np.random.choice(a, size=len(a), replace=True)
        sb = np.random.choice(b, size=len(b), replace=True)
        boot.append(np.mean(sa[:, None] > sb[None, :]))
    return point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def plot_performance_profile(returns_dict, title="Performance Profile"):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"SAC": "#4c8cbf", "PPO": "#e07b3a", "Random": "#aaaaaa"}
    all_vals = np.concatenate(list(returns_dict.values()))
    taus = np.linspace(all_vals.min(), all_vals.max(), 300)
    for label, rets in returns_dict.items():
        rets = np.array(rets)
        frac = [np.mean(rets >= tau) for tau in taus]
        ax.plot(taus, frac, label=label, color=colors.get(label, None), lw=2)
    ax.axhline(0.5, color="black", lw=0.8, linestyle="--", alpha=0.5, label="50% line")
    ax.set_xlabel("Score threshold tau")
    ax.set_ylabel("Fraction of episodes with return >= tau")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_eval_charts(agent_label, agent_color, agent_returns,
                     other_label, other_color, other_returns,
                     eval_eps, flight_mode):
    """Bar (bootstrap CI) + box plot + IQM for agent vs other."""
    all_rets = [agent_returns, other_returns]
    labels   = [agent_label, other_label]
    means    = [float(np.mean(r)) for r in all_rets]
    colors   = [agent_color, other_color]

    ci_los  = [bootstrap_ci(r)[1] for r in all_rets]
    ci_his  = [bootstrap_ci(r)[2] for r in all_rets]
    yerr_lo = [m - lo for m, lo in zip(means, ci_los)]
    yerr_hi = [hi - m  for m, hi in zip(means, ci_his)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    bars = axes[0].bar(labels, means, yerr=[yerr_lo, yerr_hi], capsize=6,
                       color=colors, edgecolor='black', linewidth=0.8)
    axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[0].set_ylabel('Mean episode return')
    axes[0].set_title(f'Waypoints eval  flight_mode={flight_mode}\n'
                      f'(bootstrap 95% CI, n={eval_eps})')
    for bar, mean in zip(bars, means):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5, f'{mean:.1f}',
                     ha='center', va='bottom', fontsize=9)

    bp = axes[1].boxplot(all_rets, labels=labels, patch_artist=True,
                         medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[1].set_ylabel('Episode return')
    axes[1].set_title('Return distribution')

    iqms = [iqm(r) for r in all_rets]
    axes[2].bar(labels, iqms, color=colors, edgecolor='black', linewidth=0.8)
    axes[2].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[2].set_ylabel('IQM episode return')
    axes[2].set_title('Interquartile Mean (Agarwal et al. 2021)')
    for i, val in enumerate(iqms):
        axes[2].text(i, val + 0.5, f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def plot_training_curve_vs(train_returns, other_mean, other_label,
                            agent_label, agent_color, other_color,
                            flight_mode):
    """Training curve for agent vs other agent's eval mean."""
    fig, ax = plt.subplots(figsize=(10, 4))
    if train_returns:
        ax.plot(train_returns, alpha=0.2, color=agent_color)
        if len(train_returns) >= 20:
            smoothed = np.convolve(train_returns, np.ones(20) / 20, mode='valid')
            ax.plot(smoothed, color=agent_color, lw=2, label=f'{agent_label} (smoothed)')
    else:
        ax.text(0.5, 0.5, 'No episodes completed', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='grey')
    ax.axhline(other_mean, color=other_color, lw=2,
               label=f'{other_label} eval mean ({other_mean:.1f})')
    ax.axhline(0, color='black', lw=0.8, linestyle='--', label='zero')
    ax.set_xlabel(f'{agent_label} episode')
    ax.set_ylabel('Episode return (shaped)')
    ax.set_title(f'{agent_label} training curve vs {other_label} - '
                 f'{ENV_ID}  flight_mode={flight_mode}')
    ax.legend()
    plt.tight_layout()
    return fig


def print_stats_table(entries, flight_mode):
    print(f"\nflight_mode={flight_mode}")
    print(f"{'Agent':<10} {'Mean':>8} {'IQM':>8} {'95% CI (bootstrap)':>25}")
    print("-" * 55)
    for label, rets in entries:
        mean, ci_lo, ci_hi = bootstrap_ci(rets)
        i = iqm(rets)
        print(f"{label:<10} {mean:>8.2f} {i:>8.2f}  [{ci_lo:9.2f}, {ci_hi:9.2f}]")


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


def save_training_curve(returns, stage_at_ep, stage_history, flight_mode, agent, color="steelblue"):
    fig, ax1 = plt.subplots(figsize=(12, 4))

    # background stage shading
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
                ax1.axvspan(prev_ep, end, color=stage_colors[prev_stage], alpha=0.5, label=lbl)
                if lbl:
                    labeled.add(prev_stage)
                prev_ep, prev_stage = i, stage

    # raw + smoothed returns (exactly like hover)
    if returns:
        ax1.plot(returns, alpha=0.25, color=color, label='raw')
        if len(returns) >= 20:
            ax1.plot(range(len(smooth(returns))), smooth(returns),
                     color=color, lw=2, label='smoothed (w=20)')
    else:
        ax1.text(0.5, 0.5, 'No episodes completed', transform=ax1.transAxes,
                 ha='center', va='center', fontsize=12, color='grey')

    ax1.axhline(0, color="black", lw=0.8, linestyle="--")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode return (shaped)")
    ax1.set_title(f"{agent} curriculum - {ENV_ID}  flight_mode={flight_mode}")

    # stage step line on right axis
    if len(arr) > 0:
        ax2 = ax1.twinx()
        ax2.step(range(len(arr)), arr, color="black", lw=1.2,
                 linestyle="--", alpha=0.6, where="post", label="stage")
        ax2.set_ylabel("Curriculum stage (# waypoints)")
        ax2.set_yticks([1, 2, 3, 4])
        ax2.set_ylim(0.5, 4.5)
        ax2.legend(loc="center right", fontsize=8)

    ax1.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    path = f"logs/{agent.lower()}_curve_mode{flight_mode}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════
# Episode logging callback
# ═══════════════════════════════════════════════════════════════

class EpisodeLogCallback(BaseCallback):
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
    current_stage = 1
    total_ts      = 0
    stage_history = []
    stage_ref     = [current_stage]
    all_returns   = []
    all_stage_at  = []

    while current_stage <= MAX_WAYPOINTS:
        print(f"\n  [{label}] Stage {current_stage}/{MAX_WAYPOINTS} "
              f"({TIMESTEPS_PER_STAGE} timesteps) ...")

        env_stage = make_env(flight_mode, current_stage)
        log_cb    = EpisodeLogCallback(stage_ref, label)
        wandb_cb  = WandbCallback(gradient_save_freq=0, verbose=0)

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
            print(f"  [{label}] -> Advancing to {current_stage} waypoints")
            wandb.log({"curriculum_advance": current_stage, "total_ts": total_ts})
        else:
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

    sac_train_returns, sac_stage_at, sac_stage_hist, sac_ts = train_curriculum(
        sac, FLIGHT_MODE, "SAC"
    )

    # SAC standalone training curve
    sac_curve_path = save_training_curve(
        sac_train_returns, sac_stage_at, sac_stage_hist, FLIGHT_MODE, "SAC", color="steelblue"
    )

    sac_mean, sac_std, sac_eval = eval_agent(
        lambda obs: sac.predict(obs, deterministic=True)[0],
        "SAC (curriculum)", FLIGHT_MODE
    )
    sac_boot_mean, sac_ci_lo, sac_ci_hi = bootstrap_ci(sac_eval)
    sac_iqm = iqm(sac_eval)

    # SAC performance profile
    fig_sac_pp = plot_performance_profile(
        {"SAC": sac_eval},
        title=f"SAC Performance Profile - {ENV_ID}  flight_mode={FLIGHT_MODE}",
    )
    wandb.log({
        "eval_mean_return"   : sac_mean,
        "eval_std_return"    : sac_std,
        "eval_iqm"           : sac_iqm,
        "eval_ci_lo"         : sac_ci_lo,
        "eval_ci_hi"         : sac_ci_hi,
        "total_ts"           : sac_ts,
        "training_curve"     : wandb.Image(sac_curve_path),
        "performance_profile": wandb.Image(fig_sac_pp),
    })
    fig_sac_pp.savefig(f"logs/sac_perf_profile_mode{FLIGHT_MODE}.png", dpi=120)
    plt.close(fig_sac_pp)

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

    ppo_train_returns, ppo_stage_at, ppo_stage_hist, ppo_ts = train_curriculum(
        ppo, FLIGHT_MODE, "PPO"
    )

    # PPO standalone training curve
    ppo_curve_path = save_training_curve(
        ppo_train_returns, ppo_stage_at, ppo_stage_hist, FLIGHT_MODE, "PPO", color="#e07b3a"
    )

    ppo_mean, ppo_std, ppo_eval = eval_agent(
        lambda obs: ppo.predict(obs, deterministic=True)[0],
        "PPO (curriculum)", FLIGHT_MODE
    )
    ppo_boot_mean, ppo_ci_lo, ppo_ci_hi = bootstrap_ci(ppo_eval)
    ppo_iqm = iqm(ppo_eval)

    # PPO performance profile
    fig_ppo_pp = plot_performance_profile(
        {"PPO": ppo_eval},
        title=f"PPO Performance Profile - {ENV_ID}  flight_mode={FLIGHT_MODE}",
    )
    wandb.log({
        "eval_mean_return"   : ppo_mean,
        "eval_std_return"    : ppo_std,
        "eval_iqm"           : ppo_iqm,
        "eval_ci_lo"         : ppo_ci_lo,
        "eval_ci_hi"         : ppo_ci_hi,
        "total_ts"           : ppo_ts,
        "training_curve"     : wandb.Image(ppo_curve_path),
        "performance_profile": wandb.Image(fig_ppo_pp),
    })
    fig_ppo_pp.savefig(f"logs/ppo_perf_profile_mode{FLIGHT_MODE}.png", dpi=120)
    plt.close(fig_ppo_pp)

    ppo_ckpt = f"checkpoints/ppo_waypoints_mode{FLIGHT_MODE}_curriculum-{TIMESTEPS_PER_STAGE}"
    ppo.save(ppo_ckpt)
    wandb.save(ppo_ckpt + ".zip")
    wandb.finish()
    print("PPO run closed.")

    # ══════════════════════════════════════════════════════════
    # Post-run combined analysis (Agarwal et al. 2021)
    # ══════════════════════════════════════════════════════════

    # 1. Stats table
    print_stats_table([("SAC", sac_eval), ("PPO", ppo_eval)], FLIGHT_MODE)

    # 2. Probability of improvement
    p_sac_ppo, p_lo,  p_hi  = probability_of_improvement(sac_eval, ppo_eval)
    p_ppo_sac, p_lo2, p_hi2 = probability_of_improvement(ppo_eval, sac_eval)
    print(f"\nP(SAC > PPO) = {p_sac_ppo:.2f}  95% CI=[{p_lo:.2f}, {p_hi:.2f}]")
    print(f"P(PPO > SAC) = {p_ppo_sac:.2f}  95% CI=[{p_lo2:.2f}, {p_hi2:.2f}]")

    # 3. Combined eval charts (SAC vs PPO)
    fig_eval = plot_eval_charts(
        "SAC", "steelblue", sac_eval,
        "PPO", "#e07b3a",  ppo_eval,
        EVAL_EPS, FLIGHT_MODE
    )
    fig_eval.savefig(f"logs/waypoints_eval_mode{FLIGHT_MODE}.png", dpi=120)
    plt.close(fig_eval)

    # 4. Combined performance profile
    fig_combined_pp = plot_performance_profile(
        {"SAC": sac_eval, "PPO": ppo_eval},
        title=f"Performance Profile - {ENV_ID}  flight_mode={FLIGHT_MODE}",
    )
    fig_combined_pp.savefig(f"logs/waypoints_perf_profile_mode{FLIGHT_MODE}.png", dpi=120)
    plt.close(fig_combined_pp)

    # 5. SAC training curve vs PPO eval mean
    fig_sac_vs_ppo = plot_training_curve_vs(
        sac_train_returns, ppo_mean, "PPO",
        "SAC", "steelblue", "#e07b3a", FLIGHT_MODE
    )
    fig_sac_vs_ppo.savefig(f"logs/sac_vs_ppo_mode{FLIGHT_MODE}.png", dpi=120)
    plt.close(fig_sac_vs_ppo)

    # 6. PPO training curve vs SAC eval mean
    fig_ppo_vs_sac = plot_training_curve_vs(
        ppo_train_returns, sac_mean, "SAC",
        "PPO", "#e07b3a", "steelblue", FLIGHT_MODE
    )
    fig_ppo_vs_sac.savefig(f"logs/ppo_vs_sac_mode{FLIGHT_MODE}.png", dpi=120)
    plt.close(fig_ppo_vs_sac)

    print(f"\n-- Mode {FLIGHT_MODE} summary -------------------------------")
    print(f"  SAC  mean={sac_mean:.2f}  std={sac_std:.2f}  IQM={sac_iqm:.2f}  "
          f"ts={sac_ts:,}  95% CI=[{sac_ci_lo:.2f}, {sac_ci_hi:.2f}]")
    print(f"  PPO  mean={ppo_mean:.2f}  std={ppo_std:.2f}  IQM={ppo_iqm:.2f}  "
          f"ts={ppo_ts:,}  95% CI=[{ppo_ci_lo:.2f}, {ppo_ci_hi:.2f}]")

print("\n" + "="*60)
print("  All flight modes done.")
print("="*60)
