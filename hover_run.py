import numpy as np
import gymnasium as gym
import PyFlyt.gym_envs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb
import os
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback
from wandb.integration.sb3 import WandbCallback

FLIGHT_MODES = [7, 6, 4, 0, -1]

# ── Agarwal et al. (2021) statistical tools ───────────────────────────────────

def bootstrap_ci(returns, n_bootstrap=2000, confidence=0.95):
    """
    Stratified bootstrap confidence interval for the mean.
    Returns (mean, ci_low, ci_high).
    """
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
    """
    Interquartile Mean - mean of the middle 50% of episodes.
    Less sensitive to outlier crashes than the plain mean.
    """
    returns = np.sort(np.array(returns, dtype=float))
    n  = len(returns)
    lo = int(np.floor(0.25 * n))
    hi = int(np.ceil(0.75 * n))
    return float(np.mean(returns[lo:hi]))


def probability_of_improvement(returns_a, returns_b, n_bootstrap=2000):
    """
    P(A > B) estimated via bootstrap over all (a, b) pairs.
    Returns (point_estimate, ci_lo, ci_hi).
    """
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
    """
    CDF-style plot: fraction of episodes scoring >= tau, for a range of tau.
    """
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


def plot_eval_charts(agent_label, agent_color, agent_returns, rand_returns,
                     rand_mean, eval_eps, env_id, flight_mode):
    """
    Bar (bootstrap CI) + box plot + IQM chart for one agent vs random.
    Returns the figure.
    """
    all_rets  = [rand_returns, agent_returns]
    labels    = ["Random", agent_label]
    means     = [float(np.mean(rand_returns)), float(np.mean(agent_returns))]
    colors    = ["#aaaaaa", agent_color]

    ci_los  = [bootstrap_ci(r)[1] for r in all_rets]
    ci_his  = [bootstrap_ci(r)[2] for r in all_rets]
    yerr_lo = [m - lo for m, lo in zip(means, ci_los)]
    yerr_hi = [hi - m  for m, hi in zip(means, ci_his)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # Bar chart with bootstrap CIs
    bars = axes[0].bar(labels, means, yerr=[yerr_lo, yerr_hi], capsize=6,
                       color=colors, edgecolor='black', linewidth=0.8)
    axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[0].set_ylabel('Mean episode return')
    axes[0].set_title(f'{env_id}  flight_mode={flight_mode}\n(bootstrap 95% CI, n={eval_eps})')
    for bar, mean in zip(bars, means):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1, f'{mean:.1f}',
                     ha='center', va='bottom', fontsize=9)

    # Box plot
    bp = axes[1].boxplot(all_rets, labels=labels, patch_artist=True,
                         medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[1].set_ylabel('Episode return')
    axes[1].set_title('Return distribution')

    # IQM bar chart
    iqms = [iqm(r) for r in all_rets]
    axes[2].bar(labels, iqms, color=colors, edgecolor='black', linewidth=0.8)
    axes[2].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[2].set_ylabel('IQM episode return')
    axes[2].set_title('Interquartile Mean (Agarwal et al. 2021)')
    for i, val in enumerate(iqms):
        axes[2].text(i, val + 0.5, f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def plot_training_curve(train_returns, rand_mean, other_mean, other_label,
                        agent_label, agent_color, other_color,
                        env_id, flight_mode, smooth_fn):
    """
    Training curve for agent_label vs the other agent's eval mean and random baseline.
    Returns the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    if train_returns:
        ax.plot(train_returns, alpha=0.2, color=agent_color)
        if len(train_returns) >= 20:
            ax.plot(smooth_fn(train_returns), color=agent_color,
                    lw=2, label=f'{agent_label} (smoothed)')
    else:
        ax.text(0.5, 0.5, 'No episodes completed', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='grey')
    ax.axhline(other_mean,  color=other_color, lw=2,
               label=f'{other_label} eval mean ({other_mean:.1f})')
    ax.axhline(rand_mean, color='grey', linestyle='--', lw=1,
               label=f'Random ({rand_mean:.1f})')
    ax.set_xlabel(f'{agent_label} episode')
    ax.set_ylabel('Episode return')
    ax.set_title(f'{agent_label} training curve vs {other_label} & random'
                 f' - {env_id}  flight_mode={flight_mode}')
    ax.legend()
    plt.tight_layout()
    return fig


def print_stats_table(entries):
    """entries: list of (label, returns). Prints mean, IQM, bootstrap 95% CI."""
    print(f"\n{'Agent':<10} {'Mean':>8} {'IQM':>8} {'95% CI (bootstrap)':>25}")
    print("-" * 55)
    for label, rets in entries:
        mean, ci_lo, ci_hi = bootstrap_ci(rets)
        i = iqm(rets)
        print(f"{label:<10} {mean:>8.2f} {i:>8.2f}  [{ci_lo:9.2f}, {ci_hi:9.2f}]")


# ── main loop ─────────────────────────────────────────────────────────────────

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

for flight_mode in FLIGHT_MODES:

    # -- reproducibility ------------------------------------------
    SEED = 42
    np.random.seed(SEED)

    # -- environment ----------------------------------------------
    ENV_ID      = "PyFlyt/QuadX-Hover-v4"

    # -- eval config ----------------------------------------------
    MAX_STEPS  = 500
    EVAL_EPS   = 10
    EVAL_SEEDS = list(range(SEED, SEED + EVAL_EPS))

    # -- SAC hyperparameters --------------------------------------
    SAC_TOTAL_TIMESTEPS = 150_000
    SAC_BATCH_SIZE      = 256
    SAC_GAMMA           = 0.99
    SAC_TAU             = 0.005
    SAC_ALPHA           = 0.2
    SAC_LR              = 3e-4
    SAC_BUFFER          = 500_000
    SAC_LEARNING_STARTS = 1_000

    # -- PPO hyperparameters --------------------------------------
    PPO_TIMESTEPS = 150_000
    PPO_N_STEPS   = 2048
    PPO_BATCH     = 64
    PPO_EPOCHS    = 10
    PPO_GAMMA     = 0.99
    PPO_GAE       = 0.95
    PPO_CLIP      = 0.2
    PPO_LR        = 3e-4

    # -- WandB ----------------------------------------------------
    WANDB_PROJECT = "hover"

    shared_config = {
        "env_id"      : ENV_ID,
        "flight_mode" : flight_mode,
        "max_steps"   : MAX_STEPS,
        "eval_eps"    : EVAL_EPS,
    }

    def smooth(x, w=20):
        return np.convolve(x, np.ones(w) / w, mode='valid')

    print(f"Environment : {ENV_ID}  |  flight_mode = {flight_mode}")

    def eval_agent(policy_fn, label, n_episodes=EVAL_EPS, seeds=EVAL_SEEDS, max_steps=MAX_STEPS):
        env     = gym.make(ENV_ID, flight_mode=flight_mode)
        returns = []
        for i, seed in enumerate(seeds):
            obs, _ = env.reset(seed=seed)
            total  = 0.0
            for _ in range(max_steps):
                action = policy_fn(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            returns.append(total)
            print(f"  [{label}] episode {i+1}/{n_episodes}  return={total:.2f}")
        env.close()
        mean, std = float(np.mean(returns)), float(np.std(returns))
        print(f"  {label:<16}  mean={mean:8.2f}  std={std:7.2f}")
        return mean, std, returns

    _env_tmp      = gym.make(ENV_ID, flight_mode=flight_mode)
    _action_space = _env_tmp.action_space
    _env_tmp.close()

    rand_mean, rand_std, rand_returns = eval_agent(
        lambda obs: _action_space.sample(), "Random"
    )

    class EpisodeReturnCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episode_returns = []

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                if "episode" in info:
                    ep_r = float(info["episode"]["r"])
                    self.episode_returns.append(ep_r)
                    wandb.log({
                        "episode_return": ep_r,
                        "episode"       : len(self.episode_returns),
                    })
            return True

    # ══════════════════════════════════════════════════════════════
    # SAC
    # ══════════════════════════════════════════════════════════════
    wandb.init(
        project = WANDB_PROJECT,
        name    = f"SAC-mode{flight_mode}",
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
    print(f"WandB SAC run : {wandb.run.name}  ({wandb.run.url})")

    env_sac = gym.make(ENV_ID, flight_mode=flight_mode)
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

    sac_ep_cb = EpisodeReturnCallback()
    sac.learn(
        total_timesteps = SAC_TOTAL_TIMESTEPS,
        callback        = [sac_ep_cb, WandbCallback(gradient_save_freq=0, verbose=0)],
    )
    env_sac.close()

    sac_train_returns = sac_ep_cb.episode_returns
    print(f"\nSAC training done. Episodes: {len(sac_train_returns)}  "
          f"Last-10 mean: {np.mean(sac_train_returns[-10:]):.2f}")

    # SAC standalone training curve
    fig, ax = plt.subplots(figsize=(10, 4))
    if sac_train_returns:
        ax.plot(sac_train_returns, alpha=0.25, color='steelblue', label='raw')
        if len(sac_train_returns) >= 20:
            ax.plot(range(len(smooth(sac_train_returns))), smooth(sac_train_returns),
                    color='steelblue', lw=2, label='smoothed (w=20)')
    else:
        ax.text(0.5, 0.5, 'No episodes completed', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='grey')
    ax.axhline(rand_mean, color='grey', linestyle='--', label=f'random baseline ({rand_mean:.1f})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode return')
    ax.set_title(f'SAC training curve - {ENV_ID}  flight_mode={flight_mode}')
    ax.legend()
    plt.tight_layout()
    wandb.log({"training_curve": wandb.Image(fig)})
    plt.savefig(f"logs/hover_sac_training_mode{flight_mode}.png", dpi=120)
    plt.close(fig)

    sac_mean, sac_std, sac_returns = eval_agent(
        lambda obs: sac.predict(obs, deterministic=True)[0], "SAC"
    )
    sac_boot_mean, sac_ci_lo, sac_ci_hi = bootstrap_ci(sac_returns)
    sac_iqm = iqm(sac_returns)

    # SAC eval charts (bar + box + IQM)
    fig_sac_eval = plot_eval_charts(
        "SAC", "steelblue", sac_returns, rand_returns,
        rand_mean, EVAL_EPS, ENV_ID, flight_mode
    )
    wandb.log({"eval_charts": wandb.Image(fig_sac_eval)})
    fig_sac_eval.savefig(f"logs/hover_sac_eval_mode{flight_mode}.png", dpi=120)
    plt.close(fig_sac_eval)

    # SAC performance profile
    fig_sac_pp = plot_performance_profile(
        {"Random": rand_returns, "SAC": sac_returns},
        title=f"SAC Performance Profile - {ENV_ID}  flight_mode={flight_mode}",
    )
    wandb.log({"performance_profile": wandb.Image(fig_sac_pp)})
    fig_sac_pp.savefig(f"logs/hover_sac_perf_profile_mode{flight_mode}.png", dpi=120)
    plt.close(fig_sac_pp)

    wandb.log({
        "eval_mean_return" : sac_mean,
        "eval_std_return"  : sac_std,
        "eval_iqm"         : sac_iqm,
        "eval_ci_lo"       : sac_ci_lo,
        "eval_ci_hi"       : sac_ci_hi,
    })

    sac_ckpt_path = f"checkpoints/sac_hover_mode{flight_mode}"
    sac.save(sac_ckpt_path)
    wandb.save(sac_ckpt_path + ".zip")
    wandb.finish()
    print("SAC run closed.")

    # ══════════════════════════════════════════════════════════════
    # PPO
    # ══════════════════════════════════════════════════════════════
    wandb.init(
        project = WANDB_PROJECT,
        name    = f"PPO-mode{flight_mode}",
        config  = {
            **shared_config,
            "agent"        : "PPO",
            "ppo_timesteps": PPO_TIMESTEPS,
            "ppo_n_steps"  : PPO_N_STEPS,
            "ppo_batch"    : PPO_BATCH,
            "ppo_epochs"   : PPO_EPOCHS,
            "ppo_gamma"    : PPO_GAMMA,
            "ppo_gae"      : PPO_GAE,
            "ppo_clip"     : PPO_CLIP,
            "ppo_lr"       : PPO_LR,
        }
    )
    print(f"WandB PPO run : {wandb.run.name}  ({wandb.run.url})")

    env_ppo = gym.make(ENV_ID, flight_mode=flight_mode)
    ppo = PPO(
        policy        = "MlpPolicy",
        env           = env_ppo,
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

    ppo_ep_cb = EpisodeReturnCallback()
    ppo.learn(
        total_timesteps = PPO_TIMESTEPS,
        callback        = [ppo_ep_cb, WandbCallback(gradient_save_freq=0, verbose=0)],
    )
    env_ppo.close()
    print("PPO training done.")

    ppo_train_returns = ppo_ep_cb.episode_returns

    # PPO standalone training curve
    fig, ax = plt.subplots(figsize=(10, 4))
    if ppo_train_returns:
        ax.plot(ppo_train_returns, alpha=0.25, color='#e07b3a', label='raw')
        if len(ppo_train_returns) >= 20:
            ax.plot(range(len(smooth(ppo_train_returns))), smooth(ppo_train_returns),
                    color='#e07b3a', lw=2, label='smoothed (w=20)')
    else:
        ax.text(0.5, 0.5, 'No episodes completed', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='grey')
    ax.axhline(rand_mean, color='grey', linestyle='--', label=f'random baseline ({rand_mean:.1f})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode return')
    ax.set_title(f'PPO training curve - {ENV_ID}  flight_mode={flight_mode}')
    ax.legend()
    plt.tight_layout()
    wandb.log({"training_curve": wandb.Image(fig)})
    plt.savefig(f"logs/hover_ppo_training_mode{flight_mode}.png", dpi=120)
    plt.close(fig)

    ppo_mean, ppo_std, ppo_returns = eval_agent(
        lambda obs: ppo.predict(obs, deterministic=True)[0], "PPO"
    )
    ppo_boot_mean, ppo_ci_lo, ppo_ci_hi = bootstrap_ci(ppo_returns)
    ppo_iqm = iqm(ppo_returns)

    # PPO eval charts (bar + box + IQM)
    fig_ppo_eval = plot_eval_charts(
        "PPO", "#e07b3a", ppo_returns, rand_returns,
        rand_mean, EVAL_EPS, ENV_ID, flight_mode
    )
    wandb.log({"eval_charts": wandb.Image(fig_ppo_eval)})
    fig_ppo_eval.savefig(f"logs/hover_ppo_eval_mode{flight_mode}.png", dpi=120)
    plt.close(fig_ppo_eval)

    # PPO performance profile
    fig_ppo_pp = plot_performance_profile(
        {"Random": rand_returns, "PPO": ppo_returns},
        title=f"PPO Performance Profile - {ENV_ID}  flight_mode={flight_mode}",
    )
    wandb.log({"performance_profile": wandb.Image(fig_ppo_pp)})
    fig_ppo_pp.savefig(f"logs/hover_ppo_perf_profile_mode{flight_mode}.png", dpi=120)
    plt.close(fig_ppo_pp)

    wandb.log({
        "eval_mean_return" : ppo_mean,
        "eval_std_return"  : ppo_std,
        "eval_iqm"         : ppo_iqm,
        "eval_ci_lo"       : ppo_ci_lo,
        "eval_ci_hi"       : ppo_ci_hi,
    })

    ppo_ckpt_path = f"checkpoints/ppo_hover_mode{flight_mode}"
    ppo.save(ppo_ckpt_path)
    wandb.save(ppo_ckpt_path + ".zip")
    wandb.finish()
    print("PPO run closed.")

    # ══════════════════════════════════════════════════════════════
    # Post-run combined analysis (Agarwal et al. 2021)
    # ══════════════════════════════════════════════════════════════

    # 1. Stats table: mean, IQM, bootstrap 95% CI
    print_stats_table([
        ("Random", rand_returns),
        ("SAC",    sac_returns),
        ("PPO",    ppo_returns),
    ])

    # 2. Probability of improvement
    p_sac_ppo, p_lo,  p_hi  = probability_of_improvement(sac_returns, ppo_returns)
    p_ppo_sac, p_lo2, p_hi2 = probability_of_improvement(ppo_returns, sac_returns)
    print(f"\nP(SAC > PPO) = {p_sac_ppo:.2f}  95% CI=[{p_lo:.2f}, {p_hi:.2f}]")
    print(f"P(PPO > SAC) = {p_ppo_sac:.2f}  95% CI=[{p_lo2:.2f}, {p_hi2:.2f}]")

    # 3. Combined performance profile (Random + SAC + PPO)
    fig_combined_pp = plot_performance_profile(
        {"Random": rand_returns, "SAC": sac_returns, "PPO": ppo_returns},
        title=f"Performance Profile - {ENV_ID}  flight_mode={flight_mode}",
    )
    fig_combined_pp.savefig(f"logs/hover_perf_profile_mode{flight_mode}.png", dpi=120)
    plt.close(fig_combined_pp)

    # 4. SAC training curve vs PPO eval mean & random
    fig_sac_vs_ppo = plot_training_curve(
        sac_train_returns, rand_mean, ppo_mean, "PPO",
        "SAC", "steelblue", "#e07b3a",
        ENV_ID, flight_mode, smooth
    )
    fig_sac_vs_ppo.savefig(f"logs/hover_sac_vs_ppo_mode{flight_mode}.png", dpi=120)
    plt.close(fig_sac_vs_ppo)

    # 5. PPO training curve vs SAC eval mean & random
    fig_ppo_vs_sac = plot_training_curve(
        ppo_train_returns, rand_mean, sac_mean, "SAC",
        "PPO", "#e07b3a", "steelblue",
        ENV_ID, flight_mode, smooth
    )
    fig_ppo_vs_sac.savefig(f"logs/hover_ppo_vs_sac_mode{flight_mode}.png", dpi=120)
    plt.close(fig_ppo_vs_sac)

    # 6. Reload sanity check
    env_reload = gym.make(ENV_ID, flight_mode=flight_mode)

    sac_loaded = SAC.load(f"checkpoints/sac_hover_mode{flight_mode}", env=env_reload)
    sac_rl_mean, _, _ = eval_agent(
        lambda obs: sac_loaded.predict(obs, deterministic=True)[0], "SAC (reloaded)"
    )

    ppo_loaded = PPO.load(f"checkpoints/ppo_hover_mode{flight_mode}", env=env_reload)
    ppo_rl_mean, _, _ = eval_agent(
        lambda obs: ppo_loaded.predict(obs, deterministic=True)[0], "PPO (reloaded)"
    )

    env_reload.close()

    print(f"\nSanity check:")
    print(f"  SAC  original={sac_mean:.2f}  reloaded={sac_rl_mean:.2f}")
    print(f"  PPO  original={ppo_mean:.2f}  reloaded={ppo_rl_mean:.2f}")

    print(f"\nRun from the terminal:")
    print(f"  python scripts/evaluate.py --model checkpoints/sac_hover_mode{flight_mode}.zip "
          f"--env hover --flight_mode {flight_mode}")
    print(f"  python scripts/evaluate.py --model checkpoints/ppo_hover_mode{flight_mode}.zip "
          f"--env hover --flight_mode {flight_mode}")
