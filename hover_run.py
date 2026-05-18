import numpy as np
import gymnasium as gym
import PyFlyt.gym_envs
import matplotlib.pyplot as plt
import wandb
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback
from wandb.integration.sb3 import WandbCallback

FLIGHT_MODES = [7, 6, 4, 0, -1]

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

    # -- WandB -----------------------------------------------------
    WANDB_PROJECT = "hover"

    shared_config = {
        "env_id"      : ENV_ID,
        "flight_mode" : flight_mode,
        "max_steps"   : MAX_STEPS,
        "eval_eps"    : EVAL_EPS,
    }

    print(f"Environment : {ENV_ID}  |  flight_mode = {flight_mode}")

    def eval_agent(policy_fn, label, n_episodes=EVAL_EPS, seeds=EVAL_SEEDS, max_steps=MAX_STEPS):
        """
        Evaluate policy_fn(obs) -> action on n_episodes with fixed seeds.
        Returns (mean, std, list_of_returns).
        """
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

    class EpisodeReturnCallback(BaseCallback):
        """Collects per-episode returns and logs them to WandB."""

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

    ep_return_cb = EpisodeReturnCallback()
    wandb_cb     = WandbCallback(gradient_save_freq=0, verbose=0)

    sac.learn(
        total_timesteps = SAC_TOTAL_TIMESTEPS,
        callback        = [ep_return_cb, wandb_cb],
    )
    env_sac.close()

    sac_train_returns = ep_return_cb.episode_returns
    print(f"\nSAC training done. Episodes: {len(sac_train_returns)}  "
        f"Last-10 mean: {np.mean(sac_train_returns[-10:]):.2f}")


    def smooth(x, w=20):
        return np.convolve(x, np.ones(w) / w, mode='valid')


    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sac_train_returns, alpha=0.25, color='steelblue', label='raw')
    ax.plot(range(len(smooth(sac_train_returns))), smooth(sac_train_returns),
            color='steelblue', lw=2, label='smoothed (w=20)')
    ax.axhline(rand_mean, color='grey', linestyle='--', label=f'random baseline ({rand_mean:.1f})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode return')
    ax.set_title(f'SAC training curve — {ENV_ID}  flight_mode={flight_mode}')
    ax.legend()
    plt.tight_layout()
    wandb.log({"training_curve": wandb.Image(fig)})
    plt.show()


    sac_mean, sac_std, sac_returns = eval_agent(
        lambda obs: sac.predict(obs, deterministic=True)[0], "SAC"
    )

    wandb.log({"eval_mean_return": sac_mean, "eval_std_return": sac_std})

    import os
    os.makedirs("checkpoints", exist_ok=True)

    sac_ckpt_path = f"checkpoints/sac_hover_mode{flight_mode}"
    sac.save(sac_ckpt_path)
    wandb.save(sac_ckpt_path + ".zip")

    wandb.finish()
    print("SAC run closed.")


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
        policy      = "MlpPolicy",
        env         = env_ppo,
        learning_rate = PPO_LR,
        n_steps     = PPO_N_STEPS,
        batch_size  = PPO_BATCH,
        n_epochs    = PPO_EPOCHS,
        gamma       = PPO_GAMMA,
        gae_lambda  = PPO_GAE,
        clip_range  = PPO_CLIP,
        verbose     = 0,
        seed        = SEED,
    )

    ppo.learn(
        total_timesteps = PPO_TIMESTEPS,
        callback        = WandbCallback(gradient_save_freq=0, verbose=0),
    )
    env_ppo.close()
    print("PPO training done.")


    ppo_mean, ppo_std, ppo_returns = eval_agent(
        lambda obs: ppo.predict(obs, deterministic=True)[0], "PPO"
    )

    wandb.log({"eval_mean_return": ppo_mean, "eval_std_return": ppo_std})

    ppo_ckpt_path = f"checkpoints/ppo_hover_mode{flight_mode}"
    ppo.save(ppo_ckpt_path)
    wandb.save(ppo_ckpt_path + ".zip")

    wandb.finish()
    print("PPO run closed.")


    print(f"{'Agent':<12} {'Mean return':>14} {'Std':>10} {'95% CI':>22}")
    print("-" * 62)
    for label, mean, std, rets in [
        ("Random", rand_mean, rand_std, rand_returns),
        ("SAC",    sac_mean,  sac_std,  sac_returns),
        ("PPO",    ppo_mean,  ppo_std,  ppo_returns),
    ]:
        n     = len(rets)
        se    = std / np.sqrt(n)
        ci_lo = mean - 1.96 * se
        ci_hi = mean + 1.96 * se
        print(f"{label:<12} {mean:>14.2f} {std:>10.2f}  [{ci_lo:9.2f}, {ci_hi:9.2f}]")


    labels = ["Random", "SAC", "PPO"]
    means  = [rand_mean, sac_mean, ppo_mean]
    stds   = [rand_std,  sac_std,  ppo_std]
    colors = ["#aaaaaa", "#4c8cbf", "#e07b3a"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # bar chart
    bars = axes[0].bar(labels, means, yerr=stds, capsize=6,
                    color=colors, edgecolor='black', linewidth=0.8)
    axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[0].set_ylabel('Mean episode return')
    axes[0].set_title(f'{ENV_ID}  flight_mode={flight_mode}\n(±1 std, n={EVAL_EPS})')
    for bar, mean, std in zip(bars, means, stds):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                    mean + std + 1, f'{mean:.1f}', ha='center', va='bottom', fontsize=9)

    # box plot
    bp = axes[1].boxplot(
        [rand_returns, sac_returns, ppo_returns],
        labels=labels, patch_artist=True,
        medianprops=dict(color='black', linewidth=2),
    )
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[1].set_ylabel('Episode return')
    axes[1].set_title('Return distribution')

    plt.tight_layout()
    plt.show()


    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sac_train_returns, alpha=0.2, color='steelblue')
    ax.plot(smooth(sac_train_returns), color='steelblue', lw=2, label='SAC (smoothed)')
    ax.axhline(ppo_mean,  color='#e07b3a', lw=2, label=f'PPO eval mean ({ppo_mean:.1f})')
    ax.axhline(rand_mean, color='grey', linestyle='--', lw=1, label=f'Random ({rand_mean:.1f})')
    ax.set_xlabel('SAC episode')
    ax.set_ylabel('Episode return')
    ax.set_title(f'SAC training curve vs PPO & random — {ENV_ID}  flight_mode={flight_mode}')
    ax.legend()
    plt.tight_layout()
    plt.show()


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


    print("Run from the terminal:")
    print(f"  python scripts/evaluate.py --model checkpoints/sac_hover_mode{flight_mode}.zip "
        f"--env hover --flight_mode {flight_mode}")
    print(f"  python scripts/evaluate.py --model checkpoints/ppo_hover_mode{flight_mode}.zip "
        f"--env hover --flight_mode {flight_mode}")
