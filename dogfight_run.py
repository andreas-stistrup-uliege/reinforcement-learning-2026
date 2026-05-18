import copy
import os
import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque
import wandb
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback
from wandb.integration.sb3 import WandbCallback

from scripts.dogfight_wrapper import DogfightSelfPlayEnv

# -- reproducibility -------------------------------------------
SEED = 42
np.random.seed(SEED)

# -- obs / action dims -----------------------------------------
_env_tmp   = DogfightSelfPlayEnv()
STATE_DIM  = _env_tmp.observation_space.shape[0]
ACTION_DIM = _env_tmp.action_space.shape[0]
ACTION_LOW  = _env_tmp.action_space.low
ACTION_HIGH = _env_tmp.action_space.high
_env_tmp.close()

# -- eval config -----------------------------------------------
MAX_STEPS = 1800
EVAL_EPS  = 10

# -- self-play config ------------------------------------------
OPPONENT_UPDATE_FREQ      = 50
PPO_OPPONENT_UPDATE_STEPS = 180_000
SNAPSHOT_POOL_SIZE        = 5

# -- reward shaping --------------------------------------------
VEL_SLICE     = slice(7, 10)
REL_SLICE     = slice(26, 29)
CLOSING_SCALE = 0.01
HEADING_SCALE = 0.005

# -- SAC hyperparameters  --------------------------------------
SAC_TOTAL_TIMESTEPS = 900_000
SAC_BATCH_SIZE      = 256
SAC_GAMMA           = 0.99
SAC_TAU             = 0.005
SAC_ALPHA           = 0.1 
SAC_LR              = 3e-4
SAC_BUFFER          = 500_000
SAC_LEARNING_STARTS = 1_000

# -- PPO hyperparameters  --------------------------------------
PPO_TIMESTEPS = 900_000
PPO_N_STEPS   = 4096
PPO_BATCH     = 128
PPO_EPOCHS    = 10
PPO_GAMMA     = 0.99
PPO_GAE       = 0.95
PPO_CLIP      = 0.2
PPO_LR        = 3e-4
PPO_ENT_COEF  = 0.01

# -- WandB -----------------------------------------------------
WANDB_PROJECT = "dogfight"

shared_config = {
    "state_dim"                : STATE_DIM,
    "action_dim"               : ACTION_DIM,
    "max_steps"                : MAX_STEPS,
    "opponent_update_freq"     : OPPONENT_UPDATE_FREQ,
    "ppo_opponent_update_steps": PPO_OPPONENT_UPDATE_STEPS,
    "snapshot_pool_size"       : SNAPSHOT_POOL_SIZE,
    "closing_scale"            : CLOSING_SCALE,
    "heading_scale"            : HEADING_SCALE,
}

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

print(f"State dim  : {STATE_DIM}")
print(f"Action dim : {ACTION_DIM}")
print(f"SAC steps  : {SAC_TOTAL_TIMESTEPS:,}")
print(f"PPO steps  : {PPO_TIMESTEPS:,}")


# ═══════════════════════════════════════════════════════════════
# Reward shaping wrapper
# ═══════════════════════════════════════════════════════════════

class DogfightShapingWrapper(gym.Wrapper):
    def __init__(self, env, closing_scale=CLOSING_SCALE, heading_scale=HEADING_SCALE):
        super().__init__(env)
        self.closing_scale = closing_scale
        self.heading_scale = heading_scale
        self._prev_dist    = None

    @staticmethod
    def _dist(obs):
        return float(np.linalg.norm(obs[REL_SLICE]))

    @staticmethod
    def _heading(obs):
        vel, rel = obs[VEL_SLICE], obs[REL_SLICE]
        vn, rn = np.linalg.norm(vel), np.linalg.norm(rel)
        if vn < 1e-6 or rn < 1e-6:
            return 0.0
        return float(np.dot(vel / vn, rel / rn))

    def reset(self, **kwargs):
        obs, info       = self.env.reset(**kwargs)
        self._prev_dist = self._dist(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        curr_dist = self._dist(obs)
        if self._prev_dist is not None:
            reward += self.closing_scale * (self._prev_dist - curr_dist)
        reward += self.heading_scale * self._heading(obs)
        self._prev_dist = curr_dist
        return obs, reward, terminated, truncated, info

    def set_opponent_policy(self, policy):
        self.env.set_opponent_policy(policy)


# ═══════════════════════════════════════════════════════════════
# Self-play
# ═══════════════════════════════════════════════════════════════

class FrozenPolicy:
    """
    Frozen copy of an SB3 agent's policy.
    Exposes .predict(obs, deterministic) for set_opponent_policy().
    Works for both SAC and PPO since both are SB3.
    """

    def __init__(self, agent):
        self._policy = copy.deepcopy(agent.policy)
        self._policy.set_training_mode(False)

    def predict(self, obs, deterministic=True):
        return self._policy.predict(obs, deterministic=deterministic)


class SnapshotPool:
    def __init__(self, env, max_size=SNAPSHOT_POOL_SIZE, update_freq=OPPONENT_UPDATE_FREQ):
        self.env         = env
        self.max_size    = max_size
        self.update_freq = update_freq
        self._pool       = deque(maxlen=max_size)
        self.n_updates   = 0

    def maybe_update(self, episode, agent):
        if (episode + 1) % self.update_freq != 0:
            return False
        self._pool.append(FrozenPolicy(agent))
        opponent = self._pool[np.random.randint(len(self._pool))]
        self.env.set_opponent_policy(opponent)
        self.n_updates += 1
        return True

    @property
    def pool_size(self):
        return len(self._pool)


# ═══════════════════════════════════════════════════════════════
# Env factory & eval
# ═══════════════════════════════════════════════════════════════

def make_env():
    env = DogfightSelfPlayEnv()
    env = DogfightShapingWrapper(env)
    return env


def eval_agent(policy_fn, label, n_episodes=EVAL_EPS):
    env, returns = make_env(), []
    for i in range(n_episodes):
        obs, _ = env.reset()
        total  = 0.0
        for _ in range(MAX_STEPS):
            action = policy_fn(obs)
            obs, r, terminated, truncated, _ = env.step(action)
            total += r
            if terminated or truncated:
                break
        returns.append(total)
        print(f"  [{label}] ep {i+1}/{n_episodes}  return={total:.2f}")
    env.close()
    mean, std = float(np.mean(returns)), float(np.std(returns))
    win_rate  = float(np.mean([r > 0 for r in returns]))
    print(f"  {label:<20}  mean={mean:8.2f}  std={std:7.2f}  win_rate={win_rate:.2f}")
    return mean, std, returns, win_rate


def smooth(x, w=20):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def save_curve(returns, label):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(returns, alpha=0.2, color="steelblue", label="raw")
    if len(returns) >= 20:
        ax.plot(range(len(smooth(returns))), smooth(returns),
                color="steelblue", lw=2, label="smoothed (w=20)")
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode return (shaped)")
    ax.set_title(f"{label} — Dogfight self-play snapshots")
    ax.legend()
    plt.tight_layout()
    path = f"logs/{label.lower()}_curve_dogfight.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════

class DogfightSACCallback(BaseCallback):
    """
    Handles episode logging, win rate tracking, and self-play
    snapshot updates for SB3 SAC in the dogfight env.
    """

    def __init__(self, pool):
        super().__init__()
        self.pool            = pool
        self.episode_returns = []
        self.win_history     = deque(maxlen=50)

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_r    = float(info["episode"]["r"])
                ep_idx  = len(self.episode_returns)
                self.episode_returns.append(ep_r)
                self.win_history.append(float(ep_r > 0))

                updated = self.pool.maybe_update(ep_idx, self.model)

                wandb.log({
                    "episode_return": ep_r,
                    "win_rate"      : float(np.mean(self.win_history)),
                    "snapshot_pool" : self.pool.pool_size,
                    "episode"       : ep_idx,
                })
                if updated:
                    wandb.log({"opponent_update": self.pool.n_updates, "episode": ep_idx})
                    print(f"  [ep {ep_idx+1}] Opponent updated (pool={self.pool.pool_size})")

                if (ep_idx + 1) % 50 == 0:
                    print(f"  [SAC] ep {ep_idx+1}  "
                          f"last-50 mean={np.mean(self.episode_returns[-50:]):.2f}  "
                          f"win_rate={float(np.mean(self.win_history)):.2f}  "
                          f"pool={self.pool.pool_size}")
        return True


class SelfPlaySnapshotCallback(BaseCallback):
    """Updates opponent snapshot every update_freq timesteps for PPO."""

    def __init__(self, env, update_freq=PPO_OPPONENT_UPDATE_STEPS,
                 pool_size=SNAPSHOT_POOL_SIZE, verbose=0):
        super().__init__(verbose)
        self.env         = env
        self.update_freq = update_freq
        self._pool       = deque(maxlen=pool_size)
        self._ep_returns = deque(maxlen=50)
        self.n_updates   = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_r = float(info["episode"]["r"])
                self._ep_returns.append(ep_r)
                wandb.log({
                    "episode_return": ep_r,
                    "win_rate"      : float(np.mean([r > 0 for r in self._ep_returns])),
                    "timestep"      : self.num_timesteps,
                })

        if self.num_timesteps > 0 and self.num_timesteps % self.update_freq == 0:
            frozen = FrozenPolicy(self.model)
            self._pool.append(frozen)
            opponent = self._pool[np.random.randint(len(self._pool))]
            self.env.set_opponent_policy(opponent)
            self.n_updates += 1
            wandb.log({
                "opponent_update": self.n_updates,
                "snapshot_pool"  : len(self._pool),
                "timestep"       : self.num_timesteps,
            })
            print(f"  [PPO ts={self.num_timesteps}] Opponent updated (pool={len(self._pool)})")
        return True


# ═══════════════════════════════════════════════════════════════
# SAC
# ═══════════════════════════════════════════════════════════════
print("\n-- SAC -------------------------------------------------\n")

wandb.init(
    project = WANDB_PROJECT,
    name    = "SAC-selfplay-snapshots",
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

env_sac  = make_env()
sac_pool = SnapshotPool(env_sac)
sac_cb   = DogfightSACCallback(sac_pool)
wandb_cb = WandbCallback(gradient_save_freq=0, verbose=0)

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
    callback        = [sac_cb, wandb_cb],
)
env_sac.close()

sac_train_returns = sac_cb.episode_returns
print(f"\nSAC done. Episodes: {len(sac_train_returns)}"
      f"  Last-20 mean: {np.mean(sac_train_returns[-20:]):.2f}")

curve_path = save_curve(sac_train_returns, "SAC")

sac_mean, sac_std, sac_returns, sac_wr = eval_agent(
    lambda obs: sac.predict(obs, deterministic=True)[0], "SAC"
)
wandb.log({
    "eval_mean_return": sac_mean,
    "eval_std_return" : sac_std,
    "eval_win_rate"   : sac_wr,
    "training_curve"  : wandb.Image(curve_path),
})

sac_ckpt = "checkpoints/sac_dogfight"
sac.save(sac_ckpt)
wandb.save(sac_ckpt + ".zip")
wandb.finish()
print("SAC run closed.")


# ═══════════════════════════════════════════════════════════════
# PPO
# ═══════════════════════════════════════════════════════════════
print("\n-- PPO -------------------------------------------------\n")

wandb.init(
    project = WANDB_PROJECT,
    name    = "PPO-selfplay-snapshots",
    config  = {
        **shared_config,
        "agent"                    : "PPO",
        "ppo_timesteps"            : PPO_TIMESTEPS,
        "ppo_n_steps"              : PPO_N_STEPS,
        "ppo_batch"                : PPO_BATCH,
        "ppo_epochs"               : PPO_EPOCHS,
        "ppo_gamma"                : PPO_GAMMA,
        "ppo_gae"                  : PPO_GAE,
        "ppo_clip"                 : PPO_CLIP,
        "ppo_lr"                   : PPO_LR,
        "ppo_ent_coef"             : PPO_ENT_COEF,
        "ppo_opponent_update_steps": PPO_OPPONENT_UPDATE_STEPS,
    }
)

env_ppo     = make_env()
snapshot_cb = SelfPlaySnapshotCallback(env=env_ppo, verbose=1)
wandb_cb    = WandbCallback(gradient_save_freq=0, verbose=0)

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
    ent_coef      = PPO_ENT_COEF,
    verbose       = 0,
    seed          = SEED,
)

ppo.learn(
    total_timesteps     = PPO_TIMESTEPS,
    callback            = [snapshot_cb, wandb_cb],
    reset_num_timesteps = True,
)
env_ppo.close()
print("PPO training done.")

ppo_mean, ppo_std, ppo_returns, ppo_wr = eval_agent(
    lambda obs: ppo.predict(obs, deterministic=True)[0], "PPO"
)
wandb.log({
    "eval_mean_return": ppo_mean,
    "eval_std_return" : ppo_std,
    "eval_win_rate"   : ppo_wr,
})

ppo_ckpt = "checkpoints/ppo_dogfight"
ppo.save(ppo_ckpt)
wandb.save(ppo_ckpt + ".zip")
wandb.finish()
print("PPO run closed.")


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  Dogfight training complete.")
print(f"  SAC  mean={sac_mean:.2f}  std={sac_std:.2f}  win_rate={sac_wr:.2f}")
print(f"  PPO  mean={ppo_mean:.2f}  std={ppo_std:.2f}  win_rate={ppo_wr:.2f}")
print("="*60)
