"""
algorithms.py
=============
Implements:
  - FittedQIterationContinuous  (batch, offline, sklearn regressor)
  - SACAgent                    (online, off-policy, custom PyTorch)
  - PPOAgent                    (online, on-policy,  SB3 wrapper)

Shared interface required by evaluate.py / tournament.py:
  model.predict(obs, deterministic=True) -> (action, info)
"""

# ─────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────
import numpy as np
import random
import gymnasium as gym
from collections import deque
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════
# 1.  FITTED Q-ITERATION  (unchanged)
# ══════════════════════════════════════════════════════════════
class FittedQIterationContinuous:
    """
    Offline / batch Q-learning with a sklearn-compatible regression model.

    The action space is continuous; we approximate max_a Q(s,a) by
    sampling n_action_samples random candidate actions and keeping
    the best one.

    Interface note
    --------------
    FQI does NOT expose the tournament `predict()` interface because it
    requires offline training from a fixed dataset.  Use SACAgent or
    PPOAgent for the tournament submission.
    """

    def __init__(
        self,
        model,
        gamma: float,
        action_low: np.ndarray,
        action_high: np.ndarray,
        n_action_samples: int = 50,
    ):
        self.model = model
        self.gamma = gamma
        self.action_low = np.array(action_low)
        self.action_high = np.array(action_high)
        self.action_dim = len(action_low)
        self.n_action_samples = n_action_samples
        self.is_fitted = False

    def sample_actions(self, n=None):
        n = n or self.n_action_samples
        return np.random.uniform(
            self.action_low, self.action_high, size=(n, self.action_dim)
        )

    def predict_Q(self, state: np.ndarray, actions: np.ndarray):
        inputs = [np.concatenate([state, a]) for a in actions]
        return self.model.predict(inputs)

    def max_Q(self, state: np.ndarray):
        if not self.is_fitted:
            return 0.0
        actions = self.sample_actions()
        return np.max(self.predict_Q(state, actions))

    def train(
        self,
        experience_replay: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]],
        max_iterations: int = 10,
    ):
        states = [s for (s, _, _, _) in experience_replay]

        for _ in tqdm(range(max_iterations)):
            inputs, targets = [], []
            for s, a, r, ns in experience_replay:
                inputs.append(np.concatenate([s, a]))
                targets.append(r + self.gamma * self.max_Q(ns))

            previous_pred = np.array([self.max_Q(s) for s in states])
            self.model.fit(inputs, targets)
            self.is_fitted = True
            next_pred = np.array([self.max_Q(s) for s in states])

            if np.max(np.abs(next_pred - previous_pred)) < 1e-3:
                break

    def predict_action(self, state: np.ndarray, n_candidates: int = 100):
        actions = self.sample_actions(n_candidates)
        return actions[np.argmax(self.predict_Q(state, actions))]


# ══════════════════════════════════════════════════════════════
# 2.  SAC — Soft Actor-Critic  (custom PyTorch)
# ══════════════════════════════════════════════════════════════

# ── reward normaliser ─────────────────────────────────────────

class _RunningNormalizer:
    """
    Online Welford mean/variance estimator for scalar rewards.

    Why this is needed
    ------------------
    SAC stores rewards in the replay buffer and the critic regresses on
    them directly.  When the raw reward range is large and asymmetric
    (Hover gives rewards in [-102, +3]), the critic's output scale grows
    large, making the actor loss gradient noisy.  In the worst case the
    critic diverges and the actor converges to a fixed bad policy —
    exactly the collapse observed on modes 0 and 6.

    Normalising each reward to approximately zero mean / unit variance
    before storing it keeps Q-value targets in a stable numerical range
    regardless of the environment's reward scale, without requiring any
    hand-tuned reward-scaling constant per environment.

    The normaliser is updated online with every raw reward seen during
    training (inside `store`).  At evaluation time (`predict` /
    `env_action`) no normalisation is applied — evaluation measures the
    true cumulative return.

    Algorithm
    ---------
    Uses Welford's online algorithm, which computes the exact running
    mean and variance in a single pass and is numerically stable for any
    number of samples:

        delta  = x - mean
        mean  += delta / n
        delta2 = x - mean          # note: uses updated mean
        M2    += delta * delta2
        var    = M2 / n

    The normalised value is clipped to [-10, 10] to prevent pathological
    outliers (e.g. a massive crash penalty on the first step) from
    destabilising the very first gradient updates.
    """

    def __init__(self, eps: float = 1e-8):
        self.mean  = 0.0
        self.var   = 1.0
        self._M2   = 0.0
        self.count = 0
        self.eps   = eps

    def update(self, x: float):
        """Update running statistics with one new scalar reward."""
        self.count += 1
        delta       = x - self.mean
        self.mean  += delta / self.count
        delta2      = x - self.mean
        self._M2   += delta * delta2
        self.var    = self._M2 / self.count if self.count > 1 else 1.0

    def normalize(self, x: float) -> float:
        """Return (x - mean) / std, clipped to [-10, 10] for stability."""
        normed = (x - self.mean) / (np.sqrt(self.var) + self.eps)
        return float(np.clip(normed, -10.0, 10.0))


# ── sub-modules ───────────────────────────────────────────────

class _Actor(nn.Module):
    """
    Gaussian policy.  Outputs (mean, log_std) for each action dimension.
    Actions are squashed through tanh -> live in (-1, 1) before rescaling.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
            nn.Linear(256, action_dim * 2),
        )

    def forward(self, state):
        return self.net(state)


class _CriticNet(nn.Module):
    """
    Q-network Q(s, a).
    Takes state and action as *separate* tensors and concatenates them
    internally.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),                    nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


# ── agent ─────────────────────────────────────────────────────

class SACAgent:
    """
    Soft Actor-Critic for continuous action spaces.

    Key design choices
    ------------------
    * Off-policy: learns from a replay buffer -> sample-efficient.
    * Maximum-entropy objective: the entropy term (controlled by alpha)
      encourages exploration and prevents premature convergence.
    * Two critics (Q1, Q2) + two target critics with soft updates
      -> reduces overestimation bias (clipped double-Q trick).
    * Actor outputs a squashed Gaussian: tanh(Normal(mu, sigma)).
      Actions are then linearly rescaled to the environment's bounds.
    * Reward normalisation (normalize_rewards=True by default): raw
      rewards are normalised to ~N(0,1) before being stored in the
      replay buffer, preventing critic divergence on environments with
      large or asymmetric reward ranges.

    Action representation
    ---------------------
    Internally the actor always works in normalised space (-1, 1).
    `_scale_action` maps that to the environment's [low, high] range.
    The *stored* action in the replay buffer is the normalised one so
    that gradients through the actor remain well-conditioned.

    Reward normalisation
    --------------------
    Pass the *raw* environment reward to `store()`.  The agent calls
    `_reward_norm.update(r)` then `_reward_norm.normalize(r)` and
    stores the result.  `predict()` / `env_action()` are unaffected —
    they only depend on the actor, not the critic or the buffer.

    Tournament interface
    --------------------
    `predict(obs, deterministic=True)` matches the interface expected by
    `scripts/evaluate.py` and `scripts/tournament.py`.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low,
        action_high,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        lr: float = 3e-4,
        buffer_size: int = 1_000_000,
        normalize_rewards: bool = True,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau   = tau
        self.alpha = alpha

        # Action rescaling tensors (kept on device for fast GPU ops)
        self.action_low  = torch.tensor(action_low,  dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32, device=self.device)

        # ── reward normaliser ─────────────────────────────────
        self.normalize_rewards = normalize_rewards
        self._reward_norm = _RunningNormalizer()

        # ── replay buffer ─────────────────────────────────────
        self.buffer: deque = deque(maxlen=buffer_size)

        # ── networks ──────────────────────────────────────────
        self.actor = _Actor(state_dim, action_dim).to(self.device)

        self.q1   = _CriticNet(state_dim, action_dim).to(self.device)
        self.q2   = _CriticNet(state_dim, action_dim).to(self.device)
        self.q1_t = _CriticNet(state_dim, action_dim).to(self.device)
        self.q2_t = _CriticNet(state_dim, action_dim).to(self.device)

        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())

        # ── optimisers ────────────────────────────────────────
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_opt    = optim.Adam(self.q1.parameters(),    lr=lr)
        self.q2_opt    = optim.Adam(self.q2.parameters(),    lr=lr)

    # ── replay buffer ─────────────────────────────────────────

    def store(self, s, a, r: float, ns, d: float):
        """
        Push one transition into the replay buffer.

        Parameters
        ----------
        s  : current observation
        a  : normalised action in (-1, 1)  — from act()
        r  : *raw* environment reward — normalised internally here
        ns : next observation
        d  : 1.0 if done (terminated OR truncated), else 0.0
        """
        if self.normalize_rewards:
            self._reward_norm.update(r)
            r = self._reward_norm.normalize(r)
        self.buffer.append((s, a, float(r), ns, float(d)))

    def _sample_batch(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.array, zip(*batch))
        to = lambda x: torch.FloatTensor(x).to(self.device)
        return (
            to(s),
            to(a),
            to(r).unsqueeze(1),
            to(ns),
            to(d).unsqueeze(1),
        )

    # ── stochastic action (used during training) ──────────────

    def _sample_action(self, state: torch.Tensor):
        """
        Reparameterised sample from the squashed Gaussian policy.

        Returns
        -------
        a    : tanh-squashed action in (-1, 1)
        logp : log-probability (with tanh Jacobian correction)
        """
        x = self.actor(state)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        a = torch.tanh(z)

        logp = dist.log_prob(z) - torch.log(1.0 - a.pow(2) + 1e-6)
        logp = logp.sum(dim=-1, keepdim=True)

        return a, logp

    # ── action rescaling ──────────────────────────────────────

    def _scale_action(self, a: torch.Tensor) -> torch.Tensor:
        """Map normalised action from (-1, 1) to environment bounds."""
        return self.action_low + (a + 1.0) * 0.5 * (self.action_high - self.action_low)

    # ── acting API ────────────────────────────────────────────

    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Return normalised action in (-1, 1)."""
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                x = self.actor(s)
                mean, _ = torch.chunk(x, 2, dim=-1)
                a = torch.tanh(mean)
            else:
                a, _ = self._sample_action(s)
        return a.cpu().numpy()[0]

    def env_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """act() + rescale to environment bounds in one call."""
        a = self.act(state, deterministic=deterministic)
        return self._scale_action(torch.tensor(a, device=self.device)).cpu().numpy()

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        """
        Tournament / evaluate.py interface.
        Returns (action_in_env_bounds, {}).
        Reward normalisation does NOT affect this path.
        """
        return self.env_action(obs, deterministic=deterministic), {}

    # ── gradient update ───────────────────────────────────────

    def update(self, batch_size: int = 256):
        """One gradient step on a random mini-batch from the replay buffer."""
        if len(self.buffer) < batch_size:
            return

        s, a, r, ns, d = self._sample_batch(batch_size)

        with torch.no_grad():
            na, logp = self._sample_action(ns)
            q_next = torch.min(self.q1_t(ns, na), self.q2_t(ns, na)) - self.alpha * logp
            target = r + (1.0 - d) * self.gamma * q_next

        q1_loss = ((self.q1(s, a) - target) ** 2).mean()
        self.q1_opt.zero_grad(); q1_loss.backward(); self.q1_opt.step()

        q2_loss = ((self.q2(s, a) - target) ** 2).mean()
        self.q2_opt.zero_grad(); q2_loss.backward(); self.q2_opt.step()

        na, logp = self._sample_action(s)
        q_pi = torch.min(self.q1(s, na), self.q2(s, na))
        actor_loss = (self.alpha * logp - q_pi).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        self._soft_update(self.q1_t, self.q1)
        self._soft_update(self.q2_t, self.q2)

    def _soft_update(self, target: nn.Module, source: nn.Module):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(self.tau * s.data + (1.0 - self.tau) * t.data)


# ══════════════════════════════════════════════════════════════
# 3.  PPO — Proximal Policy Optimisation  (SB3 wrapper)
# ══════════════════════════════════════════════════════════════

class PPOAgent:
    """
    Proximal Policy Optimisation via stable-baselines3.

    Why PPO vs SAC?
    ---------------
    * On-policy: collects fresh rollouts each update -> no replay buffer,
      no off-policy corrections needed.
    * Uses a clipped surrogate objective to limit the size of each policy
      update (the "proximal" part), which makes training stable.
    * Generally faster to get reasonable behaviour, but less
      sample-efficient than SAC because old data is discarded.

    Tournament interface
    --------------------
    `predict(obs, deterministic=True)` delegates directly to SB3's PPO,
    which already implements this signature.
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        verbose: int = 1,
        seed: int = 0,
        device: str = "auto",
    ):
        from stable_baselines3 import PPO
        self._model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            verbose=verbose,
            seed=seed,
            device=device,
        )

    def train(self, total_timesteps: int = 200_000):
        self._model.learn(total_timesteps=total_timesteps)

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        return self._model.predict(obs, deterministic=deterministic)

    def save(self, path: str):
        self._model.save(path)

    @classmethod
    def load(cls, path: str, env: gym.Env = None):
        from stable_baselines3 import PPO
        instance = cls.__new__(cls)
        instance._model = PPO.load(path, env=env)
        return instance


# ══════════════════════════════════════════════════════════════
# 4.  EVALUATION UTILITIES
# ══════════════════════════════════════════════════════════════

def evaluate_policy(env, policy_fn, episodes: int = 10, max_steps: int = 500, seed: int = 0):
    """
    Evaluate any callable policy over multiple episodes.

    Parameters
    ----------
    env       : Gymnasium environment
    policy_fn : callable  obs -> action
    episodes  : number of evaluation episodes
    max_steps : maximum steps per episode
    seed      : base random seed (episode i uses seed+i)

    Returns
    -------
    (mean_return, std_return)
    """
    returns = []
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        total = 0.0
        for _ in range(max_steps):
            action = policy_fn(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))


def run_comparison(models: dict, env_name: str, modes: list, episodes: int = 10):
    """
    Evaluate multiple named policies across several flight modes.

    Parameters
    ----------
    models   : {name: policy_fn}  where policy_fn(obs) -> action
    env_name : Gymnasium env id
    modes    : list of flight_mode integers
    episodes : episodes per (model, mode) pair

    Returns
    -------
    results[mode][name] = (mean, std)
    """
    results = {}
    for mode in modes:
        env = gym.make(env_name, flight_mode=mode)
        results[mode] = {}
        for name, policy_fn in models.items():
            mean, std = evaluate_policy(env, policy_fn, episodes=episodes)
            results[mode][name] = (mean, std)
            print(f"Mode {mode:>2} | {name:<20} : {mean:8.2f} +/- {std:.2f}")
        env.close()
    return results
