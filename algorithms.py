import numpy as np
from typing import List, Tuple
from tqdm import tqdm

# Fitteq Q Iteration
class FittedQIterationContinuous:
    def __init__(
        self,
        model,
        gamma: float,
        action_low: np.ndarray,
        action_high: np.ndarray,
        n_action_samples: int = 50,
    ):
        """
        FQI adapted for continuous UAV action spaces.

        Parameters:
        - model: regression model (e.g. sklearn)
        - gamma: discount factor
        - action_low: lower bound of action space (shape: [4])
        - action_high: upper bound of action space (shape: [4])
        - n_action_samples: number of samples to approximate max over actions
        """
        self.model = model
        self.gamma = gamma

        self.action_low = np.array(action_low)
        self.action_high = np.array(action_high)

        self.action_dim = len(action_low)
        self.n_action_samples = n_action_samples

        self.is_fitted = False

    # -------------------------
    # Sampling actions
    # -------------------------
    def sample_actions(self, n=None):
        n = n or self.n_action_samples
        return np.random.uniform(
            self.action_low,
            self.action_high,
            size=(n, self.action_dim),
        )

    # -------------------------
    # Q prediction
    # -------------------------
    def predict_Q(self, state: np.ndarray, actions: np.ndarray):   
        inputs = [np.concatenate([state, a]) for a in actions]
        return self.model.predict(inputs)

    # -------------------------
    # Max over actions (approx)
    # -------------------------
    def max_Q(self, state: np.ndarray):
        if not self.is_fitted:
            return 0.0

        actions = self.sample_actions()
        q_vals = self.predict_Q(state, actions)
        return np.max(q_vals)

    # -------------------------
    # Training
    # -------------------------
    def train(
        self,
        experience_replay: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]],
        max_iterations: int = 10,
    ):
        """
        experience_replay: (state, action_vector, reward, next_state)
        """


        states = [s for (s, _, _, _) in experience_replay]

        for _ in tqdm(range(max_iterations)):
            inputs = []
            targets = []

            for s, a, r, ns in experience_replay:
                inputs.append(np.concatenate([s, a]))

                target = r + self.gamma * self.max_Q(ns)
                targets.append(target)

            # convergence check (approximate)
            previous_pred = np.array([
                self.max_Q(s) for s in states
            ])

            # Fit regression model
            self.model.fit(inputs, targets)
            self.is_fitted = True

            next_pred = np.array([
                self.max_Q(s) for s in states
            ])

            delta = np.max(np.abs(next_pred - previous_pred))
            if delta < 1e-3:
                break
        
    # -------------------------
    # Action selection
    # -------------------------
    def predict_action(self, state: np.ndarray, n_candidates: int = 100):
        """
        Approximate argmax_a Q(s,a) via sampling
        """
        actions = self.sample_actions(n_candidates)
        q_vals = self.predict_Q(state, actions)
        return actions[np.argmax(q_vals)]


# SAC
# https://www.researchgate.net/publication/396370894_Reinforcement_learning_for_UAV_flight_controls_Evaluating_continuous_space_reinforcement_learning_algorithms_for_fixed-wing_UAVs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


# =========================================================
# SAC Agent (single class)
# =========================================================
class SACAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        action_low,
        action_high,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        lr=3e-4,
        buffer_size=1_000_000,
        device=None,
    ):

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.action_low = torch.tensor(action_low, dtype=torch.float32).to(self.device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32).to(self.device)

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # -------------------------
        # Replay buffer
        # -------------------------
        self.buffer = deque(maxlen=buffer_size)

        # -------------------------
        # Actor
        # -------------------------
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim * 2),
        ).to(self.device)

        # -------------------------
        # Critics
        # -------------------------
        def critic():
            return nn.Sequential(
                nn.Linear(state_dim + action_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            ).to(self.device)

        self.q1 = critic()
        self.q2 = critic()
        self.q1_t = critic()
        self.q2_t = critic()

        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())

        # -------------------------
        # Optimizers
        # -------------------------
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=lr)

    # =====================================================
    # Replay buffer
    # =====================================================
    def store(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.array, zip(*batch))

        return (
            torch.FloatTensor(s).to(self.device),
            torch.FloatTensor(a).to(self.device),
            torch.FloatTensor(r).unsqueeze(1).to(self.device),
            torch.FloatTensor(ns).to(self.device),
            torch.FloatTensor(d).unsqueeze(1).to(self.device),
        )

    # =====================================================
    # Action sampling
    # =====================================================
    def _sample_action(self, state):
        x = self.actor(state)
        mean, log_std = torch.chunk(x, 2, dim=-1)

        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()

        a = torch.tanh(z)

        logp = dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
        logp = logp.sum(dim=-1, keepdim=True)

        return a, logp

    # =====================================================
    # Action scaling to environment
    # =====================================================
    def _scale_action(self, a):
        return self.action_low + (a + 1) * 0.5 * (
            self.action_high - self.action_low
        )

    # =====================================================
    # Acting
    # =====================================================
    def act(self, state, eval=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if eval:
                x = self.actor(state)
                mean, _ = torch.chunk(x, 2, dim=-1)
                a = torch.tanh(mean)
            else:
                a, _ = self._sample_action(state)

        return a.cpu().numpy()[0]

    # =====================================================
    # Learning step
    # =====================================================
    def update(self, batch_size=256):
        if len(self.buffer) < batch_size:
            return

        s, a, r, ns, d = self.sample(batch_size)

        # -------------------------
        # Target Q
        # -------------------------
        with torch.no_grad():
            na, logp = self._sample_action(ns)

            q1_t = self.q1_t(ns, na)
            q2_t = self.q2_t(ns, na)
            q_t = torch.min(q1_t, q2_t) - self.alpha * logp

            target = r + (1 - d) * self.gamma * q_t

        # -------------------------
        # Critic loss
        # -------------------------
        q1_loss = ((self.q1(s, a) - target) ** 2).mean()
        q2_loss = ((self.q2(s, a) - target) ** 2).mean()

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        # -------------------------
        # Actor loss
        # -------------------------
        na, logp = self._sample_action(s)
        q_pi = torch.min(self.q1(s, na), self.q2(s, na))

        actor_loss = (self.alpha * logp - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # -------------------------
        # Soft update
        # -------------------------
        self._soft_update(self.q1_t, self.q1)
        self._soft_update(self.q2_t, self.q2)

    def _soft_update(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)

    # =====================================================
    # Environment-safe action
    # =====================================================
    def env_action(self, state):
        a = self.act(state)
        return self._scale_action(torch.tensor(a)).cpu().numpy()


### EVALUATION
# - on the same env;
# - same # of episodes
# - multiple random seeds
# - report mean and std
# - identical evaluation function
import numpy as np

def evaluate_policy(env, policy_fn, episodes=10, max_steps=500, seed=0):

    returns = []

    for ep in range(episodes):

        state, _ = env.reset(seed=seed + ep)
        total = 0

        for _ in range(max_steps):

            action = policy_fn(state)

            state, reward, terminated, truncated, _ = env.step(action)

            total += reward

            if terminated or truncated:
                break

        returns.append(total)

    return np.mean(returns), np.std(returns)

def run_comparison(models, env_name, modes, episodes=10):

    results = {}

    for mode in modes:

        env = gym.make(env_name, flight_mode=mode)

        results[mode] = {}

        for name, policy_fn in models.items():

            mean, std = evaluate_policy(env, policy_fn, episodes=episodes)

            results[mode][name] = (mean, std)

            print(f"Mode {mode} | {name}: {mean:.2f} ± {std:.2f}")

        env.close()

    return results