"""Gymnasium wrapper for PyFlyt MAFixedwingDogfightEnvV2 self-play training.

Wraps the PettingZoo ParallelEnv into a standard Gymnasium env by controlling
one agent while opponents use a frozen copy of the policy (self-play).
"""

import gymnasium
import numpy as np


class DogfightSelfPlayEnv(gymnasium.Env):
    """Wraps MAFixedwingDogfightEnvV2 as a single-agent Gymnasium env for self-play.

    Controls agent 'uav_0' (team 0) with the RL policy.
    All other agents use a frozen opponent policy (updated periodically).

    Handles pybullet disconnection by recreating the PZ env when needed.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        team_size=1,
        opponent_policy=None,
        flatten_observation=True,
        render_mode=None,
        **env_kwargs,
    ):
        super().__init__()
        self.team_size = team_size
        self.opponent_policy = opponent_policy
        self.render_mode = render_mode
        self.flatten_observation = flatten_observation
        self.env_kwargs = env_kwargs

        # Our controlled agent
        self.agent_id = "uav_0"

        # Create initial PZ env
        self.pz_env = self._make_pz_env()

        # Observation and action spaces from PZ env
        self.observation_space = self.pz_env.observation_space(self.agent_id)
        self.action_space = self.pz_env.action_space(self.agent_id)

        self._last_obs = {}
        self._all_agents = None

    def _make_pz_env(self):
        """Create a fresh PZ environment instance."""
        from PyFlyt.pz_envs import MAFixedwingDogfightEnvV2
        return MAFixedwingDogfightEnvV2(
            team_size=self.team_size,
            assisted_flight=True,
            flatten_observation=self.flatten_observation,
            render_mode=self.render_mode,
            **self.env_kwargs,
        )

    def _get_opponent_action(self, obs):
        """Get action for an opponent agent."""
        if self.opponent_policy is not None:
            action, _ = self.opponent_policy.predict(obs, deterministic=False)
            return action
        else:
            return self.action_space.sample()

    def reset(self, seed=None, options=None):
        # Try reset; if pybullet disconnected, recreate the env
        for attempt in range(3):
            try:
                observations, infos = self.pz_env.reset(seed=seed)
                self._all_agents = list(self.pz_env.agents)
                self._last_obs = observations
                obs = observations.get(self.agent_id)
                info = infos.get(self.agent_id, {})
                return obs, info
            except Exception:
                # pybullet disconnected — recreate env
                try:
                    self.pz_env.close()
                except Exception:
                    pass
                self.pz_env = self._make_pz_env()

        # Final attempt without catching
        observations, infos = self.pz_env.reset(seed=seed)
        self._all_agents = list(self.pz_env.agents)
        self._last_obs = observations
        return observations.get(self.agent_id), infos.get(self.agent_id, {})

    def step(self, action):
        # Build actions for all alive agents
        actions = {}
        for agent in self.pz_env.agents:
            if agent == self.agent_id:
                actions[agent] = action
            else:
                actions[agent] = self._get_opponent_action(
                    self._last_obs.get(agent, np.zeros(self.observation_space.shape))
                )

        observations, rewards, terminations, truncations, infos = self.pz_env.step(actions)

        # Store observations for next opponent actions
        self._last_obs = observations

        # Get our agent's results
        obs = observations.get(self.agent_id, np.zeros(self.observation_space.shape))
        reward = rewards.get(self.agent_id, 0.0)
        terminated = terminations.get(self.agent_id, True)
        truncated = truncations.get(self.agent_id, False)
        info = infos.get(self.agent_id, {})

        # If our agent was removed from env.agents, episode is done
        if self.agent_id not in self.pz_env.agents:
            terminated = True

        # End on win too. PyFlyt's ParallelEnv is team-size-generic and doesn't
        # stop when the match is decided; without this, post-kill steps give
        # no learning signal and the policy drifts into limit cycles. For 1v1
        # (team_size=1), "match over" = only our agent left in env.agents.
        if self.team_size == 1 and len(self.pz_env.agents) <= 1:
            terminated = True

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.pz_env.render()

    def close(self):
        try:
            self.pz_env.close()
        except Exception:
            pass

    def set_opponent_policy(self, policy):
        """Update the opponent policy (for self-play curriculum)."""
        self.opponent_policy = policy
