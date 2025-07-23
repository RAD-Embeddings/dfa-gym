import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dfa_samplers import DFASampler, RADSampler

from typing import Any

__all__ = ["DFAWrapper"]

class DFAWrapper(gym.Wrapper):
    metadata = {"render_modes": ["human", "ansi"], "name": "dfa_wrapper"}

    def __init__(
        self,
        env: str | gym.Env,
        n_agents: int = 1,
        sampler: DFASampler | None = None,
        label_f: callable = None,
        r_agg_f: callable = None
    ):
        if isinstance(env, str):
            super().__init__(gym.make(env))
        else:
            super().__init__(env)

        self.n_agents = n_agents
        self.sampler = sampler if sampler is not None else RADSampler(n_tokens=self.env.unwrapped.n_tokens)
        self.label_f = label_f if label_f is not None else lambda obs: np.random.choice(self.sampler.n_tokens)
        self.r_agg_f = r_agg_f if r_agg_f is not None else lambda _, dfa_reward: dfa_reward

        assert self.env.render_mode in self.metadata["render_modes"]
        assert self.env.unwrapped.n_tokens == self.sampler.n_tokens

        self.possible_agents = [f"A_{i}" for i in range(self.n_agents)]

        self.size_bound = self.sampler.get_size_bound()

        self.action_space = self.env.action_space if self.n_agents == 1 else gym.spaces.Dict({
            agent: self.env.action_space[agent] for agent in self.possible_agents
        })
        self.observation_space = spaces.Dict({
            "obs": self.env.observation_space,
            "dfa_obs": spaces.Box(low=0, high=9, shape=(self.size_bound,), dtype=np.int64)
        }) if self.n_agents == 1 else gym.spaces.Dict({
            agent: spaces.Dict({
                "obs": self.env.observation_space[agent],
                "dfa_obs": spaces.Box(low=0, high=9, shape=(self.n_agents, self.size_bound), dtype=np.int64)
            }) for agent in self.possible_agents
        })

        self.dfas = {}
        self.dfa_dones = {}
        self.episode_rewards = {}
        self.t = 0

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        np.random.seed(seed)
        observations, info = self.env.reset(seed=seed, options=options)
        self.agents = self.possible_agents.copy()
        self.dfas = {agent: self.sampler.sample() for agent in self.agents}
        self.dfa_dones = {agent: False for agent in self.agents}
        self.episode_rewards = {agent: [] for agent in self.agents}
        self.t = 0
        if self.n_agents == 1:
            agent = next(iter(self.agents)) # Get the only element of the set
            observations = {"obs": observations, "dfa_obs": self._dfa2obs(self.dfas[agent])}
        else:
            observations = {
                agent: {"obs": observations[agent],
                        "dfa_obs": np.array([self._dfa2obs(self.dfas[agent])] + [self._dfa2obs(self.dfas[other]) for other in self.agents if other != agent])
                } for agent in self.agents
            }
        return observations, info

    def step(self, action: int) -> tuple[np.ndarray, int, bool, bool, dict[str, Any]]:
        observations, rewards, terminations, truncations, infos = self.env.step(action)
        if self.n_agents == 1:
            agent = next(iter(self.agents)) # Get the only element of the set
            symbol = self.label_f(observations, self.sampler.n_tokens)
            if symbol is not None:
                self.dfas[agent] = self.dfas[agent].advance([symbol]).minimize()
            wrapped_obs = {"obs": observations, "dfa_obs": self._dfa2obs(self.dfas[agent])}
            dfa_reward = 0   
            if self.dfas[agent]._label(self.dfas[agent].start):
                dfa_reward = 1
            elif self.dfas[agent].find_word() is None:
                dfa_reward = -1
            rewards = self.r_agg_f(rewards, dfa_reward)
            terminations = terminations or dfa_reward != 0
            self.t += 1
            return wrapped_obs, rewards, terminations, truncations, infos
        else:
            wrapped_obs = {}
            for agent in observations:
                if not terminations[agent] and not truncations[agent]:
                    symbol = self.label_f(observations[agent], self.sampler.n_tokens)
                    old_dfa = self.dfas[agent]
                    if symbol is not None:
                        self.dfas[agent] = self.dfas[agent].advance([symbol]).minimize()
                    wrapped_obs[agent] = {
                        "obs": observations[agent],
                        "dfa_obs": np.array([self._dfa2obs(self.dfas[agent])] + [self._dfa2obs(self.dfas[other]) for other in self.agents if other != agent])
                    }
                    dfa_reward = self._dfa2rew(self.dfas[agent])
                    if dfa_reward != 0:
                        self.dfa_dones[agent] = True
                    current_dfa_reward = dfa_reward if old_dfa is None or old_dfa.to_int() != self.dfas[agent].to_int() else 0
                    rewards[agent] = self.r_agg_f(rewards[agent], current_dfa_reward)
            if all(self.dfa_dones[agent] for agent in observations if not terminations[agent] and not truncations[agent]):
                for agent in observations:
                    rewards[agent] += 1e-2 * sum(self._dfa2rew(self.dfas[other]) for other in observations if other != agent)
                    terminations[agent] = True
            for agent in observations:
                if rewards[agent] != 0:
                    self.episode_rewards[agent].append({"reward": rewards[agent], "t": self.t})
            for agent in self.possible_agents:
                infos[agent] = {"episode_rewards": self.episode_rewards[agent].copy()}
            self.t += 1
            return wrapped_obs, rewards, terminations, truncations, infos

    def _dfa2obs(self, dfa) -> np.ndarray:
        dfa_arr = np.array([int(i) for i in str(dfa.to_int())])
        dfa_obs = np.pad(dfa_arr, (self.size_bound - dfa_arr.shape[0], 0), constant_values=0)
        return dfa_obs

    def _dfa2rew(self, dfa) -> int:
        rew = 0
        if dfa._label(dfa.start):
            rew = 1
        elif dfa.find_word() is None:
            rew = -1
        return rew

    def render(self):
        out = ""
        for agent in self.possible_agents:
            out += "****\n"
            out += f"{agent}'s DFA:\n"
            out += f"{self.dfas[agent]}\n"
        if self.render_mode == "human":
            self.env.render()
            print(out)
        else:
            return self.env.render() + out

