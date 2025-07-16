import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dfa_samplers import DFASampler, RADSampler

from typing import Any

__all__ = ["DFAWrapper"]

class DFAWrapper(gym.Wrapper):
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
        self.sampler = sampler if sampler is not None else RADSampler()
        self.label_f = label_f if label_f is not None else lambda obs: np.random.choice(self.sampler.n_tokens)
        self.r_agg_f = r_agg_f if r_agg_f is not None else lambda _, dfa_reward: dfa_reward

        self.size_bound = self.sampler.get_size_bound()

        # Agent identifiers
        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]

        self.action_space = self.env.action_space if self.n_agents == 1 else gym.spaces.Dict({
            agent: self.env.action_space[agent] for agent in self.possible_agents
        })
        self.observation_space = spaces.Dict({
            "obs": self.env.observation_space,
            "dfa_obs": spaces.Box(low=0, high=9, shape=(self.size_bound,), dtype=np.int64),
        }) if self.n_agents == 1 else gym.spaces.Dict({
            agent: spaces.Dict({
                "obs": self.env.observation_space[agent],
                "dfa_obs": spaces.Box(low=0, high=9, shape=(self.size_bound,), dtype=np.int64),
            }) for agent in self.possible_agents
        })

        self.dfas = {}

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        np.random.seed(seed)
        obs, info = self.env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        self.active_agents = self.possible_agents[:]
        self.dfas = {agent: self.sampler.sample() for agent in self.agents}
        if self.n_agents == 1:
            obs = {"obs": obs, "dfa_obs": self._dfa2obs(self.dfas[self.agents[0]])}
        else:
            obs = {agent: {"obs": obs[agent], "dfa_obs": self._dfa2obs(self.dfas[agent])} for agent in self.agents}
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, int, bool, bool, dict[str, Any]]:
        obs, reward, done, truncated, info = self.env.step(action)
        if self.n_agents == 1:
            symbol = self.label_f(obs)
            if symbol is not None:
                self.dfas[self.agents[0]] = self.dfas[self.agents[0]].advance([symbol]).minimize()
            obs = {"obs": obs, "dfa_obs": self._dfa2obs(self.dfas[self.agents[0]])}
            dfa_reward = 0
            if self.dfas[self.agents[0]]._label(self.dfas[self.agents[0]].start):
                dfa_reward = 1
            elif self.dfas[self.agents[0]].find_word() is None:
                dfa_reward = -1
            reward = self.r_agg_f(reward, dfa_reward)
            done = done or dfa_reward != 0
            return obs, reward, done, truncated, info
        else:
            wrapped_obs = {}
            for agent in self.possible_agents:
                if agent in self.active_agents:
                    symbol = self.label_f(obs[agent])
                    if symbol is not None:
                        self.dfas[agent] = self.dfas[agent].advance([symbol]).minimize()
                    wrapped_obs[agent] = {"obs": obs[agent], "dfa_obs": self._dfa2obs(self.dfas[agent])}
                    dfa_reward = 0
                    if self.dfas[agent]._label(self.dfas[agent].start):
                        dfa_reward = 1
                    elif self.dfas[agent].find_word() is None:
                        dfa_reward = -1
                    reward[agent] = self.r_agg_f(reward[agent], dfa_reward)
                    done[agent] = done[agent] or dfa_reward != 0
                    if done[agent]:
                        self.active_agents.remove(agent)
                else:
                    wrapped_obs[agent] = {"obs": np.zeros_like(obs[agent], dtype=np.uint8), "dfa_obs": np.zeros(shape=(self.size_bound,), dtype=np.int64)}
                    reward[agent] = 0
                    done[agent] = True
                # info[agent]["episode"]["r"] = reward[agent]
            # self.agents = [agent for agent in self.agents if not done[agent] and not truncated[agent]]
            env_is_done = all(list(done.values()))
            total_dones = {agent: env_is_done for agent in self.possible_agents}
            return wrapped_obs, reward, total_dones, truncated, info

    def _dfa2obs(self, dfa) -> np.ndarray:
        dfa_arr = np.array([int(i) for i in str(dfa.to_int())])
        dfa_obs = np.pad(dfa_arr, (self.size_bound - dfa_arr.shape[0], 0), constant_values=0)
        return dfa_obs
