import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dfa_samplers import DFASampler, RADSampler

from typing import Any

__all__ = ["DFAWrapper"]

class DFAWrapper(gym.Wrapper):
    def __init__(
        self,
        env_id: str,
        sampler: DFASampler | None = None,
        label_f: callable = None,
        r_agg_f: callable = None
    ):
        super().__init__(gym.make(env_id))
        self.sampler = sampler if sampler is not None else RADSampler()
        self.label_f = label_f if label_f is not None else lambda obs: np.random.choice(self.sampler.n_tokens)
        self.r_agg_f = r_agg_f if r_agg_f is not None else lambda _, dfa_reward: dfa_reward
        self.size_bound = self.sampler.get_size_bound()
        self.action_space = self.env.action_space
        self.observation_space = spaces.Dict({
            "obs": self.env.observation_space,
            "dfa_obs": spaces.Box(low=0, high=9, shape=(self.size_bound,), dtype=np.int64),
        })
        self.dfa = None

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        np.random.seed(seed)
        obs, info = self.env.reset(seed=seed, options=options)
        self.dfa = self.sampler.sample()
        dfa_obs = self._get_dfa_obs()
        obs = {"obs": obs, "dfa_obs": dfa_obs}
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, int, bool, bool, dict[str, Any]]:
        obs, reward, done, truncated, info = self.env.step(action)
        symbol = self.label_f(obs)
        if symbol is not None:
            self.dfa = self.dfa.advance([symbol]).minimize()
        dfa_obs = self._get_dfa_obs()
        obs = {"obs": obs, "dfa_obs": dfa_obs}
        dfa_reward = 0
        if self.dfa._label(self.dfa.start):
            dfa_reward = 1
        elif self.dfa.find_word() is None:
            dfa_reward = -1
        reward = self.r_agg_f(reward, dfa_reward)
        done = done or dfa_reward != 0
        return obs, reward, done, truncated, info

    def _get_dfa_obs(self) -> np.ndarray:
        dfa_arr = np.array([int(i) for i in str(self.dfa.to_int())])
        dfa_obs = np.pad(dfa_arr, (self.size_bound - dfa_arr.shape[0], 0), constant_values=0)
        return dfa_obs
