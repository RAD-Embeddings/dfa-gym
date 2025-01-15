import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dfa_samplers import DFASampler, RADSampler

from typing import Any

__all__ = ["DFAEnv"]

class DFAEnv(gym.Env):
    def __init__(
        self,
        sampler: DFASampler | None = None,
        timeout: int = 100
    ):
        super().__init__()
        self.sampler = sampler if sampler is not None else RADSampler()
        self.size_bound = self.sampler.get_size_bound()
        self.action_space = spaces.Discrete(self.sampler.n_tokens)
        self.observation_space = spaces.Box(low=0, high=9, shape=(self.size_bound,), dtype=np.int64)
        self.dfa = None
        self.timeout = timeout
        self.t = None

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        np.random.seed(seed)
        self.dfa = self.sampler.sample()
        self.t = 0
        return self._get_dfa_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, int, bool, bool, dict[str, Any]]:
        self.dfa = self.dfa.advance([action]).minimize()
        reward = 0
        if self.dfa._label(self.dfa.start):
            reward = 1
        elif self.dfa.find_word() is None:
            reward = -1
        self.t += 1
        done = reward != 0 or self.t > self.timeout
        return self._get_dfa_obs(), reward, done, False, {}

    def _get_dfa_obs(self) -> np.ndarray:
        dfa_obs = np.array([int(i) for i in str(self.dfa.to_int())])
        obs = np.pad(dfa_obs, (self.size_bound - dfa_obs.shape[0], 0), constant_values=0)
        return obs
