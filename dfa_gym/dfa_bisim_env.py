from dfa import DFA
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dfa_samplers import DFASampler, RADSampler

from typing import Any

__all__ = ["DFABisimEnv"]

class DFABisimEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "name": "dfa_env"}

    def __init__(
        self,
        sampler: DFASampler | None = None,
        timeout: int = 100,
        render_mode: str = "human"
    ):
        super().__init__()
        assert render_mode in self.metadata["render_modes"]

        self.sampler = sampler if sampler is not None else RADSampler()
        self.timeout = timeout
        self.render_mode = render_mode

        self.size_bound = self.sampler.get_size_bound()
        self.action_space = spaces.Discrete(self.sampler.n_tokens)
        self.observation_space = spaces.Box(low=0, high=9, shape=(2*self.size_bound,), dtype=np.int64)
        self.dfa_pair = [None, None]
        self.t = None

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        np.random.seed(seed)
        self.dfa_pair = [None, None]
        while self.dfa_pair[0] == self.dfa_pair[1]:
            self.dfa_pair[0] = self.sampler.sample()
            self.dfa_pair[1] = self.sampler.sample()
        self.t = 0
        obs = np.concatenate([DFABisimEnv.dfa2obs(self.dfa_pair[0], size_bound=self.size_bound), DFABisimEnv.dfa2obs(self.dfa_pair[1], size_bound=self.size_bound)])
        return obs, {}

    def step(self, action: int | list[int]) -> tuple[np.ndarray, int, bool, bool, dict[str, Any]]:
        self.dfa_pair[0] = self.dfa_pair[0].advance([action]).minimize()
        self.dfa_pair[1] = self.dfa_pair[1].advance([action]).minimize()
        reward0 = DFABisimEnv.dfa2rew(self.dfa_pair[0])
        reward1 = DFABisimEnv.dfa2rew(self.dfa_pair[1])
        reward = reward0 - reward1
        self.t += 1
        done = (reward0 != 0 or reward1 != 0) or self.t > self.timeout
        obs = np.concatenate([DFABisimEnv.dfa2obs(self.dfa_pair[0], size_bound=self.size_bound), DFABisimEnv.dfa2obs(self.dfa_pair[1], size_bound=self.size_bound)])
        return obs, reward, done, False, {}

    @staticmethod
    def dfa2rew(dfa: DFA) -> int:
        reward = 0
        if dfa._label(dfa.start):
            reward = 1
        elif dfa.find_word() is None:
            reward = -1
        return reward

    @staticmethod
    def dfa2obs(dfa: DFA, size_bound: int) -> np.ndarray:
        dfa_obs = np.array([int(i) for i in str(dfa.to_int())])
        obs = np.pad(dfa_obs, (size_bound - dfa_obs.shape[0], 0), constant_values=0)
        return obs

    def render(self):
        out = f"Left DFA:\n {self.dfa_pair[0]}\nRight DFA:\n {self.dfa_pair[1]}"
        if self.render_mode == "human":
            print(out)
        else:
            return out

