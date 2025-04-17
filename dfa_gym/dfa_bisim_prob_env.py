from dfa import DFA, dfa2dict
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dfa_samplers import DFASampler, RADSampler

from typing import Any

def softmax(x):
  """
  Computes the softmax function for a given input numpy array.

  Args:
    x: A numpy array of any shape.

  Returns:
    A numpy array with the same shape as x, where each element
    represents the softmax output.
  """
  # Subtract the maximum value for numerical stability
  e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
  return e_x / np.sum(e_x, axis=-1, keepdims=True)

__all__ = ["DFABisimProbEnv"]

class DFABisimProbEnv(gym.Env):
    def __init__(
        self,
        sampler: DFASampler | None = None,
        timeout: int = 100
    ):
        super().__init__()
        self.sampler = sampler if sampler is not None else RADSampler()
        self.size_bound = self.sampler.get_size_bound()
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.sampler.n_tokens,), dtype=np.float32)
        self.observation_space = spaces.Dict({"dfa_left": spaces.Box(low=0, high=9, shape=(self.size_bound,), dtype=np.int64),
                                              "state_belief_left": spaces.Box(low=0, high=1, shape=(self.sampler.max_size,), dtype=np.float32),
                                              "dfa_right": spaces.Box(low=0, high=9, shape=(self.size_bound,), dtype=np.int64),
                                              "state_belief_right": spaces.Box(low=0, high=1, shape=(self.sampler.max_size,), dtype=np.float32)})
        self.dfa_pair = [None, None]
        self.state_beliefs = np.zeros((2, self.sampler.max_size))
        self.state_beliefs[0][0] = 1
        self.state_beliefs[1][0] = 1
        self.timeout = timeout
        self.t = None

        self.c = 0.9

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        np.random.seed(seed)
        self.dfa_pair = [None, None]
        self.state_beliefs = np.zeros((2, self.sampler.max_size), dtype=np.float32)
        self.state_beliefs[0][0] = 1
        self.state_beliefs[1][0] = 1
        while self.dfa_pair[0] == self.dfa_pair[1]:
            self.dfa_pair[0] = self.sampler.sample()
            self.dfa_pair[1] = self.sampler.sample()
        self.t = 0
        obs = {"dfa_left": DFABisimProbEnv.dfa2obs(self.dfa_pair[0], size_bound=self.size_bound),
               "state_belief_left": self.state_beliefs[0],
               "dfa_right": DFABisimProbEnv.dfa2obs(self.dfa_pair[1], size_bound=self.size_bound),
               "state_belief_right": self.state_beliefs[1]}
        return obs, {}

    def step(self, action: int | list[int]) -> tuple[np.ndarray, int, bool, bool, dict[str, Any]]:
        # action = action / action.sum()
        action = softmax(action)
        self.state_beliefs[0] = DFABisimProbEnv.get_next_state_belief(self.dfa_pair[0], self.state_beliefs[0], action)
        self.state_beliefs[1] = DFABisimProbEnv.get_next_state_belief(self.dfa_pair[1], self.state_beliefs[1], action)
        reward0 = DFABisimProbEnv.dfa2rew(self.dfa_pair[0], self.state_beliefs[0])
        reward1 = DFABisimProbEnv.dfa2rew(self.dfa_pair[1], self.state_beliefs[1])
        reward = reward0 - reward1
        self.t += 1
        done = (abs(reward0) > self.c or abs(reward1) > self.c) or self.t > self.timeout
        obs = {"dfa_left": DFABisimProbEnv.dfa2obs(self.dfa_pair[0], size_bound=self.size_bound),
               "state_belief_left": self.state_beliefs[0],
               "dfa_right": DFABisimProbEnv.dfa2obs(self.dfa_pair[1], size_bound=self.size_bound),
               "state_belief_right": self.state_beliefs[1]}
        return obs, reward, done, False, {}

    @staticmethod
    def get_next_state_belief(dfa: DFA, state_belief: np.ndarray, action: np.ndarray) -> np.ndarray:
        dfa_mat = np.zeros((state_belief.size, state_belief.size))
        dfa_dict, init_state = dfa2dict(dfa)
        for s in dfa_dict:
            for a in dfa_dict[s][1]:
                dfa_mat[s][dfa_dict[s][1][a]] += action[a]
        next_state_belief = np.matmul(state_belief, dfa_mat)
        next_state_belief = next_state_belief / np.sqrt(np.sum(next_state_belief**2))
        print(next_state_belief)
        return next_state_belief

    @staticmethod
    def dfa2rew(dfa: DFA, state_belief: np.ndarray) -> int:
        reward = 0
        for s in dfa.states():
            r = 0
            if dfa._label(s):
                r = 1
            elif sum(dfa._transition(s, a) != s for a in dfa.inputs) == 0:
                r = -1
            reward += r * state_belief[s]
        return reward

    @staticmethod
    def dfa2obs(dfa: DFA, size_bound: int) -> np.ndarray:
        dfa_obs = np.array([int(i) for i in str(dfa.to_int())])
        obs = np.pad(dfa_obs, (size_bound - dfa_obs.shape[0], 0), constant_values=0)
        return obs

