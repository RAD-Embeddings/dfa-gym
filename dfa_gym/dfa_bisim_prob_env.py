from dfa import DFA
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dfa_samplers import DFASampler, RADSampler
from scipy.special import log_softmax

from typing import Any

def softmax(x):
  """
  Computes the softmax function for a given input array.

  Args:
    x: A NumPy array.

  Returns:
    A NumPy array with the softmax values.
  """
  e_x = np.exp(x - np.max(x)) # Subtract max for numerical stability
  return e_x / e_x.sum(axis=-1, keepdims=True)

__all__ = ["DFABisimProbEnv"]

class DFABisimProbEnv(gym.Env):
    def __init__(
        self,
        sampler: DFASampler | None = None,
        timeout: int = 100,
        success_reward: float = 0.9
    ):
        super().__init__()
        self.sampler = sampler if sampler is not None else RADSampler()
        self.size_bound = self.sampler.get_size_bound()
        self.action_space = spaces.Box(low=-10, high=10, shape=(self.sampler.n_tokens,), dtype=np.float32)
        # self.action_space = spaces.Discrete(self.sampler.n_tokens)
        self.observation_space = spaces.Dict({"dfa_left": spaces.Box(low=0, high=9, shape=(self.size_bound,), dtype=np.int64),
                                              "dfa_left_state_belief": spaces.Box(low=0.0, high=1.0, shape=(self.sampler.n_tokens,), dtype=np.float32),
                                              "dfa_right": spaces.Box(low=0, high=9, shape=(self.size_bound,), dtype=np.int64),
                                              "dfa_right_state_belief": spaces.Box(low=0.0, high=1.0, shape=(self.sampler.n_tokens,), dtype=np.float32)})
        self.timeout = timeout
        self.success_reward = success_reward
        self.dfa_left = None
        self.dfa_right = None
        self.dfa_left_state_belief = None
        self.dfa_right_state_belief = None
        self.t = 0

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        np.random.seed(seed)
        self.dfa_left = None
        self.dfa_right = None
        self.dfa_left_state_belief = np.zeros((self.sampler.max_size,), dtype=np.float32)
        self.dfa_right_state_belief = np.zeros((self.sampler.max_size,), dtype=np.float32)
        while self.dfa_left == self.dfa_right:
            self.dfa_left = self.sampler.sample()
            self.dfa_right = self.sampler.sample()
        self.dfa_left_state_belief[0] = 1
        self.dfa_right_state_belief[0] = 1
        self.t = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # print(action.sum(), action)
        # input()
        # if not (np.all(action >= 0.0) and np.all(action <= 1.0) and np.isclose(action.sum(), 1.0)):
        #     # numerically stable softmax
        #     shifted = action - np.max(action)
        #     exp_vals = np.exp(shifted)
        #     action = exp_vals / exp_vals.sum()
        # print(action.sum(), softmax(action).sum())
        # input()
        # action = softmax(action * 100)

        action = np.exp(log_softmax(action))

        self.dfa_left_state_belief = DFABisimProbEnv.get_next_state_belief(self.dfa_left, self.dfa_left_state_belief, action)
        self.dfa_right_state_belief = DFABisimProbEnv.get_next_state_belief(self.dfa_right, self.dfa_right_state_belief, action)
        reward_left = DFABisimProbEnv.dfa2rew(self.dfa_left, self.dfa_left_state_belief, self.t, action)
        reward_right = DFABisimProbEnv.dfa2rew(self.dfa_right, self.dfa_right_state_belief, self.t, action)
        reward = reward_left - reward_right
        self.t += 1
        done = (reward_left != 0 or reward_right != 0) or self.t > self.timeout
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        return {"dfa_left": DFABisimProbEnv.dfa2obs(self.dfa_left, size_bound=self.size_bound),
                "dfa_left_state_belief": self.dfa_left_state_belief,
                "dfa_right": DFABisimProbEnv.dfa2obs(self.dfa_right, size_bound=self.size_bound),
                "dfa_right_state_belief": self.dfa_right_state_belief}

    @staticmethod
    def get_next_state_belief(dfa: DFA, state_belief, action):
        if type(action) == np.int64:
            temp = np.zeros((len(dfa.inputs),))
            temp[action] = 1
            action = temp
        prob_transition_mat = np.zeros((state_belief.size, state_belief.size), dtype=np.float32)
        for s in dfa.states():
            for a in dfa.inputs:
                t = dfa._transition(s, a)
                prob_transition_mat[s][t] += action[a]
        return np.matmul(state_belief, prob_transition_mat)

    @staticmethod
    def dfa2rew(dfa: DFA, state_belief, t, action, success_prob=0.9):
        for s in dfa.states():
            if state_belief[s] >= success_prob:
                if dfa._label(s):
                    return 1
                elif sum(dfa._transition(s, a) != s for a in dfa.inputs) == 0: # No outgoing edges, so it is a rejecting sink
                    return -1
        return 0

    @staticmethod
    def dfa2obs(dfa: DFA, size_bound: int) -> np.ndarray:
        dfa_obs = np.array([int(i) for i in str(dfa.to_int())])
        obs = np.pad(dfa_obs, (size_bound - dfa_obs.shape[0], 0), constant_values=0)
        return obs
