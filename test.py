import gymnasium as gym

import numpy as np

from dfa_gym import DFAEnv, DFAWrapper

from stable_baselines3.common.env_checker import check_env

if __name__ == "__main__":
    # # Test DFAEnv
    # dfa_env = gym.make("DFAEnv-v1")
    # obs, info = dfa_env.reset()
    # for _ in range(1000):
    #     action = dfa_env.action_space.sample()  # Random action
    #     obs, reward, done, truncated, info = dfa_env.step(action)
    #     if done:
    #         break
    # dfa_env.close()
    # # Test DFAWrapper
    # env_cls = "CartPole-v1"
    # wrapped_env = DFAWrapper(env_cls)
    # observation, info = wrapped_env.reset()
    # done = False
    # i = 0
    # while not done:
    #     i += 1
    #     action = wrapped_env.action_space.sample()
    #     observation, reward, terminated, truncated, info = wrapped_env.step(action)
    #     done = terminated or truncated
    # wrapped_env.close()
    # # Test DFAEnv
    # dfa_env = gym.make("DFABisimEnv-v2")
    # obs, info = dfa_env.reset()
    # for _ in range(1000):
    #     action = dfa_env.action_space.sample()  # Random action
    #     obs, reward, done, truncated, info = dfa_env.step(action)
    #     if done:
    #         break
    # dfa_env.close()
    # Test DFABisimProbEnv
    dfa_env = gym.make("DFABisimProbEnv-v1")
    check_env(dfa_env)
    obs, info = dfa_env.reset()
    for _ in range(1000):
        # action = dfa_env.action_space.sample()  # Random action
        action = np.zeros((10,))
        action [0] = 0.5
        action [1] = 0.5
        obs, reward, done, truncated, info = dfa_env.step(action)
        if done:
            break
    dfa_env.close()
