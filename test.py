import token_env
import gymnasium as gym
from pettingzoo.test import parallel_api_test

from dfa_gym import DFAEnv, DFAWrapper, gym2zoo

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
    # Test DFAWrapper
    # env_cls = "CartPole-v1"
    # wrapped_env = DFAWrapper(env_cls)
    # n_agents = 2
    # env = token_env.TokenEnv(n_agents=n_agents, n_tokens=10, n_token_repeat=2, size=(7, 7), timeout=100, use_fixed_map=False)
    # wrapped_env = DFAWrapper(env=env, n_agents=n_agents)
    # observation, info = wrapped_env.reset()
    # done = False
    # i = 0
    # while (not isinstance(done, bool) or not done) and (not isinstance(done, dict) or not all(done.values())):
    #     i += 1
    #     action = wrapped_env.action_space.sample()
    #     observation, reward, terminated, truncated, info = wrapped_env.step(action)
    #     print(observation, reward, terminated, truncated, info)
    #     input()
    #     done = terminated or truncated
    # wrapped_env.close()
    # # Test DFAEnv
    # dfa_env = gym.make("DFABisimEnv-v1")
    # obs, info = dfa_env.reset()
    # for _ in range(1000):
    #     action = dfa_env.action_space.sample()  # Random action
    #     obs, reward, done, truncated, info = dfa_env.step(action)
    #     if done:
    #         break
    # dfa_env.close()
    # # Test DFABisimProbEnv
    # env = gym.make("DFABisimProbEnv-v1")
    # check_env(env)
    # obs, info = env.reset()
    # for _ in range(1000):
    #     action = env.action_space.sample()  # Random action
    #     obs, reward, done, truncated, info = env.step(action)
    #     if done:
    #         break
    # env.close()

    ####

    # env = token_env.TokenEnv(
    #     n_agents=2,
    #     n_tokens=5,
    #     n_token_repeat=2,
    #     size=(7,7),
    #     timeout=100,
    #     use_fixed_map=True,
    #     slip_prob=(0.0, 0.0)
    # )
    # env = DFAWrapper(env=env, n_agents=env.n_agents, label_f=token_env.TokenEnv.label_f)
    # obs, info = env.reset()
    # env.render()
    # done = False
    # step = 0

    # while not done:
    #     actions = env.action_space.sample()
    #     print(actions)
    #     obs, rewards, terms, truncs, infos = env.step(actions)
    #     env.render()
    #     print(step, rewards, terms, truncs, infos)
    #     input(">>")

    #     # for a in rewards:
    #     #     print(f" {a}: reward={rewards[a]}, done={terms[a] or truncs[a]}")
    #     done = all(terms.values())
    #     step += 1

    ####

    env = token_env.TokenEnv(
        n_agents=2,
        n_tokens=5,
        n_token_repeat=2,
        size=(7,7),
        timeout=100,
        use_fixed_map=True,
        slip_prob=(0.0, 0.0)
    )
    env = DFAWrapper(env=env, n_agents=env.n_agents, label_f=token_env.TokenEnv.label_f, r_agg_f=token_env.TokenEnv.r_agg_f)
    env = gym2zoo(env)
    obs, info = env.reset()
    env.render()
    # input(">>")
    done = False
    step = 0

    while not done:
        actions = {agent: env.action_space(agent).sample() for agent in obs}
        print({agent: env.env.env.action_parser[actions[agent]] for agent in actions})
        # actions = env.action_space.sample()
        # print({agent: env.env.action_parser[actions[agent]] for agent in actions})
        obs, rewards, terms, truncs, infos = env.step(actions)
        env.render()
        print(step, rewards, terms, truncs, infos)
        # input(">>")

        # for a in rewards:
        #     print(f" {a}: reward={rewards[a]}, done={terms[a] or truncs[a]}")
        done = all(terms.values())
        step += 1

    # env = gym2zoo(env)

    # parallel_api_test(env)

    # env = ss.black_death_v3(env)
    # env = ss.pettingzoo_env_to_vec_env_v1(env)
    # env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
    # env = VecMonitor(env)


    # Sanity test: run one episode with random actions
    # obs, info = env.reset()

    # done = False

    # step = 0
    # while not done:
    #     print(obs)
    #     print(actions)
    #     actions = {a: gym.spaces.Discrete(4).sample() for a in info}
    #     # actions = env.action_space.sample()
    #     obs, rewards, terms, truncs, infos = env.step(actions)
    #     print(f"\nStep {step}")

    #     for a in rewards:
    #         print(f" {a}: reward={rewards[a]}, done={terms[a] or truncs[a]}")
    #     done = all(terms.values())
    #     step += 1

    # env.close()
    # print("\nEpisode finished.")
