import token_env
import gymnasium as gym
from pettingzoo.test import parallel_api_test
from dfa_gym import DFAEnv, DFAWrapper, gym2zoo

def test(env):
    if isinstance(env, str):
        env = gym.make(env)
    obs, info = env.reset()
    env.render()
    step = 0
    for _ in range(1000):
        if isinstance(env.action_space, gym.Space):
            action = env.action_space.sample()
        else:
            action = {agent: env.action_space(agent).sample() for agent in obs}
        obs, reward, done, truncated, info = env.step(action)
        step += 1
        print(action)
        print(step, reward, done, truncated, info)
        env.render()
        if done:
            break
    env.close()

if __name__ == "__main__":
    ####
    test("DFAEnv-v1")
    ####
    test("DFABisimEnv-v1")
    ####
    test("DFABisimEnv-5-tokens")
    ####
    test("DFABisimEnv-10-tokens")
    ####
    env = token_env.TokenEnv(
        n_agents=1,
        use_fixed_map=True
    )
    env = DFAWrapper(env=env, n_agents=env.n_agents, r_agg_f=token_env.TokenEnv.r_agg_f, label_f=token_env.TokenEnv.label_f)
    test(env)
    ####
    env = token_env.TokenEnv(
        n_agents=2,
        use_fixed_map=True
    )
    env = DFAWrapper(env=env, n_agents=env.n_agents, r_agg_f=token_env.TokenEnv.r_agg_f, label_f=token_env.TokenEnv.label_f)
    zoo_env = gym2zoo(env)
    test(env)
    parallel_api_test(zoo_env)
    test(zoo_env)
    ####
    env = token_env.TokenEnv(
        n_agents=3,
        use_fixed_map=True
    )
    env = DFAWrapper(env=env, n_agents=env.n_agents, r_agg_f=token_env.TokenEnv.r_agg_f, label_f=token_env.TokenEnv.label_f)
    zoo_env = gym2zoo(env)
    test(env)
    parallel_api_test(zoo_env)
    test(zoo_env)
    print("Tests completed.")

