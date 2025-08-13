import jax
from dfa_gym import TokenEnv, DFABisimEnv

def test(env):

    key = jax.random.PRNGKey(30)

    n = 1_000

    for i in range(n):

        key, subkey = jax.random.split(key)
        obs, state = env.reset(key=subkey)
        done = False
        steps = 0

        while not done:
            keys = jax.random.split(key, env.n_agents + 1)
            key, subkeys = keys[0], keys[1:]
            actions = {agent: env.action_space(agent).sample(subkeys[i]) for i, agent in enumerate(env.agents)}
            key, subkey = jax.random.split(key)
            obs, state, rewards, dones, info = env.step(actions=actions, state=state, key=subkey)
            done = dones["__all__"]
            steps += 1

        print(f"Test completed for {i + 1} samples.", end="\r")

    print(f"Test completed for {n} samples.")

if __name__ == '__main__':
    test(env=TokenEnv())
    test(env=DFABisimEnv())

