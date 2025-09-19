import jax
import dfax
import chex
import jax.numpy as jnp
from flax import struct
from dfa_gym import spaces
from functools import partial
from typing import Tuple, Dict, Callable
from dfax.utils import list2batch, batch2graph
from dfa_gym.env import MultiAgentEnv, State
from dfax.samplers import DFASampler, RADSampler


@struct.dataclass
class DFAWrapperState(State):
    dfas: Dict[str, dfax.DFAx]
    env_obs: chex.Array
    env_state: State

class DFAWrapper(MultiAgentEnv):

    def __init__(
        self,
        env: MultiAgentEnv,
        gamma: float | None = 0.99,
        sampler: DFASampler = RADSampler()
    ) -> None:
        super().__init__(num_agents=env.num_agents)
        self.env = env
        self.gamma = gamma
        self.sampler = sampler

        assert self.sampler.n_tokens == self.env.n_tokens

        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        self.action_spaces = {
            agent: self.env.action_space(agent)
            for agent in self.agents
        }
        max_dfa_size = self.sampler.max_size
        n_tokens = self.sampler.n_tokens
        if self.num_agents == 1:
            self.observation_spaces = {
                agent: spaces.Dict({
                    "obs": self.env.observation_space(agent),
                    "guarantee": spaces.Dict({
                        "node_features": spaces.Box(low=0, high=1, shape=(max_dfa_size, 4), dtype=jnp.float32),
                        "edge_features": spaces.Box(low=0, high=1, shape=(max_dfa_size*max_dfa_size, n_tokens + 8), dtype=jnp.float32),
                        "edge_index": spaces.Box(low=0, high=max_dfa_size, shape=(2, max_dfa_size*max_dfa_size), dtype=jnp.int32),
                        "current_state": spaces.Box(low=0, high=max_dfa_size, shape=(1,), dtype=jnp.int32),
                        "n_states": spaces.Box(low=0, high=max_dfa_size, shape=(max_dfa_size,), dtype=jnp.int32)
                    })
                })
                for agent in self.agents
            }
        else:
            n_other = self.num_agents - 1
            self.observation_spaces = {
                agent: spaces.Dict({
                    "assume": spaces.Dict({
                        "node_features": spaces.Box(low=0, high=1, shape=(max_dfa_size*n_other, 4), dtype=jnp.float32),
                        "edge_features": spaces.Box(low=0, high=1, shape=(max_dfa_size*n_other*max_dfa_size*n_other, n_tokens + 8), dtype=jnp.float32),
                        "edge_index": spaces.Box(low=0, high=max_dfa_size*n_other, shape=(2, max_dfa_size*n_other*max_dfa_size*n_other), dtype=jnp.int32),
                        "current_state": spaces.Box(low=0, high=max_dfa_size*n_other, shape=(n_other,), dtype=jnp.int32),
                        "n_states": spaces.Box(low=0, high=max_dfa_size*n_other, shape=(max_dfa_size*n_other,), dtype=jnp.int32)
                    }),
                    "obs": self.env.observation_space(agent),
                    "guarantee": spaces.Dict({
                        "node_features": spaces.Box(low=0, high=1, shape=(max_dfa_size, 4), dtype=jnp.float32),
                        "edge_features": spaces.Box(low=0, high=1, shape=(max_dfa_size*max_dfa_size, n_tokens + 8), dtype=jnp.float32),
                        "edge_index": spaces.Box(low=0, high=max_dfa_size, shape=(2, max_dfa_size*max_dfa_size), dtype=jnp.int32),
                        "current_state": spaces.Box(low=0, high=max_dfa_size, shape=(1,), dtype=jnp.int32),
                        "n_states": spaces.Box(low=0, high=max_dfa_size, shape=(max_dfa_size,), dtype=jnp.int32)
                    })
                })
                for agent in self.agents
            }

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: chex.PRNGKey
    ) -> Tuple[Dict[str, chex.Array], DFAWrapperState]:

        key, subkey = jax.random.split(key)
        env_obs, env_state = self.env.reset(subkey)

        def sample_dfas(key):
            keys = jax.random.split(key, self.num_agents + 1)
            key, subkeys = keys[0], keys[1:]
            dfas = {agent: self.sampler.sample(subkeys[i]) for i, agent in enumerate(self.agents)}
            return key, dfas

        def cond_fun(carry):
            key, dfas = carry
            n_states = jnp.array([dfa.n_states for dfa in dfas.values()])
            return jnp.all(n_states <= 1)

        def body_fun(carry):
            key, _ = carry
            return sample_dfas(key)

        key, dfas = jax.lax.while_loop(cond_fun, body_fun, sample_dfas(key))

        state = DFAWrapperState(
            dfas=dfas,
            env_obs=env_obs,
            env_state=env_state
        )
        obs = self.get_obs(state=state)

        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: DFAWrapperState,
        action: int,
    ) -> Tuple[Dict[str, chex.Array], DFAWrapperState, Dict[str, float], Dict[str, bool], Dict]:

        env_obs, env_state, env_rewards, env_dones, env_info = self.env.step_env(key, state.env_state, action)

        symbols = self.env.label_f(env_state)

        dfas = {
            agent: state.dfas[agent].advance(symbols[agent]).minimize()
            for agent in self.agents
        }

        dones = {
            agent: jnp.logical_or(env_dones[agent], dfas[agent].reward(binary=True) != 0.0)
            for agent in self.agents
        }
        _dones = jnp.array([dones[agent] for agent in self.agents])
        dones.update({"__all__": jnp.all(_dones)})

        dfa_rewards_min = jnp.min(jnp.array([dfas[agent].reward(binary=True) for agent in self.agents]))
        rewards = {
            agent: jax.lax.cond(
                dones["__all__"],
                lambda _: env_rewards[agent] + dfa_rewards_min,
                lambda _: env_rewards[agent],
                operand=None
            )
            for agent in self.agents
        }

        if self.gamma is not None:
            rewards = {
                agent: rewards[agent] + self.gamma * dfas[agent].reward(binary=True) - state.dfas[agent].reward(binary=True)
                for agent in self.agents
            }

        infos = {}

        state = DFAWrapperState(
            dfas=dfas,
            env_obs=env_obs,
            env_state=env_state
        )

        obs = self.get_obs(state=state)

        return obs, state, rewards, dones, infos

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(
        self,
        state: DFAWrapperState
    ) -> Dict[str, chex.Array]:
        if self.num_agents == 1:
            return {
                agent: {
                    "obs": state.env_obs[agent],
                    "guarantee": state.dfas[agent].to_graph()
                }
                for agent in self.agents
            }

        graphs = [state.dfas[agent].to_graph() for agent in self.agents]
        assumes = {agent: batch2graph(list2batch(graphs[:i] + graphs[i + 1:])) for i, agent in enumerate(self.agents)}
        guarantees = {agent: graphs[i] for i, agent in enumerate(self.agents)}
        return {
            agent: {
                "assume": assumes[agent],
                "obs": state.env_obs[agent],
                "guarantee": guarantees[agent]
            }
            for agent in self.agents
        }

    def render(self, state: DFAWrapperState):
        out = ""
        for agent in self.agents:
            out += "****\n"
            out += f"{agent}'s DFA:\n"
            out += f"{state.dfas[agent]}\n"
        self.env.render(state.env_state)
        print(out)

