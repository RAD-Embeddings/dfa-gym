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
from dfax.samplers import DFASampler, RADSampler, ConflictSampler


@struct.dataclass
class DFAWrapperState(State):
    dfas: Dict[str, dfax.DFAx]
    env_obs: chex.Array
    env_state: State
    dfa_dones: Dict[str, bool]

class DFAWrapper(MultiAgentEnv):

    def __init__(
        self,
        env: MultiAgentEnv,
        sampler: DFASampler = RADSampler(),
        gamma: float = 0.99
    ) -> None:
        super().__init__(num_agents=env.num_agents)
        self.env = env
        self.sampler = sampler
        self.gamma = gamma

        assert self.sampler.n_tokens == self.env.n_tokens
        assert not isinstance(self.sampler, ConflictSampler) or self.sampler.n_agents == self.env.n_agents

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
                        "node_features": spaces.Box(low=0, high=1, shape=(max_dfa_size, 4), dtype=jnp.uint16),
                        "edge_features": spaces.Box(low=0, high=1, shape=(max_dfa_size*max_dfa_size, n_tokens + 8), dtype=jnp.uint16),
                        "edge_index": spaces.Box(low=0, high=max_dfa_size, shape=(2, max_dfa_size*max_dfa_size), dtype=jnp.uint16),
                        "current_state": spaces.Box(low=0, high=max_dfa_size, shape=(1,), dtype=jnp.uint16),
                        "n_states": spaces.Box(low=0, high=max_dfa_size, shape=(max_dfa_size,), dtype=jnp.uint16)
                    })
                })
                for agent in self.agents
            }
        else:
            n_other = self.num_agents - 1
            self.observation_spaces = {
                agent: spaces.Dict({
                    "assume": spaces.Dict({
                        "node_features": spaces.Box(low=0, high=1, shape=(max_dfa_size*n_other, 4), dtype=jnp.uint16),
                        "edge_features": spaces.Box(low=0, high=1, shape=(max_dfa_size*n_other*max_dfa_size*n_other, n_tokens + 8), dtype=jnp.uint16),
                        "edge_index": spaces.Box(low=0, high=max_dfa_size*n_other, shape=(2, max_dfa_size*n_other*max_dfa_size*n_other), dtype=jnp.uint16),
                        "current_state": spaces.Box(low=0, high=max_dfa_size*n_other, shape=(n_other,), dtype=jnp.uint16),
                        "n_states": spaces.Box(low=0, high=max_dfa_size*n_other, shape=(max_dfa_size*n_other,), dtype=jnp.uint16)
                    }),
                    "obs": self.env.observation_space(agent),
                    "guarantee": spaces.Dict({
                        "node_features": spaces.Box(low=0, high=1, shape=(max_dfa_size, 4), dtype=jnp.uint16),
                        "edge_features": spaces.Box(low=0, high=1, shape=(max_dfa_size*max_dfa_size, n_tokens + 8), dtype=jnp.uint16),
                        "edge_index": spaces.Box(low=0, high=max_dfa_size, shape=(2, max_dfa_size*max_dfa_size), dtype=jnp.uint16),
                        "current_state": spaces.Box(low=0, high=max_dfa_size, shape=(1,), dtype=jnp.uint16),
                        "n_states": spaces.Box(low=0, high=max_dfa_size, shape=(max_dfa_size,), dtype=jnp.uint16)
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

        if isinstance(self.sampler, ConflictSampler):
            key, subkey = jax.random.split(key)
            dfas = self.sampler.sample(subkey)
        else:
            keys = jax.random.split(key, self.num_agents + 1)
            key, subkeys = keys[0], keys[1:]
            dfas = {agent: self.sampler.sample(subkeys[i]) for i, agent in enumerate(self.agents)}

        state = DFAWrapperState(
            dfas=dfas,
            env_obs=env_obs,
            env_state=env_state,
            dfa_dones={agent: dfas[agent].reward() != 0 for agent in self.agents}
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
            agent: jax.lax.cond(
                state.dfa_dones[agent],
                lambda _: state.dfas[agent],
                lambda _: state.dfas[agent].advance(symbols[agent]).minimize(),
                operand=None
            )
            for agent in self.agents
        }

        dfa_dones = {
            agent: dfas[agent].reward() != 0
            for agent in self.agents
        }

        dones = {
            agent: jnp.logical_or(env_dones[agent], dfa_dones[agent])
            for agent in self.agents
        }
        _dones = jnp.array([dones[agent] for agent in self.agents])
        dones.update({"__all__": jnp.all(_dones)})

        dfa_reward_sum = jnp.sum(jnp.array([dfas[agent].reward() for agent in self.agents]))
        rewards = {
            agent: jax.lax.cond(
                jnp.logical_and(dones["__all__"], jnp.abs(dfa_reward_sum) == self.num_agents),
                lambda _: env_rewards[agent] + 1,
                lambda _: env_rewards[agent],
                operand=None
            )
            for agent in self.agents
        }

        potential_v1 = lambda agent, dfas: dfas[agent].reward()
        potential_v2 = lambda agent, dfas: jnp.sum(jnp.array([dfas[other].reward() for other in self.agents]))
        potential_v3 = lambda agent, dfas: (potential_v2(agent, dfas) - potential_v1(agent, dfas)) * 0.1 + potential_v1(agent, dfas)

        rewards = {
            agent: rewards[agent] + self.gamma * potential_v1(agent, dfas) - potential_v1(agent, state.dfas)
            for agent in self.agents
        }

        infos = {}

        state = DFAWrapperState(
            dfas=dfas,
            env_obs=env_obs,
            env_state=env_state,
            dfa_dones=dfa_dones
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

