import jax
import dfax
import chex
import jax.numpy as jnp
from flax import struct
from functools import partial
from typing import Tuple, Dict, Callable
from dfa_gym import spaces
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
        label_f: Callable,
        r_agg_f: Callable,
        sampler: DFASampler = RADSampler()
    ) -> None:
        super().__init__(num_agents=env.num_agents)
        self.env = env
        self.sampler = sampler
        self.label_f = label_f
        self.r_agg_f = r_agg_f

        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        self.action_spaces = {
            agent: self.env.action_space(agent)
            for agent in self.agents
        }
        max_dfa_size = self.sampler.max_size
        n_tokens = self.sampler.n_tokens
        self.observation_spaces = {
            agent: spaces.Dict({
                "graph": spaces.Dict({
                    "node_features": spaces.Box(low=0, high=1, shape=(max_dfa_size, 3), dtype=jnp.uint16),
                    "edge_features": spaces.Box(low=0, high=1, shape=(max_dfa_size*max_dfa_size, n_tokens), dtype=jnp.uint16),
                    "edge_index": spaces.Box(low=0, high=max_dfa_size, shape=(2, max_dfa_size*max_dfa_size), dtype=jnp.uint16),
                    "current_state": spaces.Box(low=0, high=max_dfa_size, shape=(1,), dtype=jnp.uint16),
                    "n_states": spaces.Box(low=0, high=max_dfa_size, shape=(max_dfa_size,), dtype=jnp.uint16)
                }),
                "obs": self.env.observation_space(agent)
            })
            for agent in self.agents
        }

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: chex.PRNGKey
    ) -> Tuple[Dict[str, chex.Array], DFAWrapperState]:

        keys = jax.random.split(key, self.num_agents + 2)
        key = keys[0]
        k_env = keys[1]
        k_dfas = keys[2:]

        env_obs, env_state = self.env.reset(k_env)

        dfas = {agent: self.sampler.sample(k_dfas[i]) for i, agent in enumerate(self.agents)}

        state = DFAWrapperState(dfas=dfas, env_obs=env_obs, env_state=env_state)
        obs = self.get_obs(state=state)

        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: DFAWrapperState,
        action: int
    ) -> Tuple[Dict[str, chex.Array], DFAWrapperState, Dict[str, float], Dict[str, bool], Dict]:

        env_obs, env_state, env_rewards, env_dones, env_info = self.env.step_env(key, state.env_state, action)

        symbols = self.label_f(env_state)

        dfas = {agent: state.dfas[agent].advance(symbols[agent]).minimize() for agent in self.agents}

        state = DFAWrapperState(dfas=dfas, env_obs=env_obs, env_state=env_state)

        rewards = {agent: self.r_agg_f(dfas[agent].reward(), env_rewards[agent]) for agent in self.agents}

        _dones = jnp.array([jnp.logical_or(rewards[agent] != 0, env_dones[agent]) for agent in self.agents])
        dones = {agent: _dones[i] for i, agent in enumerate(self.agents)}
        dones.update({"__all__": jnp.all(_dones)})

        obs = self.get_obs(state=state)

        infos = {}

        return obs, state, rewards, dones, infos

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(
        self,
        state: DFAWrapperState
    ) -> Dict[str, chex.Array]:
        return {
            agent: {"graph": state.dfas[agent].to_graph(), "obs": state.env_obs[agent]}
            for agent in self.agents
        }

