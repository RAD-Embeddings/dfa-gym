import jax
import chex
import jax.numpy as jnp
from flax import struct
from functools import partial
from typing import Tuple, Dict
from dfa_gym import spaces
from dfa_gym.env import MultiAgentEnv

import dfax
from dfax.samplers import DFASampler, RADSampler

ACTION_MAP = jnp.array([[1, 0], [0, 1], [-1, 0], [0, -1], [0, 0]])

@struct.dataclass
class State:
    dfa_l: dfax.DFAx
    dfa_r: dfax.DFAx
    time: int

class DFABisimEnv(MultiAgentEnv):

    def __init__(
        self,
        sampler: DFASampler = RADSampler(),
        max_steps_in_episode: int = 100
    ) -> None:
        super().__init__(num_agents=1)
        self.n_agents = self.num_agents
        self.sampler = sampler
        self.max_steps_in_episode = max_steps_in_episode

        self.agents = [f"agent_{i}" for i in range(self.n_agents)]

        self.action_spaces = {
            agent: spaces.Discrete(self.sampler.n_tokens)
            for agent in self.agents
        }
        max_dfa_size = self.sampler.max_size
        n_tokens = self.sampler.n_tokens
        self.observation_spaces = {
            agent: spaces.Dict({
                "graph_l": spaces.Dict({
                    "node_features": spaces.Box(low=0, high=1, shape=(max_dfa_size, 3), dtype=jnp.uint16),
                    "edge_features": spaces.Box(low=0, high=1, shape=(max_dfa_size*max_dfa_size, n_tokens), dtype=jnp.uint16),
                    "edge_index": spaces.Box(low=0, high=max_dfa_size, shape=(2, max_dfa_size*max_dfa_size), dtype=jnp.uint16),
                    "current_state": spaces.Box(low=0, high=max_dfa_size, shape=(1,), dtype=jnp.uint16),
                    "n_states": spaces.Box(low=0, high=max_dfa_size, shape=(max_dfa_size,), dtype=jnp.uint16)
                }),
                "graph_r": spaces.Dict({
                    "node_features": spaces.Box(low=0, high=1, shape=(max_dfa_size, 3), dtype=jnp.uint16),
                    "edge_features": spaces.Box(low=0, high=1, shape=(max_dfa_size*max_dfa_size, n_tokens), dtype=jnp.uint16),
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
    ) -> Tuple[Dict[str, chex.Array], State]:

        key, kl, kr = jax.random.split(key, 3)
        dfa_l = self.sampler.sample(kl)
        dfa_r = self.sampler.sample(kr)

        state = State(dfa_l=dfa_l, dfa_r=dfa_r, time=0)
        obs = self.get_obs(state=state)
        return {self.agents[0]: obs}, state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: State,
        action: int
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:

        dfa_l = state.dfa_l.advance(action[self.agents[0]]).minimize()
        dfa_r = state.dfa_r.advance(action[self.agents[0]]).minimize()

        reward_l = dfa_l.reward()
        reward_r = dfa_r.reward()
        reward = reward_l - reward_r

        new_state = State(
            dfa_l=dfa_l,
            dfa_r=dfa_r,
            time=state.time+1
        )

        done = jnp.logical_or(jnp.logical_or(reward_l != 0, reward_r != 0), new_state.time > self.max_steps_in_episode)

        obs = self.get_obs(state=new_state)
        info = {}

        return {self.agents[0]: obs}, new_state, {self.agents[0]: reward}, {self.agents[0]: done, "__all__": done}, info

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(
        self,
        state: State
    ) -> Dict[str, chex.Array]:
        return {
            "graph_l": state.dfa_l.to_graph(),
            "graph_r": state.dfa_r.to_graph()
        }

