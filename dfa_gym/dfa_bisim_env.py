import jax
import chex
import jax.numpy as jnp
from flax import struct
from functools import partial
from typing import Tuple, Dict
from jaxmarl.environments import spaces
from jaxmarl.environments.multi_agent_env import MultiAgentEnv

import dfax
from dfax.samplers import DFASampler, RADSampler

ACTION_MAP = jnp.array([(1, 0), (0, 1), (-1, 0), (0, -1), (0, 0)])

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
        self.observation_spaces = {
            agent: spaces.Dict({
                "start_l": spaces.Discrete(self.sampler.max_size),
                "labels_l": spaces.Box(low=0, high=1, shape=(self.sampler.max_size,), dtype=jnp.uint8),
                "transitions_l": spaces.Box(low=0, high=self.sampler.max_size, shape=(self.sampler.max_size, self.sampler.n_tokens), dtype=jnp.uint8),
                "start_r": spaces.Discrete(self.sampler.max_size),
                "labels_r": spaces.Box(low=0, high=1, shape=(self.sampler.max_size,), dtype=jnp.uint8),
                "transitions_r": spaces.Box(low=0, high=self.sampler.max_size, shape=(self.sampler.max_size, self.sampler.n_tokens), dtype=jnp.uint8)
            })
            for agent in self.agents
        }

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: chex.PRNGKey
    ) -> Tuple[Dict[str, chex.Array], State]:

        def body_fn(carry):
            key, dfa_l, dfa_r = carry
            key, kl, kr = jax.random.split(key, 3)
            dfa_l = self.sampler.sample(kl)
            dfa_r = self.sampler.sample(kr)
            return (key, dfa_l, dfa_r)

        def cond_fn(carry):
            _, dfa_l, dfa_r = carry
            return dfa_l == dfa_r

        key, kl, kr = jax.random.split(key, 3)
        dfa_l = self.sampler.sample(kl)
        dfa_r = self.sampler.sample(kr)

        _, dfa_l, dfa_r = jax.lax.while_loop(cond_fn, body_fn, (key, dfa_l, dfa_r))

        state = State(dfa_l=dfa_l, dfa_r=dfa_r, time=0)
        obs = self.get_obs(state=state)
        return {"agent_0": obs}, state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: State,
        action: int
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:

        dfa_l = state.dfa_l.advance(action["agent_0"]).minimize()
        dfa_r = state.dfa_r.advance(action["agent_0"]).minimize()

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

        return {"agent_0": obs}, new_state, {"agent_0": reward}, {"agent_0": done, "__all__": done}, info

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(
        self,
        state: State
    ) -> Dict[str, chex.Array]:
        return {
            "start_l": state.dfa_l.start,
            "labels_l": state.dfa_l.labels,
            "transitions_l": state.dfa_l.transitions,
            "start_r": state.dfa_r.start,
            "labels_r": state.dfa_r.labels,
            "transitions_r": state.dfa_r.transitions
        }

