import jax
import chex
import numpy as np
import jax.numpy as jnp
from flax import struct
from functools import partial
from typing import Tuple, Dict
from dfa_gym import spaces
from dfa_gym.env import MultiAgentEnv, State


ACTION_MAP = jnp.array([(1, 0), (0, 1), (-1, 0), (0, -1), (0, 0)])

@struct.dataclass
class TokenEnvState(State):
    agent_positions: jax.Array
    token_positions: jax.Array
    is_alive: jax.Array
    time: int

class TokenEnv(MultiAgentEnv):

    def __init__(
        self,
        n_agents: int = 3,
        n_tokens: int = 10,
        n_token_repeat: int = 2,
        grid_shape: Tuple[int, int] = (7, 7),
        fixed_map_seed: int | None = None,
        max_steps_in_episode: int = 100,
        collision_reward = -1e1,
        black_death = True
    ) -> None:
        super().__init__(num_agents=n_agents)
        self.n_agents = n_agents
        self.n_tokens = n_tokens
        self.n_token_repeat = n_token_repeat
        self.grid_shape = grid_shape
        self.grid_shape_arr = jnp.array(self.grid_shape)
        self.fixed_map_seed = fixed_map_seed
        self.max_steps_in_episode = max_steps_in_episode
        self.collision_reward = collision_reward
        self.black_death = black_death

        self.agents = [f"agent_{i}" for i in range(self.n_agents)]

        self.action_spaces = {
            agent: spaces.Discrete(len(ACTION_MAP))
            for agent in self.agents
        }
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=1, shape=(self.n_tokens + self.n_agents - 1, *self.grid_shape), dtype=jnp.uint8)
            for agent in self.agents
        }

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: chex.PRNGKey
    ) -> Tuple[Dict[str, chex.Array], TokenEnvState]:
        if self.fixed_map_seed is not None:
            key = jax.random.PRNGKey(self.fixed_map_seed)
        key, subkey = jax.random.split(key)
        grid_points = jnp.stack(jnp.meshgrid(jnp.arange(self.grid_shape[0]), jnp.arange(self.grid_shape[1])), -1)
        grid_flat = grid_points.reshape(-1,2)
        idx = jax.random.choice(subkey, grid_flat.shape[0], (self.n_agents + self.n_tokens * self.n_token_repeat,), replace=False)
        locs = grid_flat[idx]
        agent_positions = locs[:self.n_agents]
        token_positions = locs[self.n_agents:].reshape(self.n_tokens, self.n_token_repeat, 2)
        state = TokenEnvState(agent_positions=agent_positions,
                         token_positions=token_positions,
                         is_alive=jnp.array([True for _ in jnp.arange(self.n_agents)]),
                         time=0)
        obs = self.get_obs(state=state)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: TokenEnvState,
        actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], TokenEnvState, Dict[str, float], Dict[str, bool], Dict]:

        _actions = jnp.array([actions[agent] for agent in self.agents])

        def move_agent(pos, a, is_agent_alive):
            return jnp.where(
                is_agent_alive,
                (pos + jnp.array(ACTION_MAP[a])) % self.grid_shape_arr,
                pos
            )

        new_positions = jax.vmap(move_agent, in_axes=(0, 0, 0))(state.agent_positions, _actions, state.is_alive)

        eq = (new_positions[:, None, :] == new_positions[None, :, :]).all(axis=-1)
        eq = eq.at[jnp.diag_indices(self.n_agents)].set(False)
        collisions = jnp.any(eq, axis=1)

        rewards = jnp.where(jnp.logical_and(state.is_alive, collisions), self.collision_reward, 0.0)
        rewards = {agent: rewards[i] for i, agent in enumerate(self.agents)}

        new_state = TokenEnvState(agent_positions=new_positions,
                             token_positions=state.token_positions,
                             is_alive=jnp.logical_and(state.is_alive, jnp.logical_not(collisions)),
                             time=state.time + 1)

        _dones = jnp.logical_or(collisions, new_state.time >= self.max_steps_in_episode)
        dones = {a: _dones[i] for i, a in enumerate(self.agents)}
        dones.update({"__all__": jnp.all(_dones)})

        obs = self.get_obs(state=new_state)
        info = {}

        return obs, new_state, rewards, dones, info

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(
        self,
        state: TokenEnvState
    ) -> Dict[str, chex.Array]:

        def obs_for_agent(i):
            base = jnp.zeros((self.n_tokens + self.n_agents - 1, *self.grid_shape), dtype=jnp.uint8)
            offset = (self.grid_shape_arr // 2) - state.agent_positions[i]

            def place_token(token_idx, val):
                rel = (state.token_positions[token_idx] + offset) % self.grid_shape_arr
                return val.at[token_idx, rel[:, 0], rel[:, 1]].set(1)
            b1 = jax.lax.fori_loop(0, self.n_tokens, place_token, base)

            def place_other(other_idx, val):
                rel = (state.agent_positions[other_idx + (other_idx >= i)] + offset) % self.grid_shape_arr
                return val.at[self.n_tokens + other_idx, rel[0], rel[1]].set(1)
            b2 = jax.lax.fori_loop(0, self.n_agents - 1, place_other, b1)

            return jnp.where(jnp.logical_or(jnp.logical_not(self.black_death), state.is_alive[i]), b2, base)

        obs = jax.vmap(obs_for_agent)(jnp.arange(self.n_agents))
        return {agent: obs[i] for i, agent in enumerate(self.agents)}

    @partial(jax.jit, static_argnums=(0,))
    def label_f(self, state: TokenEnvState) -> Dict[str, int]:

        diffs = state.agent_positions[:, None, None, :] - state.token_positions[None, :, :, :]
        matches = jnp.all(diffs == 0, axis=-1)
        matches_any = jnp.any(matches, axis=-1)

        has_match = jnp.any(matches_any, axis=1)
        token_idx = jnp.argmax(matches_any, axis=1)

        agent_token_matches = jnp.where(jnp.logical_and(has_match, state.is_alive), token_idx, -1)

        return {self.agents[agent_idx]: token_idx for agent_idx, token_idx in enumerate(agent_token_matches)}

    def render(self, state: TokenEnvState):
        empty_cell = "."
        grid = np.full(self.grid_shape, empty_cell, dtype=object)

        for token, positions in enumerate(state.token_positions):
            for pos in positions:
                grid[pos[0], pos[1]] = f"{token}"

        for agent in range(self.n_agents):
            pos = state.agent_positions[agent]
            current = grid[pos[0], pos[1]]
            if current == empty_cell:
                grid[pos[0], pos[1]] = f"A_{agent}"
            else:
                grid[pos[0], pos[1]] = f"A_{agent},{current}"

        max_width = max(len(str(cell)) for row in grid for cell in row)

        out = ""
        h_line = "+" + "+".join(["-" * (max_width + 2) for _ in range(self.grid_shape[1])]) + "+"
        out += h_line + "\n"
        for row in grid:
            row_str = "| " + " | ".join(f"{str(cell):<{max_width}}" for cell in row) + " |"
            out += row_str + "\n"
            out += h_line + "\n"

        print(out)

