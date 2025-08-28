import jax
import chex
import numpy as np
import jax.numpy as jnp
from flax import struct
from functools import partial
from typing import Tuple, Dict
from dfa_gym import spaces
from dfa_gym.env import MultiAgentEnv, State


ACTION_MAP = jnp.array([
    [ 1,  0], # DOWN
    [ 0,  1], # RIGHT
    [-1,  0], # UP
    [ 0, -1], # LEFT
    [ 0,  0]] # NOOP
)

@struct.dataclass
class TokenEnvState(State):
    agent_positions: jax.Array
    token_positions: jax.Array
    wall_positions: jax.Array
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
        collision_reward: int | None = None,
        black_death: bool = True,
        is_circular: bool = False,
        is_walled: bool = False
    ) -> None:
        super().__init__(num_agents=n_agents)
        assert not is_walled or grid_shape[1] >= 3
        assert (grid_shape[0] * grid_shape[1]) >= (n_agents + n_tokens * n_token_repeat)
        self.n_agents = n_agents
        self.n_tokens = n_tokens
        self.n_token_repeat = n_token_repeat
        self.grid_shape = grid_shape
        self.grid_shape_arr = jnp.array(self.grid_shape)
        self.fixed_map_seed = fixed_map_seed
        self.max_steps_in_episode = max_steps_in_episode
        self.collision_reward = collision_reward
        self.black_death = black_death
        self.is_circular = is_circular
        self.is_walled = is_walled

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

        n_walls = (self.grid_shape[1] - 1) // 2
        n_wall_cells = (self.grid_shape[0] - 1) * n_walls
        wall_positions = jnp.full((n_wall_cells, 2), -1)
        if self.is_walled:
            def sample_wall_idx(key):
                n_walls = (self.grid_shape[1] - 1) // 2
                wall_freq = self.grid_shape[1] // n_walls

                key, subkey = jax.random.split(key)
                wall_mark = jax.random.randint(subkey, (), 2, wall_freq) - 1

                wall_idx = wall_mark + wall_freq * jnp.arange(n_walls)
                return wall_idx

            key, subkey = jax.random.split(key)
            wall_idx = sample_wall_idx(subkey)

            door_idx = jax.random.randint(subkey, wall_idx.shape, 0, self.grid_shape[0])

            is_wall_grid = jnp.zeros(self.grid_shape).at[:, wall_idx].set(1).at[door_idx, wall_idx].set(0)
            _, flat_idx = jax.lax.top_k(is_wall_grid.flatten(), n_wall_cells)
            wall_positions = jnp.stack(jnp.divmod(flat_idx, self.grid_shape[1]), axis=-1)

        grid_points = jnp.stack(jnp.meshgrid(jnp.arange(self.grid_shape[0]), jnp.arange(self.grid_shape[1])), -1)
        grid_flat = grid_points.reshape(-1, 2)

        is_avail = jnp.all(
            jnp.any(
                grid_flat[:, None, :] != wall_positions[None, :, :] # [N, M, 2]
            , axis=-1) # [N, M]
        , axis=-1) # [N,]

        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, grid_flat.shape[0])
        idx = jnp.argsort(is_avail[perm], descending=True)
        grid_flat_sorted = grid_flat[perm][idx]

        agent_positions = grid_flat_sorted[:self.n_agents]
        token_positions = grid_flat_sorted[self.n_agents: self.n_agents + self.n_tokens * self.n_token_repeat].reshape(self.n_tokens, self.n_token_repeat, 2)

        state = TokenEnvState(
            agent_positions=agent_positions,
            token_positions=token_positions,
            wall_positions=wall_positions,
            is_alive=jnp.array([True for _ in jnp.arange(self.n_agents)]),
            time=0
        )

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

        # No agents
        def move_agent(pos, a, is_agent_alive, wall_positions):
            new_pos = pos + ACTION_MAP[a]
            new_pos_circ = new_pos % self.grid_shape_arr

            if self.is_walled:
                wall_col = jnp.any(
                    jnp.all(
                        new_pos[None, :] == wall_positions # [N, 2]
                    , axis=-1) # [N,]
                , axis=-1) # [1,]
                new_pos = jnp.where(wall_col, pos, new_pos)
                wall_col_circ = jnp.any(
                    jnp.all(
                        new_pos_circ[None, :] == wall_positions # [N, 2]
                    , axis=-1) # [N,]
                , axis=-1) # [1,]
                new_pos_circ = jnp.where(wall_col_circ, pos, new_pos_circ)

            if self.is_circular:
                return jnp.where(
                    is_agent_alive,
                    new_pos_circ,
                    pos
                )
            else:
                return jnp.where(
                    jnp.logical_and(is_agent_alive, jnp.all(new_pos == new_pos_circ)),
                    new_pos,
                    pos
                )

        new_agent_pos = jax.vmap(move_agent, in_axes=(0, 0, 0, None))(state.agent_positions, _actions, state.is_alive, state.wall_positions)

        # Handle collisions
        # TODO: When collision_reward is not None, there might be unintended behavior.
        # +-------+-------+-------+-------+-------+-------+-------+
        # | 0     | #     | .     | #     | .     | #     | 2     |
        # +-------+-------+-------+-------+-------+-------+-------+
        # | .     | #     | .     | #     | 1     | #     | .     |
        # +-------+-------+-------+-------+-------+-------+-------+
        # | 4     | #     | 9     | 5     | 3     | #     | 7     |
        # +-------+-------+-------+-------+-------+-------+-------+
        # | .     | .     | 8     | #     | .     | #     | 7     |
        # +-------+-------+-------+-------+-------+-------+-------+
        # | 6     | #     | 2     | #     | 9     | #     | 3     |
        # +-------+-------+-------+-------+-------+-------+-------+
        # | 8     | #     | 0     | #     | .     | #     | A_1,5 |
        # +-------+-------+-------+-------+-------+-------+-------+
        # | .     | #     | 1     | #     | 4     | A_2   | A_0,6 |
        # +-------+-------+-------+-------+-------+-------+-------+
        # Action for agent_0
        # 3
        # Action for agent_1
        # 0
        # Action for agent_2
        # 1
        # Gives
        # {'agent_0': Array(-100., dtype=float32), 'agent_1': Array(-100., dtype=float32), 'agent_2': Array(-100., dtype=float32)}
        # {'__all__': Array(True, dtype=bool), 'agent_0': Array(True, dtype=bool), 'agent_1': Array(True, dtype=bool), 'agent_2': Array(True, dtype=bool)}
        def compute_collisions(mask):
            positions = jnp.where(mask[:, None], state.agent_positions, new_agent_pos)

            collision_grid = jnp.zeros(self.grid_shape)
            collision_grid, _ = jax.lax.scan(
                lambda grid, pos: (grid.at[pos[0], pos[1]].add(1), None),
                collision_grid,
                positions,
            )

            collision_mask = collision_grid > 1

            collisions = jax.vmap(lambda p: collision_mask[p[0], p[1]])(positions)
            return jnp.logical_and(state.is_alive, collisions)

        collisions = jax.lax.while_loop(
            lambda mask: jnp.any(compute_collisions(mask)),
            lambda mask: jnp.logical_or(mask, compute_collisions(mask)),
            jnp.zeros((self.n_agents,), dtype=bool)
        )

        if self.collision_reward is None:
            new_agent_pos = jnp.where(collisions[:, None], state.agent_positions, new_agent_pos)
            collisions = jnp.full(collisions.shape, False)

        # Handle swaps
        def compute_swaps(original_positions, new_positions):
            original_pos_expanded = jnp.expand_dims(original_positions, axis=0)
            new_pos_expanded = jnp.expand_dims(new_positions, axis=1)

            swap_mask = (original_pos_expanded == new_pos_expanded).all(axis=-1)
            swap_mask = jnp.fill_diagonal(swap_mask, False, inplace=False)

            swap_pairs = jnp.logical_and(swap_mask, swap_mask.T)

            swaps = jnp.any(swap_pairs, axis=0)
            return swaps

        swaps = compute_swaps(state.agent_positions, new_agent_pos)
        new_agent_pos = jnp.where(swaps[:, None], state.agent_positions, new_agent_pos)

        _rewards = jnp.zeros((self.n_agents,), dtype=jnp.float32)
        if self.collision_reward is not None:
            _rewards = jnp.where(jnp.logical_and(state.is_alive, collisions), self.collision_reward, _rewards)
        rewards = {agent: _rewards[i] for i, agent in enumerate(self.agents)}

        new_state = TokenEnvState(
            agent_positions=new_agent_pos,
            token_positions=state.token_positions,
            wall_positions=state.wall_positions,
            is_alive=jnp.logical_and(state.is_alive, jnp.logical_not(collisions)),
            time=state.time + 1
        )

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
        wall_cell = "#"
        grid = np.full(self.grid_shape, empty_cell, dtype=object)

        for pos in state.wall_positions:
            grid[pos[0], pos[1]] = f"{wall_cell}"

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

