import numpy as np
import gymnasium as gym
from token_env import TokenEnv
from pettingzoo.utils import ParallelEnv


def gym2zoo(env, black_death=True):
    return ParallelTokenEnv(env=env, black_death=black_death)

class ParallelTokenEnv(ParallelEnv):
    metadata = {"render_modes": [], "name": "parallel_tokenenv"}

    def __init__(self, env, black_death=True, render_mode: str | None = None):
        self.env = env
        self.black_death = black_death
        self.render_mode = render_mode

        # Agent identifiers
        self.possible_agents = list(self.env.possible_agents)

    def reset(self, seed=None, options=None):
        obss, infos = self.env.reset(seed=seed, options=options)

        self.agents = self.possible_agents[:]

        if not self.black_death:
            return obss, infos

        black_obs = {
            agent: {"obs": np.zeros_like(self.observation_space(agent)["obs"].low),
                    "dfa_obs": np.zeros_like(self.observation_space(agent)["dfa_obs"].low)}
            for agent in self.agents
            if agent not in obss
        }

        return {**obss, **black_obs}, infos

    def step(self, actions):
        active_actions = {agent: actions[agent] for agent in self.env.agents}
        obss, rews, terms, truncs, infos = self.env.step(active_actions)

        if not self.black_death:
            return obss, rews, terms, truncs, infos

        black_obs = {
            agent: {"obs": np.zeros_like(self.observation_space(agent)["obs"].low),
                    "dfa_obs": np.zeros_like(self.observation_space(agent)["dfa_obs"].low)}
            for agent in self.agents
            if agent not in obss
        }
        black_rews = {agent: 0 for agent in self.agents if agent not in obss}
        black_infos = {agent: {} for agent in self.agents if agent not in obss}
        terminations = np.fromiter(terms.values(), dtype=bool)
        truncations = np.fromiter(truncs.values(), dtype=bool)
        env_is_done = (terminations | truncations).all() # It was previously (terminations & truncations).all()
        total_obs = {**black_obs, **obss}
        total_rews = {**black_rews, **rews}
        total_infos = {**black_infos, **infos}
        total_dones = {agent: env_is_done for agent in self.agents}
        if env_is_done:
            self.agents.clear()
        return total_obs, total_rews, total_dones, total_dones, total_infos

    def observation_space(self, agent):
        return self.env.observation_space[agent]
    
    def action_space(self, agent):
        return self.env.action_space[agent]

    def render(self, agent=None):
        return self.env.render(agent=agent)

