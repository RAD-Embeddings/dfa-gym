from dfa_gym.dfa_env import *
from dfa_gym.dfa_wrapper import *
from dfa_gym.dfa_bisim_env import *

from dfa_samplers import RADSampler
from gymnasium.envs.registration import register

from rad_embeddings import Encoder

register(
    id='DFAEnv-v0',
    entry_point='dfa_gym.dfa_env:DFAEnv',
    kwargs = {"sampler": RADSampler(n_tokens=12), "timeout": 75}
)

register(
    id='DFAEnv-v1',
    entry_point='dfa_gym.dfa_env:DFAEnv'
)


register(
    id='DFABisimEnv-v1',
    entry_point='dfa_gym.dfa_bisim_env:DFABisimEnv'
)


register(
    id='DFABisimEnv-v2',
    entry_point='dfa_gym.dfa_bisim2_env:DFABisim2Env'
)

