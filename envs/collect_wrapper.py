import gym_minigrid
import gym
import numpy as np
from gym_minigrid.envs.collect import CollectEnvStochastic9x9

def generate_collect_stochastic():
    return CollectStochasticWrapper()

class CollectStochasticWrapper(CollectEnvStochastic9x9):

    def __init__(self):
        super().__init__()

    def get_state(self):
        return self.agent_pos, self.agent_dir

    def get_signature(self):
        sig = {'agent_pos': np.copy(self.agent_pos),
               'agent_dir': np.copy(self.agent_dir)}
        return sig

    def set_signature(self, sig):
        self.agent_pos = np.copy(sig['agent_pos'])
        self.agent_dir = np.copy(sig['agent_dir'])
