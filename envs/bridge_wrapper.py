from gym_minigrid.envs.bridge import BridgeEnvStochastic9x9
import numpy as np

def generate_bridge_stochastic():
    return BridgeStochasticWrapper()


class BridgeStochasticWrapper(BridgeEnvStochastic9x9):
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


if __name__ == '__main__':
    env = BridgeStochasticWrapper()
    while True:
        env.render(mode='human', close=False)
        env.step(1)