import numpy as np
from envs.FiniteMDP import FiniteMDP
from gym_minigrid.register import register


def generate_toy(**game_params):
    if game_params is None:
        game_params = {}
    return ToyEnv(**game_params)


class ToyEnv(FiniteMDP):
    def __init__(self, scale_reward=False, gamma=0.99, horizon=2):
        self.horizon = horizon = 2
        nA = 2
        nS = 7
        p = compute_probabilities(nS, nA)
        r = compute_rewards(nS, nA)
        if scale_reward:
            r /= np.max(r)
        mu = compute_mu(nS)
        super().__init__(p, r, mu, gamma, horizon)


def compute_probabilities(nS, nA):
    p = np.zeros((nS, nA, nS))
    p[0, 0, 1] = 1
    p[0, 1, 2] = p[0, 1, 3] = 0.5
    p[1, 1, 4] = p[1, 0, 4] = 1
    p[2, 0, 5] = p[2, 1, 6] = 1
    p[3, 0, 6] = p[3, 1, 5] = 1
    p[4, 0, 4] = p[4, 1, 4] = p[5, 0, 5] = p[5, 1, 5] = p[6, 0, 6] = p[6, 1, 6] = 1
    return p


def compute_rewards(nS, nA):
    r = np.zeros((nS, nA, nS))
    r[1, 0, 4] = r[1, 1, 4] = 1
    r[2, 0, 5] = r[3, 1, 5] = -2
    r[3, 0, 6] = r[2, 1, 6] = 2
    return r


def compute_mu(nS):
    mu = np.zeros(nS)
    mu[0] = 1.
    return mu


register(
    id='MiniGrid-ToyEnv-v0',
    entry_point='envs.toy_env:ToyEnv'
)