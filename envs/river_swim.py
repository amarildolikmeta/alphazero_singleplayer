import numpy as np
from envs.FiniteMDP import FiniteMDP
from gym_minigrid.register import register


def generate_river(n=6, gamma=0.95, small=5, large=10000, horizon=10, scale_reward=True):
    return RiverSwim(n, gamma, small, large, horizon, scale_reward)


class RiverSwim(FiniteMDP):
    def __init__(self, n=6, gamma=0.95, small=5, large=10000, horizon=10, scale_reward=True):
        nA = 2
        nS = n
        p = compute_probabilities(nS, nA)
        r = compute_rewards(nS, nA, small, large)
        if scale_reward:
            r /= large
        mu = compute_mu(nS)
        super().__init__(p, r, mu, gamma, horizon)

def compute_probabilities(nS, nA):
    p = np.zeros((nS, nA, nS))
    for i in range(1, nS):
        p[i, 0, i - 1] = 1
        if i != nS - 1:
            p[i, 1, i - 1] = 0.1
            p[i, 1, i] = 0.6
        else:
            p[i, 1, i - 1] = 0.7
            p[i, 1, i] = 0.3
    for i in range(nS - 1):
        p[i, 1, i + 1] = 0.3
    # state 0
    p[0, 0, 0] = 1
    p[0, 1, 0] = 0.7

    return p


def compute_rewards(nS, nA, small, large):
    r = np.zeros((nS, nA, nS))
    r[0, 0, 0] = small
    r[nS - 1, 1, nS - 1] = large
    return r


def compute_mu(nS):
    mu = np.zeros(nS)
    mu[1] = 0.5
    mu[2] = 0.5
    return mu


register(
    id='MiniGrid-RiverSwim-v0',
    entry_point='envs.river_swim:RiverSwim'
)