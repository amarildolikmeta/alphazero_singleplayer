import numpy as np
from envs.FiniteMDP import FiniteMDP


def generate_river(**game_params):
    if game_params is None:
        game_params = {}
    return RiverSwim(**game_params)


class RiverSwim(FiniteMDP):

    def __init__(self, dim=6, gamma=0.95, small=1, large=100, horizon=10, fail_prob=0.5, scale_reward=True):
        nA = 2
        nS = dim
        p = compute_probabilities(nS, nA, fail_prob)
        r = compute_rewards(nS, nA, small, large)
        if scale_reward:
            r /= large
        mu = compute_mu(nS)
        super(RiverSwim, self).__init__(p, r, mu, gamma, horizon)
        self.reset()


def compute_probabilities(nS, nA, fail_prob=0.1):
    p = np.zeros((nS, nA, nS))
    for i in range(1, nS):
        p[i, 0, i - 1] = 1
        if i != nS - 1:
            p[i, 1, i - 1] = 0.2 * fail_prob  # 0.1
            p[i, 1, i] = 0.8 * fail_prob  # 0.6
        else:
            p[i, 1, i - 1] = fail_prob  # 0.7
            p[i, 1, i] = 1 - fail_prob  # 0.3
    for i in range(nS - 1):
        p[i, 1, i + 1] = 1 - fail_prob  # 0.3
    # state 0
    p[0, 0, 0] = 1
    p[0, 1, 0] = fail_prob  # 0.7

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


if __name__ == '__main__':
    gamma = 1
    fail_prob = 0
    horizon = 20
    delta_fail = 0.1
    means = []
    stds = []
    n = 500
    action_labels = ['left', 'right']
    xs = []
    while fail_prob < 0.91:
        mdp = RiverSwim(dim=7, scale_reward=True, horizon=horizon, gamma=gamma, fail_prob=fail_prob)
        row_mean = []
        row_std = []
        for a in [0, 1]:
            results = []
            for i in range(n):
                ret = 0
                mdp.reset()
                for t in range(horizon):
                    _, r, _, _ = mdp.step(a)
                    ret += gamma ** t * r
                results.append(ret)
            row_mean.append(np.mean(results))
            row_std.append(np.std(results))
        xs.append(fail_prob)
        means.append(row_mean)
        stds.append(row_std)
        fail_prob += delta_fail
    from matplotlib import pyplot as plt

    means = np.array(means)
    stds = np.array(stds)
    errors = stds / np.sqrt(n)
    colors = ['r', 'b']
    for i, action_label in enumerate(action_labels):
        plt.plot(xs, means[:, i], label=action_label, color=colors[i])
        plt.fill_between(xs, means[:, i] - 2 * errors[:, i], means[:, i] + 2 * errors[:, i], alpha=0.2,
                         color=colors[i])
    plt.legend()
    plt.show()