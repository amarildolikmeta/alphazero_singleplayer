import numpy as np
from mdp import random_biased_mdp, random_mdp
from matplotlib import pyplot
import time
from test_simple_estimator import Estimator


if __name__ == '__main__':
    action_length = 10
    num_actions = 3
    num_states = 10
    gamma = 1.
    alpha = 0.0
    delta_alpha = 0.1
    max_alpha = 1.05
    n = 200
    budget = 500
    bins = 50
    true_mean_samples = 10000

    action_sequence = np.random.choice(num_actions, size=action_length)
    mdp = random_mdp(n_states=num_states, n_actions=num_actions)
    P2 = np.zeros((num_states, num_actions, num_states))

    for s in range(num_states):
        for a in range(num_actions):
            ps = np.ones(num_states)
            ps[s] = 0  # don't stay in same state
            ps = ps / ps.sum()
            next_state = np.random.choice(num_states, p=ps)
            P2[s, a, next_state] = 1

    signature = mdp.get_signature()
    P = np.array(mdp.P)
    alphas = []
    ys_mc = []

    ys_particle_simple = []
    samples_p_simple = []
    ess_p_simple = []

    ys_particle_bh = []
    samples_p_bh = []
    ess_p_bh = []
    while alpha < max_alpha:
        # reset mdp
        new_P = (1 - alpha) * P + alpha * P2
        new_P = new_P / new_P.sum(axis=-1)[:, :, np.newaxis]
        mdp.P = new_P
        mdp.set_signature(signature)
        estimator = Estimator(mdp, action_sequence, gamma=gamma)

        estimations_mc = estimator.run_monte_carlo_estimation(true_mean_samples, action_length)
        mean = np.mean(estimations_mc)
        std_hat = np.std(estimations_mc, ddof=1)

        estimations_mc = estimator.run_monte_carlo_estimation(n, budget)
        error = ((np.array(estimations_mc) - mean) ** 2).mean()
        ys_mc.append(error)
        print("Finished MC with alpha=" + str(alpha))

        estimations_particle, ess, _, counts = estimator.run_particle_estimation(n, budget, bh=False)
        error = ((np.array(estimations_particle) - mean) ** 2).mean()
        ys_particle_simple.append(error)
        samples_p_simple.append(np.mean(counts))
        ess_p_simple.append(np.mean(ess))
        print("Finished Particle Simple with alpha=" + str(alpha))

        estimations_particle, ess, _, counts = estimator.run_particle_estimation(n, budget, bh=True)
        error = ((np.array(estimations_particle) - mean) ** 2).mean()
        ys_particle_bh.append(error)
        samples_p_bh.append(np.mean(counts))
        ess_p_bh.append(np.mean(ess))
        print("Finished Particle BH with alpha=" + str(alpha))

        print("Finished alpha " + str(alpha))
        alphas.append(alpha)
        alpha += delta_alpha

    xs = np.array(alphas)
    pyplot.plot(xs, ys_mc, alpha=0.8, label='MC error', marker='x')
    pyplot.plot(xs, ys_particle_simple, alpha=0.5, label='particle_simple ', marker='o')
    pyplot.plot(xs, ys_particle_bh, alpha=0.5, label='particle_bh', marker='o')
    pyplot.legend(loc='upper right')
    pyplot.xlabel("Samples")
    pyplot.ylabel("Error")
    pyplot.savefig("Error_alpha.pdf")
    pyplot.show()
