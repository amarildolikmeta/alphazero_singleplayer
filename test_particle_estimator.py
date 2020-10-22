import numpy as np
from mdp import random_mdp
from matplotlib import pyplot
from test_simple_estimator import Estimator
import time
import multiprocessing
import os
import pickle

if __name__ == '__main__':
    action_length = 10
    num_actions = 3
    num_states = 10
    gamma = 1.
    alpha = 0.0
    delta_alpha = 0.1
    max_alpha = 1.05
    n = 200
    budget = 50
    bins = 50
    true_mean_samples = 10000
    min_alpha = alpha
    max_workers = 10
    num_deterministic = 7

    action_sequence = np.random.choice(num_actions, size=action_length)
    mdp = random_mdp(n_states=num_states, n_actions=num_actions)
    mdp.P0 = np.zeros(num_states)
    mdp.P0[0] = 1
    deterministic_P = np.random.rand(num_states, num_actions,  num_states)
    deterministic_P = deterministic_P / deterministic_P.sum(axis=-1)[:, :, np.newaxis]

    for s in range(num_deterministic):
        next_state = s + 1
        deterministic_P[s, action_sequence[s], :] = 0
        deterministic_P[s, action_sequence[s], next_state] = 1

    mdp.reset()
    signature = mdp.get_signature()
    random_P = np.array(mdp.P)

    alphas = []
    ys_mc = []
    stds_mc = []

    ys_particle_simple = []
    samples_p_simple = []
    ess_p_simple = []

    ys_particle_bh = []
    stds_bh = []
    samples_p_bh = []
    ess_p_bh = []


    def evaluate(alpha):
        new_P = (1 - alpha) * random_P + alpha * deterministic_P
        new_P = new_P / new_P.sum(axis=-1)[:, :, np.newaxis]
        mdp.P = new_P
        mdp.reset()
        mdp.set_signature(signature)
        estimator = Estimator(mdp, action_sequence, gamma=gamma)

        estimations_mc = estimator.run_monte_carlo_estimation(true_mean_samples, action_length)
        mean = np.mean(estimations_mc)

        estimations_mc = estimator.run_monte_carlo_estimation(n, budget)
        error_mc = ((np.array(estimations_mc) - mean) ** 2).mean()
        error_mc_std = ((np.array(estimations_mc) - mean) ** 2).std()
        print("Finished MC with alpha=" + str(alpha))

        # estimations_particle, ess, _, counts = estimator.run_particle_estimation(n, budget, bh=False)
        # error_simple = ((np.array(estimations_particle) - mean) ** 2).mean()
        # counts_simple = np.mean(counts)
        # ess_simple = np.mean(ess)
        # print("Finished Particle Simple with alpha=" + str(alpha))

        estimations_particle, ess, _, counts = estimator.run_particle_estimation(n, budget, bh=True)
        error_bh = ((np.array(estimations_particle) - mean) ** 2).mean()
        error_bh_std = ((np.array(estimations_particle) - mean) ** 2).std()
        counts_bh = np.mean(counts)
        ess_bh = np.mean(ess)
        print("Finished Particle BH with alpha=" + str(alpha))
        print("Finished alpha " + str(alpha))

        return error_mc, error_mc_std, error_bh, error_bh_std, counts_bh, ess_bh# error_simple, counts_simple, ess_simple,

    while alpha < max_alpha:
        # reset mdp
        # new_P = (1 - alpha) * random_P + alpha * deterministic_P
        # new_P = new_P / new_P.sum(axis=-1)[:, :, np.newaxis]
        # mdp.P = new_P
        # mdp.reset()
        # mdp.set_signature(signature)
        # estimator = Estimator(mdp, action_sequence, gamma=gamma)
        #
        # estimations_mc = estimator.run_monte_carlo_estimation(true_mean_samples, action_length)
        # mean = np.mean(estimations_mc)
        # std_hat = np.std(estimations_mc, ddof=1)
        #
        # estimations_mc = estimator.run_monte_carlo_estimation(n, budget)
        # error = ((np.array(estimations_mc) - mean) ** 2).mean()
        # ys_mc.append(error)
        # print("Finished MC with alpha=" + str(alpha))
        #
        # estimations_particle, ess, _, counts = estimator.run_particle_estimation(n, budget, bh=False)
        # error = ((np.array(estimations_particle) - mean) ** 2).mean()
        # ys_particle_simple.append(error)
        # samples_p_simple.append(np.mean(counts))
        # ess_p_simple.append(np.mean(ess))
        # print("Finished Particle Simple with alpha=" + str(alpha))
        #
        # estimations_particle, ess, _, counts = estimator.run_particle_estimation(n, budget, bh=True)
        # error = ((np.array(estimations_particle) - mean) ** 2).mean()
        # ys_particle_bh.append(error)
        # samples_p_bh.append(np.mean(counts))
        # ess_p_bh.append(np.mean(ess))
        # print("Finished Particle BH with alpha=" + str(alpha))
        #
        # print("Finished alpha " + str(alpha))
        alphas.append(alpha)
        alpha += delta_alpha

    out_dir = 'logs/particle_estimator_alpha_exp/'
    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        res = []

    # Run the evaluation on multiple threads
    num_alpha = len(alphas)
    start = time.time()
    n_workers = min(num_alpha, multiprocessing.cpu_count())
    n_workers = min(n_workers, max_workers)


    iterations = max(num_alpha // n_workers, 1)
    remainder = num_alpha % n_workers if n_workers < num_alpha else 0
    print(alphas)
    for it in range(iterations):
        p = multiprocessing.Pool(n_workers)
        results = p.starmap(evaluate, [(alpha,) for alpha in alphas[it * n_workers: (it + 1) * n_workers]])
        print("Time to perform evaluation episodes:", time.time() - start, "s")

        # Unpack results
        for r in results:
            ys_mc.append(np.array(r[0]))
            stds_mc.append(r[1])
            ys_particle_bh.append(r[2])
            stds_bh.append(r[3])
            samples_p_bh.append(np.array(r[4]))
            ess_p_bh.append(r[5])
            # ys_particle_simple.append(np.array(r[4]))
            # samples_p_simple.append(np.array(r[5]))
            # ess_p_simple.append(r[6])
        p.close()
    if remainder > 0:
        p = multiprocessing.Pool(remainder)
        results = p.starmap(evaluate, [(alpha,) for alpha in alphas[-remainder:]])
        print("Time to perform evaluation episodes:", time.time() - start, "s")
        # Unpack results
        for r in results:
            ys_mc.append(np.array(r[0]))
            stds_mc.append(r[1])
            ys_particle_bh.append(r[2])
            stds_bh.append(r[3])
            samples_p_bh.append(np.array(r[4]))
            ess_p_bh.append(r[5])
            # ys_particle_simple.append(np.array(r[4]))
            # samples_p_simple.append(np.array(r[5]))
            # ess_p_simple.append(r[6])
    xs = np.array(alphas)

    data = {
        "alphas": alphas,
        "ys_mc": ys_mc,
        "stds_mc": stds_mc,
        "ys_particle_bh": ys_particle_bh,
        "stds_bh": stds_bh,
        "samples_p_bh": samples_p_bh,
        "ess_p_bh": ess_p_bh,
        # "ys_particle_simple": ys_particle_simple,
        # "samples_p_simple": samples_p_simple,
        # "ess_p_simple": ess_p_simple,
    }
    with open(out_dir + 'results.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    lower_mc = np.array(ys_mc) - 2 * np.array(stds_mc) / np.sqrt(n)
    upper_mc = np.array(ys_mc) + 2 * np.array(stds_mc) / np.sqrt(n)
    lower_bh = np.array(ys_particle_bh) - 2 * np.array(stds_bh) / np.sqrt(n)
    upper_bh = np.array(ys_particle_bh) + 2 * np.array(stds_bh) / np.sqrt(n)
    pyplot.plot(xs, ys_mc, label='MC', marker='x', color='c')
    # pyplot.fill_between(xs, lower_mc, upper_mc, alpha=0.2, color='c')
    # pyplot.plot(xs, ys_particle_simple, alpha=0.5, label='particle_simple ', marker='o')
    pyplot.plot(xs, ys_particle_bh, label='PARTICLE BH', marker='o', color='purple')
    # pyplot.fill_between(xs, lower_bh, upper_bh, alpha=0.2, color='purple')
    pyplot.legend(loc='upper right')
    pyplot.xlabel("alpha (larger -> more deterministic)")
    pyplot.ylabel("Error")
    pyplot.savefig(out_dir + "Error_alpha.pdf")
    pyplot.show()
