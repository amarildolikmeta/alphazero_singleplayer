import numpy as np
from rl.make_game import make_game
from mdp import random_mdp
from matplotlib import pyplot

class Particle(object):
    """Class storing information about a particle"""

    def __init__(self, state, signature, reward, terminal, weight, parent_particle=None):
        self.state = state
        self.signature = signature
        self.reward = reward
        self.terminal = terminal
        self.weight = weight
        self.parent_particle = parent_particle

    def __str__(self):
        return str(self.state)


class Node:
    def __init__(self, depth=0,  parent_node=None):
        self.n = 0
        self.V = 0
        self.r = 0
        self.depth = depth
        self.num_particles = 0
        self.parent_node = parent_node
        self.child_node = None
        self.particles = []
        self.weights = []
        self.xs = []

    def add_child(self, node):
        self.child_node = node

    def add_particle(self, particle):
        self.particles.append(particle)
        self.r = particle.reward
        self.weights.append(particle.weight)
        self.num_particles += 1

    def update(self, r, weights):
        self.xs.append(r)
        if self.n == 0:
            self.V = r
        else:
            #self.V = (self.V * self.n + r * weight) / (self.n + weight)
            assert len(self.xs) == weights.shape[0], "Weights and samples don't match"
            self.V = np.dot(weights, self.xs)
        self.n += 1


class Estimator:
    def __init__(self, env, action_sequence, gamma=1.):
        self.env = env
        self.action_sequence = action_sequence
        self.gamma = gamma
        self.multi_steps_Ps = []
        self.compute_multi_steps()
        self.root = None
        self.full_cost = len(action_sequence)
        self.starting_state = self.env.reset()
        self.starting_signature = self.env.get_signature()
        self.reset()

    def reset(self):
        self.weights = []
        self.ess = 0
        self.J = 0
        self.ps = []
        self.qs = []
        self.xs = []

    def compute_ess(self, weights):
        #ess = np.linalg.norm(weights, ord=1) ** 2 / np.linalg.norm(weights, ord=2) ** 2
        weights = np.array(weights)
        weights = weights / weights.sum()
        ess = 1 / np.linalg.norm(weights, ord=2) ** 2
        return ess

    def get_weights_balance_heuristic(self):
        denom = np.sum(self.qs)
        weights = np.array(self.ps) / denom
        weights = weights / np.sum(weights)
        self.weights = weights
        return weights

    def get_new_weights_balance_heuristic(self, p, q):
        denom = np.sum(self.qs + [q])
        weights = np.array(self.ps + [p]) / denom
        weights = weights / np.sum(weights)
        return weights

    def get_sample_weight(self, p, q):
        return p / np.sum(self.qs + [q])

    def estimate_f(self, xs):
        assert len(xs) == self.weights.shape[0], "Weights and samples don't match"
        return np.dot(xs, self.weights)

    def should_resample(self, node, starting_state):
        particles = node.particles
        weights = node.weights
        candidate_particle = np.random.choice(particles, p=weights / np.sum(weights))
        ending_state = candidate_particle.state
        p_x = self.multi_step_model(starting_state, node.depth - 1, ending_state, candidate_particle)
        p_f_x = candidate_particle.weight / np.sum(weights)
        new_weight = self.get_sample_weight(p_x, p_f_x)
        new_weights = self.get_new_weights_balance_heuristic(p_x, p_f_x)
        ess = self.compute_ess(new_weights)
        full_resampling_weights = self.get_new_weights_balance_heuristic(p_x, p_x)
        sample_size = self.compute_ess(full_resampling_weights)
        full_cost = self.full_cost
        resampling_cost = full_cost - node.depth
        full_error_reduction = (1 / self.ess - 1 / sample_size)
        resampling_error_reduction = (1 / self.ess - 1 / ess)
        # assert sample_size > ess, "Sample sizes don't match"
        # assert full_error_reduction > resampling_error_reduction, "Error reduction costs don't match"
        should_resample = full_error_reduction / full_cost < resampling_error_reduction / resampling_cost
        margin = resampling_error_reduction / resampling_cost - full_error_reduction / full_cost
        if not should_resample:
            p_f_x = p_x
        return should_resample, candidate_particle, new_weight, margin, p_x, p_f_x

    def compute_multi_steps(self):
        P = self.env.P
        prev_p = np.eye(P.shape[0])
        for a in self.action_sequence:
            P_ = P[:, a, :]
            multi = np.dot(prev_p, P_)
            self.multi_steps_Ps.append(multi)
            prev_p = multi

    def multi_step_model(self, starting_state, depth, ending_state, particle):
        p = 1.
        while particle.parent_particle is not None:
            prev_state = particle.parent_particle.state
            state = particle.state
            p *= self.env.P[prev_state, self.action_sequence[depth], state]
            depth -= 1
            particle = particle.parent_particle
        assert depth == -1, "Something wrong"
        # return self.multi_steps_Ps[depth][starting_state, ending_state]
        return p

    def run_monte_carlo_estimation(self, n=1000, budget=1000):
        evaluations = []
        signature = self.starting_signature
        m = int(np.ceil(budget / self.full_cost))
        stds = []
        for i in range(n):
            #print("MC estimation #" + str(i + 1))
            self.env.set_signature(signature)
            self.env.seed()
            sum_returns = 0
            rets = []
            for j in range(m):
                ret = 0
                self.env.set_signature(signature)
                for t, a in enumerate(self.action_sequence):
                    s, r, _, _ = self.env.step(a)
                    ret += r * self.gamma ** t
                sum_returns += ret
                rets.append(ret)
            evaluations.append(sum_returns / m)
            stds.append(np.std(rets))

        return evaluations, stds

    def backup(self, node, particle):
        #weight = particle.weight
        weights = self.get_weights_balance_heuristic()
        R = 0
        while node.parent_node is not None:
            node.update(R, weights)
            R = particle.reward + self.gamma * R
            node = node.parent_node
            particle = particle.parent_particle
        node.update(R, weights)

    def resample_from_particle(self, node, particle, weight, depth, p, q):
        self.env.set_signature(particle.signature)
        self.env.seed()
        parent_particle = particle
        self.ps.append(p)
        self.qs.append(q)
        self.ess = self.compute_ess(self.get_weights_balance_heuristic())
        self.J += 1
        for d, a in enumerate(self.action_sequence[depth:]):
            s, r, done, _ = self.env.step(a)
            particle = Particle(s, self.env.get_signature(), r, done, weight=1,
                                parent_particle=parent_particle)
            new_node = node.child_node
            new_node.r = r
            new_node.add_particle(particle)
            node = new_node
            parent_particle = particle
        return particle

    def run_particle_estimation(self, n=1000, budget=1000, resample_prob=0.2):
        evaluations = []
        starting_state = self.starting_state
        signature = self.starting_signature
        resamplings = []
        depths = np.zeros(self.full_cost)
        counts = []
        ess = []
        std = []
        for i in range(n):
            # print("Particle estimation #" + str(i + 1))
            root_particle = Particle(starting_state, signature=signature, reward=0, terminal=False, weight=1)
            self.root = root = Node(depth=0)
            root.add_particle(root_particle)
            self.last_node = last_node = None
            self.ess = 0
            self.J = 0
            self.ps = []
            self.qs = []
            self.xs = []
            resampling_count = 0
            resampling_depths = np.zeros(self.full_cost)
            j = 0
            remaining_budget = budget
            count = 0
            while remaining_budget > 0:
                self.env.set_signature(signature)
                node = root
                if j == 0:
                    # first sample
                    parent_particle = root_particle
                    for depth, a in enumerate(action_sequence):
                        s, r, done, _ = self.env.step(a)
                        particle = Particle(s, self.env.get_signature(),  r, done, weight=1,
                                            parent_particle=parent_particle)
                        new_node = Node(parent_node=node, depth=depth+1)
                        new_node.add_particle(particle)
                        node.add_child(new_node)
                        node = new_node
                        parent_particle = particle
                    remaining_budget -= self.full_cost
                    j = 1
                    last_node = node
                    self.ps.append(1)
                    self.qs.append(1)
                    self.ess = 1
                    self.J = 1
                    self.get_weights_balance_heuristic()
                    self.backup(last_node, particle)
                    self.last_node = last_node
                else:
                    node = last_node.parent_node
                    resampled = False
                    max_margin = - np.inf
                    while node.parent_node is not None:
                        if np.random.random() < resample_prob:
                            should_resample, particle, weight, margin, p, q = self.should_resample(node, starting_state)
                            if should_resample:
                                resampled = True
                                if margin > max_margin:
                                    max_margin = margin
                                    starting_particle = particle
                                    sample_weight = weight
                                    sample_depth = node.depth
                                    starting_node = node
                                    sample_p = p
                                    sample_q = q
                            else:
                                root_sample_p = p
                                root_sample_q = q
                        else:
                            root_sample_p = 1
                            root_sample_q = 1
                        node = node.parent_node
                    if not resampled:
                        starting_particle = self.root.particles[0]
                        sample_weight = 1
                        sample_depth = 0
                        sample_p = root_sample_p
                        sample_q = root_sample_q
                    else:
                        resampling_count += 1
                        resampling_depths[sample_depth - 1] += 1
                        node = starting_node
                    generated_particle = self.resample_from_particle(node=node, particle=starting_particle,
                                                                     weight=sample_weight, depth=sample_depth,
                                                                     p=sample_p, q=sample_q)
                    remaining_budget -= (self.full_cost - sample_depth)
                    self.backup(self.last_node, generated_particle)
                count += 1

            evaluations.append(self.root.V)
            resamplings.append(resampling_count)
            depths += resampling_depths
            counts.append(count)
            ess.append(self.compute_ess(self.get_weights_balance_heuristic()))
        return evaluations, ess, depths / n, counts

    def basic_one_step_estimator(self, n=1000, budget=1000, m=20):
        evaluations = []
        starting_state = self.starting_state
        signature = self.starting_signature

        for i in range(n):
            #print("Particle estimation #" + str(i + 1))
            self.ess = 0
            particles = []
            weights = []
            root_particle = Particle(starting_state, signature=signature, reward=0, terminal=False, weight=1)
            for i in range(m):
                self.env.set_signature(signature)
                self.env.seed()
                s, r, done, _ = self.env.step(self.action_sequence[0])
                particle = Particle(s, self.env.get_signature(), r, done, weight=1,
                                    parent_particle=root_particle)
                particles.append(particle)
                weights.append(1)
            remaining_budget = budget
            samples = []
            sample_weights = []
            while remaining_budget > self.full_cost - 2:
                candidate_particle = np.random.choice(particles, p=weights / np.sum(weights))
                ending_state = candidate_particle.state
                p_x = self.multi_step_model(starting_state, 0, ending_state)
                p_f_x = candidate_particle.weight / np.sum(weights)
                new_weight = p_x / p_f_x
                self.env.set_signature(candidate_particle.signature)
                self.env.seed()
                self.weights.append(new_weight)
                self.ess = self.compute_ess(self.weights)
                ret = candidate_particle.reward
                for d, a in enumerate(self.action_sequence[1:]):
                    s, r, done, _ = self.env.step(a)
                    ret += r * self.gamma ** (d+1)
                samples.append(ret)
                sample_weights.append(new_weight)
                remaining_budget -= (self.full_cost - 1)
            samples = np.array(samples)
            sample_weights = np.array(sample_weights)
            estimate = np.sum(sample_weights * samples) / (np.sum(sample_weights))
            evaluations.append(estimate)
        return evaluations


if __name__ == '__main__':
    game_params = {'horizon': 20}
    action_length = 10
    action_sequence = np.random.choice(3, size=action_length)
    mdp = random_mdp(n_states=10, n_actions=3)
    estimator = Estimator(mdp, action_sequence, gamma=1.)
    n = 300

    budget = 300
    bins = 50

    print("Doing " + str(n) + " MC estimations with " + str(budget) + " samples")
    estimations_mc, stds = estimator.run_monte_carlo_estimation(n, budget)
    pyplot.hist(estimations_mc, bins, alpha=0.5, label='MC', density=True)
    #
    # print("Doing " + str(n) + " simple particle estimations with " + str(budget) + " samples")
    # estimations_p = estimator.basic_one_step_estimator(n, budget)
    # pyplot.hist(estimations_p, bins, alpha=0.5, label='Particle One Step')
    #
    print("Doing " + str(n) + " Particle estimations with " + str(budget) + " samples")
    estimations_particle, ess, depths, counts = estimator.run_particle_estimation(n, budget, resample_prob=1.)
    pyplot.hist(estimations_particle, bins, alpha=0.5, label='PARTICLE', density=True)
    pyplot.xlabel("Return")
    pyplot.legend(loc='upper right')
    pyplot.savefig('Estimators.pdf')
    pyplot.show()
    #
    pyplot.hist(ess, bins, alpha=0.5, label='ess')
    pyplot.legend(loc='upper right')
    pyplot.show()

    pyplot.plot(np.arange(action_length), depths, alpha=0.5, label='depths')
    pyplot.legend(loc='upper right')
    pyplot.show()

    pyplot.hist(counts, bins, alpha=0.5, label='particle samples')
    pyplot.legend(loc='upper right')
    pyplot.show()

    print("MC:" + str(np.mean(estimations_mc)) + " +/- " + str(np.std(estimations_mc)) + " with " + str(budget / action_length) + " samples")
    print("Particle:" + str(np.mean(estimations_particle)) + " +/- " + str(np.std(estimations_particle)) + " with " + str(np.mean(counts)) + " samples")

    ## Check Error rates
    budgets = [10, 20, 30, 40, 50, 70, 80, 100, 200, 300, 400, 500]
    budgets = budgets[:]
    n = 200
    ys_mc = []
    ys_p = []
    ys_particle = []
    ys_particle_2 = []
    ys_particle_3 = []
    ys_particle_4 = []
    true_mean_samples = 20000
    estimations_mc = estimator.run_monte_carlo_estimation(true_mean_samples, 10)
    mean = np.mean(estimations_mc)
    std_hat = np.std(estimations_mc, ddof=1)
    print("Mean=" + str(mean) + "+/- " + str(2 * std_hat / np.sqrt(true_mean_samples)))
    samples_p = []
    ess_p = []
    samples_p_2 = []
    ess_p_2 = []
    samples_p_3 = []
    ess_p_3 = []
    samples_p_4 = []
    ess_p_4 = []
    for b in budgets:
        estimations_mc = estimator.run_monte_carlo_estimation(n, b)
        error = ((np.array(estimations_mc) - mean) ** 2).mean()
        ys_mc.append(error)
        # estimations_p = estimator.basic_one_step_estimator(n, b)
        # error = ((np.array(estimations_p) - mean) ** 2).mean()
        # ys_p.append(error)
        estimations_particle, ess, _, counts = estimator.run_particle_estimation(n, b, resample_prob=0.2)
        error = ((np.array(estimations_particle) - mean) ** 2).mean()
        ys_particle.append(error)
        samples_p.append(np.mean(counts))
        ess_p.append(np.mean(ess))

        estimations_particle, ess, _, counts = estimator.run_particle_estimation(n, b, resample_prob=0.5)
        error = ((np.array(estimations_particle) - mean) ** 2).mean()
        ys_particle_2.append(error)
        samples_p_2.append(np.mean(counts))
        ess_p_2.append(np.mean(ess))

        estimations_particle, ess, _, counts = estimator.run_particle_estimation(n, b, resample_prob=0.05)
        error = ((np.array(estimations_particle) - mean) ** 2).mean()
        ys_particle_3.append(error)
        samples_p_3.append(np.mean(counts))
        ess_p_3.append(np.mean(ess))


        estimations_particle, ess, _, counts = estimator.run_particle_estimation(n, b, resample_prob=1.)
        error = ((np.array(estimations_particle) - mean) ** 2).mean()
        ys_particle_4.append(error)
        samples_p_4.append(np.mean(counts))
        ess_p_4.append(np.mean(ess))
        print("Finished budget " + str(b))

    xs = np.array(budgets) / action_length
    pyplot.plot(xs, ys_mc, alpha=0.5, label='MC error', marker='x')

    pyplot.plot(samples_p, ys_particle, alpha=0.5, label='particle_error(N)', marker='o')
    pyplot.plot(ess_p, ys_particle, alpha=0.5, label='particle_error(ess)', marker='o')
    # pyplot.plot(xs, ys_p, alpha=0.5, label='particle error')
    # if ys_mc[0] / ys_particle[0] < 0.95:
    #     pyplot.plot(ess_p, np.array(ys_particle) * (ys_mc[0] / ys_particle[0]), alpha=0.5,
    #                 label='particle_error_corrected(ess)', marker='o')
    pyplot.plot(xs,  std_hat**2 / np.array(xs), alpha=0.5, label='1/x')
    pyplot.legend(loc='upper right')
    pyplot.xlabel("Samples")
    pyplot.ylabel("Error")
    pyplot.savefig("Error.pdf")
    pyplot.show()

    # pyplot.plot(samples_p, ys_particle, alpha=0.5, label='particle_error(N)', marker='o')
    pyplot.plot(ess_p, ys_particle, alpha=0.5, label='p=0.2(ess)', marker='o')

    # pyplot.plot(samples_p_2, ys_particle_2, alpha=0.5, label='p=0.5(N)', marker='o')
    pyplot.plot(ess_p_2, ys_particle_2, alpha=0.5, label='p=0.5(ess)', marker='o')

    # pyplot.plot(samples_p_3, ys_particle_3, alpha=0.5, label='p=0.05(N)', marker='o')
    pyplot.plot(ess_p_3, ys_particle_3, alpha=0.5, label='p=0.05(ess)', marker='o')

    # pyplot.plot(samples_p_2, ys_particle_2, alpha=0.5, label='p=0.5(N)', marker='o')
    pyplot.plot(ess_p_4, ys_particle_4, alpha=0.5, label='p=1(ess)', marker='o')

    pyplot.plot(xs, std_hat ** 2 / np.array(xs), alpha=0.5, label='1/x')
    pyplot.legend(loc='upper right')
    pyplot.xlabel("Samples")
    pyplot.ylabel("Particle Error")
    pyplot.savefig("Error_p.pdf")
    pyplot.show()

    pyplot.plot(xs, samples_p, alpha=0.5, label='samples(budget)', marker='o')
    pyplot.plot(xs, ess_p, alpha=0.5, label='ess(budget)', marker='x')
    pyplot.legend(loc='lower right')
    pyplot.xlabel("Samples")
    pyplot.savefig("samples.pdf")
    pyplot.show()


    resample_probs = np.arange(0.05, 1.02, 0.05)
    budget = 200
    ys_particle = []
    samples_p = []
    ess_p = []
    for p in resample_probs:
        estimations_particle, ess, _, counts = estimator.run_particle_estimation(n, budget, resample_prob=p)
        error = ((np.array(estimations_particle) - mean) ** 2).mean()
        ys_particle.append(error)
        samples_p.append(np.mean(counts))
        ess_p.append(np.mean(ess))
        print("Finished resaple_prob " + str(p))


    pyplot.plot(resample_probs, ys_particle, alpha=0.5, label='particle error', marker='o')
    #ax1.plot(resample_probs, resample_probs**2, alpha=0.5, label='x^2', marker='x')
    #ax1.plot(resample_probs, np.ones_like(resample_probs) * ys_mc[-4], alpha=0.5, label='mc_error', marker='o')
    pyplot.xlabel("resample probability")
    pyplot.ylabel("Error")
    pyplot.legend(loc='lower right')
    pyplot.show()
    pyplot.savefig("Error_prob.pdf")

    pyplot.plot(resample_probs, samples_p, alpha=0.5, label='Number of Samples', marker='+')
    # pyplot.plot(resample_probs, resample_probs * (samples_p[-1] / resample_probs[-1]), alpha=0.5, label='x', marker='+')
    pyplot.plot(resample_probs, ess_p, alpha=0.5, label='ESS', marker='+')
    pyplot.ylabel('samples')
    pyplot.xlabel("resample probability")
    pyplot.legend(loc='lower right')
    pyplot.savefig("samples.pdf")
    pyplot.show()
