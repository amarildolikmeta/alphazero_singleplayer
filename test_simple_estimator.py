import numpy as np
from mdp import random_mdp
from matplotlib import pyplot
import time


def check_sub_trajectory(nu, distribution, depth):
    for particle in distribution:
        t = depth
        p = nu[t]
        sampled_p = particle
        while particle.state == p.state and particle.parent_particle is not None:  # and particle.reward == p.reward
            particle = particle.parent_particle
            t -= 1
            p = nu[t]
        if particle.parent_particle is None:
            return True, sampled_p
    return False, None


def compute_ess(weights):
    weights = np.array(weights) / np.sum(weights)
    ess = 1 / np.linalg.norm(weights, ord=2) ** 2
    return ess


class Particle(object):
    """Class storing information about a particle"""

    def __init__(self, state, signature, reward, terminal, weight, prob, parent_particle=None):
        self.state = state
        self.signature = signature
        self.reward = reward
        self.terminal = terminal
        self.weight = weight
        self.parent_particle = parent_particle
        self.prob = prob

    def __str__(self):
        return str(self.state)


class Node:
    def __init__(self, depth=0,  parent_node=None):
        self.n = 0
        self.V = 0
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
        self.weights.append(particle.weight)
        self.num_particles += 1

    def update(self, r, weights):
        self.xs.append(r)
        if self.n == 0:
            self.V = r
        else:
            assert len(self.xs) == weights.shape[0], "Weights and samples don't match"
            self.V = np.dot(weights, self.xs)
        self.n += 1


class Estimator:
    def __init__(self, env, action_sequence, gamma=1.):
        self.env = env
        self.action_sequence = action_sequence
        self.gamma = gamma
        self.root = None
        self.full_cost = len(action_sequence)
        self.starting_state = self.env.reset()
        self.starting_signature = self.env.get_signature()

    def reset(self, bh=False):
        self.ess = 0
        if bh:
            self.T = 0
            self.xs = []
            self.depths = []
            self.distributions = []
            self.trajectories = []
            self.particles = []
            self.ps = []
            self.partial_q_sums = []
        else:
            self.weights = []

    def compute_bh_weights_fast(self, trajectories, particles, depths, distributions, T):
        ps = [x for x in self.ps]
        partial_q_sums = [x for x in self.partial_q_sums]
        assert len(ps) == T - 1

        last_nu = trajectories[-1]
        last_particle = particles[-1]
        last_depth = depths[-1]
        last_distribution = distributions[-1]
        last_p_nu = last_particle.prob
        last_num_particles = distributions[-1][1]
        qs_new = []
        partial_q_sum_new = 0
        for j in range(T):
            # compute q_j(\nu_T)
            num_particles_j = distributions[j][1]
            if depths[j] > last_depth:
                qs_new.append(0)
            else:
                valid, valid_particle = check_sub_trajectory(nu=last_nu,
                                                             distribution=distributions[j][0][:num_particles_j],
                                                             depth=depths[j])
                # if i == j:
                #     assert valid, "Error in checking validity of behavioral distribution"
                if valid:
                    if depths[j] == last_depth:
                        p_nu_j = 1
                    else:
                        p_nu_j = last_p_nu / valid_particle.prob
                    qs_new.append(p_nu_j / num_particles_j)
                    partial_q_sum_new += p_nu_j / num_particles_j
                else:
                    qs_new.append(0)
            # compute q_T(\nu_j)
            if j == T - 1:
                break
            if last_depth > depths[j]:
                partial_q_sums[j] += 0
            else:
                valid, valid_particle = check_sub_trajectory(nu=trajectories[j],
                                                             distribution=last_distribution[0][:last_num_particles],
                                                             depth=last_depth)
                # if i == j:
                #     assert valid, "Error in checking validity of behavioral distribution"
                if valid:
                    if depths[j] == last_depth:
                        p_nu_j = 1
                    else:
                        p_nu_j = last_particle.prob / valid_particle.prob
                    partial_q_sums[j] + p_nu_j / last_num_particles
                else:
                    partial_q_sums[j] += 0

        partial_q_sums.append(partial_q_sum_new)
        ps.append(last_p_nu)
        new_weights = np.array(ps) / np.array(partial_q_sums)
        return new_weights, ps, partial_q_sums

    def compute_bh_weights(self, trajectories, particles, depths, distributions, T, fast=False):
        assert len(trajectories) == len(particles) and len(particles) == len(depths) and \
               len(depths) == len(distributions) and len(distributions) == T, "Error in computing bh weights"
        if fast:
            return self.compute_bh_weights_fast(trajectories, particles, depths, distributions, T)
        weights = []
        partial_sums = []
        ps = []
        for i in range(T):
            sample_depth = depths[i]
            nu = trajectories[i]
            p_nu = particles[i].prob
            ps.append(p_nu)
            qs = []
            for j in range(T):
                num_particles_j = distributions[j][1]
                if depths[j] > sample_depth:
                    qs.append(0)
                else:
                    valid, valid_particle = check_sub_trajectory(nu=nu, distribution=distributions[j][0][:num_particles_j],
                                                                 depth=depths[j])
                    # if i == j:
                    #     assert valid, "Error in checking validity of behavioral distribution"
                    if valid:
                        if depths[j] == sample_depth:
                            p_nu_j = 1
                        else:
                            p_nu_j = particles[i].prob / valid_particle.prob
                        qs.append(p_nu_j / num_particles_j)
                    else:
                        qs.append(0)
            partial_sum = np.sum(qs)
            weights.append(p_nu / partial_sum)
            partial_sums.append(partial_sum)
        weights = np.array(weights) / np.sum(weights)
        return weights, ps, partial_sums

    def get_weights_balance_heuristic(self, fast=False):
        trajectories = self.trajectories
        particles = self.particles
        depths = self.depths
        distributions = self.distributions
        T = self.T
        weights, ps, partial_sums = self.compute_bh_weights(depths=depths, distributions=distributions,
                                                            trajectories=trajectories,
                                                            particles=particles, T=T, fast=fast)
        self.ps = ps
        self.partial_q_sums = partial_sums
        self.weights = weights
        return weights

    def get_new_weights_balance_heuristic(self, trajectory, particle, depth, distribution, fast=False):
        trajectories = [x for x in self.trajectories] + [trajectory]
        particles = [x for x in self.particles] + [particle]
        depths = [x for x in self.depths] + [depth]
        distributions = [x for x in self.distributions] + [distribution]
        T = self.T + 1
        weights, _, _ = self.compute_bh_weights(depths=depths, distributions=distributions, trajectories=trajectories,
                                                particles=particles, T=T, fast=fast)
        return weights

    def get_new_weights_simple(self, new_weight):
        return [x for x in self.weights] + [new_weight]

    def should_resample(self, node, bh=False, full_resampling_weights=None, fast=False):
        particles = node.particles
        weights = node.weights
        candidate_particle = np.random.choice(particles, p=weights / np.sum(weights))
        p_x = candidate_particle.prob
        p_f_x = candidate_particle.weight / np.sum(weights)
        if bh:
            p = candidate_particle
            trajectory = [p]
            while p.parent_particle is not None:
                p = p.parent_particle
                trajectory.append(p)
            trajectory = trajectory[::-1]
            new_weights = self.get_new_weights_balance_heuristic(trajectory=trajectory, depth=node.depth,
                                                                 particle=candidate_particle,
                                                                 distribution=(node.particles, node.num_particles),
                                                                 fast=fast)

        else:
            new_weight = p_x / p_f_x
            new_weights = self.get_new_weights_simple(new_weight)

        ess = compute_ess(new_weights)
        sample_size = compute_ess(full_resampling_weights)

        full_cost = self.full_cost
        resampling_cost = full_cost - node.depth
        full_error_reduction = (1 / self.ess - 1 / sample_size)
        resampling_error_reduction = (1 / self.ess - 1 / ess)
        # if sample_size <= ess:
        #     print("What")
        # assert sample_size >= ess, "Sample sizes don't match"
        # assert full_error_reduction >= resampling_error_reduction, "Error reduction costs don't match"
        should_resample = full_error_reduction / full_cost < resampling_error_reduction / resampling_cost
        margin = resampling_error_reduction / resampling_cost - full_error_reduction / full_cost
        if should_resample:
            new_weight = new_weights[-1]
        else:
            new_weight = full_resampling_weights[-1]
        return should_resample, candidate_particle, new_weight, margin

    def multi_step_model(self,  depth,  particle, len=None):
        p = 1.
        while particle.parent_particle is not None:
            prev_state = particle.parent_particle.state
            state = particle.state
            p *= self.env.P[prev_state, self.action_sequence[depth], state]
            depth -= 1
            particle = particle.parent_particle
            if len is not None:
                len -= 1
                if len == 0:
                    break
        return p, depth

    def run_monte_carlo_estimation(self, n=1000, budget=1000):
        evaluations = []
        signature = self.starting_signature
        m = int(np.ceil(budget / self.full_cost))
        for i in range(n):
            self.env.set_signature(signature)
            self.env.seed()
            sum_returns = 0
            for j in range(m):
                ret = 0
                self.env.set_signature(signature)
                for t, a in enumerate(self.action_sequence):
                    s, r, _, _ = self.env.step(a)
                    ret += r * self.gamma ** t
                sum_returns += ret
            evaluations.append(sum_returns / m)
        return evaluations

    def backup(self, node, particle, weight, sampled_particle, sampled_node, bh=False, fast=False):
        if bh:
            last_node = node
            last_particle = particle
            trajectory = []
            while node.parent_node is not None:
                trajectory.append(particle)
                node = node.parent_node
                particle = particle.parent_particle
            trajectory.append(particle)
            self.trajectories.append(trajectory[::-1])
            node = last_node
            particle = last_particle
            self.particles.append(sampled_particle)
            self.distributions.append((sampled_node.particles, sampled_node.num_particles))
            self.depths.append(sampled_node.depth)
            self.T += 1
            weights = self.get_weights_balance_heuristic(fast=fast)
        else:
            self.weights.append(weight)
            weights = self.weights
        weights = np.array(weights) / np.sum(weights)
        self.ess = compute_ess(weights)
        R = 0
        while node.parent_node is not None:
            if not bh:
                node.update(R, weights)
            R = particle.reward + self.gamma * R
            node = node.parent_node
            particle = particle.parent_particle
        if bh:
            self.xs.append(R)
        else:
            node.update(R, weights)

    def resample_from_particle(self, node, particle):
        self.env.set_signature(particle.signature)
        self.env.seed()
        depth = node.depth
        parent_particle = particle
        prob = particle.prob
        prev_state = particle.state
        for a in self.action_sequence[depth:]:
            s, r, done, _ = self.env.step(a)
            prob = prob * self.env.P[prev_state, a, s]
            particle = Particle(s, self.env.get_signature(), r, done, weight=1, prob=prob,
                                parent_particle=parent_particle)
            new_node = node.child_node
            new_node.add_particle(particle)
            node = new_node
            parent_particle = particle
            prev_state = s
        return particle

    def run_particle_estimation(self, n=1000, budget=1000, bh=False, fast=True):
        evaluations = []
        starting_state = self.starting_state
        signature = self.starting_signature
        resamplings = []
        depths = np.zeros(self.full_cost)
        counts = []
        ess = []
        for i in range(n):
            self.root_particle = root_particle = Particle(starting_state, signature=signature, reward=0, terminal=False,
                                                          prob=1., weight=1)
            self.root = root = Node(depth=0)
            root.add_particle(root_particle)
            self.last_node = last_node = None
            self.reset(bh=bh)
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
                    parent_particle = particle = root_particle
                    prob = 1.
                    prev_state = parent_particle.state
                    for depth, a in enumerate(action_sequence):
                        s, r, done, _ = self.env.step(a)
                        prob = prob * self.env.P[prev_state, a, s]
                        particle = Particle(s, self.env.get_signature(),  r, done, weight=1, prob=prob,
                                            parent_particle=parent_particle)
                        new_node = Node(parent_node=node, depth=depth+1)
                        new_node.add_particle(particle)
                        node.add_child(new_node)
                        node = new_node
                        parent_particle = particle
                        prev_state = s
                    remaining_budget -= self.full_cost
                    j = 1
                    last_node = node
                    self.backup(last_node, particle, sampled_node=root, sampled_particle=root_particle,
                                bh=bh, weight=1, fast=fast)
                    self.last_node = last_node
                else:
                    node = last_node.parent_node
                    resampled = False
                    max_margin = -np.inf
                    starting_node = self.root
                    starting_particle = self.root_particle
                    sample_weight = 1
                    if bh:
                        full_resampling_weights = self.get_new_weights_balance_heuristic(trajectory=[self.root_particle],
                                                                                         depth=0,
                                                                                         particle=self.root_particle,
                                                                                         distribution=(
                                                                                         [self.root_particle], 1),
                                                                                         fast=fast)
                    else:
                        full_resampling_weights = self.get_new_weights_simple(1)
                    while node.parent_node is not None:
                        should_resample, particle, weight, margin = \
                            self.should_resample(node, bh=bh, full_resampling_weights=full_resampling_weights,
                                                 fast=fast)
                        if should_resample:
                            resampled = True
                            if margin > max_margin:
                                max_margin = margin
                                starting_particle = particle
                                sample_weight = weight
                                starting_node = node
                        node = node.parent_node
                    if resampled:
                        resampling_count += 1
                        resampling_depths[starting_node.depth - 1] += 1
                    generated_particle = self.resample_from_particle(node=starting_node, particle=starting_particle)
                    remaining_budget -= (self.full_cost - starting_node.depth)
                    self.backup(self.last_node, generated_particle, bh=bh, sampled_node=starting_node,
                                sampled_particle=starting_particle, weight=sample_weight, fast=fast)
                count += 1
            if bh:
                weights = self.get_weights_balance_heuristic(fast=False)
                weights = np.array(weights) / np.sum(weights)
                evaluations.append(np.dot(weights, self.xs))
            else:
                weights = self.weights
                evaluations.append(self.root.V)
            resamplings.append(resampling_count)
            depths += resampling_depths
            counts.append(count)
            ess.append(compute_ess(weights))

        return evaluations, ess, depths / n, counts
    #
    # def basic_one_step_estimator(self, n=1000, budget=1000, m=20):
    #     evaluations = []
    #     starting_state = self.starting_state
    #     signature = self.starting_signature
    #
    #     for i in range(n):
    #         self.ess = 0
    #         particles = []
    #         weights = []
    #         root_particle = Particle(starting_state, signature=signature, reward=0, terminal=False, weight=1)
    #         for i in range(m):
    #             self.env.set_signature(signature)
    #             self.env.seed()
    #             s, r, done, _ = self.env.step(self.action_sequence[0])
    #             particle = Particle(s, self.env.get_signature(), r, done, weight=1,
    #                                 parent_particle=root_particle)
    #             particles.append(particle)
    #             weights.append(1)
    #         remaining_budget = budget
    #         samples = []
    #         sample_weights = []
    #         while remaining_budget > self.full_cost - 2:
    #             candidate_particle = np.random.choice(particles, p=weights / np.sum(weights))
    #             ending_state = candidate_particle.state
    #             p_x, _ = self.multi_step_model(starting_state, 0, ending_state)
    #             p_f_x = candidate_particle.weight / np.sum(weights)
    #             new_weight = p_x / p_f_x
    #             self.env.set_signature(candidate_particle.signature)
    #             self.env.seed()
    #             self.weights.append(new_weight)
    #             self.ess =compute_ess(self.weights)
    #             ret = candidate_particle.reward
    #             for d, a in enumerate(self.action_sequence[1:]):
    #                 s, r, done, _ = self.env.step(a)
    #                 ret += r * self.gamma ** (d+1)
    #             samples.append(ret)
    #             sample_weights.append(new_weight)
    #             remaining_budget -= (self.full_cost - 1)
    #         samples = np.array(samples)
    #         sample_weights = np.array(sample_weights)
    #         estimate = np.sum(sample_weights * samples) / (np.sum(sample_weights))
    #         evaluations.append(estimate)
    #     return evaluations


if __name__ == '__main__':
    action_length = 10
    num_actions = 3
    num_states = 10
    gamma = 1.

    action_sequence = np.random.choice(num_actions, size=action_length)
    mdp = random_mdp(n_states=num_states, n_actions=num_actions)
    estimator = Estimator(mdp, action_sequence, gamma=gamma)
    n = 400
    budget = 800
    bins = 50
    samples = n
    print("Doing " + str(n) + " MC estimations with " + str(budget) + " budget")
    start = time.time()
    estimations_mc = estimator.run_monte_carlo_estimation(n, budget)
    end = time.time()
    print("Time Elapsed:" + str(end - start))
    mean = np.mean(estimations_mc)
    std_hat = np.std(estimations_mc, ddof=1)
    print("Mean=" + str(mean) + "+/- " + str(2 * std_hat / np.sqrt(n)))
    pyplot.hist(estimations_mc, bins, alpha=0.5, label='MC', density=True, color='c')

    # print("Doing " + str(n) + " Particle Simple estimations with " + str(budget) + " budget")
    # start = time.time()
    # estimations_particle, ess, depths, counts = estimator.run_particle_estimation(n, budget, bh=False)
    # end = time.time()
    # print("Time Elapsed:" + str(end - start))
    # mean = np.mean(estimations_particle)
    # std_hat = np.std(estimations_particle, ddof=1)
    # print("Mean=" + str(mean) + "+/- " + str(2 * std_hat / np.sqrt(n)))
    # pyplot.hist(estimations_particle, bins, alpha=0.5, label='PARTICLE SIMPLE', density=True)

    print("Doing " + str(n) + " Particle BH estimations with " + str(budget) + " budget")
    start = time.time()
    estimations_particle_bh, ess_bh, depths_bh, counts_bh = estimator.run_particle_estimation(n, budget, bh=True)
    end = time.time()
    print("Time Elapsed:" + str(end - start))
    mean = np.mean(estimations_particle_bh)
    std_hat = np.std(estimations_particle_bh, ddof=1)
    print("Mean=" + str(mean) + "+/- " + str(2 * std_hat / np.sqrt(n)))
    pyplot.hist(estimations_particle_bh, bins, alpha=0.5, label='PARTICLE BH', density=True, color='k')

    pyplot.xlabel("Return")
    pyplot.title("Return - " + str(n) + " samples with budget " + str(budget))
    pyplot.legend(loc='upper right')
    pyplot.savefig('Estimators.pdf')
    pyplot.show()

    pyplot.xlabel("ESS")
    # pyplot.hist(ess, bins, alpha=0.5, label='simple estimator')
    pyplot.hist(ess_bh, bins, alpha=0.5, label='bh estimator')
    pyplot.legend(loc='upper right')
    pyplot.show()

    pyplot.xlabel("Resampling Depths")
    # pyplot.plot(np.arange(action_length) + 1, depths, alpha=0.5, label='simple estimator')
    pyplot.plot(np.arange(action_length) + 1, depths_bh, alpha=0.5, label='bh estimator')
    pyplot.legend(loc='upper right')
    pyplot.show()

    pyplot.xlabel("Number of Samples")
    # pyplot.hist(counts, bins, alpha=0.5, label='simple estimator')
    pyplot.hist(counts_bh, bins, alpha=0.5, label='bh estimator')
    pyplot.legend(loc='upper right')
    pyplot.show()

    ## Check Error rates
    budgets = [10, 20, 30, 40, 50, 70, 80, 100, 200, 300, 400, 500, 1000] #,
    budgets = budgets[:]
    n = 300
    ys_mc = []

    ys_particle_simple = []
    samples_p_simple = []
    ess_p_simple = []

    ys_particle_bh = []
    samples_p_bh = []
    ess_p_bh = []

    true_mean_samples = 20000
    estimations_mc = estimator.run_monte_carlo_estimation(true_mean_samples, 10)
    mean = np.mean(estimations_mc)
    std_hat = np.std(estimations_mc, ddof=1)
    print("Mean=" + str(mean) + "+/- " + str(2 * std_hat / np.sqrt(true_mean_samples)))

    for b in budgets:
        estimations_mc = estimator.run_monte_carlo_estimation(n, b)
        error = ((np.array(estimations_mc) - mean) ** 2).mean()
        ys_mc.append(error)

        estimations_particle, ess, _, counts = estimator.run_particle_estimation(n, b, bh=False)
        error = ((np.array(estimations_particle) - mean) ** 2).mean()
        ys_particle_simple.append(error)
        samples_p_simple.append(np.mean(counts))
        ess_p_simple.append(np.mean(ess))

        estimations_particle, ess, _, counts = estimator.run_particle_estimation(n, b, bh=True)
        error = ((np.array(estimations_particle) - mean) ** 2).mean()
        ys_particle_bh.append(error)
        samples_p_bh.append(np.mean(counts))
        ess_p_bh.append(np.mean(ess))
        print("Finished budget " + str(b))

    xs = np.array(budgets) / action_length
    pyplot.plot(xs, ys_mc, alpha=0.8, label='MC error', marker='x')

    pyplot.plot(samples_p_simple, ys_particle_simple, alpha=0.5, label='particle_simple_error(N)', marker='o')
    pyplot.plot(samples_p_bh, ys_particle_bh, alpha=0.5, label='particle_bh_error(N)', marker='o')
    # pyplot.plot(ess_p_simple, ys_particle_simple, alpha=0.5, label='particle_simple_error(ess)', marker='o')
    # pyplot.plot(xs, ys_particle_simple, alpha=0.5, label='particle_simple_error(Budget)', marker='o')

    pyplot.plot(xs,  std_hat**2 / np.array(xs), alpha=0.5, label='1/x')
    pyplot.legend(loc='upper right')
    pyplot.xlabel("Samples")
    pyplot.ylabel("Error")
    pyplot.savefig("Error_simple.pdf")
    pyplot.show()

    # pyplot.plot(xs, ys_mc, alpha=0.5, label='MC error', marker='x')
    # pyplot.plot(ess_p_bh, ys_particle_bh, alpha=0.5, label='particle_bh_error(ess)', marker='o')
    # pyplot.plot(xs, ys_particle_bh, alpha=0.5, label='particle_bh_error(Budget)', marker='o')
    # pyplot.plot(xs, std_hat ** 2 / np.array(xs), alpha=0.5, label='1/x')
    # pyplot.legend(loc='upper right')
    # pyplot.xlabel("Samples")
    # pyplot.ylabel("Error")
    # pyplot.savefig("Error_bh.pdf")
    # pyplot.show()

    # pyplot.plot(xs, samples_p_simple, alpha=0.5, label='samples_simple(budget)', marker='o')
    # pyplot.plot(xs, ess_p_simple, alpha=0.5, label='ess_simple(budget)', marker='x')
    # pyplot.plot(xs, samples_p_bh, alpha=0.5, label='samples_bh(budget)', marker='o')
    # pyplot.plot(xs, ess_p_bh, alpha=0.5, label='ess_bh(budget)', marker='x')
    # pyplot.legend(loc='lower right')
    # pyplot.xlabel("Samples")
    # pyplot.savefig("samples.pdf")
    # pyplot.show()


