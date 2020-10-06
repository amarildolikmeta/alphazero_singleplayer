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

    def add_child(self, node):
        self.child_node = node

    def add_particle(self, particle):
        self.particles.append(particle)
        self.r = particle.reward
        self.weights.append(particle.weight)
        self.num_particles += 1

    def update(self, r, weight):
        if self.n == 0:
            self.V = r
        else:
            self.V = (self.V * self.n + r * weight) / (self.n + weight)
        self.n += weight


class Estimator:
    def __init__(self, env, action_sequence, gamma=1.):
        self.env = env
        self.action_sequence = action_sequence
        self.gamma = gamma
        self.multi_steps_Ps = []
        self.compute_multi_steps()
        self.root = None
        self.full_cost = len(action_sequence)
        self.ess = 0
        self.weights = []
        self.starting_state = self.env.reset()
        self.starting_signature = self.env.get_signature()

    def reset(self):
        self.weights = []
        self.ess = 0

    def compute_ess(self, weights):
        return np.linalg.norm(weights, ord=1) ** 2 / np.linalg.norm(weights, ord=2) ** 2

    def should_resample(self, node, starting_state):
        particles = node.particles
        weights = node.weights
        candidate_particle = np.random.choice(particles, p=weights / np.sum(weights))
        ending_state = candidate_particle.state
        p_x = self.multi_step_model(starting_state, node.depth - 1, ending_state)
        p_f_x = candidate_particle.weight / np.sum(weights)
        new_weight = p_x / p_f_x
        new_weights = [x for x in self.weights]
        new_weights.append(new_weight)
        ess = self.compute_ess(new_weights)
        #ess = self.ess + new_weight
        full_resampling_weights = [x for x in self.weights]
        full_resampling_weights.append(1)
        sample_size = self.compute_ess(full_resampling_weights)
        full_cost = self.full_cost
        resampling_cost = full_cost - node.depth
        full_error_reduction = (1 / self.ess - 1 / sample_size)
        resampling_error_reduction = (1 / self.ess - 1 / ess)
        # assert sample_size > ess, "Sample sizes don't match"
        # assert full_error_reduction > resampling_error_reduction, "Error reduction costs don't match"
        should_resample = full_error_reduction / full_cost < resampling_error_reduction / resampling_cost
        margin = resampling_error_reduction / resampling_cost - full_error_reduction / full_cost
        return should_resample, candidate_particle, new_weight, margin

    def compute_multi_steps(self):
        P = self.env.P
        prev_p = np.eye(P.shape[0])
        for a in self.action_sequence:
            P_ = P[:, a, :]
            multi = np.dot(prev_p, P_)
            self.multi_steps_Ps.append(multi)
            prev_p = multi

    def multi_step_model(self, starting_state, depth, ending_state):
        return self.multi_steps_Ps[depth][starting_state, ending_state]

    def run_monte_carlo_estimation(self, n=1000, budget=1000):
        evaluations = []
        signature = self.starting_signature
        m = int(np.ceil(budget / self.full_cost))
        for i in range(n):
            print("MC estimation #" + str(i + 1))
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

    def backup(self, node, particle, weight):
        #weight = particle.weight
        R = 0
        while node.parent_node is not None:
            node.update(R, weight)
            R = particle.reward + self.gamma * R
            node = node.parent_node
            particle = particle.parent_particle
        node.update(R, weight)

    def resample_from_particle(self, node, particle, weight, depth):
        self.env.set_signature(particle.signature)
        self.env.seed()
        parent_particle = particle
        self.weights.append(weight)
        self.ess = self.compute_ess(self.weights)
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

    def run_particle_estimation(self, n=1000, budget=1000):
        evaluations = []
        starting_state = self.starting_state
        signature = self.starting_signature
        resamplings = []
        depths = np.zeros(self.full_cost)
        counts = []
        for i in range(n):
            print("Particle estimation #" + str(i + 1))
            root_particle = Particle(starting_state, signature=signature, reward=0, terminal=False, weight=1)
            self.root = root = Node(depth=0)
            root.add_particle(root_particle)
            self.last_node = last_node = None
            self.ess = 0
            self.weights = []
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
                    self.backup(last_node, particle, 1)
                    self.last_node = last_node
                    self.ess = 1
                    self.weights.append(1)
                else:
                    node = last_node.parent_node
                    resampled = False
                    max_margin = - np.inf
                    while node.parent_node is not None:
                        should_resample, particle, weight, margin = self.should_resample(node, starting_state)
                        if should_resample:
                            resampled = True
                            if margin > max_margin:
                                max_margin = margin
                                starting_particle = particle
                                sample_weight = weight
                                sample_depth = node.depth
                                starting_node = node
                        #     starting_particle = particle
                        #     sample_weight = weight
                        #     sample_depth = node.depth
                        #     resampling_count += 1
                        #     resampling_depths[sample_depth - 1] += 1
                        #     break
                        # else:
                        node = node.parent_node
                    if not resampled:
                        starting_particle = self.root.particles[0]
                        sample_weight = 1
                        sample_depth = 0
                    else:
                        resampling_count += 1
                        resampling_depths[sample_depth - 1] += 1
                        node = starting_node
                    generated_particle = self.resample_from_particle(node=node, particle=starting_particle,
                                                                     weight=sample_weight, depth=sample_depth)
                    remaining_budget -= (self.full_cost - sample_depth)
                    self.backup(self.last_node, generated_particle, sample_weight)
                count += 1

            evaluations.append(self.root.V)
            resamplings.append(resampling_count)
            depths += resampling_depths
            counts.append(count)
        return evaluations, resamplings, depths / n, counts

    def basic_one_step_estimator(self, n=1000, budget=1000, m=20):
        evaluations = []
        starting_state = self.starting_state
        signature = self.starting_signature

        for i in range(n):
            print("Particle estimation #" + str(i + 1))
            self.ess = 0
            self.weights = []
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
    n = 1000

    budget = 300
    bins = 50

    print("Doing " + str(n) + " MC estimations with " + str(budget) + " samples")
    estimations_mc = estimator.run_monte_carlo_estimation(n, budget)
    pyplot.hist(estimations_mc, bins, alpha=0.5, label='MC')

    print("Doing " + str(n) + " MC estimations with " + str(budget) + " samples")
    estimations_p = estimator.basic_one_step_estimator(n, budget, m=100)
    pyplot.hist(estimations_p, bins, alpha=0.5, label='Particle One Step')

    print("Doing " + str(n) + " Particle estimations with " + str(budget) + " samples")
    estimations_particle, resamplings, depths, counts = estimator.run_particle_estimation(n, budget)
    pyplot.hist(estimations_particle, bins, alpha=0.5, label='PARTICLE')
    #
    pyplot.legend(loc='upper right')
    pyplot.show()
    #
    # pyplot.hist(resamplings, bins, alpha=0.5, label='resamplings')
    # pyplot.legend(loc='upper right')
    # pyplot.show()
    #
    # pyplot.plot(np.arange(action_length), depths, alpha=0.5, label='depths')
    # pyplot.legend(loc='upper right')
    # pyplot.show()
    #
    # pyplot.hist(counts, bins, alpha=0.5, label='particle samples')
    # pyplot.legend(loc='upper right')
    # pyplot.show()


