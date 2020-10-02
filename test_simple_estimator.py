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
        self.ess = 0

    def add_child(self, node):
        self.child_node = node

    def add_particle(self, particle):
        self.particles.append(particle)
        self.r = particle.reward
        self.weights.append(particle.weight)
        self.num_particles += 1
        self.ess = np.linalg.norm(self.weights, ord=1) ** 2 / np.linalg.norm(self.weights, ord=2) ** 2

    def update(self, r, weight):
        if self.n == 0:
            self.V = r / weight
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

    def should_resample(self, node, starting_state):
        particles = node.particles
        weights = node.weights
        candidate_particle = np.random.choice(particles, p=weights)
        ending_state = candidate_particle.state
        p_x = self.multi_step_model(starting_state, node.depth - 1, ending_state)
        p_f_x = candidate_particle.weight
        new_weight = p_x / p_f_x
        new_weights = [x for x in weights]
        new_weights.append(new_weight)
        ess = np.linalg.norm(new_weights, ord=1) ** 2 / np.linalg.norm(new_weights, ord=2) ** 2
        sample_size = node.ess + 1
        full_cost = self.full_cost
        resampling_cost = full_cost - node.depth
        full_error_reduction = (1 / node.ess - 1 / sample_size)
        resampling_error_reduction = (1 / node.ess - 1 / ess)
        assert full_error_reduction > resampling_error_reduction, "Error reduction costs don't match"
        should_resample = full_error_reduction / full_cost < resampling_error_reduction / resampling_cost
        return should_resample, candidate_particle, new_weight

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

    def run_monte_carlo_estimation(self, n=1000, m=1000):
        evaluations = []
        self.env.reset()
        signature = self.env.get_signature()
        for i in range(n):
            print("MC estimation #" + str(i + 1))
            self.env.set_signature(signature)
            sum_returns = 0
            for j in range(m):
                ret = 0
                self.env.set_signature(signature)
                for a in self.action_sequence:
                    s, r, _, _ = self.env.step(a)
                    ret += r
                sum_returns += ret
            evaluations.append(sum_returns / m)
        return evaluations

    def backup(self, node, weight=1):
        R = 0
        while node.parent_node is not None:
            node.update(R, weight)
            R = node.r + self.gamma * R
            node = node.parent_node
        node.update(R, weight)

    def resample_from_particle(self, node, particle, weight, depth):
        self.env.set_signature(particle.signature)
        self.env.seed()
        parent_particle = particle
        for d, a in enumerate(self.action_sequence[depth:]):
            s, r, done, _ = self.env.step(a)
            particle = Particle(s, self.env.get_signature(), r, done, weight=weight,
                                parent_particle=parent_particle)
            new_node = node.child_node
            new_node.r = r
            new_node.add_particle(particle)
            node = new_node
            parent_particle = particle

    def run_particle_estimation(self, n=1000, m=1000):
        evaluations = []
        starting_state = self.env.reset()
        signature = self.env.get_signature()
        resamplings = []
        for i in range(n):
            print("Particle estimation #" + str(i + 1))
            self.root = root = Node(depth=0)
            root_particle = Particle(starting_state, signature=signature, reward=0, terminal=False, weight=1)
            self.last_node = last_node = None
            resampling_count = 0
            for j in range(m):
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

                    last_node = node
                    self.backup(last_node)
                    self.last_node = last_node
                else:
                    node = last_node.parent_node
                    resampled = False
                    while node.parent_node is not None:
                        should_resample, particle, weight = self.should_resample(node, starting_state)
                        if should_resample:
                            resampled = True
                            starting_particle = particle
                            sample_weight = weight
                            sample_depth = node.depth
                            resampling_count += 1
                            break
                        else:
                            node = node.parent_node
                    if not resampled:
                        starting_particle = self.root.particles[0]
                        sample_weight = 1
                        sample_depth = 0
                    self.resample_from_particle(node=node, particle=starting_particle, weight=sample_weight,
                                                depth=sample_depth)
                    self.backup(self.last_node, sample_weight)

            evaluations.append(self.root.V)
            resamplings.append(resampling_count)
        return evaluations, resamplings


if __name__ == '__main__':
    game_params = {'horizon': 20}
    action_sequence = np.random.choice(3, size=10)
    mdp = random_mdp(n_states=10, n_actions=3)
    estimator = Estimator(mdp, action_sequence, gamma=1.)
    n = 300
    m = 300
    bins = 50

    print("Doing " + str(n) + "MC estimations with " + str(m) + " samples")
    estimations_mc = estimator.run_monte_carlo_estimation(n, m)
    pyplot.hist(estimations_mc, bins, alpha=0.5, label='MC')

    print("Doing " + str(n) + "Particle estimations with " + str(m) + " samples")
    estimations_particle, resamplings = estimator.run_particle_estimation(n, m)
    pyplot.hist(estimations_particle, bins, alpha=0.5, label='PARTICLE')

    pyplot.legend(loc='upper right')
    pyplot.show()
    pyplot.hist(resamplings, bins, alpha=0.5, label='resamplings')

    pyplot.legend(loc='upper right')
    pyplot.show()


