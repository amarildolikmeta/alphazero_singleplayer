import copy
from rl.make_game import is_atari_game
import numpy as np
from particle_filtering.pf_uct import PFState, PFAction
from particle_filtering.ol_uct import OL_MCTS
from test_particle_tree_estimator import check_sub_trajectory, Particle, compute_ess
from helpers import stable_normalizer, argmax, max_Q


class Action(PFAction):
    """ Action object """

    def __init__(self, index, parent_state, sequence, root_depth=0):
        self.index = index
        self.parent_state = parent_state
        self.W = 0.0
        self.n = 0
        self.Q = 0
        self.ess = 0
        self.rewards = []
        self.child_state = None
        self.xs = []
        self.sampling_depths = []
        self.trajectories = []
        self.approx_ps = []
        self.sampling_distributions = []
        self.passing_particles = []
        self.last_particles = []
        self.ps = []
        self.partial_q_sums = []
        self.estimated_n = 0
        self.action_sequence = sequence
        self.root_depth = root_depth

    def sample_from_particle(self, source_particle, env, budget):
        env.set_signature(source_particle.state)
        env.seed(np.random.randint(1e7))
        s, r, done, _ = env.step(self.index)
        budget -= 1
        new_particle = Particle(env.get_signature(), None, r, done, parent_particle=source_particle)
        return new_particle, budget

    def sample_from_parent_state(self, env, budget):
        state = self.parent_state
        if state.root:
            parent_particle = state.particles[0]
        else:
            parent_particle = np.random.choice(state.particles)
        new_particle, budget = self.generate_new_particle(env, parent_particle, budget)
        return new_particle, budget

    def generate_new_particle(self, env, particle, budget):
        """Generate the successor particle for a given particle"""
        # Do not give any reward if a particle is being generated from a terminal state particle
        if particle.terminal:
            return Particle(particle.state, None, 0, True, parent_particle=particle), budget
        # Apply the selected action to the state encapsulated by the particle and store the new state and reward
        env.set_signature(particle.state)
        env.seed(np.random.randint(1e7))
        s, r, done, _ = env.step(self.index)
        budget -= 1
        return Particle(env.get_signature(), None, r, done, parent_particle=particle, prob=1), budget

    def add_child_state(self, env, budget, max_depth=200, depth=0, source_particle=None):
        # if source_particle is not None:
        #     new_particle, budget = self.sample_from_particle(source_particle, env, budget)
        # else:
        #     new_particle, budget = self.sample_from_parent_state(env, budget)
        self.child_state = State(parent_action=self, na=self.parent_state.na, env=env, root=False, max_depth=max_depth,
                                 budget=budget, particle=source_particle, depth=depth,
                                 sequence=self.action_sequence, root_depth=self.root_depth)
        last_particle = self.child_state.final_particle

        return self.child_state, self.child_state.remaining_budget, last_particle

    def update(self, r, sampling_depth, sampling_distribution, trajectory, approx_p, passing_particle, last_particle,
               estimate=False):
        self.xs.append(r)
        self.trajectories.append(trajectory)
        self.sampling_depths.append(sampling_depth)
        self.approx_ps.append(approx_p)
        self.sampling_distributions.append(sampling_distribution)
        self.passing_particles.append(passing_particle)
        self.last_particles.append(last_particle)
        # if passing_particle.signature['t'] != self.root_depth + self.parent_state.depth:
        #     print("Whatt")
        if estimate:
            self.estimate_value()
        self.n += 1

    def estimate_value(self):
        if self.estimated_n == self.n:
            return
        ps = self.ps
        trajectories = self.trajectories
        sampling_depths = self.sampling_depths
        sampling_distributions = self.sampling_distributions
        partial_q_sums = self.partial_q_sums
        passing_particles = self.passing_particles
        final_particles = self.last_particles
        approx_state_distributions = self.approx_ps
        T = self.n

        assert len(ps) == self.estimated_n
        last_nus = trajectories[self.estimated_n:]
        last_sampling_depths = sampling_depths[self.estimated_n:]
        last_sampling_distributions = sampling_distributions[self.estimated_n:]
        last_passing_particles = passing_particles[self.estimated_n:]
        last_final_particles = final_particles[self.estimated_n:]
        last_approx_state_distributions = approx_state_distributions[self.estimated_n:]
        for i in range(self.n - self.estimated_n):
            # for every "new" trajectory
            current_nu = last_nus[i]
            current_sampling_depth = last_sampling_depths[i]
            current_sampling_distribution = last_sampling_distributions[i][0]
            current_sampling_num_particles = last_sampling_distributions[i][1]
            current_passing_particle = last_passing_particles[i]
            current_passing_state = current_passing_particle.state
            current_approx_state_distribution = last_approx_state_distributions[i]
            current_final_particle = last_final_particles[i]
            # compute the true trajectory distribution
            current_p_nu = current_final_particle.prob / current_passing_particle.prob * \
                           self.parent_state.true_P[current_passing_state]
            if isinstance(current_p_nu, np.ndarray):
                current_p_nu = current_p_nu[0]
            qs_new = []
            partial_q_sum_new = 0
            for j in range(T):
                # weight the current trajectory in all sample distributions
                num_particles_j = sampling_distributions[j][1]
                if sampling_depths[j] == self.parent_state.depth:
                    weight = current_final_particle.prob / current_passing_particle.prob * \
                             approx_state_distributions[j][current_passing_state]

                elif sampling_depths[j] > self.parent_state.depth:
                    valid, lk_particle = check_sub_trajectory(nu=current_nu, depth=sampling_depths[j],
                                                              distribution=sampling_distributions[j][0][
                                                                           :num_particles_j])
                    if valid:
                        prob = current_final_particle.prob / lk_particle.prob
                        weight = prob / num_particles_j
                    else:
                        weight = 0
                else:
                    if sampling_depths[j] > 0:

                        prob = approx_state_distributions[j]
                        for k in range(self.parent_state.depth - sampling_depths[j]):
                            prob = np.dot(prob, self.parent_state.P[:, self.action_sequence[k + sampling_depths[j]], :])
                        prob_ = prob[current_passing_state]
                    else:
                        prob_ = self.parent_state.true_P[current_passing_state]
                    prob = prob_ * current_final_particle.prob / current_passing_particle.prob
                    weight = prob
                if isinstance(weight, np.ndarray):
                    weight = weight[0]
                qs_new.append(weight)
                partial_q_sum_new += weight
                # weight old trajectories in the current sample distribution
                if j < self.estimated_n:
                    if current_sampling_depth == self.parent_state.depth:
                        weight = final_particles[j].prob / passing_particles[j].prob * \
                                 current_approx_state_distribution[passing_particles[j].state]
                    elif current_sampling_depth > self.parent_state.depth:
                        valid, lk_particle = \
                            check_sub_trajectory(nu=trajectories[j], depth=current_sampling_depth,
                                                 distribution=current_sampling_distribution[:
                                                                                        current_sampling_num_particles])
                        if valid:
                            prob = final_particles[j].prob / lk_particle.prob
                            weight = prob / current_sampling_num_particles
                        else:
                            weight = 0
                    else:
                        if current_sampling_depth > 0:
                            prob = current_approx_state_distribution
                            for k in range(self.parent_state.depth - current_sampling_depth):
                                prob = np.dot(prob, self.parent_state.P[:,
                                                    self.action_sequence[k + current_sampling_depth], :])
                            prob_ = prob[passing_particles[j].state]
                        else:
                            prob_ = self.parent_state.true_P[current_passing_state]
                        prob = prob_ * final_particles[j].prob / passing_particles[j].prob
                        weight = prob
                    if isinstance(weight, np.ndarray):
                        weight = weight[0]
                    partial_q_sums[j] += weight
            partial_q_sums.append(partial_q_sum_new)
            ps.append(current_p_nu)
        new_weights = np.array(ps) / np.array(partial_q_sums)
        new_weights = np.array(new_weights) / np.sum(new_weights)
        self.ps = ps
        self.partial_q_sums = partial_q_sums
        self.weights = new_weights
        self.V = np.dot(self.xs, self.weights)
        self.ess = compute_ess(self.weights)
        self.estimated_n = self.n
        if np.isnan(new_weights).any():
            print("What")


class State(PFState):
    """ State object """

    def __init__(self, parent_action, na, env, particle, budget, root=False, max_depth=200, depth=0, sequence=[],
                 root_depth=0):
        """ Initialize a new state """
        self.parent_action = parent_action
        # Child actions
        self.na = na
        self.remaining_budget = budget
        self.depth = depth
        self.terminal = self.is_terminal(max_depth, env)
        self.reward = particle.reward
        self.root = root
        self.n = 0
        self.child_actions = [Action(a, parent_state=self, sequence=sequence + [a],
                                     root_depth=root_depth) for a in range(na)]
        self.n_particles = 1
        self.particles = [particle]
        self.visits = np.zeros(env.P.shape[0])
        self.visits[particle.state] = 1
        self.P = env.P
        if self.depth == 0:
            self.true_P = np.zeros(env.P.shape[0])
            self.true_P[particle.state] = 1
        else:
            P = self.parent_action.parent_state.true_P
            P = np.dot(P, env.P[:, sequence[-1], :])
            self.true_P = P
        self.sequence = sequence
        final_particle = None
        if self.terminal or root or particle.terminal:
            self.V = 0
            final_particle = Particle(particle.state, particle.signature, 0, True, prob=particle.prob)
        elif env is None:
            print("Warning, no environment was provided, initializing to 0 the value of the state!")
            self.V = 0
        else:
            env.set_signature(particle.signature)
            self.V, self.remaining_budget, final_particle = self.evaluate(env, budget, particle, max_depth)
            final_particle = final_particle
        self.last_particle = particle
        self.final_particle = final_particle

    def evaluate(self, env, budget, particle, max_depth=200):
        actions = np.arange(self.na, dtype=int)
        return_, budget, last_particle = self.random_rollout(actions, env, budget, particle, max_depth)
        return return_, budget, last_particle

    def random_rollout(self, actions, env, budget, particle, max_depth=200):
        """Rollout from the current state following a random policy up to hitting a terminal state"""
        terminal = particle.terminal
        if terminal:
            particle = Particle(particle.state, particle.signature, 0, True, prob=particle.prob)
            return 0, budget, particle

        done = False
        env.set_signature(particle.signature)
        env.seed(np.random.randint(1e7))
        ret = 0
        t = 0
        prob = particle.prob
        prev_state = particle.state

        while t < max_depth and not done:
            action = np.random.choice(actions)
            s, r, done, _ = env.step(action)
            prob_ = prob * env.P[prev_state, action, s]
            if prob_ == 0:
                print("What")
            prob = prob_
            ret += r
            budget -= 1
            t += 1
            prev_state = s
        final_particle = Particle(prev_state, env.get_signature(), r, done, prob=prob)
        budget = max(budget, 0)
        return ret, budget, final_particle

    def get_n_particles(self):
        return self.n_particles

    def get_approx_state_dist(self):
        return np.copy(self.visits / self.n_particles)

    def get_true_state_dist(self):
        return self.true_P

    def add_particle(self, particle):
        self.particles.append(particle)
        self.n_particles += 1
        self.reward = particle.reward  # to be used in the backup
        self.last_particle = particle
        self.visits[particle.state] += 1

    def select(self, c=1.5):
        """
         Select one of the child actions based on UCT rule
         :param c: UCB exploration constant
         :param csi: exploration constant
         :param b: parameter such that the rewards belong to [0, b]
         """
        ess = 0
        for child_action in self.child_actions:
            if self.depth != 0:
                child_action.estimate_value()
            ess += child_action.ess
        uct_upper_bound = np.array(
            [child_action.Q + c * np.sqrt(np.log(ess) / (child_action.ess)) if child_action.ess > 0 else np.inf
             for child_action in self.child_actions])
        winner = argmax(uct_upper_bound)
        return self.child_actions[winner]


class PFModelBasedMCTS(OL_MCTS):
    """ MCTS object """
    def __init__(self, root, root_index, na, gamma, model=None, depth_based_bias=False):
        super(PFModelBasedMCTS, self).__init__(root, root_index, na, gamma, model, variance=False,
                                      depth_based_bias=depth_based_bias)
        self.na = na
        self.reset()

    def reset(self):
        self.ess = [0 for _ in range(self.na)]
        self.T = [0 for _ in range(self.na)]
        self.xs = [[] for _ in range(self.na)]
        self.depths = [[] for _ in range(self.na)]
        self.distributions = [[] for _ in range(self.na)]
        self.trajectories = [[] for _ in range(self.na)]
        self.particles = [[] for _ in range(self.na)]
        self.ps = [[] for _ in range(self.na)]
        self.partial_q_sums = [[] for _ in range(self.na)]
        self.weights = [[] for _ in range(self.na)]

    def create_root(self, env, budget):
        if self.root is None:
            self.reset()
            signature = env.get_signature()
            state = env.get_state()
            self.root_signature = signature
            self.root_state = state
            particle = Particle(state=state, signature=signature, reward=0, terminal=False, parent_particle=None,
                                prob=1)
            self.root = State(parent_action=None, na=self.na, env=env, particle=particle, root=True,
                                budget=budget, depth=0, root_depth=signature['t'])
            return particle
        else:
            raise (NotImplementedError("Need to reset the tree"))

    def compute_bh_weights(self, trajectories, particles, depths, distributions, T, action):
        assert len(trajectories) == len(particles) and len(particles) == len(depths) and \
               len(depths) == len(distributions) and len(distributions) == T, "Error in computing bh weights"
        weights, ps, partial_sums = self.compute_bh_weights_fast(trajectories, particles, depths, distributions, T,
                                                                 action)
        return weights, ps, partial_sums

    def get_new_weights_balance_heuristic(self, trajectory, particle, depth, distribution, action):
        trajectories = [x for x in self.trajectories[action]] + [trajectory]
        particles = [x for x in self.particles[action]] + [particle]
        depths = [x for x in self.depths[action]] + [depth]
        distributions = [x for x in self.distributions[action]] + [distribution]
        T = self.T[action] + 1
        weights, _, _ = self.compute_bh_weights(depths=depths, distributions=distributions, trajectories=trajectories,
                                                particles=particles, T=T, action=action)
        return weights

    def compute_bh_weights_fast(self, trajectories, particles, depths, distributions, T, action):
        ps = [x for x in self.ps[action]]
        partial_q_sums = [x for x in self.partial_q_sums[action]]
        assert len(ps) == T - 1

        last_nu = trajectories[-1]
        last_particle = particles[-1]
        last_depth = depths[-1]
        last_distribution = distributions[-1]
        last_p_nu = last_particle.prob
        if isinstance(last_p_nu, np.ndarray):
            last_p_nu = last_p_nu[0]
        last_num_particles = distributions[-1][1]
        qs_new = []
        partial_q_sum_new = 0
        for j in range(T):
            # compute new trajectory weight under old trajectory distribution
            num_particles_j = distributions[j][1]
            if depths[j] > last_depth:
                qs_new.append(0)
            else:
                valid, valid_particle = check_sub_trajectory(nu=last_nu,
                                                             distribution=distributions[j][0][:num_particles_j],
                                                             depth=depths[j])
                if valid:
                    if depths[j] == last_depth:
                        p_nu_j = 1
                    else:
                        p_nu_j = last_p_nu / valid_particle.prob
                    if isinstance(p_nu_j, np.ndarray):
                        p_nu_j = p_nu_j[0]
                    qs_new.append(p_nu_j / num_particles_j)
                    partial_q_sum_new += p_nu_j / num_particles_j
                else:
                    qs_new.append(0)
            if j == T - 1:
                break
            if last_depth > depths[j]:
                partial_q_sums[j] += 0
            else:
                valid, valid_particle = check_sub_trajectory(nu=trajectories[j],
                                                             distribution=last_distribution[0][:last_num_particles],
                                                             depth=last_depth)
                if valid:
                    if depths[j] == last_depth:
                        p_nu_j = 1
                    else:
                        p_nu_j = last_particle.prob / valid_particle.prob
                    if isinstance(p_nu_j, np.ndarray):
                        p_nu_j = p_nu_j[0]
                    partial_q_sums[j] += p_nu_j / last_num_particles
                else:
                    partial_q_sums[j] += 0

        partial_q_sums.append(partial_q_sum_new)
        ps.append(last_p_nu)
        new_weights = np.array(ps) / np.array(partial_q_sums)
        new_weights = np.array(new_weights) / np.sum(new_weights)
        try:
            if np.isnan(new_weights).any():
                print("What")
        except:
            print("Whaaaaat")
        return new_weights, ps, partial_q_sums

    def should_resample(self, node, action, full_resampling_weights=None):
        if node.depth == 0:
            return node.particles[0], full_resampling_weights[-1], 0
        particles = node.particles
        candidate_particle = np.random.choice(particles)
        p = candidate_particle
        trajectory = [p]
        while p.parent_particle is not None:
            p = p.parent_particle
            trajectory.append(p)

        trajectory = trajectory[::-1]
        new_weights = self.get_new_weights_balance_heuristic(trajectory=trajectory, depth=node.depth,
                                                             particle=candidate_particle,
                                                             distribution=(node.particles, node.n_particles),
                                                             action=action)
        ess = compute_ess(new_weights)
        sample_size = compute_ess(full_resampling_weights)

        full_cost = self.full_cost
        resampling_cost = full_cost - node.depth
        full_error_reduction = (1 / self.ess[action] - 1 / sample_size)
        resampling_error_reduction = (1 / self.ess[action] - 1 / ess)
        should_resample = full_error_reduction / full_cost < resampling_error_reduction / resampling_cost
        margin = resampling_error_reduction / resampling_cost - full_error_reduction / full_cost
        if should_resample:
            new_weight = new_weights[-1]
        else:
            new_weight = full_resampling_weights[-1]
        return candidate_particle, new_weight, margin

    def resample_from_particle(self, env, node, particle, action_sequence, budget):
        env.set_signature(particle.signature)
        env.seed()
        depth = node.depth
        parent_particle = particle
        prob = particle.prob
        prev_state = particle.state
        if particle.terminal:
            budget -= 1
            # return particle, budget, False, node
            i = 0
            while i < len(action_sequence[depth:]):
                a = action_sequence[depth + i].index
                budget -= 1
                particle = Particle(particle.state, particle.signature, 0, True, prob=prob,
                                    parent_particle=parent_particle)
                new_node = node.child_actions[a].child_state
                if new_node is not None:
                    new_node.add_particle(particle)
                node = new_node
                parent_particle = particle
                i += 1
        else:
            # expand = True
            for i, a in enumerate(action_sequence[depth:]):

                a = a.index
                s, r, done, _ = env.step(a)
                budget -= 1
                prob = prob * env.P[prev_state, a, s]
                particle = Particle(s, env.get_signature(), r, done, prob=prob,
                                    parent_particle=parent_particle)
                new_node = node.child_actions[a].child_state
                if new_node is not None:
                    new_node.add_particle(particle)
                    node = new_node
                parent_particle = particle
                prev_state = s
                if done:
                    i += 1
                    while i < len(action_sequence[depth:]):
                        a = action_sequence[depth + i].index
                        budget -= 1
                        particle = Particle(s, particle.signature, 0, done, prob=prob,
                                            parent_particle=parent_particle)
                        new_node = node.child_actions[a].child_state
                        if new_node is not None:
                            new_node.add_particle(particle)
                        node = new_node
                        parent_particle = particle
                        i += 1
                    # if new_node is not None:
                    #     expand = False
                    # else:
                    #     new_node = node
                    break
        return particle, budget#, expand, new_node

    def get_weights_balance_heuristic(self, action):
        trajectories = self.trajectories[action]
        particles = self.particles[action]
        depths = self.depths[action]
        distributions = self.distributions[action]
        T = self.T[action]
        weights, ps, partial_sums = self.compute_bh_weights(depths=depths, distributions=distributions,
                                                            trajectories=trajectories, particles=particles, T=T,
                                                            action=action)
        self.ps[action] = ps
        self.partial_q_sums[action] = partial_sums
        self.weights[action] = weights
        return weights

    def backup(self, state, last_particle, sampled_particle, source_particle, sampled_node, action_sequence):
        last_state = state
        trajectory = []
        action = action_sequence[0].index
        particle = source_particle
        while state.parent_action is not None:
            trajectory.append(particle)
            state = state.parent_action.parent_state
            particle = particle.parent_particle
        # trajectory.append(particle)
        self.trajectories[action].append(trajectory[::-1])
        state = last_state
        self.particles[action].append(sampled_particle)
        distribution = (sampled_node.particles, sampled_node.n_particles)
        self.distributions[action].append(distribution)
        self.depths[action].append(sampled_node.depth)
        self.T[action] += 1
        weights = self.get_weights_balance_heuristic(action)
        weights = np.array(weights) / np.sum(weights)
        ess = compute_ess(weights)
        self.ess[action] = ess
        R = state.V
        state.update()
        particle = source_particle
        while state.parent_action is not None:
            r = particle.reward
            if not particle.terminal:
                R = r + self.gamma * R
            else:
                R = r
            action = state.parent_action
            if action.parent_state.parent_action is not None:

                action.update(R, sampling_depth=sampled_node.depth,
                              trajectory=trajectory, estimate=False, sampling_distribution=distribution,
                              passing_particle=particle.parent_particle, last_particle=last_particle,
                              approx_p=sampled_node.get_approx_state_dist())
            state = action.parent_state
            state.update()
            particle = particle.parent_particle
        action = action_sequence[0].index
        self.xs[action].append(R)
        self.root.child_actions[action].ess = ess
        self.root.child_actions[action].Q = np.dot(weights, self.xs[action])
        self.root.child_actions[action].n += 1

    def search(self, n_mcts, c, Env, mcts_env, budget, max_depth=200, fixed_depth=True):
        """ Perform the MCTS search from the root """
        assert hasattr(Env, 'p'), "Need the transition matrix P"
        env = copy.deepcopy(Env)
        root_particle = self.create_root(env, budget)
        if self.root.terminal:
            raise (ValueError("Can't do tree search from a terminal state"))

        is_atari = is_atari_game(env)
        if is_atari:
            raise NotImplementedError
        self.full_cost = max_depth
        self.num_resamplings = 0
        while budget > 0:

            state = self.root  # reset to root for new trace

            if not is_atari:
                mcts_env = copy.deepcopy(Env)  # copy original Env to rollout from
            else:
                raise NotImplementedError
            mcts_env.seed()
            st = 0
            bias = c * self.gamma ** st / (1 - self.gamma) if self.depth_based_bias else c
            action = state.select(c=bias).index
            full_resampling_weights = self.get_new_weights_balance_heuristic(trajectory=[root_particle], depth=0,
                                                                             particle=root_particle,
                                                                             distribution=([root_particle], 1),
                                                                             action=action)
            action_sequence = []
            max_margin = -np.inf
            starting_node = self.root
            starting_particle = root_particle
            generated_particle = None
            last_particle = None
            while not state.terminal:
                bias = c * self.gamma ** st / (1 - self.gamma) if self.depth_based_bias else c
                action = state.select(c=bias)
                action_sequence.append(action)
                st += 1
                particle, weight, margin = self.should_resample(state, action=action_sequence[0].index,
                                                                full_resampling_weights=full_resampling_weights)
                if margin > max_margin:
                    max_margin = margin
                    starting_particle = particle
                    starting_node = state
                    sampling_depth = state.depth
                if action.child_state is not None:
                    state = action.child_state
                else:
                    generated_particle, budget = self.resample_from_particle(env=env, node=starting_node,
                                                                             particle=starting_particle,
                                                                             action_sequence=action_sequence,
                                                                             budget=budget) #, expand, node
                    rollout_depth = max_depth if fixed_depth else max_depth - st
                    state, budget, last_particle = action.add_child_state(source_particle=generated_particle,
                                                                          env=mcts_env, budget=budget,
                                                                          max_depth=rollout_depth, depth=st,)
                    # if expand:
                    #
                    #     state, budget, last_particle = \
                    #         action.add_child_state(source_particle=generated_particle, env=mcts_env, budget=budget,
                    #                                max_depth=rollout_depth, depth=st,)
                    # else:
                    #     state = node
                    #     last_particle = generated_particle
                    break
                if state.terminal:
                    generated_particle, budget = self.resample_from_particle(env=env, node=starting_node,
                                                                                 particle=starting_particle,
                                                                                 action_sequence=action_sequence,
                                                                                 budget=budget)#, expand, node
                    # if not expand:
                    #     state = node
                    last_particle = generated_particle
            # Back-up
            if sampling_depth > 0:
                self.num_resamplings += 1
            self.backup(state, last_particle=last_particle, source_particle=generated_particle,
                        sampled_node=starting_node, sampled_particle=starting_particle,
                        action_sequence=action_sequence)


    def return_results(self, temp, on_visits=False):
        """ Process the output at the root node """
        counts = np.array([child_action.ess for child_action in self.root.child_actions])
        Q = np.array([child_action.Q for child_action in self.root.child_actions])
        if on_visits:
            pi_target = stable_normalizer(counts, temp)
        else:
            pi_target = max_Q(Q)
        V_target = np.sum((counts / np.sum(counts)) * Q)[None]
        return self.root_signature, pi_target, V_target