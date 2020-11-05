import copy
from rl.make_game import is_atari_game
import numpy as np
from particle_filtering.pf_uct import PFState, PFAction
from particle_filtering.ol_uct import OL_MCTS


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


class Action(PFAction):
    """ Action object """

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
        return Particle(env.get_signature(), None, r, done, parent_particle=particle), budget

    def add_child_state(self, env, budget,  max_depth=200, depth=0,  source_particle=None,):
        if source_particle is not None:
            new_particle, budget = self.sample_from_particle(source_particle, env, budget)
        else:
            new_particle, budget = self.sample_from_parent_state(env, budget)
        self.child_state = PFState(parent_action=self,
                                   na=self.parent_state.na,
                                   env=env,
                                   root=False,
                                   max_depth=max_depth,
                                   budget=budget,
                                   particle=new_particle,
                                   depth=depth)

        return self.child_state, self.child_state.remaining_budget, new_particle


class State(PFState):
    """ State object """

    def __init__(self, parent_action, na, env, particle, budget, root=False, max_depth=200, depth=0):
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
        self.child_actions = [PFAction(a, parent_state=self) for a in range(na)]
        self.n_particles = 1
        self.particles = [particle]

        if self.terminal or root or particle.terminal:
            self.V = 0
        elif env is None:
            print("Warning, no environment was provided, initializing to 0 the value of the state!")
            self.V = 0
        else:
            env.set_signature(particle.state)
            self.V, self.remaining_budget = self.evaluate(env, budget, max_depth, particle.terminal)
        self.last_particle = particle

    def get_n_particles(self):
        return self.n_particles

    def add_particle(self, particle):
        self.particles.append(particle)
        self.n_particles += 1
        self.reward = particle.reward  # to be used in the backup
        self.last_particle = particle

    def sample_reward(self):
        p = np.random.choice(self.particles)
        self.reward = p.reward
        self.last_particle = p
        return p


class PFModelBasedMCTS(OL_MCTS):
    """ MCTS object """
    def __init__(self, root, root_index, na, gamma, model=None, variance=False, depth_based_bias=False):
        super(OL_MCTS, self).__init__(root, root_index, na, gamma, model, variance, depth_based_bias)

    def create_root(self, env, budget):
        if self.root is None:
            signature = env.get_signature()
            state = env.get_state()
            self.root_signature = signature
            self.root_state = state
            particle = Particle(state=state, signature=signature, reward=0, terminal=False, parent_particle=None,
                                prob=1, weight=1)
            self.root = PFState(parent_action=None, na=self.na, env=env, particle=particle, root=True,
                              budget=budget, depth=0)
        else:
            raise (NotImplementedError("Need to reset the tree"))

    def get_new_weights_balance_heuristic(self, trajectory, particle, depth, distribution):
        return 0

    def get_new_weights_simple(self, new_weight):
        return 0

    def should_resample(self, node, bh=False, full_resampling_weights=None):
        return 0, 0, 0, 0

    def search(self, n_mcts, c, Env, mcts_env, budget, max_depth=200, fixed_depth=True, bh=True):
        """ Perform the MCTS search from the root """
        assert hasattr(Env, 'p'), "Need the transition matrix P"
        env = copy.deepcopy(Env)
        self.create_root(env, budget)
        if self.root.terminal:
            raise (ValueError("Can't do tree search from a terminal state"))

        is_atari = is_atari_game(env)
        if is_atari:
            raise NotImplementedError

        while budget > 0:

            state = self.root  # reset to root for new trace
            if not is_atari:
                mcts_env = copy.deepcopy(Env)  # copy original Env to rollout from
            else:
                raise NotImplementedError
            mcts_env.seed()
            st = 0
            terminal = False
            path = []
            while not state.terminal:
                path.append(state)
                bias = c * self.gamma ** st / (1 - self.gamma) if self.depth_based_bias else c
                action = state.select(c=bias, variance=self.variance)
                st += 1
                if action.child_state is not None:
                    state = action.child_state  # select
                    # terminal, budget = state.sample(mcts_env, action.index, budget)
                    if terminal:
                        break
                else:
                    rollout_depth = max_depth if fixed_depth else max_depth - st
                    # state, budget = action.add_child_state(mcts_env, budget, rollout_depth, depth=st)  # expand
                    # if not state.terminal(path.append(state))
                    break

            # Back-up

            max_margin = -np.inf
            starting_node = self.root
            root_particle = self.root.particles[0]
            starting_particle = root_particle
            sample_weight = 1
            i = len(path) - 1
            node = path[i]
            if bh:
                full_resampling_weights = self.get_new_weights_balance_heuristic(trajectory=[root_particle],
                                                                                 depth=0,
                                                                                 particle=root_particle,
                                                                                 distribution=(
                                                                                     [root_particle], 1))
            else:
                full_resampling_weights = self.get_new_weights_simple(1)

            while i >= 0:
                should_resample, particle, weight, margin = \
                    self.should_resample(node, bh=bh, full_resampling_weights=full_resampling_weights)
                if should_resample:
                    if margin > max_margin:
                        max_margin = margin
                        starting_particle = particle
                        sample_weight = weight
                        starting_node = node
                node = node.parent_node
            generated_particle = self.resample_from_particle(node=starting_node, particle=starting_particle)
            budget -= (self.full_cost - starting_node.depth)
            self.backup(self.last_node, generated_particle, bh=bh, sampled_node=starting_node,
                        sampled_particle=starting_particle, weight=sample_weight, fast=fast)

            R = state.V
            state.update()
            while state.parent_action is not None:  # loop back-up until root is reached
                if not terminal:
                    R = state.reward + self.gamma * R
                else:
                    R = state.reward
                    terminal = False
                action = state.parent_action
                action.update(R)
                state = action.parent_state
                state.update()
