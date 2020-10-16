import copy
from rl.make_game import is_atari_game
import numpy as np
from particle_filtering.ol_uct import OL_MCTS, State, Action


class Particle(object):
    """Class storing information about a particle"""

    def __init__(self, state, seed, reward, terminal, parent_particle=None):
        self.state = state
        self.seed = seed
        self.reward = reward
        self.terminal = terminal
        self.parent_particle = parent_particle


class PFAction(Action):
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


class PFState(State):
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
            print("[WARNING] No environment was provided, initializing to 0 the value of the state!")
            self.V = 0
        else:
            env.set_signature(particle.state)
            # TODO remove, only for debugging raceStrategy
            if hasattr(env, "get_available_actions") and hasattr(env, "get_next_agent"):
                owner = env.get_next_agent()
                action_list = env.get_available_actions(owner)
                self.child_actions = [PFAction(a, parent_state=self) for a in action_list]
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


class PFMCTS(OL_MCTS):
    """ MCTS object """
    def __init__(self, root, root_index, na, gamma, model=None, variance=False,
                 depth_based_bias=False, alpha=0.6, beta=1):
        super(PFMCTS, self).__init__(root, root_index, na, gamma, model, variance, depth_based_bias)
        assert 0 < alpha <= 1, "Alpha must be between 0 and 1"
        self.alpha = alpha
        self.beta = beta

    def create_root(self, env, budget):
        if self.root is None:
            signature = env.get_signature()
            self.root_signature = signature
            particle = Particle(state=signature, seed=None, reward=0, terminal=False, parent_particle=None)
            self.root = PFState(parent_action=None, na=self.na, env=env, particle=particle, root=True,
                              budget=budget, depth=0)
        else:
            raise (NotImplementedError("Need to reset the tree"))

    def search(self, n_mcts, c, Env, mcts_env, budget, max_depth=200, fixed_depth=True):
        """ Perform the MCTS search from the root """
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
            mcts_env.seed(np.random.randint(1e7))
            st = 0
            flag = False
            source_particle = None
            while not state.terminal:
                bias = c * self.gamma ** st / (1 - self.gamma) if self.depth_based_bias else c
                action = state.select(c=bias, variance=self.variance)
                st += 1
                k = np.ceil(self.beta * action.n ** self.alpha)
                if action.child_state is not None:
                    state = action.child_state  # select
                    add_particle = k >= state.get_n_particles()
                    if add_particle and not flag:
                        flag = True
                        source_particle, budget = action.sample_from_parent_state(mcts_env, budget)
                        state.add_particle(source_particle)
                        if source_particle.terminal:
                            break
                    elif flag:
                        source_particle, budget = action.sample_from_particle(source_particle, mcts_env, budget)
                        state.add_particle(source_particle)
                        if source_particle.terminal:
                            break
                    elif state.terminal:
                        source_particle = np.random.choice(state.particles)
                        budget -= 1  # sample from the terminal states particles

                else:
                    rollout_depth = max_depth if fixed_depth else max_depth - st
                    state, budget, source_particle = action.add_child_state(mcts_env, budget, max_depth=rollout_depth,
                                                                            source_particle=source_particle,
                                                                            depth=st)  # expand
                    break


            # Back-up
            R = state.V
            state.update()
            particle = source_particle
            while state.parent_action is not None:  # loop back-up until root is reached
                r = particle.reward
                if not particle.terminal:
                    R = r + self.gamma * R
                else:
                    R = r
                action = state.parent_action
                action.update(R)
                state = action.parent_state
                state.update()
                particle = particle.parent_particle
