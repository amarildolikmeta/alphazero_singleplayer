import copy
import numpy as np

from particle_filtering.pf_uct import PFState, PFMCTS, PFAction, Particle
from race_components.race_ol_uct import strategic_rollout
from rl.make_game import is_atari_game


class RacePFState(PFState):
    def __init__(self, parent_action, na, env, particle, budget, root=False, max_depth=200, depth=0, owner=None):
        """ Initialize a new state """
        assert owner is not None, "Owner parameter must be specified for RacePFState class constructor"
        self.owner = owner
        super().__init__(parent_action, na, env, particle, budget, root, max_depth, depth)
        action_list = env.get_available_actions(owner)
        self.child_actions = [RacePFAction(a, parent_state=self, owner=owner) for a in action_list]

    def random_rollout(self, actions, env, budget, max_depth=200, terminal=False):
        return strategic_rollout(env, budget, max_depth=200, terminal=terminal, root_owner=self.owner)

    def is_terminal(self, max_depth, env):
        return env.is_terminal()

    # def add_particle(self, particle):
    #     super(RacePFState, self).add_particle(particle)
    #     self.reward = self.reward[self.owner]
    #
    # def sample_reward(self):
    #     super(RacePFState, self).sample_reward()
    #     self.reward = self.reward[self.owner]


class RacePFAction(PFAction):
    def __init__(self, index, parent_state, owner=None):
        assert owner is not None, "Owner parameter must be specified for RacePFAction class constructor"
        self.owner = owner
        super(RacePFAction, self).__init__(index, parent_state)

    def sample_from_particle(self, source_particle, env, budget):
        env.set_signature(source_particle.state)
        env.seed(np.random.randint(1e7))
        owner = env.get_next_agent()
        s, r, done, _ = env.partial_step(self.index, owner)
        if not done:
            r = r[self.owner]
        if env.has_transitioned() or done:
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
        owner = env.get_next_agent()
        env.seed(np.random.randint(1e7))
        s, r, done, _ = env.partial_step(self.index, owner)
        if not done:
            r = r[self.owner]
        if env.has_transitioned() or done:
            budget -= 1
        return Particle(env.get_signature(), None, r, done, parent_particle=particle), budget

    def add_child_state(self, env, budget, max_depth=200, depth=0, source_particle=None):
        if source_particle is not None:
            new_particle, budget = self.sample_from_particle(source_particle, env, budget)
        else:
            new_particle, budget = self.sample_from_parent_state(env, budget)

        child_owner = env.get_next_agent()

        self.child_state = RacePFState(parent_action=self,
                                       na=self.parent_state.na,
                                       env=env,
                                       root=False,
                                       max_depth=max_depth,
                                       budget=budget,
                                       particle=new_particle,
                                       depth=depth,
                                       owner=child_owner)

        return self.child_state, self.child_state.remaining_budget, new_particle


class RacePFMCTS(PFMCTS):

    def __init__(self, root, root_index, na, gamma, model=None, variance=False,
                 depth_based_bias=False, alpha=0.6, beta=1, owner=None):

        assert owner is not None, "Owner must be specified for the RacePFMCTS constructor"
        self.owner = owner
        self.NAME = "PF_MCTS"
        super(RacePFMCTS, self).__init__(root, root_index, na, gamma, model, variance, depth_based_bias, alpha, beta)

    def create_root(self, env, budget):
        if self.root is None:
            signature = env.get_signature()
            self.root_signature = signature
            particle = Particle(state=signature, seed=None, reward=0, terminal=False, parent_particle=None)
            self.root = RacePFState(parent_action=None, na=self.na, env=env, particle=particle, root=True,
                                    budget=budget, depth=0, owner=self.owner)
        else:
            raise (NotImplementedError("Need to reset the tree"))

    def search(self, n_mcts, c, env, mcts_env, budget, max_depth=200, fixed_depth=True):
        """ Perform the MCTS search from the root """
        env = copy.deepcopy(env)
        self.create_root(env, budget)
        if self.root.terminal:
            raise (ValueError("Can't do tree search from a terminal state"))

        is_atari = is_atari_game(env)
        if is_atari:
            raise NotImplementedError
        while budget > 0:
            state = self.root  # reset to root for new trace
            if not is_atari:
                mcts_env = copy.deepcopy(env)  # copy original Env to rollout from
            else:
                raise NotImplementedError
            mcts_env.seed(np.random.randint(1e7))
            st = 0
            flag = False
            source_particle = None

            while not state.terminal:

                bias = c * self.gamma ** st / (1 - self.gamma) if self.depth_based_bias else c
                action = state.select(c=bias, variance=self.variance)

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

                if mcts_env.has_transitioned():
                    st += 1

            # Back-up

            R = {}
            for agent in range(env.agents_number):
                R[agent] = 0

            if not state.terminal:
                R[state.owner] = state.V

            state.update()
            particle = source_particle
            while state.parent_action is not None:  # loop back-up until root is reached
                r = particle.reward
                if not particle.terminal:
                    R[state.owner] = r + self.gamma * R[state.owner]
                else:
                    R = copy.deepcopy(r)
                action = state.parent_action
                action.update(R[action.owner])
                state = action.parent_state
                state.update()
                particle = particle.parent_particle
