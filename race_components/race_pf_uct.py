import copy
import numpy as np

from particle_filtering.pf_uct import PFState, PFMCTS, PFAction, Particle
from rl.make_game import is_atari_game

def random_rollout(actions, env, budget, max_depth=200, terminal=False, root_owner=None):
    """Rollout from the current state following a random policy up to hitting a terminal state"""
    done = False
    if terminal:
        return 0, budget
    env.seed(np.random.randint(1e7))
    ret = 0
    t = 0

    agent_queue = env.get_agent_standings()

    # The root owner might not be the leading agent, discard agents that have already acted
    while agent_queue.get() != root_owner:
        pass
    agent = root_owner

    while budget > 0 and t/3 < max_depth and not done:
        action = np.random.choice(actions)
        s, r, done, _ = env.partial_step(action, agent)
        if t % env.agents_number == 0:
            ret += r
        t += 1

        if agent_queue.empty():
            budget -= 1
            agent_queue = env.get_agent_standings()

        agent = agent_queue.get()
    return ret, budget


class RacePFState(PFState):
    def __init__(self, parent_action, na, env, particle, budget, root=False, max_depth=200, depth=0, owner=None):

        """ Initialize a new state """
        self.parent_action = parent_action
        # Child actions
        self.na = na
        self.remaining_budget = budget
        self.depth = depth
        self.terminal = (depth/3 >= max_depth)
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

        assert owner is not None, "Owner parameter must be specified for RacePFState class constructor"
        self.owner = owner
        self.child_actions = [RacePFAction(a, parent_state=self, owner=owner) for a in range(na)]

    def evaluate(self, env, budget, max_depth=200, terminal=False):
        actions = np.arange(self.na, dtype=int)
        if budget > 0:
            return_, budget = random_rollout(actions, env, budget, max_depth, terminal)
        else:
            return_ = 0
        return return_, budget


class RacePFAction(PFAction):
    def __init__(self, index, parent_state, owner=None):
        super(RacePFAction, self).__init__(index, parent_state)
        assert owner is not None, "Owner parameter must be specified for RacePFAction class constructor"
        self.owner = owner

    def sample_from_particle(self, source_particle, env, budget, last_agent=False):
        env.set_signature(source_particle.state)
        env.seed(np.random.randint(1e7))
        s, r, done, _ = env.partial_step(self.index, self.owner)
        if last_agent:
            budget -= 1
        new_particle = Particle(env.get_signature(), None, r[self.owner], done, parent_particle=source_particle)
        return new_particle, budget

    def sample_from_parent_state(self, env, budget, last_agent=False):
        state = self.parent_state
        if state.root:
            parent_particle = state.particles[0]
        else:
            parent_particle = np.random.choice(state.particles)
        new_particle, budget = self.generate_new_particle(env, parent_particle, budget, last_agent=last_agent)
        return new_particle, budget

    def generate_new_particle(self, env, particle, budget, last_agent=False):
        """Generate the successor particle for a given particle"""
        # Do not give any reward if a particle is being generated from a terminal state particle
        if particle.terminal:
            return Particle(particle.state, None, 0, True, parent_particle=particle), budget
        # Apply the selected action to the state encapsulated by the particle and store the new state and reward
        env.set_signature(particle.state)
        env.seed(np.random.randint(1e7))
        s, r, done, _ = env.partial_step(self.index, self.owner)
        if last_agent:
            budget -= 1
        return Particle(env.get_signature(), None, r[self.owner], done, parent_particle=particle), budget

    def add_child_state(self, env, budget, max_depth=200, depth=0, source_particle=None, owner=None):
        assert owner is not None, "Owner parameter must be specified to add a new state"
        if source_particle is not None:
            new_particle, budget = self.sample_from_particle(source_particle, env, budget)
        else:
            new_particle, budget = self.sample_from_parent_state(env, budget)
        self.child_state = RacePFState(parent_action=self,
                                       na=self.parent_state.na,
                                       env=env,
                                       root=False,
                                       max_depth=max_depth,
                                       budget=budget,
                                       particle=new_particle,
                                       depth=depth,
                                       owner=owner)

        return self.child_state, self.child_state.remaining_budget, new_particle


class RacePFMCTS(PFMCTS):

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
            agent_queue = env.get_agent_standings()

            # The root owner might not be the leading agent, discard agents that have already acted
            while agent_queue.get() != state.owner:
                pass
            while not state.terminal:
                next_agent = agent_queue.get()

                bias = c * self.gamma ** st / (1 - self.gamma) if self.depth_based_bias else c
                action = state.select(c=bias, variance=self.variance)
                # TODO check assumption
                if agent_queue.empty():
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
                    elif state.terminal and agent_queue.empty():
                        source_particle = np.random.choice(state.particles)
                        budget -= 1  # sample from the terminal states particles

                else:
                    rollout_depth = max_depth if fixed_depth else max_depth - st
                    state, budget, source_particle = action.add_child_state(mcts_env, budget, max_depth=rollout_depth,
                                                                            source_particle=source_particle,
                                                                            depth=st, owner=next_agent)  # expand
                    break

                # If there are no more agent in the decision queue, a lap has been completed
                # and the ordering of the agents must be re-evaluated
                if agent_queue.empty():
                    agent_queue = env.get_agent_standings()

            # Back-up

            R = {}
            for agent in range(env.agents_number):
                R[agent] = 0

            if not state.terminal:
                R[state.owner] = state.V

            state.update()
            particle = source_particle
            while state.parent_action is not None:  # loop back-up until root is reached
                r = particle.reward[state.owner]
                if not particle.terminal:
                    R[state.owner] = r + self.gamma * R[state.owner]
                else:
                    R[state.owner] = r
                action = state.parent_action
                action.update(R[action.owner])
                state = action.parent_state
                state.update()
                particle = particle.parent_particle
