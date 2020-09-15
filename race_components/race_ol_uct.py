import copy

from particle_filtering.ol_uct import Action, State, OL_MCTS
import numpy as np

from rl.make_game import is_atari_game


def sample(env, action, budget, agent):
    env.seed(np.random.randint(1e7))
    _, r, done, _ = env.partial_step(action, agent)
    if env.has_transitioned():
        budget -= 1
    return r, done, budget


def random_rollout(actions, env, budget, max_depth=200, terminal=False, root_owner=None):
    """Rollout from the current state following a random policy up to hitting a terminal state"""
    done = False
    if terminal:
        return 0, budget
    env.seed(np.random.randint(1e7))
    ret = 0
    t = 0

    agent_queue = env.get_agents_standings()

    # The root owner might not be the leading agent, discard agents that have already acted
    while agent_queue.get() != root_owner:
        pass
    agent = root_owner

    while budget > 0 and t / 3 < max_depth and not done:
        action = np.random.choice(actions)
        s, r, done, _ = env.partial_step(action, agent)
        if t % env.agents_number == 0:
            ret += r
        t += 1

        # Get the agent ranking to specify the turn order
        if env.has_transitioned():
            budget -= 1
            agent_queue = env.get_agents_standings()

        agent = agent_queue.get()
    return ret, budget


class RaceAction(Action):
    def __init__(self, index, parent_state, owner):
        super(RaceAction, self).__init__(index, parent_state)
        self.owner = owner

    def add_child_state(self, env, budget, max_depth=200, depth=0, child_owner=None):
        assert child_owner is not None, "Child state owner must be specified"
        reward, terminal, budget = sample(env, self.index, budget, self.owner)
        if not terminal:
            reward = reward[self.owner]
        self.child_state = RaceState(parent_action=self,
                                     na=self.parent_state.na,
                                     env=env,
                                     root=False,
                                     max_depth=max_depth,
                                     budget=budget,
                                     reward=reward,
                                     terminal=terminal,
                                     depth=depth,
                                     owner=child_owner)

        return self.child_state, self.child_state.remaining_budget


class RaceState(State):
    def __init__(self, parent_action, na, env, budget, root=False, max_depth=200, reward=0, terminal=False, depth=0,
                 owner=None):
        super(RaceState, self).__init__(parent_action, na, env, budget, root, max_depth, reward, terminal, depth)
        self.owner = owner
        self.child_actions = [RaceAction(a, parent_state=self) for a in range(na)]

    @staticmethod
    def random_rollout(actions, env, budget, max_depth=200, terminal=False, root_owner=None):
        return random_rollout(actions, env, budget, max_depth=200, terminal=False, root_owner=root_owner)

    def sample(self, env, action,  budget):
        r, done, budget = sample(env, action, budget)
        self.reward = r[self.owner]
        return done, budget

class RaceOL_MCTS(OL_MCTS):
    def __init__(self, root, root_index, na, gamma, model=None, variance=False,
                 depth_based_bias=False, owner=None):
        assert owner is not None, "Owner must be specified for RaceOL_MCTS constructor"
        super(RaceOL_MCTS, self).__init__(root, root_index, na, gamma, model, variance, depth_based_bias)
        self.owner = owner

    def create_root(self, env, budget):
        if self.root is None:
            signature = env.get_signature()
            self.root_signature = signature
            self.root = RaceState(parent_action=None, na=self.na, env=env, root=True, budget=budget, owner=self.owner)
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
            terminal = False

            agent_queue = env.get_agents_standings()

            # The root owner might not be the leading agent, discard agents that have already acted
            while agent_queue.get() != state.owner:
                pass
            while not state.terminal:
                next_agent = agent_queue.get()

                bias = c * self.gamma ** st / (1 - self.gamma) if self.depth_based_bias else c
                action = state.select(c=bias, variance=self.variance)
                if agent_queue.empty():
                    st += 1
                if action.child_state is not None:
                    state = action.child_state  # select
                    terminal, budget = state.sample(mcts_env, action.index, budget)
                    if terminal:
                        break
                else:
                    rollout_depth = max_depth if fixed_depth else max_depth - st
                    state, budget = action.add_child_state(mcts_env, budget, rollout_depth, depth=st)  # expand
                    break

                # If there are no more agent in the decision queue, a lap has been completed
                # and the ordering of the agents must be re-evaluated
                if agent_queue.empty():
                    agent_queue = env.get_agents_standings()

            # Back-up

            R = {}
            for agent in range(env.agents_number):
                R[agent] = 0

            if not state.terminal:
                R[state.owner] = state.V

            state.update()
            while state.parent_action is not None:  # loop back-up until root is reached
                if not terminal:
                    R[state.owner] = state.reward + self.gamma * R[state.owner]
                else:
                    R[state.owner] = state.reward
                action = state.parent_action
                action.update(R[action.owner])
                state = action.parent_state
                state.update()
