import copy

from particle_filtering.ol_uct import Action, State, OL_MCTS
import numpy as np

from rl.make_game import is_atari_game

DEFAULT_STRATEGY = [0.91, 0.03, 0.03, 0.03]
MAX_P = [1]


def sample(env, action, budget, agent):
    env.seed(np.random.randint(1e7))
    _, r, done, _ = env.partial_step(action, agent)
    if env.has_transitioned() or done:
        budget -= 1
    return r, done, budget


def strategic_rollout(env, budget, max_depth=200, terminal=False, root_owner=None):
    """Rollout from the current state following a default policy up to hitting a terminal state"""
    done = False
    if terminal:
        return 0, budget
    env.seed(np.random.randint(1e7))
    ret = 0
    t = 0

    agent = root_owner

    while budget > 0 and t / env.agents_number < max_depth and not done:
        actions = env.get_available_actions(agent)
        if len(actions) > 1: # Pit-stop can be done
            prob = DEFAULT_STRATEGY
        else: # Pit-stop is not available
            prob = MAX_P
        action = np.random.choice(actions, p = prob)
        s, r, done, _ = env.partial_step(action, agent)
        if agent == root_owner or done:
            ret += r[root_owner]
        t += 1

        # Get the agent ranking to specify the turn order
        if env.has_transitioned():
            budget -= 1

        agent = env.get_next_agent()

    return ret, budget


class RaceAction(Action):
    def __init__(self, index, parent_state, owner=None):
        assert owner is not None, "Owner must be specified for constructor of RaceAction class"
        self.owner = owner
        super(RaceAction, self).__init__(index, parent_state)

    def add_child_state(self, env, budget, max_depth=200, depth=0):
        reward, terminal, budget = sample(env, self.index, budget, self.owner)

        child_owner = env.get_next_agent()

        if not env.is_terminal():
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
        assert owner is not None, "Owner parameter must be specified for RaceState class constructor"
        self.owner = owner
        super(RaceState, self).__init__(parent_action, na, env, budget, root, max_depth, reward, terminal, depth)
        self.child_actions = [RaceAction(a, parent_state=self, owner=self.owner) for a in range(na)]
        # if not self.terminal and len(self.reward) > 1:
        #     self.reward = self.reward[owner]

    def random_rollout(self, actions, env, budget, max_depth=200, terminal=False):
        return strategic_rollout(env, budget, max_depth=200, terminal=False, root_owner=self.owner)

    def sample(self, env, action, budget, parent_owner=None):
        r, done, budget = sample(env, action, budget, parent_owner)
        if not env.is_terminal():
            self.reward = r[self.owner]
        else:
            self.reward = r
        return done, budget

    def is_terminal(self, max_depth, env):
        return env.is_terminal()


class RaceOL_MCTS(OL_MCTS):
    def __init__(self, root, root_index, na, gamma, model=None, variance=False, depth_based_bias=False, owner=None):
        assert owner is not None, "Owner must be specified for RaceOL_MCTS constructor"
        self.owner = owner
        self.NAME = "OL_MCTS"
        super(RaceOL_MCTS, self).__init__(root, root_index, na, gamma, model, variance, depth_based_bias)

    def create_root(self, env, budget):
        if self.root is None:
            signature = env.get_signature()
            self.root_signature = signature
            self.root = RaceState(parent_action=None, na=self.na, env=env, root=True, budget=budget, owner=self.owner)
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
            terminal = False

            while not state.terminal:

                bias = c * self.gamma ** st / (1 - self.gamma) if self.depth_based_bias else c
                action = state.select(c=bias, variance=self.variance)
                if action.child_state is not None:
                    parent_owner = state.owner
                    state = action.child_state  # select
                    terminal, budget = state.sample(mcts_env, action.index, budget, parent_owner)
                    if terminal:
                        break
                else:
                    rollout_depth = max_depth if fixed_depth else max_depth - st
                    state, budget = action.add_child_state(mcts_env, budget, rollout_depth, depth=st)  # expand
                    break

                if mcts_env.has_transitioned():
                    st += 1

                # If there are no more agent in the decision queue, a lap has been completed
                # and the ordering of the agents must be re-evaluated

            # Back-up

            R = [0] * env.agents_number

            if not state.terminal:
                R[state.owner] = state.V

            state.update()
            while state.parent_action is not None:  # loop back-up until root is reached
                if not terminal and not state.terminal:
                    R[state.owner] = state.reward + self.gamma * R[state.owner]
                else:
                    R = copy.deepcopy(state.reward)
                    terminal = False
                action = state.parent_action
                action.update(R[action.owner])
                state = action.parent_state
                state.update()
