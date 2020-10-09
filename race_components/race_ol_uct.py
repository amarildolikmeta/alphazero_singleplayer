import copy

from particle_filtering.ol_uct import Action, State, OL_MCTS
import numpy as np

from rl.make_game import is_atari_game

MAX_P = [1]
PROB_1 = [0.95, 0.05]
PROB_2 = [0.95, 0.025, 0.025]
PROB_3 = [0.91, 0.03, 0.03, 0.03]
PROBS = {1: MAX_P, 2: PROB_1, 3: PROB_2, 4:PROB_3}


def sample(env, action, budget, agent):
    env.seed(np.random.randint(1e7))
    _, r, done, _ = env.partial_step(action, agent)
    if env.has_transitioned() or done:
        budget -= 1
    return r, done, budget


def strategic_rollout(env, budget, max_depth=200, terminal=False, root_owner=None):
    """Rollout from the current state following a default policy up to hitting a terminal state"""
    done = False
    ret = np.zeros(env.agents_number)
    if terminal or budget <= 0:
        return ret, budget
    env.seed(np.random.randint(1e7))
    t = 0

    agent = root_owner

    while budget > 0 and t / env.agents_number < max_depth and not done:
        actions = env.get_available_actions(agent)
        prob = PROBS[len(actions)]
        action = np.random.choice(actions, p=prob)
        s, r, done, _ = env.partial_step(action, agent)

        ret += r
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
        self.end_turn = env.has_transitioned()
        super(RaceState, self).__init__(parent_action, na, env, budget, root, max_depth, reward, terminal, depth)

        # Overwrite the actions generated by parent class
        action_list = env.get_available_actions(owner)
        self.child_actions = [RaceAction(a, parent_state=self, owner=owner) for a in action_list]
        if self.terminal or terminal or root:
            self.V = np.zeros(env.agents_number)

    def random_rollout(self, actions, env, budget, max_depth=200, terminal=False):
        return strategic_rollout(env, budget, max_depth=200, terminal=terminal, root_owner=self.owner)

    def sample(self, env, action, budget, parent_owner=None):
        r, done, budget = sample(env, action, budget, parent_owner)
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

            R = np.zeros(mcts_env.agents_number)

            if not state.terminal:
                R = copy.deepcopy(state.V)

            state.update()
            agents_reward = copy.deepcopy(state.reward)
            while state.parent_action is not None:  # loop back-up until root is reached
                owner = state.parent_action.owner  # rewards are stored in the state following the action, which has different owner
                if not terminal:
                    if state.end_turn:
                        agents_reward = copy.deepcopy(state.reward)
                    try:
                        R[owner] = agents_reward[owner] + self.gamma * R[owner]
                    except TypeError:
                        print("R:", R)
                        print("agents_reward:", agents_reward)
                else:
                    if state.terminal:
                        R = copy.deepcopy(state.reward)
                    else:  # ???
                        R[owner] = state.reward[owner]
                    terminal = False
                action = state.parent_action
                action.update(R[action.owner])
                state = action.parent_state
                state.update()
