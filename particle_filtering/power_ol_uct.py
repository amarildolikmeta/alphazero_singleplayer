from particle_filtering.ol_uct import *

class PowerState(State):

    def __init__(self, parent_action, na, env, budget, root=False, max_depth=200, reward=0, terminal=False, depth=0,
                 deepen=False):

        """ Initialize a new state """
        self.parent_action = parent_action
        # Child actions
        self.na = na
        self.remaining_budget = budget
        self.depth = depth
        self.terminal = self.is_terminal(max_depth, env) or terminal
        self.reward = reward
        self.root = root
        self.n = 0

        if hasattr(env, "get_available_actions") and hasattr(env, "get_next_agent"):
            owner = env.get_next_agent()
            action_list = env.get_available_actions(owner)
            self.child_actions = [PowerAction(a, parent_state=self) for a in action_list]
        else:
            self.child_actions = [PowerAction(a, parent_state=self) for a in range(na)]

        if env is None:
            print("[WARNING] No environment was provided, initializing to 0 the value of the state!")
            self.V = 0
        elif self.terminal or root or (deepen and len(self.child_actions) == 1):
            self.V = 0.
        else:
            self.V, self.remaining_budget = self.evaluate(env, budget, max_depth, terminal)

    def update(self, p=100):
        self.n += 1

        if self.n == 1: # If the state has been visited only one time, its value must be the rollout
            return

        state_value = 0
        for action in self.child_actions:
            if action.n > 0:
                temp = (action.Q ** p)
                state_value += action.n/self.n * temp
        self.V = state_value ** (1/p)

class PowerAction(Action):

    def __init__(self, index, parent_state):
        super(PowerAction, self).__init__(index, parent_state)
        self.reward_sum = 0

    def add_child_state(self, env, budget, max_depth=200, depth=0, deepen=False):
        reward, terminal, budget = sample(env, self.index, budget)
        self.child_state = PowerState(parent_action=self,
                                 na=self.parent_state.na,
                                 env=env,
                                 root=False,
                                 max_depth=max_depth,
                                 budget=budget,
                                 reward=reward,
                                 terminal=terminal,
                                 depth=depth,
                                 deepen=deepen)

        return self.child_state, self.child_state.remaining_budget

    def update(self, r, gamma=1.):
        self.reward_sum += r
        self.n += 1
        self.Q = self.reward_sum / self.n + gamma * self.child_state.V


class PowerOLMCTS(OL_MCTS):
    def __init__(self, root, root_index, na, gamma, model=None, variance=False, depth_based_bias=False, csi=1.,
                 alpha=0.1, p=100):
        super(PowerOLMCTS, self).__init__(root, root_index, na, gamma, model, variance, depth_based_bias, csi, alpha)
        self.p = p

    def create_root(self, env, budget):
        if self.root is None:
            signature = env.get_signature()
            self.root_signature = signature
            self.root = PowerState(parent_action=None, na=self.na, env=env, root=True, budget=budget)
        else:
            raise (NotImplementedError("Need to reset the tree"))

    def backup(self, state, terminal):
        # Back-up
        # if budget < 0: #finished budget before rollout
        #     break
        state.update(p = self.p)
        while state.parent_action is not None:  # loop back-up until root is reached
            # if state.reward > -85:
            #     print("WTF:", state.reward)
            r = state.reward
            action = state.parent_action
            action.update(r, self.gamma)
            state = action.parent_state
            state.update(p=self.p)