from planners.q_learning_ol_uct import QL_OL_MCTS, State, Action, sample

from helpers import argmax
import numpy as np

class OLSarsaAction(Action):
    def __init__(self, index, parent_state):
        super(OLSarsaAction, self).__init__(index, parent_state)

    def add_child_state(self, env, budget, max_depth=200, depth=0, deepen=False, q_learning=False):
        reward, terminal, budget = sample(env, self.index, budget)
        self.child_state = OLSarsaState(parent_action=self,
                                 na=self.parent_state.na,
                                 env=env,
                                 root=False,
                                 max_depth=max_depth,
                                 budget=budget,
                                 reward=reward,
                                 terminal=terminal,
                                 depth=depth,
                                 deepen=deepen,
                                 q_learning=False)

        return self.child_state, self.child_state.remaining_budget

    def update(self, R, q_learning=False, mixed_q_learning=False, alpha=0.1, gamma=1., beta=2):
        self.Q = self.child_state.V
        self.n += 1

class OLSarsaState(State):
    def __init__(self, parent_action, na, env, budget, root=False, max_depth=200, reward=0, terminal=False, depth=0,
                     deepen=False, q_learning=False):

        """ Initialize a new state """

        super(OLSarsaState, self).__init__(parent_action, na, env, budget,
                                           root=root,
                                           max_depth=max_depth,
                                           reward=reward,
                                           terminal=terminal,
                                           depth=depth,
                                           deepen=deepen,
                                           q_learning=False)

        # Replace actions with correct subclass

        if hasattr(env, "get_available_actions") and hasattr(env, "get_next_agent"):
            owner = env.get_next_agent()
            action_list = env.get_available_actions(owner)
            self.child_actions = [OLSarsaAction(a, parent_state=self) for a in action_list]
        else:
            self.child_actions = [OLSarsaAction(a, parent_state=self) for a in range(na)]

    def select(self, c=1.5, csi=1., b=1., variance=False, bias_zero=True):
        if not variance:
            uct_upper_bound = np.array(
                [child_action.child_state.V + c * np.sqrt(np.log(self.n) / child_action.n)
                 if child_action.n > 0 else np.inf
                 for child_action in self.child_actions])

            winner = argmax(uct_upper_bound, bias_zero=bias_zero)
            return self.child_actions[winner]
        else:
            raise NotImplementedError("No variance-based selection is actually supported")

class OL_Sarsa_MCTS(QL_OL_MCTS):
    def __init__(self, root, root_index, na, gamma, model=None, variance=False, depth_based_bias=False, csi=1.,
                 q_learning=False, alpha=0.1, beta=2., mixed_q_learning=True):

        super(OL_Sarsa_MCTS, self).__init__(root, root_index, na, gamma,
                                            model=model,
                                            variance=variance,
                                            depth_based_bias=depth_based_bias,
                                            csi=csi,
                                            q_learning=False,
                                            alpha=alpha,
                                            beta=beta,  # Use this parameter as the lambda parameter of the paper
                                            mixed_q_learning=False)

        self.lamda = beta

    def create_root(self, env, budget):
        if self.root is None:
            signature = env.get_signature()
            self.root_signature = signature
            self.root = OLSarsaState(parent_action=None,
                                     na=self.na,
                                     env=env,
                                     root=True,
                                     budget=budget,
                                     q_learning=self.q_learning)
        else:
            raise (NotImplementedError("Need to reset the tree"))

    def backup(self, state, terminal):
        delta_sum = 0
        v_next = 0

        # Back-up
        # if budget < 0: #finished budget before rollout
        #     break
        while state.parent_action is not None:  # loop back-up until root is reached
            v_current = state.V
            R = state.reward
            delta = R + self.gamma * v_next - v_current
            delta_sum = self.lamda * self.gamma * delta_sum + delta

            state.update(delta_sum)
            alpha = 1 / state.n
            state.V += alpha * delta_sum

            v_next = v_current

            action = state.parent_action
            state = action.parent_state

            action.update(R, q_learning=False, mixed_q_learning=False, alpha=self.alpha, gamma=self.gamma, beta=self.beta)

