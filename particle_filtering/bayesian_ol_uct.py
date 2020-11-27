import copy

from envs.planning_env import PlanningEnv
from helpers import stable_normalizer, argmax, max_Q

import numpy as np

from particle_filtering.ol_uct import OL_MCTS, State, Action, sample

DEBUG = False

DEFAULT_ALPHA = 1.1
DEFAULT_BETA = 500


class BayesianAction(Action):
    """ Action object """

    def __init__(self, index, parent_state, mu_zero=0, alpha=0, beta=0):
        super(BayesianAction, self).__init__(index, parent_state)
        self.alpha = alpha
        self.beta = beta
        self.mu_zero = mu_zero
        self.lamda = 1

    def add_child_state(self, env, budget, max_depth=200, depth=0, deepen=False):
        reward, terminal, budget = sample(env, self.index, budget)
        self.child_state = BayesianState(parent_action=self,
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

    def update(self, R):
        super(BayesianAction, self).update(R)
        self.beta += (self.lamda * (R - self.mu_zero) ** 2) / (2 * (self.lamda +1))
        self.mu_zero = (self.lamda * self.mu_zero + R) / (self.lamda + 1)
        self.alpha += 0.5
        self.lamda += 1

    def sample_normal_gamma(self):
        if self.lamda == 0:
            return np.inf
        tau = np.random.gamma(self.alpha, 1/self.beta)
        sigma = np.sqrt(1/(self.lamda * tau))
        return np.random.normal(self.mu_zero, sigma)


class BayesianState(State):
    """ State object """

    def __init__(self, parent_action, na, env: PlanningEnv, budget, root=False, max_depth=200, reward=0, terminal=False, depth=0, deepen=False):

        assert isinstance(env, PlanningEnv), "Only PlanningEnv instances are supported for bayesian OL"
        owner = env.get_next_agent()
        action_iterable = env.get_available_actions(owner)

        max_ep_length = env.get_max_ep_length()
        correction = 0.5 *(1 - (env.get_remaining_steps() / max_ep_length))
        alpha = DEFAULT_ALPHA + correction
        beta = DEFAULT_BETA + 10 * correction

        actions = [BayesianAction(a, parent_state=self,
                                  mu_zero=env.get_mean_estimation(a, owner),
                                  alpha=alpha,
                                  beta=beta) for a in action_iterable]

        super(BayesianState, self).__init__(parent_action, na, env, budget, root, max_depth, reward, terminal, depth, deepen)

        self.child_actions = actions

    # def to_json(self):
    #     js = super(BayesianState, self).to_json()

    def rollout(self, actions, env, budget, max_depth=200, terminal=False, brain_on=False, double_rollout=False, no_pit=False):
        return super(BayesianState, self).rollout(actions, env, budget, max_depth=200, terminal=False,
                                           brain_on=True,
                                           double_rollout=False,
                                           no_pit=False)

    def select(self, c=1.5, csi=1., b=1., variance=False, bias_zero=True):
        """
         Select one of the child actions based on UCT rule
         :param c: UCB exploration constant
         :param csi: exploration constant
         :param b: parameter such that the rewards belong to [0, b]
         :param variance: controls if the UCT-V selection should be applied
         :param bias_zero: gives bias to the first action if ties need to be broken
         """

        samples = np.array([child_action.sample_normal_gamma() for child_action in self.child_actions])

        winner = argmax(samples, bias_zero=bias_zero)

        return self.child_actions[winner]


class Bayesian_OL_MCTS(OL_MCTS):
    """ MCTS object """

    def create_root(self, env, budget):
        if self.root is None:
            signature = env.get_signature()
            self.root_signature = signature
            self.root = BayesianState(parent_action=None, na=self.na, env=env, root=True, budget=budget)
        else:
            raise (NotImplementedError("Need to reset the tree"))

    def return_results(self, temp, on_visits=False, on_lower=False):
        """ Process the output at the root node """
        counts = np.array([child_action.n for child_action in self.root.child_actions])
        Q = np.array([child_action.Q for child_action in self.root.child_actions])
        if DEBUG:
            print(Q)
            print(counts)

        if on_lower:
            uct_lower_bound = np.array(
                [child_action.Q - self.c * np.sqrt(
                    np.log(self.root.n) / child_action.n) if child_action.n > 0 else np.inf
                 for child_action in self.root.child_actions])
            pi_target = max_Q(uct_lower_bound)  # max_Q doesn't really take the maximum Q in this case
        else:
            pi_target = max_Q(Q)
            if np.argmax(pi_target) > 0 and DEBUG:
                print("PIT")
        V_target = np.sum((counts / np.sum(counts)) * Q)[None]
        return self.root_signature, pi_target, V_target