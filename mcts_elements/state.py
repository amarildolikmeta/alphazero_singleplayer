import json
from mcts_elements.action import ThompsonSamplingAction
from mcts_dpw import StochasticState
from helpers import argmax
import numpy as np

#
# class State:
#     ''' State object '''
#
#     def __init__(self, index, r, terminal, parent_action, na, model):
#         ''' Initialize a new state '''
#         self.index = index  # state
#         self.r = r  # reward upon arriving in this state
#         self.terminal = terminal  # whether the domain terminated in this state
#         self.parent_action = parent_action
#         self.n = 0
#         self.model = model
#
#         self.evaluate()
#         # Child actions
#         self.na = na
#         self.child_actions = [Action(a, parent_state=self, Q_init=self.V) for a in range(na)]
#         self.priors = model.predict_pi(index).flatten()
#
#     def to_json(self):
#         inf = {}
#         inf["state"] = str(self.index)
#         inf["V"] = str(self.V)
#         inf["n"] = self.n
#         inf["terminal"] = self.terminal
#         inf["priors"] = str(self.priors)
#         inf["r"] = self.r
#         return json.dumps(inf)
#
#     def select(self, c=1.5):
#         ''' Select one of the child actions based on UCT rule '''
#
#         UCT = np.array(
#             [child_action.Q + prior * c * (np.sqrt(self.n + 1) / (child_action.n + 1)) for child_action, prior in
#              zip(self.child_actions, self.priors)])
#         winner = argmax(UCT)
#         return self.child_actions[winner]
#
#     def evaluate(self):
#         ''' Bootstrap the state value '''
#         self.V = np.squeeze(self.model.predict_V(self.index)) if not self.terminal else np.array(0.0)
#
#     def update(self):
#         ''' update count on backward pass '''
#         self.n += 1
#
# class StochasticState(State):
#     ''' StochasticState object '''
#
#     def __init__(self, index, r, terminal, parent_action, na, model, signature):
#
#         self.index = index  # state
#         self.r = r  # reward upon arriving in this state
#         self.terminal = terminal  # whether the domain terminated in this state
#         self.parent_action = parent_action
#         self.n = 0
#         self.model = model
#         self.signature = signature
#         self.evaluate()
#         # Child actions
#         self.na = na
#         self.priors = model.predict_pi(index).flatten()
#         self.child_actions = [StochasticAction(a, parent_state=self, Q_init=self.V) for a in range(na)]

class ThompsonSamplingState(StochasticState):

    def __init__(self, index, r, terminal, parent_action, na, model):#, signature
        self.index = index  # state
        self.r = r  # reward upon arriving in this state
        self.terminal = terminal  # whether the domain terminated in this state
        self.parent_action = parent_action
        self.n = 0
        self.model = model
        #self.signature = signature
        # Child actions
        self.na = na
        #self.priors = model.predict_pi(index).flatten()
        self.child_actions = [ThompsonSamplingAction(a, parent_state=self, Q_init=self.model.predict(self.index, a))
                              for a in range(na)]

    def to_json(self):
        inf = {}
        inf["state"] = str(self.index)
        #inf["V"] = str(self.V)
        inf["n"] = self.n
        inf["terminal"] = self.terminal
        inf["priors"] = str(self.priors)
        inf["r"] = self.r
        return json.dumps(inf)

    def select(self, stochastic=True):
        qs = np.zeros(self.na)
        for a in self.child_actions:
            qs[a] = a.q(stochastic)
        return self.child_actions[argmax(qs)]

    # def q(self, a, depth, stochastic, dist_params, rho):
    #     # r = 0
    #     # if stochastic:
    #     #     weights = np.random.dirichlet(rho[a])
    #     # else:
    #     #     weights = rho[a] / np.sum(rho[a], axis=-1)
    #     #
    #     # for next_state in range(self.n_states):
    #     #     r += weights[next_state] * self.v(next_state, depth, stochastic)
    #     # r = self.r * self.gamma * r
    #     # return r
    #     if stochastic:
    #         mu, tau = self.sampleNG(a.Q)
    #         return mu
    #     else:
    #         return a.Q[2]
    #
    # def v(self, s, depth, stochastic):
    #     dist_params, _ = self.model.predict(s)
    #     if depth >= self.H or self.terminal:
    #         return 0
    #     if stochastic:
    #         mu, tau = self.sampleNG(dist_params)
    #         return mu
    #     else:
    #         return dist_params[2]

    def sampleNG(self, alpha, beta, mu, lamb):
        tau = np.random.gamma(alpha, beta)
        R = np.random.normal(mu, 1.0 / (lamb * tau))
        return R, tau

class ThompsonSamplingAction:
    ''' ThompsonSamplingAction object '''

    def __init__(self, index, parent_state, Q_init):
        self.index = index
        self.parent_state = parent_state
        self.child_states = []
        self.n_children = 0
        self.state_indeces = {}
        self.W = 0.0
        self.n = 0
        self.Q = Q_init

    def add_child_state(self, s1, r, terminal, model):#, signature
        child_state = ThompsonSamplingState(s1, r, terminal, self, self.parent_state.na, model)#, signature
        self.child_states.append(child_state)
        s1_hash = s1.tostring()
        self.state_indeces[s1_hash] = self.n_children
        self.n_children += 1
        return child_state

    def get_state_ind(self, s1):
        s1_hash = s1.tostring()
        try:
            index = self.state_indeces[s1_hash]
            return index
        except KeyError:
            return -1

    def q(self, stochastic):
        if stochastic:
            mu, tau = self.sampleNG(self.Q)
            return mu
        else:
            return self.Q[2]

    def sampleNG(self, alpha, beta, mu, lamb):
        tau = np.random.gamma(alpha, beta)
        R = np.random.normal(mu, 1.0 / (lamb * tau))
        return R, tau

    def sample_state(self):
        p = []
        for i, s in enumerate(self.child_states):
            s = self.child_states[i]
            p.append(s.n / self.n)
        return self.child_states[np.random.choice(a=self.n_children, p=p)]
