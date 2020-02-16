import sys
sys.path.append('..')

from mcts_elements.state import ThompsonSamplingState
import numpy as np

#
# class Action():
#     ''' Action object '''
#
#     def __init__(self, index, parent_state, Q_init=0.0):
#         self.index = index
#         self.parent_state = parent_state
#         self.W = 0.0
#         self.n = 0
#         self.Q = Q_init
#
#     def add_child_state(self, s1, r, terminal, model):
#         self.child_state = State(s1, r, terminal, self, self.parent_state.na, model)
#         return self.child_state
#
#     def update(self, R):
#         self.n += 1
#         self.W += R
#         self.Q = self.W / self.n
#
#
# class StochasticAction(Action):
#     ''' StochasticAction object '''
#
#     def __init__(self, index, parent_state, Q_init=0.0):
#         super(StochasticAction, self).__init__(index, parent_state, Q_init)
#         self.child_states = []
#         self.n_children = 0
#         self.state_indeces = {}
#
#     def add_child_state(self, s1, r, terminal, model, signature):
#         child_state = StochasticState(s1, r, terminal, self, self.parent_state.na, model, signature)
#         self.child_states.append(child_state)
#         s1_hash = s1.tostring()
#         self.state_indeces[s1_hash] = self.n_children
#         self.n_children += 1
#         return child_state
#
#     def get_state_ind(self, s1):
#         s1_hash = s1.tostring()
#         try:
#             index = self.state_indeces[s1_hash]
#             return index
#         except KeyError:
#             return -1
#
#     def sample_state(self):
#         p = []
#         for i, s in enumerate(self.child_states):
#             s = self.child_states[i]
#             p.append(s.n / self.n)
#         return self.child_states[np.random.choice(a=self.n_children, p=p)]


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

