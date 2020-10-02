import numpy as np
from helpers import argmax
from igraph import Graph
import plotly.graph_objects as go
import copy
from gym.utils import seeding

class MDP(object):
    """
    A class representing a Markov decision process (MDP).

    Parameters
    ----------
        - n_states: the number of states (S)
        - n_actions: the number of actions (A)
        - transitions: transition probability matrix P(S_{t+1}=s'|S_t=s,A_t=a). Shape: (SxAxS)
        - rewards: the reward function R(s,a) or R(s,a,s'). Shape: (S,A) or (S,A,S)
        - init_state: the initial state probabilities P(S_0=s). Shape: (S,)
        - gamma: discount factor in [0,1]

    """

    def __init__(self, n_states, n_actions, transitions, rewards, init_state, gamma):

        assert n_states > 0, "The number of states must be positive"
        self.S = n_states
        assert n_actions > 0, "The number of actions must be positive"
        self.A = n_actions

        assert 0 <= gamma < 1, "Gamma must be in [0,1)"
        self.gamma = gamma

        assert transitions.shape == (n_states, n_actions, n_states), "Wrong shape for P"
        self.P = [transitions[s] for s in range(self.S)]
        self.P = np.array(self.P)
        if rewards.shape == (n_states, n_actions, n_states):
            # Compute the expected one-step rewards
            self.R = np.sum(self.P * rewards, axis=2)
        elif rewards.shape == (n_states, n_actions):
            self.R = rewards
        else:
            raise TypeError("Wrong shape for R")
        self.R = np.array(self.R)
        assert init_state.shape == (n_states,), "Wrong shape for P0"
        self.P0 = init_state
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_signature(self):
        sig = {'state': np.copy(self.state)}
        return sig

    def set_signature(self, sig):
        self.state = np.copy(sig['state'])

    def bellmap_op(self, V):
        """
        Applies the optimal Bellman operator to a value function V.

        :param V: a value function. Shape: (S,)
        :return: the updated value function and the corresponding greedy action for each state. Shapes: (S,) and (S,)
        """

        assert V.shape == (self.S,), "V must be an {0}-dimensional vector".format(self.S)

        Q = np.empty(( self.S, self.A))
        for s in range(self.S):
            Q[s] = self.R[s] + self.gamma * self.P[s].dot(V)

        return Q.argmax(axis=1), Q.max(axis=1), Q

    def value_iteration(self, max_iter=1000, tol=1e-3, verbose=False, qs =False):
        """
        Applies value iteration to this MDP.

        :param max_iter: maximum number of iterations
        :param tol: tolerance required to converge
        :param verbose: whether to print info
        :return: the optimal policy and the optimal value function. Shapes: (S,) and (S,)
        """

        # Initialize the value function to zero
        V = np.zeros(self.S,)

        for i in range(max_iter):

            # Apply the optimal Bellman operator to V
            pi, V_new, Q = self.bellmap_op(V)

            # Check whether the difference between the new and old values are below the given tolerance
            diff = np.max(np.abs(V - V_new))

            if verbose:
                print("Iter: {0}, ||V_new - V_old||: {1}, ||V_new - V*||: {2}".format(i, diff,
                                                                                      2*diff*self.gamma/(1-self.gamma)))

            # Terminate if the change is below tolerance
            if diff <= tol:
                break

            # Set the new value function
            V = V_new
        if qs:
            return pi, V, Q
        else:
            return pi, V

    def reset(self):
        self._t = 0
        self.state = np.random.choice(self.S, p=self.P0)
        return self.state

    def step(self, action):
        next_state = np.random.choice(self.S, p=self.P[self.state, action])
        reward = self.R[self.state, action]
        self.state = next_state
        return next_state, reward, False, {}

    def average_rew(self, state, actions):
        p = np.zeros(self.S)
        p[state] = 1
        for i in range(len(actions)-1):
            a = actions[i]
            p = np.dot(self.P[:, a], p)
        reward = np.dot(self.R[:, actions[-1]])
        return reward


def random_mdp(n_states, n_actions, gamma=0.99):
    """
    Creates a random MDP.

    :param n_states: number of states
    :param n_actions: number of actions
    :param gamma: discount factor
    :return: and MDP with S state, A actions, and randomly generated transitions and rewards
    """

    # Create a random transition matrix
    P = np.random.rand(n_states, n_actions,  n_states)
    # Make sure the probabilities are normalized
    for s in range(n_states):
        for a in range(n_actions):
            P[s, a, :] = P[s, a, :] / np.sum(P[s, a, :])

    # Create a random reward matrix
    R = np.random.rand(n_states, n_actions)

    # Create a random initial-state distribution
    P0 = np.random.rand(n_states)
    # Normalize
    P0 /= np.sum(P0)

    return MDP(n_states, n_actions, P, R, P0, gamma)


class Node:
    def __init__(self, action_seq, mdp, init_state):
        self.n = 0
        self.V = 0

        self.parent_action = None
        self.r = None
        if len(action_seq) > 0:
            self.parent_action = action_seq[-1]
            self.r = mdp.average_rew(init_state, action_seq)
        if len(action_seq) < 5:
            self.children = [Node(copy.deepcopy(action_seq) + [i], mdp, init_state) for i in range(2)]
        else:
            self.children = []
        self.c = 2.

    def select(self):
        UCT = np.array(
            [child.Q + self.c * (np.sqrt(self.n + 1) / (child.n + 1)) for child in self.children])
        winner = argmax(UCT)
        return self.children[winner]


def visualize(root):
    g = Graph()
    v_label = []
    a_label = []
    nr_vertices = inorderTraversal(root, g, 0, 0, v_label, a_label)
    lay = g.layout_reingold_tilford(mode="in", root=[0])
    position = {k: lay[k] for k in range(nr_vertices)}
    Y = [lay[k][1] for k in range(nr_vertices)]
    M = max(Y)
    E = [e.tuple for e in g.es]  # list of edges

    L = len(position)
    Xn = [position[k][0] for k in range(L)]
    Yn = [2 * M - position[k][1] for k in range(L)]
    Xe = []
    Ye = []
    label_xs = []
    label_ys = []
    for edge in E:
        Xe += [position[edge[0]][0], position[edge[1]][0], None]
        Ye += [2 * M - position[edge[0]][1], 2 * M - position[edge[1]][1], None]
        label_xs.append((position[edge[0]][0] + position[edge[1]][0]) / 2)
        label_ys.append((2 * M - position[edge[0]][1] + 2 * M - position[edge[1]][1]) / 2)

    labels = v_label
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Xe,
                             y=Ye,
                             mode='lines',
                             line=dict(color='rgb(210,210,210)', width=1),
                             hoverinfo='none'
                             ))
    fig.add_trace(go.Scatter(x=Xn,
                             y=Yn,
                             mode='markers',
                             name='bla',
                             marker=dict(symbol='circle-dot',
                                         size=5,
                                         color='#6175c1',  # '#DB4551',
                                         line=dict(color='rgb(50,50,50)', width=1)
                                         ),
                             text=labels,
                             hoverinfo='text',
                             opacity=0.8
                             ))

    axis = dict(showline=False,  # hide axis line, grid, ticklabels and  title
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                )
    fig.update_layout(title='Tree with Reingold-Tilford Layout',
                      annotations=make_annotations(position, v_label, label_xs, label_ys, a_label, M, position),
                      font_size=12,
                      showlegend=False,
                      xaxis=axis,
                      yaxis=axis,
                      margin=dict(l=40, r=40, b=85, t=100),
                      hovermode='closest',
                      plot_bgcolor='rgb(248,248,248)'
                      )
    fig.show()
    print("A")


def inorderTraversal(root, g, vertex_index, parent_index, v_label, a_label):
    if root:
        g.add_vertex(vertex_index)
        # v_label.append(str(root.index) + " Value="+str(root.V))
        v_label.append(str(vertex_index))
        if root.parent_action is not None:
            g.add_edge(parent_index, vertex_index)
            a_label.append(str(root.parent_action) + " (%.2f)" %(root.r))
        par_index = vertex_index
        vertex_index += 1

        for i, a in enumerate(root.children):
            vertex_index = inorderTraversal(a, g, vertex_index, par_index, v_label, a_label)
    return vertex_index

def print_index(self):
    print(self.count)
    self.count += 1

def print_tree(self, root):
    self.print_index()
    for i, a in enumerate(root.child_actions):
        if hasattr(a, 'child_state'):
            self.print_tree(a.child_state)


def make_annotations(pos, labels, Xe, Ye, a_labels, M, position, font_size=10, font_color='rgb(250,250,250)'):
    L = len(pos)
    if len(labels) != L:
        raise ValueError('The lists pos and text must have the same len')
    annotations = []
    for k in range(L):
        annotations.append(
            dict(
                text=labels[k],  # or replace labels with a different list for the text within the circle
                x=pos[k][0] + 2, y=2 * M - position[k][1],
                xref='x1', yref='y1',
                font=dict(color=font_color, size=font_size),
                showarrow=False)
        )
    for e in range(len(a_labels)):
        annotations.append(
            dict(
                text=a_labels[e],  # or replace labels with a different list for the text within the circle
                x=Xe[e], y=Ye[e],
                xref='x1', yref='y1',
                font=dict(color='rgb(0, 0, 0)', size=font_size),
                showarrow=False)
        )
    return annotations


if __name__ == '__main__':
    mdp = random_mdp(5, 2)
    r = mdp.average_rew(0, [0, 1, 1])

    root = Node([], mdp, 0)
    visualize(root)





