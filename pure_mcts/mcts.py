import numpy as np
import copy
from helpers import (argmax, is_atari_game, copy_atari_state, restore_atari_state, stable_normalizer)
from igraph import Graph
import plotly.graph_objects as go
import json

##### MCTS functions #####


class Action(object):
    """ Action object """

    def __init__(self, index, parent_state, Q_init=0.0):
        self.index = index
        self.parent_state = parent_state
        self.W = 0.0
        self.n = 0
        self.Q = Q_init

    def add_child_state(self, s1, r, terminal, budget, env=None, max_depth=200):
        self.child_state = State(s1, r, terminal, self, self.parent_state.na, budget, env=env, max_depth=max_depth)
        return self.child_state, self.child_state.remaining_budget

    def update(self, R):
        self.n += 1
        self.W += R
        self.Q = self.W / self.n


class State(object):
    """ State object """

    def __init__(self, index, r, terminal, parent_action, na, budget,env=None, max_depth=200):
        """ Initialize a new state """
        self.index = index  # state
        self.r = r  # reward upon arriving in this state
        self.terminal = terminal  # whether the domain terminated in this state
        self.parent_action = parent_action

        # Child actions
        self.na = na

        self.remaining_budget = budget

        if env is None:
            print("Warning, no environment was provided, initializing to 0 the value of the state!")
        if terminal or env is None:
            self.V = 0
            self.remaining_budget -= 1
        else:
            self.V, self.remaining_budget = self.evaluate(copy.deepcopy(env), budget, max_depth=max_depth)
        self.n = 0

        self.child_actions = [Action(a, parent_state=self, Q_init=self.V) for a in range(na)]

    def to_json(self):
        inf = {}
        inf["state"] = str(self.index)
        inf["V"] = str(self.V)
        inf["n"] = self.n
        inf["terminal"] = self.terminal
        # inf["priors"] = str(self.priors)
        inf["r"] = self.r
        return json.dumps(inf)

    def select(self, c=1.5):
        """ Select one of the child actions based on UCT rule """
        # TODO check here
        UCT = np.array(
            [child_action.Q + c * np.sqrt(np.log(self.n) / (child_action.n)) if child_action.n > 0 else np.inf
             for child_action in self.child_actions])
        winner = argmax(UCT)
        return self.child_actions[winner]

    def update(self):
        """ update count on backward pass """
        self.n += 1

    def evaluate(self, env, budget, max_depth=200):

        return self.random_rollout(budget, env, max_depth)

    def random_rollout(self, budget, env, max_depth):
        """Run a random exploration from the state up to hitting a terminal state"""
        done = False
        t = 0
        ret = 0
        while t < max_depth and budget > 0 and not done:
            a = np.random.choice(np.arange(self.na))
            _, r, done, _ = env.step(a)
            ret += r
            t += 1
            budget -= 1
        return ret, budget


class MCTS(object):
    """ MCTS object """

    def __init__(self, root, root_index, na, gamma, dpw=False, alpha=0.6, model=None, depth_based_bias=False):
        self.root = root
        self.root_index = root_index
        self.na = na
        self.gamma = gamma
        self.dpw = dpw
        self.alpha = alpha
        self.depth_based_bias = depth_based_bias

    def search(self, n_mcts, c, Env, mcts_env, budget, max_depth=200):
        """ Perform the MCTS search from the root """
        if self.root is None:
            # initialize new root
            self.root = State(self.root_index, r=0.0, terminal=False, parent_action=None, na=self.na, env=mcts_env, budget=budget)
        else:
            self.root.parent_action = None  # continue from current root
        if self.root.terminal:
            raise (ValueError("Can't do tree search from a terminal state"))

        is_atari = is_atari_game(Env)
        if is_atari:
            snapshot = copy_atari_state(Env)  # for Atari: snapshot the root at the beginning

        while budget > 0:
            state = self.root  # reset to root for new trace
            if not is_atari:
                mcts_env = copy.deepcopy(Env)  # copy original Env to rollout from
            else:
                restore_atari_state(mcts_env, snapshot)
            st = 0
            while not state.terminal:
                bias = c * self.gamma ** st / (1 - self.gamma) if self.depth_based_bias else c
                action = state.select(c=bias)
                st += 1
                s1, r, t, _ = mcts_env.step(action.index)
                if hasattr(action, 'child_state'):
                    state = action.child_state  # select
                    if state.terminal:
                        budget -= 1
                    continue
                else:
                    state, budget = action.add_child_state(s1, r, t, budget, env=mcts_env, max_depth=max_depth-st)  # expand
                    break

            # Back-up
            R = state.V
            state.update()
            while state.parent_action is not None:  # loop back-up until root is reached
                if not state.terminal:
                    R = state.r + self.gamma * R
                else:
                    R = state.r
                action = state.parent_action
                action.update(R)
                state = action.parent_state
                state.update()

    def return_results(self, temp):
        """ Process the output at the root node """
        counts = np.array([child_action.n for child_action in self.root.child_actions])
        Q = np.array([child_action.Q for child_action in self.root.child_actions])
        pi_target = stable_normalizer(counts, temp)
        V_target = np.sum((counts / np.sum(counts)) * Q)[None]
        return self.root.index.flatten(), pi_target, V_target

    def forward(self, a, s1, r):
        """ Move the root forward """
        if not (self.root and hasattr(self.root.child_actions[a], 'child_state')):
            self.root = None
            self.root_index = s1
        elif np.linalg.norm(self.root.child_actions[a].child_state.index - s1) > 0.01:
            print('Warning: this domain seems stochastic. Not re-using the subtree for next search. ' +
                  'To deal with stochastic environments, implement progressive widening.')
            self.root = None
            self.root_index = s1
        else:
            self.root = self.root.child_actions[a].child_state

    def visualize(self):
        g = Graph()
        v_label = []
        a_label = []
        nr_vertices = self.inorderTraversal(self.root, g, 0, 0, v_label, a_label)
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
                          annotations=make_annotations(position, v_label, label_xs, label_ys, a_label,  M, position),
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



    def inorderTraversal(self, root, g, vertex_index, parent_index, v_label, a_label):
        if root:
            g.add_vertex(vertex_index)
            #v_label.append(str(root.index) + " Value="+str(root.V))
            v_label.append(root.to_json())
            if root.parent_action:
                g.add_edge(parent_index, vertex_index)
                a_label.append(root.parent_action.index)
            par_index = vertex_index
            vertex_index += 1
            for i, a in enumerate(root.child_actions):
                if hasattr(a, 'child_state'):
                    vertex_index = self.inorderTraversal(a.child_state, g, vertex_index, par_index, v_label, a_label)
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
    L=len(pos)
    if len(labels) != L:
        raise ValueError('The lists pos and text must have the same len')
    annotations = []
    for k in range(L):
        annotations.append(
            dict(
                text=labels[k], # or replace labels with a different list for the text within the circle
                x=pos[k][0]+2, y=2*M-position[k][1],
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
