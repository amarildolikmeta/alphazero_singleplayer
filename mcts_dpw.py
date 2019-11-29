import numpy as np
import copy
from helpers import (argmax, is_atari_game, copy_atari_state, restore_atari_state, stable_normalizer)
from mcts import MCTS, Action, State
from igraph import Graph, EdgeSeq
import plotly.graph_objects as go
import plotly.io as pio

class StochasticAction(Action):
    ''' StochasticAction object '''

    def __init__(self, index, parent_state, Q_init=0.0):
        super(StochasticAction, self).__init__(index, parent_state, Q_init)
        self.child_states = []
        self.n_children = 0
        self.state_indeces = {}

    def add_child_state(self, s1, r, terminal, model):
        child_state = StochasticState(s1, r, terminal, self, self.parent_state.na, model)
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

    def sample_state(self):
        p = []
        for i, s  in enumerate(self.child_states):
            s = self.child_states[i]
            p.append(s.n / self.n)
        return self.child_states[np.random.choice(a=self.n_children, p=p)]



class StochasticState(State):
    ''' StochasticState object '''

    def __init__(self, index, r, terminal, parent_action, na, model):
        self.index = index  # state
        self.r = r  # reward upon arriving in this state
        self.terminal = terminal  # whether the domain terminated in this state
        self.parent_action = parent_action
        self.n = 0
        self.model = model

        self.evaluate()
        # Child actions
        self.na = na
        self.priors = model.predict_pi(index).flatten()
        self.child_actions = [StochasticAction(a, parent_state=self, Q_init=self.V) for a in range(na)]



class MCTSStochastic(MCTS):
    ''' MCTS object '''

    def __init__(self, root, root_index, model, na, gamma, alpha=0.6):
        super(MCTSStochastic, self).__init__(root, root_index, model, na, gamma)
        self.alpha = alpha

    def search(self, n_mcts, c, Env, mcts_env):
        ''' Perform the MCTS search from the root '''
        if self.root is None:
            # initialize new root
            self.root = StochasticState(self.root_index, r=0.0, terminal=False, parent_action=None, na=self.na,
                                        model=self.model)
        else:
            self.root.parent_action = None  # continue from current root
        if self.root.terminal:
            raise (ValueError("Can't do tree search from a terminal state"))

        is_atari = is_atari_game(Env)
        if is_atari:
            snapshot = copy_atari_state(Env)  # for Atari: snapshot the root at the beginning

        for i in range(n_mcts):
            state = self.root  # reset to root for new trace
            if not is_atari:
                mcts_env = copy.deepcopy(Env)  # copy original Env to rollout from
            else:
                restore_atari_state(mcts_env, snapshot)
            mcts_env.seed()
            while not state.terminal:
                action = state.select(c=c)
                k = np.ceil(c * action.n ** self.alpha)
                if k >= action.n_children:
                    s1, r, t, _ = mcts_env.step(action.index)
                    if action.get_state_ind(s1) != -1:
                        state = action.child_states[action.get_state_ind(s1)]  # select
                        continue
                    else:
                        state = action.add_child_state(s1, r, t, self.model)  # expand
                        break
                else:
                    state = action.sample_state()

            # Back-up
            R = state.V
            state.update()
            while state.parent_action is not None:  # loop back-up until root is reached
                R = state.r + self.gamma * R
                action = state.parent_action
                action.update(R)
                state = action.parent_state
                state.update()

    def return_results(self, temp):
        ''' Process the output at the root node '''
        counts = np.array([child_action.n for child_action in self.root.child_actions])
        Q = np.array([child_action.Q for child_action in self.root.child_actions])
        pi_target = stable_normalizer(counts, temp)
        V_target = np.sum((counts / np.sum(counts)) * Q)[None]
        return self.root.index.flatten(), pi_target, V_target

    def forward(self, a, s1):
        ''' Move the root forward '''
        action = self.root.child_actions[a]
        if action.n_children > 0:
            if action.get_state_ind(s1) == -1:
                self.root = None
                self.root_index = s1
            else:
                self.root = self.root.child_actions[a].child_states[action.get_state_ind(s1)]

        else:
            self.root = None
            self.root_index = s1

    # def visualize(self):
    #     g = Graph()
    #     v_label = []
    #     nr_vertices = self.inorderTraversal(self.root, g, 0, 0, v_label)
    #     # self.count = 0
    #     # self.print_tree(self.root)v
    #     lay = g.layout_reingold_tilford(mode="in", root=[0])
    #     # igraph.plot(g, layout=lay)
    #
    #     position = {k: lay[k] for k in range(nr_vertices)}
    #     Y = [lay[k][1] for k in range(nr_vertices)]
    #     M = max(Y)
    #
    #     E = [e.tuple for e in g.es]  # list of edges
    #
    #     L = len(position)
    #     Xn = [position[k][0] for k in range(L)]
    #     Yn = [2 * M - position[k][1] for k in range(L)]
    #     Xe = []
    #     Ye = []
    #     for edge in E:
    #         Xe += [position[edge[0]][0], position[edge[1]][0], None]
    #         Ye += [2 * M - position[edge[0]][1], 2 * M - position[edge[1]][1], None]
    #
    #     labels = v_label
    #     fig = go.Figure()
    #     fig.add_trace(go.Scatter(x=Xe,
    #                              y=Ye,
    #                              mode='lines',
    #                              line=dict(color='rgb(210,210,210)', width=1),
    #                              hoverinfo='none'
    #                              ))
    #     fig.add_trace(go.Scatter(x=Xn,
    #                              y=Yn,
    #                              mode='markers',
    #                              name='bla',
    #                              marker=dict(symbol='circle-dot',
    #                                          size=18,
    #                                          color='#6175c1',  # '#DB4551',
    #                                          line=dict(color='rgb(50,50,50)', width=1)
    #                                          ),
    #                              text=labels,
    #                              hoverinfo='text',
    #                              opacity=0.8
    #                              ))
    #
    #     axis = dict(showline=False,  # hide axis line, grid, ticklabels and  title
    #                 zeroline=False,
    #                 showgrid=False,
    #                 showticklabels=False,
    #                 )
    #
    #     fig.update_layout(title='Tree with Reingold-Tilford Layout',
    #                       annotations=make_annotations(position, v_label, labels, M, position),
    #                       font_size=12,
    #                       showlegend=False,
    #                       xaxis=axis,
    #                       yaxis=axis,
    #                       margin=dict(l=40, r=40, b=85, t=100),
    #                       hovermode='closest',
    #                       plot_bgcolor='rgb(248,248,248)'
    #                       )
    #     fig.show()
    #     print("A")

    def inorderTraversal(self, root, g, vertex_index, parent_index, v_label, a_label):
        if root:
            g.add_vertex(vertex_index)
            v_label.append(str(root.index))
            if root.parent_action:
                g.add_edge(parent_index, vertex_index)
                a_label.append(root.parent_action.index)
            par_index = vertex_index
            vertex_index += 1
            for i, a in enumerate(root.child_actions):
                for s in a.child_states:
                    vertex_index = self.inorderTraversal(s, g, vertex_index, par_index, v_label, a_label)
        return vertex_index

    def print_index(self):
            print(self.count)
            self.count += 1

    def print_tree(self, root):
        self.print_index()
        for i, a in enumerate(root.child_actions):
            if hasattr(a, 'child_state'):
                self.print_tree(a.child_state)


# def make_annotations(pos, text, labels, M, position, font_size=10, font_color='rgb(250,250,250)'):
#     L = len(pos)
#     if len(text) != L:
#         raise ValueError('The lists pos and text must have the same len')
#     annotations = []
#     for k in range(L):
#         annotations.append(
#             dict(
#                 text=labels[k],  # or replace labels with a different list for the text within the circle
#                 x=pos[k][0], y=2 * M - position[k][1],
#                 xref='x1', yref='y1',
#                 font=dict(color=font_color, size=font_size),
#                 showarrow=False)
#         )
#     return annotations