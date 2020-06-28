import copy
from helpers import stable_normalizer, copy_atari_state, restore_atari_state, argmax, max_Q
from rl.make_game import is_atari_game
import numpy as np
from igraph import Graph
import plotly.graph_objects as go
import json


def sample(envs, action, terminals, budget):
        rewards = []
        dones = []
        for i, env in enumerate(envs):
            if terminals[i]:
                r = 0
                done = 1
            else:
                _, r, done, _ = env.step(action)
                budget -=1
            rewards.append(r)
            dones.append(done)
        terminals = np.array(dones)
        return rewards, terminals, budget


def random_rollout(actions, env, budget, max_depth=200, terminal=False):
    """Rollout from the current state following a random policy up to hitting a terminal state"""
    done = False
    if terminal:
        #budget -= 1
        return 0, budget
    env.seed()
    ret = 0
    t = 0
    while budget > 0 and t < max_depth and not done:
        action = np.random.choice(actions)
        s, r, done, _ = env.step(action)
        ret += r
        budget -= 1
        t += 1
    return ret, budget


class Action(object):
    """ Action object """

    def __init__(self, index, parent_state):
        self.index = index
        self.parent_state = parent_state
        self.W = 0.0
        self.M2 = 0.0
        self.n = 0
        self.Q = 0
        self.sigma = np.inf
        self.rewards = []
        # self.child_state = None

    def add_child_state(self, state, envs, terminals, budget, sampler=None, max_depth=200):
        rewards, terminals, budget = sample(envs, self.index, terminals, budget)
        rollout_budget_bound = len(envs) * max_depth
        if budget < rollout_budget_bound * 0.5:
            return state, 0

        elif 0.5 * rollout_budget_bound <= budget < rollout_budget_bound:
            budget = rollout_budget_bound

        self.child_state = State(parent_action=self,
                                 na=self.parent_state.na,
                                 envs=envs,
                                 sampler=sampler,
                                 root=False,
                                 max_depth=max_depth,
                                 budget=budget,
                                 rewards=rewards,
                                 terminals=terminals)

        return self.child_state, self.child_state.remaining_budget

    def update(self, R):
        for r in R:
            self.update_aggregate(r)
        self.finalize_aggregate()

    def update_aggregate(self, new_sample):
        self.n += 1
        delta = new_sample - self.Q
        self.Q += delta / self.n
        delta2 = new_sample - self.Q
        self.M2 += delta * delta2

    def finalize_aggregate(self):
        if self.n < 2:
            self.sigma = np.inf
        else:
            self.sigma = self.M2 / self.n
            # sample_variance = self.M2 / (self.n - 1)


class State(object):
    """ State object """

    def __init__(self, parent_action, na, envs, budget, sampler=None, root=False, max_depth=200, rewards=None,
                 terminals=None):

        """ Initialize a new state """
        self.parent_action = parent_action
        # Child actions
        self.na = na
        self.terminal = False
        self.remaining_budget = budget
        if terminals is None:
            terminals = np.zeros(len(envs))
        if rewards is None:
            rewards = np.zeros(len(envs))
        self.terminals = terminals
        self.rewards = rewards
        if root:
            self.V = 0.
        elif envs is None:
            print("Warning, no environment was provided, initializing to 0 the value of the state!")
            self.V = 0
        else:
            returns, self.remaining_budget = self.evaluate(envs, terminals, budget, max_depth)
            self.V = np.mean(returns)
            self.returns = returns
        self.n = 0
        self.V = np.array(self.V)
        self.child_actions = [Action(a, parent_state=self) for a in range(na)]

    def to_json(self):
        inf = {
            "V": str(self.V) + '<br>',
            "n": str(self.n) + '<br>'}
        return json.dumps(inf)

    def sample(self, envs, action, terminals, budget):
        self.rewards = []
        dones = []
        for i, env in enumerate(envs):
            if terminals[i] != 0:
                r = 0
                done = 1
            else:
                _, r, done, _ = env.step(action)
                budget -= 1
            self.rewards.append(r)
            dones.append(done)
        self.terminals = np.array(dones)
        return self.terminals, budget

    def select(self, c=1.5, csi=1., b=1.):
        """
         Select one of the child actions based on UCT rule
         :param c: UCB exploration constant
         :param csi: exploration constant
         :param b: parameter such that the rewards belong to [0, b]
         """
        if self.n > 0:
            logp = np.log(self.n)
        else:
            logp = -np.inf

        bound = np.array([child_action.Q + np.sqrt(
            csi * child_action.sigma * logp / child_action.n) + 3 * c * b * csi * logp / child_action.n
                          if child_action.n > 0 and not np.isinf(child_action.sigma).any() else np.inf
                          for child_action in self.child_actions])

        winner = argmax(bound)
        return self.child_actions[winner]

    def update(self, n):
        """ update count on backward pass """
        self.n += n

    def evaluate(self, envs, terminals, budget, max_depth=200):
        actions = np.arange(self.na, dtype=int)
        results = []
        for i in range(len(envs)):
            assert budget > 0, "Running out of budget during evaluation of a state should never happen"
            return_, budget = random_rollout(actions, envs[i], budget, max_depth, terminals[i])
            results.append(return_)
        return results, budget


class PFMCTS(object):
    ''' MCTS object '''

    def __init__(self, root, root_index, na, gamma, model=None, particles=2, sampler=None):
        self.root = root
        self.root_index = root_index
        self.na = na
        self.gamma = gamma
        self.n_particles = particles
        self.sampler = sampler

    def reset_root(self, envs, Env):
        for i in range(len(envs)):
            envs[i] = copy.deepcopy(Env)
            envs[i].seed()

    def search(self, n_mcts, c, Env, mcts_env, budget, max_depth=200, fixed_depth=True):
        """ Perform the MCTS search from the root """
        Envs = None
        if not self.sampler:
            Envs = [copy.deepcopy(Env) for _ in range(self.n_particles)]

        if self.root is None:
            # initialize new root with many equal particles
            signature = Env.get_signature()
            self.root_signature = signature
            box = None
            to_box = getattr(self, "index_to_box", None)
            if callable(to_box):
                box = Env.index_to_box(signature["state"])
            self.root = State(parent_action=None, na=self.na, envs=Envs, sampler=self.sampler,
                              root=True, budget=budget)
        else:
            raise (NotImplementedError("Need to reset the tree"))

        if self.root.terminal:
            raise (ValueError("Can't do tree search from a terminal state"))

        is_atari = is_atari_game(Env)
        if is_atari:
            raise NotImplementedError
            snapshot = copy_atari_state(Env)  # for Atari: snapshot the root at the beginning

        while budget > 0:
            state = self.root  # reset to root for new trace
            # reset to root state
            self.reset_root(Envs, Env)

            if not is_atari:
                mcts_envs = None
                if not self.sampler:
                    mcts_envs = Envs
            else:
                raise NotImplementedError
                restore_atari_state(mcts_env, snapshot)
            st = 0
            terminal = False
            terminals = np.zeros(self.n_particles, dtype=np.int8)
            while not terminal:
                action = state.select(c=c)
                st += 1
                # s1, r, t, _ = mcts_env.step(action.index)
                if hasattr(action, 'child_state'):
                    state = action.child_state  # select
                    terminals, budget = state.sample(mcts_envs, action.index, terminals, budget)
                    if np.all(terminals):
                        break
                else:
                    rollout_depth = max_depth if fixed_depth else max_depth - st
                    child_state, budget = action.add_child_state(state, mcts_envs, terminals, budget, self.sampler,
                                                                 rollout_depth)  # expand
                    if id(state) == id(child_state):
                        backup_ward = False
                    else:
                        backup_ward = True
                        state = child_state
                    break

            # Back-up
            if backup_ward:
                returns = np.array(state.returns) * (1 - state.terminals)
                state.update(self.n_particles)
                while state.parent_action is not None:  # loop back-up until root is reached
                    returns = state.rewards + self.gamma * returns
                    action = state.parent_action
                    action.update(returns)
                    state = action.parent_state
                    state.update(self.n_particles)

    def return_results(self, temp, on_visits=False):
        """ Process the output at the root node """
        counts = np.array([child_action.n for child_action in self.root.child_actions])
        Q = np.array([child_action.Q for child_action in self.root.child_actions])
        if on_visits:
            pi_target = stable_normalizer(counts, temp)
        else:
            pi_target = max_Q(Q)
        V_target = np.sum((counts / np.sum(counts)) * Q)[None]
        return self.root_signature, pi_target, V_target

    def forward(self, a, s1, r):
        """ Move the root forward """
        self.root = None
        self.root_index = s1

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
                                             size=18,
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

    def inorderTraversal(self, root, g, vertex_index, parent_index, v_label, a_label):
        if root:
            g.add_vertex(vertex_index)
            # v_label.append(str(root.index) + " Value="+str(root.V))
            v_label.append(root.to_json())
            if root.parent_action:
                g.add_edge(parent_index, vertex_index)
                a_label.append(root.parent_action.index)
            par_index = vertex_index
            vertex_index += 1
            for i, a in enumerate(root.child_actions):
                if hasattr(a, 'child_state'):
                    vertex_index = self.inorderTraversal(a.child_state, g, vertex_index, par_index, v_label,
                                                         a_label)
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
