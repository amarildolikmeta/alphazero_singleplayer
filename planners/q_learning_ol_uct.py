import copy

from envs.planning_env import PlanningEnv
from helpers import stable_normalizer, argmax, max_Q
from rl.make_game import is_atari_game
import numpy as np
from igraph import Graph
import plotly.graph_objects as go
import json


def sample(env, action, budget):
    env.seed(np.random.randint(1e7))
    _, r, done, _ = env.step(action)
    budget -= 1
    return r, done, budget


def random_rollout(actions, env, budget, max_depth=200, terminal=False):
    """Rollout from the current state following a random policy up to hitting a terminal state"""
    # if terminal or budget <= 0:
    #     return 0, -1

    done = False
    env.seed(np.random.randint(1e7))
    ret = 0
    t = 0
    while t < max_depth and not done:
        action = np.random.choice(actions)
        s, r, done, _ = env.step(action)
        ret += r
        budget -= 1
        t += 1
    return ret, budget

# TODO remove, only for debugging raceStrategy

MAX_P = [1]
PROB_1 = [0.95, 0.05]
PROB_2 = [0.95, 0.025, 0.025]
PROB_3 = [0.91, 0.03, 0.03, 0.03]
PROBS = {1: MAX_P, 2: PROB_1, 3: PROB_2, 4:PROB_3}

DEBUG = False


def strategic_rollout(env, budget, max_depth=200, terminal=False, root_owner=None,
                      no_pit=False,
                      brain_on=True,
                      double_rollout=False):
    """Rollout from the current state following a default policy up to hitting a terminal state"""
    if double_rollout:
        original_env = copy.deepcopy(env)

    ret = np.zeros(env.agents_number)

    for i in range(2 if double_rollout else 1):
        done = False
        # if terminal or budget <= 0:
        #     return ret, -1
        if i == 1:
            env = copy.deepcopy(original_env)

        env.seed(np.random.randint(1e7))
        t = 0

        agent = root_owner

        while t / env.agents_number < max_depth and not done:
            if brain_on:
                actions = env.get_default_strategy(root_owner)
            else:
                actions = env.get_available_actions(agent)

            if no_pit:
                action = 0
            else:
                prob = PROBS[len(actions)]
                action = np.random.choice(actions, p=prob)

            s, r, done, _ = env.partial_step(action, agent)

            ret += r
            t += 1

            # Get the agent ranking to specify the turn order
            if env.has_transitioned():
                budget -= 1

            agent = env.get_next_agent()

    if double_rollout:
        ret/=2
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
        self.child_state = None
        self.max_r = - np.inf
        self.min_r = np.inf

    def add_child_state(self, env, budget, max_depth=200, depth=0, deepen=False, q_learning=False):
        reward, terminal, budget = sample(env, self.index, budget)
        self.child_state = State(parent_action=self,
                                 na=self.parent_state.na,
                                 env=env,
                                 root=False,
                                 max_depth=max_depth,
                                 budget=budget,
                                 reward=reward,
                                 terminal=terminal,
                                 depth=depth,
                                 deepen=deepen,
                                 q_learning=q_learning)

        return self.child_state, self.child_state.remaining_budget

    def update(self, R, q_learning=False, mixed_q_learning=False, alpha=0.1, gamma=1., a=0.5, b=2):
        """
        :param R: is the reward collected by the agent
        :type R: float

        :param q_learning: enables the Q-learning update in place of the Monte-Carlo one
        :type q_learning: bool

        :param alpha: is the Q-learning algorithm learning rate
        :type alpha: float

        :param gamma: is the MDP discount factor
        :type gamma: float

        :param a: exponentiation term for learning rate schedule
        :type a: float

        :param b: constant term for learning rate schedule
        :type b: float
        """

        self.max_r = max(R, self.max_r)
        self.min_r = min(R, self.min_r)

        if mixed_q_learning:
            alpha = 1/(b + self.n ** a)
            if self.n == 0:
                self.Q = self.child_state.V + R
            else:
                no_children = True
                for a in self.child_state.child_actions:
                    no_children = no_children and (a.child_state is None)
                if no_children:
                    opt_future = self.child_state.V
                else:
                    # TODO this eliminates zeroes, probably not always correct
                    opt_future = np.max([a.Q if a.Q < 0 else -np.inf for a in self.child_state.child_actions])
                self.Q = self.Q + alpha * (R + gamma * opt_future - self.Q)
            self.n += 1

        elif q_learning:
            self.Q = self.Q + alpha * (R + gamma *  np.max([a.Q for a in self.child_state.child_actions]) - self.Q)
            self.n += 1

        else:
            self.update_aggregate(R)
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


class State(object):
    """ State object """

    def __init__(self, parent_action, na, env, budget, root=False, max_depth=200, reward=0, terminal=False, depth=0,
                 deepen=False, q_learning=False):

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
            self.child_actions = [Action(a, parent_state=self) for a in action_list]
        else:
            self.child_actions = [Action(a, parent_state=self) for a in range(na)]

        if env is None:
            print("[WARNING] No environment was provided, initializing to 0 the value of the state!")
            self.V = 0
        elif self.terminal or root or (deepen and len(self.child_actions) == 1) or q_learning:
            self.V = 0.
        else:
            self.V, self.remaining_budget = self.evaluate(env, budget, max_depth, terminal)

    def is_terminal(self, max_depth, env):
        return self.depth == max_depth or env.is_terminal()

    def to_json(self):
        inf = {
            "V": str(self.V) + '<br>',
            "Q": (str(self.parent_action.Q) if self.parent_action is not None else "0") + '<br>',
            "max": (str(self.parent_action.max_r) if self.parent_action is not None else "-inf") + '<br>',
            "min": (str(self.parent_action.min_r) if self.parent_action is not None else "inf") + '<br>',
            "sigma": (str(self.parent_action.sigma) if self.parent_action is not None else "inf") + '<br>',
            "n": str(self.n) + '<br>',
            "d": str(self.depth) + '<br>'
        }
        return json.dumps(inf)

    def sample(self, env, action, budget):
        r, done, budget = sample(env, action, budget)
        self.reward = r
        return done, budget

    def select(self, c=1.5, csi=1., b=1., variance=False, bias_zero=True):
        """
         Select one of the child actions based on UCT rule
         :param c: UCB exploration constant
         :param csi: exploration constant
         :param b: parameter such that the rewards belong to [0, b]
         :param variance: controls if the UCT-V selection should be applied
         :param bias_zero: gives bias to the first action if ties need to be broken
         """

        if not variance:
            uct_upper_bound = np.array(
                [child_action.Q + c * np.sqrt(np.log(self.n) / (child_action.n)) if child_action.n > 0 else np.inf
                 for child_action in self.child_actions])

            winner = argmax(uct_upper_bound, bias_zero=bias_zero)
            return self.child_actions[winner]

        if self.n > 0:
            logp = np.log(self.n)
        else:
            logp = -np.inf

        bound = np.array([child_action.Q +
                          np.sqrt(csi * child_action.sigma * logp / child_action.n) +
                          3 * c * b * csi * logp / child_action.n
                          if child_action.n > 0 and not np.isinf(child_action.sigma).any() else np.inf
                          for child_action in self.child_actions])
        if DEBUG and False:
            var = np.array([child_action.sigma for child_action in self.child_actions])

            qs = np.array([child_action.Q for child_action in self.child_actions])

            sq = np.array([np.sqrt(csi * child_action.sigma * logp / child_action.n)
                              if child_action.n > 0 and not np.isinf(child_action.sigma).any() else np.inf
                              for child_action in self.child_actions])

            extra = np.array([np.sqrt(csi * child_action.sigma * logp / child_action.n) +
                              3 * c * b * csi * logp / child_action.n
                              if child_action.n > 0 and not np.isinf(child_action.sigma).any() else np.inf
                              for child_action in self.child_actions])

        winner = argmax(bound, bias_zero=bias_zero)
        return self.child_actions[winner]

    def update(self):
        """ update count on backward pass """
        self.n += 1

    def evaluate(self, env, budget, max_depth=200, terminal=False):
        actions = np.arange(self.na, dtype=int)
        return_, budget = self.rollout(actions, env, budget, max_depth, terminal)
        return return_, budget

    # TODO remove, only for debugging raceStrategy
    @staticmethod
    def rollout(actions, env, budget, max_depth=200, terminal=False, brain_on=True, double_rollout=False, no_pit=False):

        if hasattr(env, "get_available_actions") and hasattr(env, "get_next_agent"):
            owner = env.get_next_agent()
            ret, budget = strategic_rollout(env, budget, max_depth, terminal, owner,
                                            double_rollout=double_rollout, brain_on=brain_on, no_pit=no_pit)
            return ret[0], budget
        else:
            return random_rollout(actions, env, budget, max_depth, terminal)


class QL_OL_MCTS(object):
    """ MCTS object """

    def __init__(self, root, root_index, na, gamma, model=None, variance=False, depth_based_bias=False, csi=1.,
                 q_learning = False, alpha=0.1, mixed_q_learning=True):
        self.root = root
        self.root_index = root_index
        self.na = na
        self.gamma = gamma
        self.depth_based_bias = depth_based_bias
        self.variance = variance
        self.csi = csi
        self.c = 1
        self.q_learning = q_learning or mixed_q_learning
        self.alpha = alpha
        self.mixed_q_learning = mixed_q_learning

    def create_root(self, env, budget):
        if self.root is None:
            signature = env.get_signature()
            self.root_signature = signature
            self.root = State(parent_action=None, na=self.na, env=env, root=True, budget=budget, q_learning=self.q_learning)
        else:
            raise (NotImplementedError("Need to reset the tree"))

    def search(self, n_mcts, c, Env: PlanningEnv, mcts_env, budget, max_depth=200, fixed_depth=True, deepen=False, visualize=False):
        """ Perform the MCTS search from the root """

        deepen = deepen and (not self.q_learning or self.mixed_q_learning)

        self.c = c

        env = copy.deepcopy(Env)

        env.enable_search_mode()

        self.create_root(env, budget)
        if self.root.terminal:
            raise (ValueError("Can't do tree search from a terminal state"))

        is_atari = is_atari_game(env)
        if is_atari:
            raise NotImplementedError
        while budget > 0:
            state = self.root  # reset to root for new trace
            if not is_atari:
                mcts_env = copy.deepcopy(Env)  # copy original Env to rollout from
            else:
                raise NotImplementedError
            mcts_env.seed(np.random.randint(1e7))
            # mcts_env.reset_stochasticity()  # Regenerate precomputed random events, if any
            st = 0
            terminal = False
            while not state.terminal:
                # Select the node
                bias = c * self.gamma ** st / (1 - self.gamma) if self.depth_based_bias else c
                action = state.select(c=bias, variance=self.variance, csi=self.csi)
                st += 1
                if action.child_state is not None:
                    # Resample from the existing tree if the node already exists
                    state = action.child_state
                    terminal, budget = state.sample(mcts_env, action.index, budget)
                    if terminal:
                        break
                else:
                    # Expand state
                    enable = self.q_learning and not self.mixed_q_learning
                    rollout_depth = max_depth if fixed_depth else max_depth - st
                    state, budget = action.add_child_state(mcts_env, budget, rollout_depth, depth=st,
                                                           deepen=deepen, q_learning=enable)
                    # If the state has only one possible action, immediately add its successor to the tree
                    while deepen and len(state.child_actions) == 1 and not state.terminal:
                        action = state.child_actions[0]
                        rollout_depth = max_depth if fixed_depth else max_depth - st
                        if budget < rollout_depth:
                            budget = rollout_depth
                        state, budget = action.add_child_state(mcts_env, budget, rollout_depth, depth=st,
                                                               deepen=deepen, q_learning=enable)
                        st += 1

                    break

            self.backup(state, terminal)

        if visualize:
            self.visualize()

    def backup(self, state, terminal):
        # Back-up
        # if budget < 0: #finished budget before rollout
        #     break
        R = state.V
        state.update()
        while state.parent_action is not None:  # loop back-up until root is reached
            # if state.reward > -85:
            #     print("WTF:", state.reward)
            if not terminal and not (self.q_learning or self.mixed_q_learning):
                R = state.reward + self.gamma * R
            else:
                R = state.reward
                terminal = False
            action = state.parent_action
            action.update(R, q_learning=self.q_learning, mixed_q_learning=self.mixed_q_learning,
                          alpha=self.alpha, gamma=self.gamma)
            state = action.parent_state
            state.update()

    def return_results(self, temp, on_visits=False, on_lower=False, on_combined=False):
        """ Process the output at the root node """
        counts = np.array([child_action.n for child_action in self.root.child_actions])
        Q = np.array([child_action.Q for child_action in self.root.child_actions])
        if DEBUG:
            print(Q)
            print(counts)
        if on_lower:
            uct_lower_bound = np.array(
                [child_action.Q - self.c * np.sqrt(np.log(self.root.n) / child_action.n) if child_action.n > 0 else np.inf
                 for child_action in self.root.child_actions])
            pi_target = max_Q(uct_lower_bound) # max_Q doesn't really take the maximum Q in this case
        elif on_visits:
            pi_target = stable_normalizer(counts, temp)
        elif on_combined:
            pi_target = max_Q([q * n for q, n in zip(Q, counts)])
        else:
            pi_target = max_Q(Q)
            if np.argmax(pi_target) > 0 and DEBUG:
                print("PIT")
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
        self.count += 1

    def print_tree(self, root):
        self.print_index()
        for i, a in enumerate(root.child_actions):
            if a.child_state is not None:
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
