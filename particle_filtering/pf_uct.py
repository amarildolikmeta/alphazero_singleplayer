import copy
from helpers import stable_normalizer, copy_atari_state, restore_atari_state, argmax, max_Q
from rl.make_game import is_atari_game
import numpy as np
from igraph import Graph
import plotly.graph_objects as go
import random
import json


def sample_particle(root_state, actions, env, budget):
    env.set_signature(root_state)
    env.seed()
    parent_particle = None
    for i, action in enumerate(actions):
        s, r, done, _ = env.step(action.index)
        budget -= 1
        new_particle = Particle(env.get_signature(), None, r, done, info=s, parent_particle=parent_particle)
        if hasattr(action, 'child_state'):
            action.child_state.add_particle(new_particle)
            #if we finish an episode in the middle of the tree
            if done:
                i += 1
                parent_particle = new_particle
                while i < len(actions):
                    action = actions[i]
                    new_particle = Particle(env.get_signature(), None, 0, done, info=s, parent_particle=parent_particle)
                    if hasattr(action, 'child_state'):
                        action.child_state.add_particle(new_particle)
                    i += 1
                    parent_particle = new_particle
                break
            else:
                parent_particle = new_particle
    return new_particle, budget


def sample_from_parent_state(state, action, env, budget):
    parent_particle = np.random.choice(state.particles)
    new_particle = generate_new_particle(env, action.index, parent_particle)
    budget -= 1
    return new_particle, budget


def sample_from_particle(source_particle, action, env, budget):
    env.set_signature(source_particle.state)
    s, r, done, _ = env.step(action.index)
    budget -= 1
    if source_particle is None:
        print("What")
    new_particle = Particle(env.get_signature(), None, r, done, info=s, parent_particle=source_particle)
    return new_particle, budget


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


def parallel_step(particle, env, action):
    """Perform a step on an environment, executing the given action"""
    if not particle.terminal:
        env.set_signature(particle.state)
        env.seed(particle.seed)
        env.step(action)
    return env


def generate_new_particle(env, action, particle):
    """Generate the successor particle for a given particle"""
    # Do not give any reward if a particle is being generated from a terminal state particle
    if particle.terminal:
        return Particle(particle.state, None, 0, True, parent_particle=particle)

    # Apply the selected action to the state encapsulated by the particle and store the new state and reward
    #env = copy.deepcopy(env)
    env.set_signature(particle.state)
    env.seed(particle.seed)
    s, r, done, _ = env.step(action)
    return Particle(env.get_signature(), None, r, done, info=s, parent_particle=particle)


class Particle(object):
    """Class storing information about a particle"""

    def __init__(self, state, seed, reward, terminal, info=None, parent_particle=None):
        self.state = state
        self.seed = seed
        self.reward = reward
        self.terminal = terminal
        self.info = info
        self.parent_particle = parent_particle

    def __str__(self):
        return str(self.info)


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

    def add_child_state(self, state, env, budget,  max_depth=200, source_particle=None,
                        depth=0):
        if source_particle is not None:
            new_particle, budget = sample_from_particle(source_particle, self, env, budget)
        else:
            # parent_particle = np.random.choice(self.parent_state.particles)
            # new_particle = generate_new_particle(env, self.index, parent_particle)
            new_particle, budget = sample_from_parent_state(state, self, env, budget)

        self.child_state = State(parent_action=self,
                                 na=self.parent_state.na,
                                 env=env,
                                 particles=[new_particle],
                                 root=False,
                                 max_depth=max_depth,
                                 budget=budget,
                                 depth=depth)

        return self.child_state, self.child_state.remaining_budget, new_particle

    def update(self, R):
        self.n += 1
        self.W += R
        self.Q = self.W / self.n


class State(object):
    """ State object """

    def __init__(self, parent_action, na, env, particles, budget, root=False, max_depth=200, depth=0):

        """ Initialize a new state """
        self.r = np.mean([particle.reward for particle in particles])
        self.mean_r = self.r
        self.sum_r = np.sum([particle.reward for particle in particles])
        self.depth = depth
        self.terminal = True  # whether the domain terminated in this state

        # A state is terminal only if all of its particles are terminal
        # for p in particles:
        #     self.terminal = self.terminal and p.terminal
        #     if not self.terminal:
        #         break
        self.terminal = depth == max_depth
        self.parent_action = parent_action
        self.particles = particles

        # Child actions
        self.na = na

        self.remaining_budget = budget

        if self.terminal or root:
            self.V = 0
            #self.remaining_budget -= len(self.particles)
        elif env is None:
            print("Warning, no environment was provided, initializing to 0 the value of the state!")
            self.V = 0
        else:
            self.V, self.remaining_budget = self.evaluate(particles[0], env, budget, max_depth)
        self.n = 0

        self.child_actions = [Action(a, parent_state=self) for a in range(na)]
        self.n_particles = len(particles)

    def to_json(self):
        inf = {
            "V": str(self.V) + '<br>',
            "n": str(self.n) + '<br>',
            "d": str(self.depth) + '<br>'
        }
        return json.dumps(inf)

    def get_n_particles(self):
        return self.n_particles

    def add_particle(self, particle):
        self.particles.append(particle)
        self.n_particles += 1
        self.sum_r += particle.reward
        self.mean_r = self.sum_r / self.n_particles
        self.r = particle.reward # to be used in the backup

    def select(self, c=1.5, csi=1., b=1., variance=False):
        """
         Select one of the child actions based on UCT rule
         :param c: UCB exploration constant
         :param csi: exploration constant
         :param b: parameter such that the rewards belong to [0, b]
         """
        if not variance:

            uct_upper_bound = np.array(
                [child_action.Q + c * np.sqrt(np.log(self.n) / (child_action.n)) if child_action.n > 0 else np.inf
                 for child_action in self.child_actions])
            winner = argmax(uct_upper_bound)
            return self.child_actions[winner]

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

    def update(self):
        """ update count on backward pass """
        self.n += 1

    def evaluate(self, particle, env, budget, max_depth=200):
        actions = np.arange(self.na, dtype=int)

        results = []
        if budget > 0:
            env.set_signature(particle.state)
            particle_return, budget = random_rollout(actions, env, budget, max_depth, particle.terminal)
            results.append(particle_return)
        else:
            results.append(0)
        return np.mean(results), budget


class PFMCTS(object):
    ''' MCTS object '''

    def __init__(self, root, root_index, na, gamma, alpha=0.6, model=None,  variance=False,
                 depth_based_bias=False, beta=1):
        self.root = root
        self.root_index = root_index
        self.na = na
        self.gamma = gamma
        self.depth_based_bias = depth_based_bias
        assert 0 < alpha <= 1, "Alpha must be between 0 and 1"
        self.alpha = alpha
        self.beta = beta
        self.variance = variance

    def search(self, n_mcts, c, Env, mcts_env, budget, max_depth=200, fixed_depth=True):
        """ Perform the MCTS search from the root """
        Envs = [copy.deepcopy(Env)]

        if self.root is None:
            # initialize new root with many equal particles
            signature = Env.get_signature()
            self.root_signature = signature
            box = None
            to_box = getattr(self, "index_to_box", None)
            if callable(to_box):
                box = Env.index_to_box(signature["state"])

            particles = [Particle(state=signature, seed=None, reward=0, terminal=False, info=box,
                                  parent_particle=None)]
            self.root = State(parent_action=None, na=self.na, env=Envs[0], particles=particles, root=True,
                              budget=budget, depth=0)
        else:
            self.root.parent_action = None  # continue from current root
        root_signature = Env.get_signature()
        if self.root.terminal:
            raise (ValueError("Can't do tree search from a terminal state"))

        is_atari = is_atari_game(Env)
        if is_atari:
            raise NotImplementedError
            snapshot = copy_atari_state(Env)  # for Atari: snapshot the root at the beginning
        while budget > 0:
            state = self.root  # reset to root for new trace
            if not is_atari:
                mcts_envs = [copy.deepcopy(Env)]  # copy original Env to rollout from
            else:
                raise NotImplementedError
                restore_atari_state(mcts_env, snapshot)
            st = 0
            actions = []
            flag = False
            source_particle = None
            while not state.terminal:
                bias = c * self.gamma ** st / (1 - self.gamma) if self.depth_based_bias else c
                action = state.select(c=bias, variance=self.variance)
                st += 1
                actions.append(action)
                k = np.ceil(self.beta * action.n ** self.alpha)

                # s1, r, t, _ = mcts_env.step(action.index)
                if hasattr(action, 'child_state'):
                    state = action.child_state  # select
                    add_particle = k >= state.get_n_particles()
                    if add_particle and not flag:
                        flag = True
                        source_particle, budget = sample_from_parent_state(action.parent_state, action, mcts_envs[0],
                                                                           budget)
                        state.add_particle(source_particle)
                    elif flag:
                        source_particle, budget = sample_from_particle(source_particle, action, mcts_envs[0], budget)
                        state.add_particle(source_particle)
                    elif state.terminal:
                        source_particle = np.random.choice(state.particles)  # sample from the terminal states particles
                    continue
                else:
                    rollout_depth = max_depth if fixed_depth else max_depth - st
                    state, budget, source_particle = action.add_child_state(state, mcts_envs[0], budget,
                                                                            max_depth=rollout_depth,
                                                                            source_particle=source_particle,
                                                                            depth=st)  # expand
                    break

            # Back-up
            R = state.V
            state.update()
            particle = source_particle
            while state.parent_action is not None:  # loop back-up until root is reached
                # r = state.r if add_particle else np.random.choice([p.reward for p in state.particles])
                try:
                    r = particle.reward
                except:
                    print("What")
                if not state.terminal:
                    R = r + self.gamma * R
                else:
                    R = r
                action = state.parent_action
                action.update(R)
                state = action.parent_state
                state.update()
                particle = particle.parent_particle

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
