import copy

from helpers import stable_normalizer, copy_atari_state, restore_atari_state, argmax, max_Q
from rl.make_game import is_atari_game
import numpy as np
import multiprocessing

from igraph import Graph
import plotly.graph_objects as go

import random
import json

MULTITHREADED = False


def random_rollout(particle, actions, env, budget, max_depth=200):
    """Rollout from the current state following a random policy up to hitting a terminal state"""
    done = False
    if particle.terminal:
        return particle.reward

    env.set_signature(particle.state)
    env.seed(particle.seed)
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
        return Particle(particle.state, particle.seed, 0, True)

    # Apply the selected action to the state encapsulated by the particle and store the new state and reward
    env = copy.deepcopy(env)
    env.set_signature(particle.state)
    env.seed(particle.seed)
    s, r, done, _ = env.step(action)

    return Particle(env.get_signature(), particle.seed, r, done, info=s)


class Particle(object):
    """Class storing information about a particle"""

    def __init__(self, state, seed, reward, terminal, info=None):
        self.state = state
        self.seed = seed
        self.reward = reward
        self.terminal = terminal
        self.info = info

    def __str__(self):
        return str(self.info)


class Action(object):
    """ Action object """

    def __init__(self, index, parent_state, Q_init=0.0):
        self.index = index
        self.parent_state = parent_state
        self.W = 0.0
        self.n = 0
        self.Q = Q_init
        # self.child_state = None

    def add_child_state(self, state, envs, budget, sampler=None, max_depth=200):
        if sampler:

            new_states = sampler.generate_next_particles(self.parent_state.particles, self.index)
            new_particles = []
            for p in new_states:
                new_particles.append(Particle(p[0], p[1], p[2], p[3]))
        elif not MULTITHREADED:
            new_particles = []

            for i in range(len(envs)):
                new_particles.append(generate_new_particle(envs[i], self.index, self.parent_state.particles[i]))

        else:
            p = multiprocessing.Pool(multiprocessing.cpu_count())

            new_particles = p.starmap(generate_new_particle,
                                      [(envs[i], self.index, self.parent_state.particles[i]) for i in range(len(envs))])

            p.close()

        self.child_state = State(parent_action=self,
                                 na=self.parent_state.na,
                                 envs=envs,
                                 particles=new_particles,
                                 sampler=sampler,
                                 root=False,
                                 max_depth=max_depth,
                                 budget=budget)

        return self.child_state, self.child_state.remaining_budget

    def update(self, R):
        self.n += 1
        self.W += R
        self.Q = self.W / self.n


class State(object):
    """ State object """

    def __init__(self, parent_action, na, envs, particles, budget, sampler=None, root=False, max_depth=200):

        """ Initialize a new state """
        self.r = np.mean([particle.reward for particle in particles])  # The reward is the mean of the particles' reward
        self.terminal = True  # whether the domain terminated in this state

        # A state is terminal only if all of its particles are terminal
        for p in particles:
            self.terminal = self.terminal and p.terminal
            if not self.terminal:
                break

        self.parent_action = parent_action
        self.particles = particles

        # Child actions
        self.na = na

        self.remaining_budget = budget

        if self.terminal or root:
            self.V = 0
            self.remaining_budget -= len(self.particles)
        elif sampler is not None:
            self.V = np.mean(sampler.evaluate(particles, max_depth))
        elif envs is None:
            print("Warning, no environment was provided, initializing to 0 the value of the state!")
            self.V = 0
        else:
            self.V, self.remaining_budget = self.evaluate(particles, copy.deepcopy(envs), budget, max_depth)
        self.n = 0

        self.child_actions = [Action(a, parent_state=self, Q_init=self.V) for a in range(na)]


    def to_json(self):
        inf = {"particles": '<br>' + str([str(p) + '<br>' for p in self.particles]),
               "V": str(self.V) + '<br>',
               "n": str(self.n) + '<br>',
               "terminal": str(self.terminal) + '<br>',
               "r": str(self.r) + '<br>'}
        return json.dumps(inf)

    def select(self, c=1.5):
        """ Select one of the child actions based on UCT rule """
        # TODO check here
        uct_upper_bound = np.array(
            [child_action.Q + c * (np.sqrt(self.n + 1) / (child_action.n + 1)) for child_action in self.child_actions])
        winner = argmax(uct_upper_bound)
        return self.child_actions[winner]

    def update(self):
        """ update count on backward pass """
        self.n += 1

    def evaluate(self, particles, envs, budget, max_depth=200):
        actions = np.arange(self.na, dtype=int)

        if not MULTITHREADED:
            results = []
            for i in range(len(particles)):
                if budget == 0:
                    break
                particle_return, budget = random_rollout(particles[i], actions, envs[i], budget, max_depth)
                results.append(particle_return)
        else:
            raise NotImplementedError("Budget handling with parallel rollout has not been implemented yet")
            p = multiprocessing.Pool(multiprocessing.cpu_count())

            r = p.starmap(random_rollout, [(particles[i], actions, envs[i], budget) for i in range(len(envs))])

            p.close()

        return np.mean(results), budget


class PFMCTS(object):
    """ MCTS object """

    def __init__(self, root, root_index, na, gamma, model=None, particles=100, sampler=None):
        self.root = root
        self.root_index = root_index
        self.na = na
        self.gamma = gamma
        self.n_particles = particles
        self.sampler = sampler

    def search(self, n_mcts, c, Env, mcts_env, budget, max_depth=200, fixed_depth=True):
        """ Perform the MCTS search from the root """
        Envs = None
        if not self.sampler:
            Envs = [copy.deepcopy(Env) for _ in range(self.n_particles)]

        if self.root is None:
            # initialize new root with many equal particles

            signature = Env.get_signature()

            box = None
            to_box = getattr(self, "index_to_box", None)
            if callable(to_box):
                box = Env.index_to_box(signature["state"])

            particles = [Particle(state=signature, seed=random.randint(0, 1e7), reward=0, terminal=False, info=box)
                         for _ in range(self.n_particles)]
            self.root = State(parent_action=None, na=self.na, envs=Envs, particles=particles, sampler=self.sampler,
                              root=True, budget=budget)
        else:
            self.root.parent_action = None  # continue from current root
            particles = self.root.particles

        if self.root.terminal:
            raise (ValueError("Can't do tree search from a terminal state"))

        is_atari = is_atari_game(Env)
        if is_atari:
            raise NotImplementedError
            snapshot = copy_atari_state(Env)  # for Atari: snapshot the root at the beginning

        while budget > 0:
            state = self.root  # reset to root for new trace
            if not is_atari:
                mcts_envs = None
                if not self.sampler:
                    mcts_envs = [copy.deepcopy(Env) for i in
                                 range(self.n_particles)]  # copy original Env to rollout from
            else:
                raise NotImplementedError
                restore_atari_state(mcts_env, snapshot)
            st = 0
            while not state.terminal:
                action = state.select(c=c)
                st += 1
                # s1, r, t, _ = mcts_env.step(action.index)
                if hasattr(action, 'child_state'):
                    state = action.child_state  # select
                    if state.terminal:
                        budget -= len(state.particles)
                    continue
                else:
                    rollout_depth = max_depth if fixed_depth else max_depth - st
                    state, budget = action.add_child_state(state, mcts_envs, budget, self.sampler,
                                                   rollout_depth)  # expand
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

    def return_results(self, temp, on_visits=False):
        """ Process the output at the root node """
        counts = np.array([child_action.n for child_action in self.root.child_actions])
        Q = np.array([child_action.Q for child_action in self.root.child_actions])
        if on_visits:
            pi_target = stable_normalizer(counts, temp)
        else:
            pi_target = max_Q(Q)
        V_target = np.sum((counts / np.sum(counts)) * Q)[None]
        return self.root.particles, pi_target, V_target

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
