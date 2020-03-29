import copy

from helpers import stable_normalizer, copy_atari_state, restore_atari_state
from rl.make_game import make_game, is_atari_game
import numpy as np
import multiprocessing
import json


def random_rollout(actions, env):
    """Rollout from the current state following a random policy up to hitting a terminal state"""
    done = False
    while not done:
        action = np.random.choice(actions)
        s, r, done, _ = env.step(action)
        if done:
            return r


def parallel_step(env, action):
    """Perform a step on an environment, executing the given action"""
    env.step(action)
    return env


def generate_new_particle(env, action, particle):
    """Generate the successor particle for a given particle"""
    # Do not give any reward if a particle is being generated from a terminal state particle
    if particle.terminal:
        return Particle(particle.state, particle.seed, 0, True)

    # Apply the selected action to the state encapsulated by the particle and store the new state and reward
    env = copy.deepcopy(env)
    env.reset(particle.state)
    env.seed = particle.seed
    s, r, done, _ = env.step(action)

    return Particle(s, particle.seed, r, done)


class Particle(object):
    """Class storing information about a particle"""
    def __init__(self, state, seed, reward, terminal):
        self.state = state
        self.seed = seed
        self.reward = reward
        self.terminal = terminal


class Action(object):
    """ Action object """

    def __init__(self, index, parent_state, Q_init=0.0):
        self.index = index
        self.parent_state = parent_state
        self.W = 0.0
        self.n = 0
        self.Q = Q_init
        self.child_state = None

    def add_child_state(self, envs):
        p = multiprocessing.Pool(multiprocessing.cpu_count())

        new_particles = p.starmap(generate_new_particle, [(envs[i], self.index) for i in range(len(envs))])

        p.close()

        self.child_state = State(parent_action=self,
                                 na=self.parent_state.na,
                                 envs=envs,
                                 particles=new_particles)
        return self.child_state

    def update(self, R):
        self.n += 1
        self.W += R
        self.Q = self.W / self.n


class State(object):
    """ State object """

    def __init__(self, parent_action, na, envs, particles):
        """ Initialize a new state """
        self.r = np.mean([particle.reward for particle in particles])  # The reward is the mean of the particles' reward
        self.terminal = True  # whether the domain terminated in this state

        # A state is terminal only if all of its particles are terminal
        for p in particles:
            self.terminal = self.terminal and p.terminal

        self.parent_action = parent_action
        self.particles = particles

        # Child actions
        self.na = na

        if envs is None:
            print("Warning, no environment was provided, initializing to 0 the value of the state!")
        if self.terminal or envs is None:
            self.V = 0
        else:
            self.V = self.evaluate(copy.deepcopy(envs))
        self.n = 0

        self.child_actions = [Action(a, parent_state=self, Q_init=self.V) for a in range(na)]

    def to_json(self):
        raise NotImplementedError

    def select(self, c=1.5):
        """ Select one of the child actions based on UCT rule """
        # TODO check here
        UCT = np.array(
            [child_action.Q + c * (np.sqrt(self.n + 1) / (child_action.n + 1)) for child_action in self.child_actions])
        winner = np.argmax(UCT)
        return self.child_actions[winner]

    def update(self):
        """ update count on backward pass """
        self.n += 1

    def evaluate(self, envs):
        actions = np.arange(self.na, dtype=int)

        p = multiprocessing.Pool(multiprocessing.cpu_count())

        results = p.starmap(random_rollout, [(actions, envs[i]) for i in range(len(envs))])
        p.close()

        return np.mean(results)


class PFMCTS(object):
    ''' MCTS object '''

    def __init__(self, root, root_index, na, gamma, model=None, n_particles=100):
        self.root = root
        self.root_index = root_index
        self.na = na
        self.gamma = gamma
        self.n_particles = n_particles

    def search(self, n_mcts, c, Envs, mcts_envs):
        """ Perform the MCTS search from the root """
        if self.root is None:
            # initialize new root with many equal particles
            particles = [Particle(state=self.root_index, seed=np.random.randint(1e7), reward=0, terminal=False)
                         for _ in range(self.n_particles)]
            self.root = State(parent_action=None, na=self.na, envs=None, particles=particles)
        else:
            self.root.parent_action = None  # continue from current root

        if self.root.terminal:
            raise (ValueError("Can't do tree search from a terminal state"))

        is_atari = is_atari_game(Envs[0])
        if is_atari:
            raise NotImplementedError
            snapshot = copy_atari_state(Env)  # for Atari: snapshot the root at the beginning

        for i in range(n_mcts):
            state = self.root  # reset to root for new trace
            if not is_atari:
                mcts_envs = copy.deepcopy(Envs)  # copy original Env to rollout from
            else:
                raise NotImplementedError
                restore_atari_state(mcts_envs, snapshot)

            while not state.terminal:
                action = state.select(c=c)

                # Do one step of the environment in parallel
                p = multiprocessing.Pool(multiprocessing.cpu_count())
                mcts_envs = p.starmap(parallel_step, [(envs[i], action.index) for i in range(len(mcts_envs))])
                p.close()

                s1, r, t, _ = mcts_env.step(action.index)
                if hasattr(action, 'child_state'):
                    state = action.child_state  # select
                    continue
                else:
                    state = action.add_child_state(mcts_envs)  # expand
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
        self.root = None
        self.root_index = s1

    def visualize(self):
        raise NotImplementedError

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
                    vertex_index = self.inorderTraversal(a.child_state, g, vertex_index, par_index, v_label, a_label)
        return vertex_index

    def print_index(self):
        raise NotImplementedError

    def print_tree(self, root):
        raise NotImplementedError


def make_annotations(pos, labels, Xe, Ye, a_labels, M, position, font_size=10, font_color='rgb(250,250,250)'):
    raise NotImplementedError
