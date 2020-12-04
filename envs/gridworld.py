import numpy as np
import gym
from gym import spaces
from builtins import AttributeError
from math import floor
from envs.FiniteMDP import FiniteMDP


def generate_gridworld(shape=(4, 4), horizon=30, gamma=0.99, randomized_initial=False):
    return GridWorld(shape=shape, horizon=horizon, gamma=gamma, randomized_initial=randomized_initial)


class GridWorld(FiniteMDP):

    ACTION_LABELS = ["UP", "RIGHT", "DOWN", "LEFT"]
    """
    A KxK discrete gridworld environment.
    
    State space: discrete in {0,1,...,K^2-1}
    Action space: discrete in {0,1,2,3}, where 0 is north, 1 is east, 2 is south, and 3 is west
    
    Reward: 1 for reaching the goal, 0 otherwise.
    
    Parameters
    ----------
        - horizon: maximum number of time steps
        - shape: shape of the grid
        - fail_prob: probability of failing an action
        - goal: goal position (x,y)
        - start: start position (x,y)
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, shape=(4, 4), horizon=100, fail_prob=0.1, goal=None, start=None, gamma=0.99,
                 rew_weights=None, randomized_initial=True, extended_features=False):

        assert shape[0] >= 3 and shape[1] >= 3, "The grid must be at least 3x3"
        self.H = 2 * shape[0] + 1 #mirrored grid
        self.W = shape[1]
        assert horizon >= 1, "The horizon must be at least 1"
        self.horizon = horizon
        assert 0 <= fail_prob <= 1, "The probability of failure must be in [0,1]"
        self.fail_prob = fail_prob

        if goal is None:
            goal = (shape[1]-1, shape[0])
        if start is None:
            start = (0, shape[0])
        self.done = False
        self.init_state = self._coupleToInt(start[0], start[1])
        self.randomized_initial = randomized_initial
        self.goal_state = self._coupleToInt(goal[0], goal[1])
        self.PrettyTable = None
        self.rendering = None
        self.gamma = gamma
        if rew_weights is None:
            rew_weights = [1, 10, 0]
        self.rew_weights = np.array(rew_weights)
        # gym attributes
        self.viewer = None
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.W * self.H)
        self.ohe = np.eye(self.W*self.H)
        self.extended_features = extended_features
        mu, p, r = self.calculate_mdp()

        self.mu = mu
        self.p = p
        self.P = p
        self.r = r
        self.n_states = self.W * self.H  # Number of states
        self.n_actions = 4  # Number of actions
        # initialize state
        self.reset()

    def generate_mdp(self):
        return FiniteMDP(self.p, self.r, self.mu, self.gamma, self.horizon)

    def _coupleToInt(self, x, y):
        return y + x * self.H
            
    def _intToCouple(self, n):
        return floor(n / self.H), n % self.H

    def get_rew_features(self, state=None):
        if state is None:
            if self.done:
                return np.zeros(3)
            state = self._state
        x, y = self._intToCouple(state)
        features = np.zeros(3)
        if state == self.goal_state: #goal state
            features[2] = 1
        elif x > 0 and x < self.W - 1 and y > 0 and y < self.H - 1:  # slow_region
            features[1] = -1
        else:
            features[0] = -1  # fast region
        return features

    def step(self, action, ohe=False):
        if self._state == self.goal_state or self._t >= self.horizon:
            return self._state, 0, True, {'features': np.zeros(3)}
        # x, y = self._intToCouple(self._state)
        # action = np.random.choice(4) if np.random.rand() < self.fail_prob else action
        #
        # if action == 0:
        #     y = min(y+1, self.H-1)
        # elif action == 1:
        #     x = min(x+1, self.W-1)
        # elif action == 2:
        #     y = max(y-1, 0)
        # elif action == 3:
        #     x = max(x-1, 0)
        # else:
        #     raise AttributeError("Illegal action")
        #
        # self._state = self._coupleToInt(x, y)
        self._state = np.random.choice(self.P.shape[0], p=self.P[self._state, action, :])
        features = self.get_rew_features()
        reward = np.sum(self.rew_weights * features)
        self._t += 1
        self.done = 1 if self._state == self.goal_state or self._t >= self.horizon else 0
        if self.extended_features:
            features = np.zeros(self.n_states)
            features[self._state] = reward

        return self.get_state(ohe), reward, self.done, {'features': features}

    def get_state(self, ohe=False):
        if ohe:
            return self.ohe[self._state]
        return self._state

    def reset(self, state=None, ohe=False):

        if state is None:
            if self.randomized_initial:
                self._state = np.random.choice(self.observation_space.n,p = self.mu)
            else:
                self._state = self.init_state
        else:
            self._state = state
        self.done = False
        self._state = self._state
        self._t = 0
        return self.get_state(ohe)

    def _render(self, mode='human', close=False):
        if not self.rendering:
            from gym.envs.classic_control import rendering as rend
            self.rendering = rend
        rendering = self.rendering
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(0, self.W, 0, self.H)

        # Draw the grid
        for i in range(self.W):
            self.viewer.draw_line((i, 0), (i, self.H))
        for i in range(self.H):
            self.viewer.draw_line((0, i), (self.W, i))

        goal = self.viewer.draw_circle(radius=0.5)
        goal.set_color(0, 0.8, 0)
        goal_x, goal_y = self._intToCouple(self.goal_state)
        goal.add_attr(rendering.Transform(translation=(goal_x + 0.5, goal_y + 0.5)))

        agent = self.viewer.draw_circle(radius=0.4)
        agent.set_color(.8, 0, 0)
        agent_x, agent_y = self._intToCouple(self._state)
        transform = rendering.Transform(translation=(agent_x + 0.5, agent_y + 0.5))
        agent.add_attr(transform)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def render_pi_v(self, pi, V):
        if self.PrettyTable is None:
            from prettytable import PrettyTable as pt
            self.PrettyTable = pt

        t = self.PrettyTable()
        pi_table = np.zeros((self.H, self.W), dtype=np.int)
        v_table = np.zeros((self.H, self.W))
        for state in range(self.W * self.H):
            x, y = self._intToCouple(state)
            pi_table[self.H - y - 1, x] = int(pi[state])
            v_table[self.H - y - 1, x] = V[state]

        for i in range(self.H):
            row = []
            for j in range(self.W):
                row.append(GridWorld.ACTION_LABELS[pi_table[i, j]] + ":%.2f" % v_table[i, j])
            t.add_row(row)

        print(t)

    def calculate_mdp(self):
        n_states = self.W * self.H  # Number of states
        n_actions = 4  # Number of actions

        # Compute the initial state distribution
        if self.randomized_initial:
            P0 = np.ones(n_states) * 1 / (n_states - 1)
            P0[self.goal_state] = 0

        else:
            P0 = np.zeros(n_states)
            P0[self.init_state] = 1

        # Compute the reward function
        R = np.zeros((n_actions, n_states, n_states))

        # Compute the transition probability matrix
        P = np.zeros((n_actions, n_states, n_states))
        p = self.fail_prob
        delta_x = [0, 1, 0, -1]  # Change in x for each action [UP, RIGHT, DOWN, LEFT]
        delta_y = [1, 0, -1, 0]  # Change in y for each action [UP, RIGHT, DOWN, LEFT]
        for s in range(n_states):
            rew = np.sum(self.get_rew_features(s) * self.rew_weights)
            R[:, :, s] = rew
            for a in range(n_actions):
                x, y = self._intToCouple(s)  # Get the coordinates of s
                x_new = max(min(x + delta_x[a], self.W - 1), 0)  # Correct next-state for a
                y_new = max(min(y + delta_y[a], self.H - 1), 0)  # Correct next-state for a
                s_new = self._coupleToInt(x_new, y_new)

                P[a, s, s_new] += 1 - p  # a does not fail with prob. 1-p
                # Suppose now a fails and try all other actions
                for a_fail in range(n_actions):
                    x_new = max(min(x + delta_x[a_fail], self.W - 1), 0)  # Correct next-state for a_fail
                    y_new = max(min(y + delta_y[a_fail], self.H - 1), 0)  # Correct next-state for a_fail
                    P[a, s, self._coupleToInt(x_new, y_new)] += p / 4  # a_fail is taken with prob. p/4
        # The goal state is terminal -> only self-loop transitions

        P[:, self.goal_state, :] = 0
        P[:, self.goal_state, self.goal_state] = 1
        for s in range(n_states):
            for a in range(n_actions):
                if np.sum(P[a, s, :]) != 1:
                    print("What")
        R[:, self.goal_state, self.goal_state] = 0  # don't get reward after reaching goal state
        P = np.transpose(P, [1, 0, 2])
        R = np.transpose(R, [1, 0, 2])
        return P0, P, R

    def get_mdp(self):

        """Returns an MDP representing this gridworld"""
        n_states = self.W * self.H  # Number of states
        n_actions = 4  # Number of actions
        P0 = self.mu
        R = self.r
        P = self.p

        return MDP(n_states, n_actions, P, R, P0, self.gamma)
