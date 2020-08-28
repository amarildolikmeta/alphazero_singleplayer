import numpy as np
import gym
from gym import spaces
from feature.rbf import build_features_gw_state
from gym.utils import seeding
from copy import copy


def generate_gridworld(**game_params):
    if game_params is None:
        game_params = {}
    return GridWorld(**game_params)


class GridWorld(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, shape=[4., 4.], horizon=30, fail_prob=0.1, goal=None, start=None, gamma=0.99,
                 rew_weights=None, randomized_initial=True, n_bases=[4, 4]):

        assert shape[0] >= 3 and shape[1] >= 3, "The grid must be at least 3x3"
        assert horizon >= 1, "The horizon must be at least 1"
        assert 0 <= fail_prob <= 1, "The probability of failure must be in [0,1]"
        self.horizon = horizon
        self.noise = fail_prob
        self.state_dim = 2
        self.time_step = 1.
        self.speed = 1.

        if goal is None:
            goal = np.array([shape[1], shape[0]], dtype=np.float64)
        if start is None:
            start = np.array([0, shape[0]], dtype=np.float64)
        size = np.array(shape)
        size[1] = 2*size[1]
        self.size = size
        self.goal_radius = 0.5
        self.done = False
        self.start = start
        self.randomized_initial = randomized_initial
        self.goal = goal

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_bases[0] * n_bases[1],))
        #delta_x, delta_y
        self.action_space = spaces.Discrete(4)
        self.PrettyTable = None
        self.rendering = None

        self._t = 0
        self.gamma = gamma
        if rew_weights is None:
            rew_weights = [1, 10, 0]
        self.rew_weights = np.array(rew_weights)

        # gym attributes
        self.viewer = None
        self.feat_func = build_features_gw_state(self.size, n_bases, 2)
        # initialize state
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_rew_features(self, state=None):
        if state is None:
            if self.done:
                return np.zeros(3)
            state = self.state
        x, y = state[:2]
        features = np.zeros(3)
        if self.reached_goal(state):  # goal state
            features[2] = 1
        elif x > 1 and x < self.size[0] -1 and y > 1 and y < self.size[1] -1:  # slow_region
            features[1] = -1
        else:
            features[0] = -1  # fast region
        return features

    def reached_goal(self, state=None):
        if state is None:
            state = self.state

        return np.linalg.norm(state - self.goal) < self.goal_radius

    def action_to_direction(self, a):
        if a == 0:
            action = np.array([0, 1])  # up
        elif a == 1:
            action = np.array([1, 0])  # right
        elif a == 2:
            action = np.array([0, -1])  # down
        elif a == 3:
            action = np.array([-1, 0])  # left

        return action

    def step(self, a, rbf=False, ohe=False):
        if self.reached_goal():
            return self.get_state(rbf=rbf, ohe=ohe), 0, 1, {'features': np.zeros(3)}

        a = self.action_to_direction(a)
        self.state += self.speed * self.time_step * np.array(a).clip([-1,-1],[1,1])
        # Add noise
        if self.noise > 0:
            self.state += np.random.normal(scale=self.noise, size=(2,))

        # Clip to make sure the agent is inside the grid

        self.state = self.state.clip([0., 0.], self.size - 1e-8)
        # Compute reward
        features = self.get_rew_features()
        reward = np.sum(self.rew_weights * features)
        reward /= np.max(self.rew_weights)
        reward += 1

        self._t += 1
        terminal = True if self._t >= self.horizon else False
        self.done = 1 if self.reached_goal() else 0
        self.done = self.done or terminal

        return self.get_state(rbf=rbf, ohe=ohe), reward, self.done, {'features': features}

    def get_state(self, rbf=False,ohe=False):
        if rbf or ohe:
            s = self.feat_func(self.state)
            return s
        return self.state

    def get_signature(self):
        sig = {'agent_pos': np.copy(self.state), 't': copy(self._t)}
        return sig

    def set_signature(self, sig):
        self.state = np.copy(sig['agent_pos'])
        self._t = copy(sig['t'])

    def reset(self, state=None, rbf=False, ohe=False):
        self.done = False
        if state is None:
            if self.randomized_initial:
                self.state = np.copy(self.goal)
                while self.reached_goal():
                    self.state = np.random.uniform(low=[0, 0], high=self.size)
            else:
                self.state = np.copy(self.start)
        else:
            self.state = np.array(state)

        self._t = 0
        return self.get_state(rbf=rbf, ohe=ohe)

    def _render(self, mode='human', close=False, a=None):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(0, self.size[0], 0, self.size[1])

        self.viewer.draw_line((1, 1), (self.size[0]-1, 1))
        self.viewer.draw_line((1, 1), ( 1, self.size[1]-1))
        self.viewer.draw_line((self.size[0] - 1, 1), (self.size[0] - 1, self.size[1] -1))
        self.viewer.draw_line(( 1, self.size[1]-1), (self.size[0] - 1, self.size[1] -1))

        if self.state is None:
            return None

        goal = self.viewer.draw_circle(radius=self.goal_radius)
        goal.set_color(0, 0.8, 0)
        goal.add_attr(rendering.Transform(translation=(self.goal[0], self.goal[1])))

        agent = self.viewer.draw_circle(radius=0.1)

        agent.set_color(.8, 0, 0)
        transform = rendering.Transform(translation=(self.state[0], self.state[1]))
        agent.add_attr(transform)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
