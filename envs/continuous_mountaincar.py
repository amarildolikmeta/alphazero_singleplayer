import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from copy import copy


def generate_mountain(**kwargs):
    if kwargs is None:
        kwargs = {}
    return ContinuousMountainCar(**kwargs)


class ContinuousMountainCar(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, horizon=250, fail_prob=0.05, goal_position=None, start_state=None, randomized_start=False):

        assert 0 <= fail_prob <= 1, "The probability of failure must be in [0,1]"
        self.horizon = horizon
        self.noise = fail_prob
        if goal_position is None:
            goal_position = 0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.randomized_start = randomized_start

        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = goal_position
        self.power = 0.0015

        if start_state is None:
            start_state = np.random.uniform(low=[-0.6, 0], high=[-0.4, 0])
        self.start_state = start_state
        self.low_state = np.array([self.min_position, -self.max_speed])
        self.high_state = np.array([self.max_position, self.max_speed])
        self.viewer = None
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.size = {'position': [self.min_position, self.max_position],
                     'speed': [-self.max_speed, self.max_speed]}
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def action_to_force(self, action):
        if action == 0:
            force = 0.
        elif action == 1:
            force = 1.
        else:
            force = -1
        return force

    def step(self, action):

        if self.done or self._t >= self.horizon:
            return self.get_state(), 0, self.done, {}
        position = self.state[0]
        velocity = self.state[1]
        force = self.action_to_force(action)
        # force = min(max(action[0], -1.0), 1.0)

        velocity += force * self.power - 0.0025 * math.cos(3*position)
        if self.noise >= 0.:
            velocity += np.random.uniform(low=-self.noise, high=self.noise)

        if velocity > self.max_speed: velocity = self.max_speed
        if velocity < -self.max_speed: velocity = -self.max_speed
        position += velocity

        if self.noise >= 0.:
            position += np.random.uniform(low=-self.noise, high=self.noise)
        if position > self.max_position: position = self.max_position
        if position < self.min_position: position = self.min_position
        if position == self.min_position and velocity < 0: velocity = 0

        self._t += 1
        self.done = bool(position >= self.goal_position) or self._t >= self.horizon
        reward = 0
        if bool(position >= self.goal_position):
            reward = 100.0
        reward -= (force ** 2) * 0.1
        reward /= 100
        self.state = np.array([position, velocity])
        return self.get_state(), reward, self.done, {}

    def reached_goal(self, state=None):
        if state is None:
            state = self.state
        return state[0] >= self.goal_position

    def get_state(self):
        return self.state

    def reset(self, state=None, rbf=False, ohe=False):
        self.done = False
        self._t = 0
        if state is None:
            if self.randomized_start:
                self.state = np.array([self.goal_position, 0])
                while self.reached_goal():
                    self.state = np.random.uniform(low=[self.min_position, 0], high=[self.max_position, 0])
            else:
                self.state = np.copy(self.start_state)
        else:
            self.state = np.array(state)
        return self.get_state()

    def get_signature(self):
        sig = {'agent_pos': np.copy(self.state), 't': copy(self._t), 'done': self.done}
        return sig

    def set_signature(self, sig):
        self.state = np.copy(sig['agent_pos'])
        self._t = copy(sig['t'])
        self.done = sig['done']

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55

    def _render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth=40
        carheight=20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position)*scale
            flagy1 = self._height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos-self.min_position)*scale, self._height(pos)*scale)
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array=mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None