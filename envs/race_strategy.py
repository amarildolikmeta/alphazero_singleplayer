import gym
from copy import copy
import numpy as np
from gym import spaces
from gym.utils import seeding
from gym import register


def generate_race(**game_params):
    if game_params is None:
        game_params = {}
    return Race(**game_params)


class Race(gym.Env):

    def __init__(self, gamma=0.95, horizon=55, mean_lap=60., std_lap=1., mean_pit_stop=10., std_pit_stop=0.5,
                 slow_lap_degradation=0.5, fast_lap_degradation=1.2, slow_lap_time=4, fast_lap_time=1, max_lap_time=100,
                 scale_reward=True, positive_reward=True, random_event=0.):

        self.horizon = horizon
        self.gamma = gamma
        self.mean_lap = mean_lap
        self.std_lap = std_lap
        self.mean_pit_stop = mean_pit_stop
        self.std_pit_stop = std_pit_stop
        self.slow_lap_degradation = slow_lap_degradation
        self.fast_lap_degradation = fast_lap_degradation
        self.slow_lap_time = slow_lap_time
        self.fast_lap_time = fast_lap_time
        self.max_lap_time = max_lap_time
        self.positive_reward = positive_reward
        self.viewer = None
        self.random_event = random_event

        self._t = 0
        self.time = 0
        self.tire_damage = 0

        self.obs_dim = 2
        self.action_space = spaces.Discrete(n=3)
        self.observation_space = spaces.Box(low=0., high=self.horizon,
                                            shape=(self.obs_dim,), dtype=np.float32)
        self.scale_reward = scale_reward
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_slow_down(self):
        return (self.tire_damage**2) / 15

    def _get_lap_time(self):
        time = self.np_random.normal(self.mean_lap, self.std_lap)
        time += self._get_slow_down()
        return time

    def _pit_stop(self):
        self.tire_damage = 0.
        delay = np.clip(self.np_random.normal(self.mean_pit_stop, self.std_pit_stop), 0, None)
        return delay

    def _slow_lap(self):
        degradation = self.np_random.normal(self.slow_lap_degradation, 0.1)
        self.tire_damage += degradation
        return self.np_random.normal(self.slow_lap_time, 0.2)

    def _fast_lap(self):
        degradation = self.np_random.normal(self.fast_lap_degradation, 0.1)
        self.tire_damage += degradation
        return self.np_random.normal(self.fast_lap_time, 0.1)

    def step(self, action):
        if self._t >= self.horizon:
            return self.get_state(), 0, True, {}

        lap_time = self._get_lap_time()
        if action == 0:  # pit_stop
            lap_time += self._fast_lap() + self._pit_stop()
        elif action == 1:  # slow lap
            lap_time += self._slow_lap()
        elif action == 2:  # fast lap
            lap_time += self._fast_lap()
            if np.random.uniform() < self.random_event:
                lap_time += 100

        lap_time = np.clip(lap_time, 0, self.max_lap_time)
        reward = -lap_time
        if self.scale_reward:
            reward /= self.max_lap_time
        self.time += lap_time
        self._t += 1
        terminal = True if self._t >= self.horizon else False
        # self.state = self.get_state()
        if self.positive_reward:
            reward = 1 + reward
        return self.get_state(), reward, terminal, {}

    def reset(self):
        self._t = 0
        self.time = 0
        self.tire_damage = 0
        return self.get_state()

    def get_state(self):
        return np.array([self._t, self.tire_damage])

    def get_signature(self):
        sig = {'state': np.copy(self.get_state()), 'time': copy(self.time)}
        return sig

    def set_signature(self, sig):
        self._time = sig["time"]
        self._t, self.tire_damage = sig["state"][0], sig["state"][1]


register(
    id='RaceStrategy-v0',
    entry_point='envs.race_strategy:Race'
)

if __name__ == '__main__':
    mdp = Race()

    s = mdp.reset()
    ret = 0
    while True:
        # print(s)
        a = 0
        s, r, done, _ = mdp.step(a)
        print("Reward:" + str(r) + " Lap Time: " + str(r * mdp.max_lap_time))
        mdp.set_signature(mdp.get_signature())
        ret += r
        if done:
            print("Return:", ret)
            print("Race Time:", mdp.time)
            break
