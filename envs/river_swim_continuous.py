from copy import copy
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from gym_minigrid.register import register


def generate_river_continuous(**game_params):
    if game_params is None:
        game_params = {}
    return RiverSwimContinuous(**game_params)


class RiverSwimContinuous(gym.Env):

    def __init__(self, dim=6, gamma=0.95, small=5, large=10000, horizon=10, fail_prob=0.4, scale_reward=True):

        self.horizon = horizon
        self.small = small
        self.large = large
        self.gamma = gamma
        self.dim = dim
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = 0
        self.max_position = dim
        self.fail_prob = fail_prob
        self.viewer = None

        self._t = 0

        # self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
        #                                shape=(1,), dtype=np.float32)
        # 0 = left, 1 = right
        self.action_space = spaces.Discrete(n=2)
        self.observation_space = spaces.Box(low=self.min_position, high=self.max_position,
                                            shape=(1,), dtype=np.float32)
        self.scale_reward = scale_reward
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_prob(self, action):

        prob = np.zeros(2)
        if action == 0:  # left
            prob[0] = 1.
            prob[1] = 0.
        else:

            prob[0] = self.fail_prob
            prob[1] = 1. - self.fail_prob

        return prob

    def step(self, action):
        if self._t >= self.horizon:
            return self.state, 0, True, {}

        # action = np.clip(action, self.action_space.low, self.action_space.high)
        prob = self._get_prob(action)
        dir = self.np_random.choice(2, p=prob)
        step = self.np_random.uniform(low=0.5, high=1.)
        if dir == 0:  # left
            new_state = self.state - step
        else:
            new_state = self.state + step

        reward = 0.
        if action == 0 and self.state <= 1:
            reward = self.small
        elif action == 1 and self.state >= self.dim - 1:
            reward = self.large

        if self.scale_reward:
            reward /= self.large

        self.state = np.clip(new_state, self.observation_space.low, self.observation_space.high)
        self._t += 1
        terminal = True if self._t >= self.horizon else False
        return self.state, reward, terminal, {}

    def reset(self):
        self._t = 0
        self.state = self.np_random.rand() * 0.5

    def get_state(self):
        return self.state

    def get_signature(self):
        sig = {'agent_pos': np.copy(self.state), 't': copy(self._t)}
        return sig

    def set_signature(self, sig):
        self.state = np.copy(sig['agent_pos'])
        self._t = copy(sig['t'])


register(
    id='MiniGrid-RiverSwim-continuous-v0',
    entry_point='envs.river_swim_continuous:RiverSwimContinuous'
)

if __name__ == '__main__':
    fail = [i * 0.1 for i in range(10)]
    for f in fail:
        mdp = RiverSwimContinuous(horizon=40, dim=10, fail=f)
        gamma = 0.99
        num_episodes = 1000
        rets = []
        for i in range(num_episodes):
            ret = 0
            t = 0
            done = False
            s = mdp.reset()
            while not done:
                a = 1
                s, r, done, _ = mdp.step(a)
                ret += r
                t += 1
            rets.append(ret)
        print("Return = %.2f +/- %.2f" %(np.mean(rets), np.std(rets) / np.sqrt(num_episodes)))
        print("Max Return:", np.max(rets))
        print("Returns:")
        print(rets)
