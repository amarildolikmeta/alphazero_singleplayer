import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class RiverSwimContinuous(gym.Env):

    def __init__(self, dim=6, gamma=0.95, small=5, large=10000, horizon=np.inf, scale_reward=True):

        self.horizon = horizon
        self.small = small
        self.large = large
        self.gamma = gamma
        self.dim = dim

        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = 0
        self.max_position = dim

        self.viewer = None

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
        if action == 0: # left
            prob[0] = 1.
            prob[1] = 0.
        else:

            prob[0] = 0.3
            prob[1] = 0.7

        return prob

    def step(self, action):

        #action = np.clip(action, self.action_space.low, self.action_space.high)
        prob = self._get_prob(action)
        dir = self.np_random.choice(2, p=prob)
        step = self.np_random.uniform(low=0.5, high=1.)
        if dir == 0: #left
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

        return self.state, reward, False, {}

    def reset(self):
        self.state = self.np_random.rand() * 0.5

    def get_state(self):
        return self.state

    def get_signature(self):
        sig = {'agent_pos': np.copy(self.state)}
        return sig

    def set_signature(self, sig):
        self.state = np.copy(sig['agent_pos'])


if __name__ == '__main__':
    mdp = RiverSwimContinuous()

    s = mdp.reset()
    while True:
        #print(s)
        a = np.random.rand() * 2 - 1
        a = 1
        s, r, _, _ = mdp.step(a)
        if s >= 5:
            print(r)