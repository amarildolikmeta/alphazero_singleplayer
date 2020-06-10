from abc import ABC
from copy import copy
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from gym import register
import pandas as pd

def generate_trade():
    return Trade()

class Trade(gym.Env):
    # 7 / 100000
    def __init__(self, fees = 0, time_lag=2, horizon = 20):
        # Initialize parameters

        # price history, previous portfolio, time
        observation_low = np.concatenate([np.full(time_lag, -1), [-1.0]])
        observation_high = np.concatenate([np.full(time_lag, +1), [+1.0]])
        self.observation_space = spaces.Box(low=observation_low, high=observation_high)
        self.action_space = spaces.Discrete(n=3)

        # Internals
        self.previous_portfolio = 0
        self.current_portfolio = 0
        self.time_lag = time_lag
        self.horizon = horizon
        self.ret_window = np.asarray([0]*self.time_lag)
        self.prices = [100]
        self.actions = []
        self.current_ret = [0]

        self.fees = fees

        self._t = 0
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_state(self):
        return np.append(self.ret_window, self.current_portfolio)

    def get_reward(self):
        new_ret = self.gmb(self.prices[-1])
        pl = self.current_portfolio * new_ret - \
                abs(self.current_portfolio - self.previous_portfolio) * self.fees

        # transform with logistic in [0,1] for algorithm
        pl = 1/(1+np.exp(-40*pl))
        return pl

    def gmb(self, s0=100, sigma=0.2, r=1, days=2, ppd=1):
        dt = 1 / (365 * ppd)
        T = days * ppd
        tmp_exp = r - 0.5 * sigma ** 2
        bm = np.cumsum([np.random.normal(0, 1, T - 1)])
        bm = np.insert(bm, 0, 0)
        s = s0 * np.exp(tmp_exp * np.arange(0, T) * dt + sigma * bm * np.sqrt(dt))
        self.prices.append(s[-1])
        s_ret = (s[1:] - s[:-1])/s[:-1]
        self.ret_window = np.append(self.ret_window[1:],s_ret.tolist())
        return s_ret[0]

    def step(self, action):
        if self._t >= self.horizon:
            return self.get_state(), 0, True, {}

        action = action -1
        # Check the action is in the range
        assert -1 <= action <= +1, "Action not in range!"
        self.actions.append(action)
        self.previous_portfolio, self.current_portfolio = self.current_portfolio, action
        reward = self.get_reward()

        self._t += 1
        if self._t >= self.horizon:
            terminal = True
        else:
            terminal = False

        return self.get_state(), reward, terminal, {}

    def reset(self):
        self._t = 0

        self.previous_portfolio = 0
        self.current_portfolio = 0
        self.prices = [100]
        self.ret_window = [0]*self.time_lag
        self.done = False

        return self.get_state()

    def get_signature(self):
        sig = {'state': np.copy(self.get_state())}
        return sig

    def set_signature(self, sig):
        self.current_portfolio = sig['state'][-1]
        self.ret_window = sig['state'][:-1]
    #

register(
    id='Trading-v0',
    entry_point='envs.trading:Trade'
)


if __name__ == '__main__':
    mdp = Trade()
    ret=0
    s = mdp.reset()
    while True:
        # print(s)
        a = 2
        s, r, done, prices = mdp.step(a)
        print("Reward:" + str(r) + " State:" + str(s) )
        mdp.set_signature(mdp.get_signature())
        ret += r - 0.5
        if done:
            print("Return:", ret)
            break

