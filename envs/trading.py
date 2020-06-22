import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from gym import register
import time


def generate_trade():
    return Trade()


class Trade(gym.Env):
    def __init__(self, fees=0.01, time_lag=1, horizon=20):
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
        self.fees = fees
        # self.actions = []
        # self.current_ret = [0]

        self._t = 0
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_state(self):
        return np.append(self.ret_window, self.current_portfolio)

    def get_reward(self):
        new_ret = self.gmb()
        pl = self.current_portfolio * new_ret - \
                abs(self.current_portfolio - self.previous_portfolio) * self.fees

        # transform with logistic in [0,1] for algorithm
        pl = 1/(1+np.exp(-100*pl))
        return pl

    def gmb(self, sigma=0.2, r=0, days=2, ppd=1):
        dt = 1 / (365 * ppd)
        T = days * ppd
        tmp_exp = r - 0.5 * sigma ** 2
        bm = self.np_random.normal(0, 1)
        s = np.exp(tmp_exp * dt + sigma * bm * np.sqrt(dt))
        s_ret = (s - 1)/1
        self.ret_window = np.append(self.ret_window[1:],s_ret.tolist())
        return s_ret

    def step(self, action):
        if self._t >= self.horizon:
            return self.get_state(), 0, True, {}
        action = int(action) -1
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

register(
    id='Trading-v0',
    entry_point='envs.trading:Trade'
)

if __name__ == '__main__':
    t0 = time.time()
    mdp = Trade()
    mt1 = time.time()
    print("initialization time", mt1 - t0)

    s = mdp.reset()
    mt1 = time.time()
    print("first reset time", mt1 - t0)
    ret=0
    for i in range(1,100):
        ft0 = time.time()
        a = 2
        s, r, done, prices = mdp.step(a)
        # print("Reward:" + str(r) + " State:" + str(s) )
        mdp.set_signature(mdp.get_signature())
        ret += r - 0.5
        # print(prices)
        if done:
            print("Return:", ret)
            rt0 = time.time()
            s = mdp.reset()
            rt1 = time.time()
            print("reset time", rt1 - rt0)
            print(s)
        ft1 = time.time()
        print("for time is ", ft1 - ft0)
        print("cumulated for time", ft1 - mt1)
    t1 = time.time()
    print("time is ", t1-t0)

