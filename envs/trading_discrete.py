import numpy as np
from gym import register
import time
from envs.FiniteMDP import FiniteMDP


def generate_trade(**game_params):
    if game_params is None:
        game_params = {}
    return Trade(**game_params)


class Trade(FiniteMDP):
    def __init__(self, fees=0.001, horizon=50, log_actions=True, save_dir='', n_ret=30, max_ret=0.07):

        self.actions = [-1, 0, 1]
        self.n_actions = len(self.actions)
        self.n_ret = n_ret
        self.n_states = self.n_ret*self.n_actions
        self.max_ret = max_ret
        self.ret = np.linspace(-max_ret, max_ret, n_ret)
        #print(self.ret)
        # Internals
        self.previous_portfolio = 0
        self.current_portfolio = 0
        self.horizon = horizon
        self.prices = [100]
        self.fees = fees
        self.rates = 100
        self._t = 0

        #Create Transition and Reward Matrix
        mu, p, r = self.calculate_mdp()
        self.mu = mu
        self.p = p
        self.P = p
        self.r = r

        # # Start logging file
        sd = self.seed()
        self.log_actions = log_actions
        if self.log_actions:
            try:
                os.makedirs(os.path.join(save_dir, "state_action"))
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise  # This was not a "directory exist" error..
            self.file_name = os.path.join(save_dir, 'state_action', str(sd[0]) + '.csv')
            text_file = open(self.file_name, 'w')
            s = ''
            s += 'p1' + ','
            s += 'a,r\n'
            text_file.write(s)
            text_file.close()
            # print('writing actions in ' + self.file_name)

        super(Trade, self).__init__(self.p, self.r, self.mu, horizon=horizon)
        self.reset()

    def write_file(self, s_a):
        with open(self.file_name, 'a') as text_file:
                prices = ','.join(str(e) for e in s_a[:-1])
                toprint = prices+','+str(s_a[-1])+'\n'
                text_file.write(toprint)

    def step(self, action):
        _, reward, absorbing, _ = super().step(action)
        current_ret = self.ret[self._state-action*self.n_ret]
        if self.log_actions:
            return self._state, reward, absorbing, {'save_path': self.file_name, 'return': current_ret}
        else:
            return self._state, reward, absorbing, _

    def reset(self):
        super().reset()
        return self._state

    def calculate_mdp(self):
        n_states = self.n_states
        n_actions = self.n_actions
        n_ret = int(n_states/n_actions)

        # Compute the initial state distribution
        P0 = np.zeros(n_states)
        P0[int(n_states/2)] = 1

        # Initialize the reward function
        R = np.zeros((n_actions, n_states, n_states))

        # Initialize the transition probability matrix
        P = np.zeros((n_actions, n_states, n_states))

        prob = 1/n_ret

        for i in range(n_actions):
            P[i, :, i*n_ret : (i+1)*n_ret].fill(prob)
            # print(np.sum(P[i, :, i*n_ret : (i+1)*n_ret]))
            reward = self.actions[i] * self.ret
            for j in range(n_actions):
                R[i, j * n_ret:(j + 1) * n_ret, i * n_ret: (i + 1) * n_ret] = reward - abs(i - j) * self.fees
        P = np.transpose(P, [1, 0, 2])
        R = np.transpose(R, [1, 0, 2])
        return P0, P, R


# register(
#     id='Trading_discrete-v0',
#     entry_point='envs.trading_discrete:Trade'
# )

if __name__ == '__main__':
    t0 = time.time()
    mdp = Trade()

    s = mdp.reset()
    ret=0
    for i in range(1,100):
        ft0 = time.time()
        a = i%3
        s, r, done, prices = mdp.step(a)
        print("Reward:" + str(r) + " State:" + str(s) )
        mdp.set_signature(mdp.get_signature())
        # ret += r - 0.5
        print(prices)
        if done:
            print("Return:", ret)
            rt0 = time.time()
            s = mdp.reset()
            # print(s)
    t1 = time.time()
    print("time is ", t1-t0)

