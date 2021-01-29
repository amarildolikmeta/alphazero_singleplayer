import numpy as np
import time
from envs.trading import Trade
from trading_paper.utils import make_model
import joblib
import pandas as pd
import copy


def generate_trade_sim(**game_params):
    if game_params is None:
        game_params = {}
    return TradeSim(**game_params)


class TradeSim(Trade):

    def __init__(self, fees=0.001, time_lag=5, horizon=30, log_actions=True, save_dir='', process="arma",
                 model_path='models/lstm', model_name='lstm', start_date='2018-07-10', end_date='2019-07-10',):
        self.start_date = start_date
        self.end_date = end_date
        self.model_name = model_name
        self.load_model_and_data(model_path)
        super(TradeSim, self).__init__(fees, time_lag, horizon, log_actions, save_dir, process)
        self.during_search = False

    def load_model_and_data(self, model_path):
        # Load data scaler
        self.scaler = joblib.load(model_path + '/scaler.gz')

        # Load data
        def date_parser(x):
            return pd.to_datetime(x, format='%d/%m/%Y')
        df = pd.read_csv(model_path + '/S&P.csv', parse_dates=['referenceDate'], date_parser=date_parser)

        df = df[(df['referenceDate'] >= self.start_date) & (df['referenceDate'] <= self.end_date)]

        # Scale data
        target_index = df.columns.get_loc('Price')
        df_log = self.scaler.transform(df.iloc[:, target_index].astype('float32').values.reshape(-1, 1))  # Close index
        self.data = pd.DataFrame(df_log).values.flatten()

        # Load model
        model_params = joblib.load(model_path + '/model_params.gz')
        model = make_model(self.model_name, **model_params)
        model.load(model_path + '/model.gz')
        self.model = model

    def set_search(self):
        self.during_search = True

    def forecast(self):
        out_logits, self.last_state = self.model.forward(self.ret_window.reshape((1, self.time_lag, 1)),
                                                         self.last_state)
        return out_logits[-1]

    def get_reward(self):
        if self.during_search:
            new_ret = self.forecast()
        else:
            new_ret = self.data[self.time_lag + self._t]
        self.ret_window = np.append(self.ret_window[1:], new_ret)
        pl = self.current_portfolio * new_ret - abs(self.current_portfolio - self.previous_portfolio) * self.fees
        return pl

    def reset(self):
        self._t = 0
        self.previous_portfolio = 0
        self.current_portfolio = 0
        self.prices = [100]
        self.ret_window = self.data[:self.time_lag].flatten()
        self.done = False
        self.last_state = copy.deepcopy(self.model.get_init_state())
        return self.get_state()

    def get_signature(self):
        sig = {'state': np.copy(self.get_state()),
               'last_model_state': np.copy(self.last_state),
               't': self._t}
        return sig

    def set_signature(self, sig):
        self.current_portfolio = sig['state'][-1]
        self.ret_window = sig['state'][:-1]
        self.last_state = sig['last_model_state']
        self._t = sig['t']


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    t0 = time.time()
    mdp = TradeSim()
    s = mdp.reset()
    sig = mdp.get_signature()
    ret = 0
    true_data = []
    for i in range(1, 100):
        a = 2
        s, r, done, prices = mdp.step(a)
        ret += r - 0.5
        if done:
            print("Return:", ret)
            rt0 = time.time()
            s = mdp.reset()
            break
        true_data.append(mdp.ret_window[-1])

    plt.plot(true_data, label='true_data')
    for j in range(5):
        mdp_search = TradeSim()
        mdp_search.set_signature(sig)
        mdp_search.set_search()
        sim_data = []
        for i in range(1, 100):
            a = 2
            s, r, done, prices = mdp_search.step(a)
            ret += r - 0.5
            if done:
                print("Return:", ret)
                rt0 = time.time()
                s = mdp_search.reset()
                break
            sim_data.append(mdp_search.ret_window[-1])
        plt.plot(sim_data, label='sim_data_' + str(j))
    plt.legend()
    plt.show()


