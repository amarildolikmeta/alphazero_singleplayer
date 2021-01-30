import numpy as np
import time
from envs.trading import Trade
from trading_paper.utils import make_model
import joblib
import pandas as pd
import copy
import os
from models import SDE_calibrations
import datetime


def generate_trade_sim(**game_params):
    if game_params is None:
        game_params = {}
    return TradeSim(**game_params)


class TradeSim(Trade):

    def __init__(self, fees=0.001, time_lag=5, horizon=30, log_actions=True, save_dir='',
                 model_path='models/', model_name='vasicek', start_date='01/03/2017', end_date='10/07/2018',):
        super(TradeSim, self).__init__(fees, time_lag, horizon, log_actions, save_dir, process=model_name)
        dataset = 'SP.CSV'
        self.start_date = self.date_parser(start_date)
        self.end_date = self.date_parser(end_date)
        self.model_name = model_name
        self.load_model_and_data(model_path, dataset)
        self.during_search = False

    def date_parser(self, x):
            return pd.to_datetime(x, format='%d/%m/%Y')

    def load_model_and_data(self, model_path, dataset):
        # Load data
        path = (os.path.join(os.getcwd(), '../'))
        df = pd.read_csv(os.path.join(path, model_path, dataset), parse_dates=['referenceDate'], date_parser=self.date_parser)
        diff = df['Price'].values[1:] / df['Price'].values[:-1] - 1
        diff = np.insert(diff,0,0)
        df['Returns'] = diff
        self.df = df

        # use df_training for initial model calibration, correct for weekends?
        df_train = self.df[(self.df['referenceDate'] >= self.start_date) & (self.df['referenceDate'] <= self.end_date)]
        self.data = df[(df['referenceDate'] >= self.end_date-datetime.timedelta(self.time_lag))].reset_index()
        # Load model
        if self.model_name == 'lstm':
            # Load data scaler
            self.scaler = joblib.load(model_path + '/scaler.gz')
            # Scale data
            target_index = self.data.columns.get_loc('Price')
            df_log = self.scaler.transform(
                self.data.iloc[:, target_index].astype('float32').values.reshape(-1, 1))  # Close index
            self.data = pd.DataFrame(df_log).values.flatten()

            model_params = joblib.load(model_path + '/model_params.gz')
            model = make_model(self.model_name, **model_params)
            model.load(model_path + '/model.gz')

        elif self.model_name == 'gbm':
            # models are just the calibrated parameters
            model = SDE_calibrations.GBM_calibration(df_train['Price'].values)

        elif self.model_name == 'arima':
            pass
        elif self.model_name == 'vasicek':
            # models are just the calibrated parameters
            model = SDE_calibrations.vasicek_calibation(df_train['Price'].values)

        self.model = model

    def set_search(self):
        self.during_search = True

    def forecast(self):
        out_logits, self.last_state = self.model.forward(self.ret_window.reshape((1, self.time_lag, 1)),
                                                         self.last_state)
        return out_logits[-1]

    def get_reward(self):

        if self.during_search:
            if self.model_name == 'lstm':
                new_ret = self.forecast()
                self.ret_window = np.append(self.ret_window[1:], new_ret)
                pl = self.current_portfolio * new_ret - abs(self.current_portfolio - self.previous_portfolio) * self.fees
            else:
                pl = super(TradeSim, self).get_reward()

        else:
            new_ret = self.data['Returns'][self.time_lag + self._t]
            self.ret_window = np.append(self.ret_window[1:], new_ret)
            pl = self.current_portfolio * new_ret - abs(self.current_portfolio - self.previous_portfolio) * self.fees

        return pl

    def reset(self):
        super(TradeSim, self).reset()

        self.ret_window = self.data['Returns'][:self.time_lag].values
        self.rates = self.data['Price'][self.time_lag]

        if self.model_name == 'lstm':
            self.ret_window = self.data[:self.time_lag].flatten()
            self.last_state = copy.deepcopy(self.model.get_init_state())
        return self.get_state()

    def get_signature(self):
        if self.model_name == 'lstm':
            sig = {'state': np.copy(self.get_state()),
                   'last_model_state': np.copy(self.last_state),
                   't': self._t}
        else:
            sig = super(TradeSim, self).get_signature()
        return sig

    def set_signature(self, sig):
        if self.model_name == 'lstm':
            self.current_portfolio = sig['state'][-1]
            self.ret_window = sig['state'][:-1]
            self._t = sig['t']
            self.last_state = sig['last_model_state']
        else:
            super(TradeSim, self).set_signature(sig)


    def step(self, action):
        state, reward, terminal, info = super(TradeSim, self).step(action)

        if not self.during_search:
            self.update()

        return state, reward, terminal, info

    def update(self):
        # update df_train as with sliding window adding new point
        # self.df_train[1:]
        # self.data[]
        self.start_date = self.start_date + datetime.timedelta(1)
        self.end_date = self.end_date + datetime.timedelta(1)
        df_train = self.df[(self.df['referenceDate'] >= self.start_date) & (self.df['referenceDate'] <= self.end_date)]

        if self.model_name == 'lstm':
            pass
        elif self.model_name == 'gbm':
            self.model = SDE_calibrations.GBM_calibration(df_train['Price'].values)

        elif self.model_name == 'arima':
            pass
        elif self.model_name == 'vasicek':
            self.model = SDE_calibrations.vasicek_calibation(df_train['Price'].values)

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
    plt.show()

    for j in range(5):
        mdp_search = TradeSim()
        mdp_search.set_signature(sig)
        mdp_search.set_search()
        s = mdp_search.reset()
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


