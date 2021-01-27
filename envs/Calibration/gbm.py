import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

def GBM_calibration(data):
    # diff = data['Price'].values[1:]/data['Price'].values[:-1]-1
    # data['perc'] = np.insert(diff, 0,0)
    # data['log'] = np.insert(log,0,0)

    log = np.log(data['Price'].values[1:]) - np.log(data['Price'].values[:-1])
    m_hat = (log[0] - log[-1])/len(log)
    v_hat = sum(np.square(log-m_hat))/len(log)

    for num_plots in range(0,10):
        log_step = [np.log(data['Price'].values[0])]
        for i in range(0,len(log)):
            log_step += [m_hat + np.sqrt(v_hat) * np.random.normal(0,1)]
        log_step = np.array(log_step)
        log_vec = log_step.cumsum()
        simulation = np.exp(log_vec)
        plt.plot(simulation, color='blue')

    plt.plot(data['Price'], color='red')

    plt.show()
    return simulation

if __name__ == '__main__':
    # Import data
    path = "/Users/edoardovittori/Code/alphazero_singleplayer/utils/SP.CSV"
    data = pd.read_csv(path)
    GBM = GBM_calibration(data)