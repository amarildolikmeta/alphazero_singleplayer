import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})
from scipy import optimize

def price2ret(data):
    diff = data['Price'].values[1:] / data['Price'].values[:-1] - 1
    return diff

def GBM_calibration(log):
    # log = np.log(data['Price'].values[1:]) - np.log(data['Price'].values[:-1])
    m_hat = (log[0] - log[-1])/len(log)
    v_hat = sum(np.square(log-m_hat))/len(log)
    return [m_hat, v_hat]

def GBM_calibration_ll(params, log):
    # log = np.log(data['Price'].values[1:]) - np.log(data['Price'].values[:-1])
    mu = params[0]
    var_t = params[1]
    LogLikelihood = 0
    for time in range(0, len(log)):
        assert var_t >= 0, "negative variance error"
        innov = np.square(log[time] - mu)
        LogLikelihood += -np.log(var_t) - innov / var_t
    mll = -LogLikelihood
    return mll

def GBM_simulation(data, params):
    m_hat = params[0]
    v_hat = params[1]
    for num_plots in range(0,10):
        log_step = [np.log(data['Price'].values[0])]
        for i in range(0,len(data)):
            log_step += [m_hat + np.sqrt(v_hat) * np.random.normal(0,1)]
        log_step = np.array(log_step)
        log_vec = log_step.cumsum()
        simulation = np.exp(log_vec)
        plt.plot(simulation, color='blue')

    plt.plot(data['Price'], color='red')

    plt.show()
    return simulation



def GARCH_calibration(params, returns):

    mu=params[0]
    omega=abs(params[1])
    alpha=abs(params[2])
    beta=abs(params[3])
    gamma=params[4]
    denum = 1-alpha-beta*(1+np.square(gamma))
    assert denum > 0, 'variance stationarity constraint breached '
    var_t = omega/denum
    LogLikelihood = 0
    for time in range(0,len(returns)):
        innov = np.square(returns[time] - mu)
        var_t = omega+alpha*var_t+beta*np.square(innov-gamma*np.sqrt(var_t))
        assert var_t >= 0, "negative variance error"
        LogLikelihood += -np.log(var_t) - innov / var_t
    mll = -LogLikelihood
    return mll

# def NL_constr(params):
#     # mu = params[0]
#     # omega = abs(params[1])
#     alpha = abs(params[2])
#     beta = abs(params[3])
#     gamma = params[4]
#     denum = 1 - alpha - beta * (1 + np.square(gamma))
#     # print(denum, denum>0)
#     return denum

# def GARCH_calibration(data):
#     returns = price2ret(data)
#     x0 = np.array([0.1, 1.0, 0.1, 0.1, 0.1])
#     params = optimize.minimize(NG_JGBM_LL, x0 = x0, args=returns, method="Nelder-Mead")
#     return mu, omega, alpha, beta, gamma

def NGARCH_simulation(N_Sim ,T,params,S0, data):
    mu=params[0]
    omega=params[1]
    alpha=params[2]
    beta=params[3]
    gamma=params[4]
    denum = 1-alpha-beta*(1+np.square(gamma))
    assert denum > 0
    for num in range(0,N_Sim):
        S = [S0]
        v = omega
        for i in range(1,T):
            mean=mu-0.5*v
            rdn = np.random.normal(0,1)
            S+=[S[i-1]*np.exp(mean+np.sqrt(v)*rdn)]
            v=omega+alpha*v+beta*np.square(np.sqrt(v)*rdn-gamma*np.sqrt(v))
        plt.plot(S, color='blue')
    plt.plot(data['Price'], color='red')
    plt.show()
        

if __name__ == '__main__':
    # Import data
    path = "/Users/edoardovittori/Code/alphazero_singleplayer/utils/SP.CSV"
    data = pd.read_csv(path)

    # calibrate and simulate GBM data with exact log likelihood
    # log_returns = np.log(data['Price'].values[1:]) - np.log(data['Price'].values[:-1])
    # params1 = GBM_calibration(log_returns)
    # print('parameters are', params1)
    # GBM_simulation(data, params1)
    #
    # # calibrate and simulate GBM data with optimized log likelihood
    # x0 = np.array([ 0.001, 0.0001])
    # params2 = optimize.minimize(GBM_calibration_ll, x0=x0, args=log_returns, method='Nelder-Mead')
    # print('parameters are', params2.x)
    # GBM_simulation(data, params2.x)

    fun = lambda x: (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2
    cons = ({'type': 'ineq', 'fun': lambda x: x[0] - 2 * x[1] + 2},
            {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
            {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
    bnds = ((0, None), (0, None))
    res = optimize.minimize(fun, (2, 0), method='SLSQP', bounds=bnds,
                   constraints=cons)

    x0 = np.array([0.1, 1.0, 0.1, 0.1, 0.1])
    returns = price2ret(data)
    bnds = ((None, None), (0, None), (0, None), (0, None), (None, None))
    cons = ({'type': 'ineq', 'fun': lambda x: 1 - x[2] - x[3]*(1+x[4]**2)})
    params3 = optimize.minimize(GARCH_calibration, x0=x0, args=returns, method="SLSQP", bounds=bnds, constraints=cons)
    print('parameters are', params3.x)
    NGARCH_simulation(10, len(data), params3.x, data['Price'].values[0], data)