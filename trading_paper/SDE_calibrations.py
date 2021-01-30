import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})
from scipy import optimize

def price2ret(data):
    diff = data['Price'].values[1:] / data['Price'].values[:-1] - 1
    return diff

def GBM_calibration(data):
    log = np.log(data[1:]) - np.log(data[:-1])
    m_hat = (log[0] - log[-1])/len(log)
    v_hat = sum(np.square(log-m_hat))/len(log)
    return [m_hat, v_hat]

def GBM_calibration_ll(params, data):
    log = np.log(data[1:]) - np.log(data[:-1])
    mu = params[0]
    var_t = params[1]
    LogLikelihood = 0
    for time in range(0, len(log)):
        assert var_t >= 0, "negative variance error"
        innov = np.square(log[time] - mu)
        LogLikelihood += -np.log(var_t) - innov / var_t
    mll = -LogLikelihood
    return mll

def GBM_simulation(data, params, N_Sim):
    m_hat = params[0]
    v_hat = params[1]
    for num_plots in range(0,N_Sim):
        log_step = [np.log(data['Price'].values[0])]
        for i in range(0,len(data)):
            log_step += [m_hat + np.sqrt(v_hat) * np.random.normal(0,1)]
        log_step = np.array(log_step)
        log_vec = log_step.cumsum()
        simulation = np.exp(log_vec)
        plt.plot(simulation, color='blue')

    plt.plot(data['Price'], color='red')

    plt.show()

def GBM_simulation2(data, params, N_Sim):
    m_hat = params[0]
    v_hat = params[1]
    dt = 1
    sigma = np.sqrt(v_hat/dt)
    mu = m_hat/dt + 1/2*sigma**2
    # sigma = 0.2
    # mu = 0
    # print("mu, sigma: ", mu, sigma)
    for num_plots in range(0,N_Sim):
        s = [data['Price'].values[0]]
        s2 = [data['Price'].values[0]]
        tmp_exp = mu - 1/2 * sigma ** 2
        for i in range(1,len(data)):
            bm = np.random.normal(0, 1)
            s2 += [s2[i-1]*np.exp( m_hat + v_hat * bm )]
            s += [s[i-1]*np.exp(tmp_exp * dt + sigma * bm * np.sqrt(dt))]
        plt.plot(np.array(s), color='green')
        # plt.plot(np.array(s), color='blue')

    plt.plot(data['Price'], color='red')

    plt.show()

def GARCH_calibration(params, returns):

    mu = params[0]
    omega = params[1]
    alpha = params[2]
    beta = params[3]
    gamma = params[4]
    denum = 1-alpha-beta*(1+np.square(gamma))
    assert denum > 0, 'variance stationarity constraint breached '
    var_t = omega/denum
    LogLikelihood = 0
    for time in range(0,len(returns)):
        innov = returns[time] - mu
        var_t = omega + var_t * (alpha + beta * np.square(innov - gamma))
        # var_t = omega+alpha*var_t+beta*np.square(innov-gamma*np.sqrt(var_t))
        assert var_t >= 0, "negative variance error"
        LogLikelihood += -np.log(var_t) - np.square(innov) / var_t
    mll = -LogLikelihood
    return mll

def NL_constr(params):
    alpha = params[2]
    beta = params[3]
    gamma = params[4]

    return 1 - alpha - beta * (1 + np.square(gamma))

def NGARCH_simulation(data, params, N_Sim):
    T = len(data)
    S0 = data['Price'].values[0]
    mu = params[0]
    omega = params[1]
    alpha = params[2]
    beta = params[3]
    gamma = params[4]
    denum = 1-alpha-beta*(1+np.square(gamma))
    assert denum > 0, 'variance stationarity constraint breached '
    for num in range(0,N_Sim):
        var_t = omega / (1 - alpha - beta)
        S = [S0]
        for i in range(1,T):
            rdn = np.random.normal(0,1)
            var_t = omega + var_t * (alpha + beta * np.square(rdn - gamma))
            S += [S[i-1]*np.exp((mu-1/2*var_t)+np.sqrt(var_t)*rdn)]


        plt.plot(S, color='blue')
    plt.plot(data['Price'], color='red')
    plt.show()

def vasicek_calibation(price):
    n=len(price)-1
    b_num = n*np.dot(price[1:],price[:-1]) - sum(price[1:])*sum(price[:-1])
    b_denum = n*sum(np.square(price[:-1]))-np.square(sum(price[:-1]))
    b = b_num/b_denum
    c = (sum(price[1:]-b*price[:-1]))/n
    delta2 = 1/n*(sum(np.square(price[1:]-b*price[:-1] - c)))

    dt = 1
    alpha = -np.log(b)/dt
    theta = c / (1 - b)
    sigma = np.sqrt(2*np.log(b) * delta2 /dt/(b**2-1))
    return [b, c, delta2, alpha, theta, sigma]

def vasicek_simulation(data, params, N_sim):
    b = params[0]
    c = params[1]
    delta = np.sqrt(params[2])

    alpha = params[3]
    theta = params[4]
    sigma = params[5]

    dt = 1
    for num_plots in range(0,N_sim):
        S = [data['Price'].values[0]]
        P = [data['Price'].values[0]]
        for i in range(1,len(data)):
            bm = np.random.normal(0,1)
            S += [c + S[i-1] * b + delta*bm]

            dr = alpha * (theta - P[i-1] ) * dt + sigma * bm * dt
            P += [P[i-1] + dr]

        plt.plot(S, color='blue')

    plt.plot(data['Price'], color='red')

    plt.show()
    return S

if __name__ == '__main__':
    # Import data
    import os
    print(os.getcwd())
    path = (os.path.join(os.getcwd(), 'SP.CSV'))
    data = pd.read_csv(path)
    returns = price2ret(data)
    N_sim = 10

    # # calibrate and simulate GBM data with exact log likelihood
    # # log_returns = np.log(data['Price'].values[1:]) - np.log(data['Price'].values[:-1])
    # params1 = GBM_calibration(data['Price'].values)
    # print('parameters are', params1)
    # GBM_simulation(data, params1, N_sim)
    # GBM_simulation2(data, params1, N_sim)
    #
    # # calibrate and simulate GBM data with optimized log likelihood
    # x0 = np.array([ 0.001, 0.0001])
    # params2 = optimize.minimize(GBM_calibration_ll, x0=x0, args=data['Price'].values, method='Nelder-Mead')
    # print('parameters are', params2.x)
    # GBM_simulation(data, params2.x, N_sim)
    # GBM_simulation2(data, params2.x, N_sim)

    # # calibrate and simulate GBM with stochastic volatility
    # x0 = np.array([0.1, 1.0, 0.1, 0.1, 0.1])
    # # mu, omega, alpha, beta, gamma
    # bnds = ((None, None), (0, None), (0, None), (0, None), (0, None))
    # # cons = ({'type': 'ineq', 'fun': lambda x: 1 - x[2] - x[3]*(1+x[4]**2)})
    # cons = ({'type': 'ineq', 'fun': NL_constr})
    # params3 = optimize.minimize(GARCH_calibration, x0=x0, args=returns, method="SLSQP", constraints=cons, bounds = bnds)
    # print('parameters are', params3.x)
    # NGARCH_simulation(data, params3.x, N_sim)

    # calibrate and simulate Vasicek
    params4 = vasicek_calibation(data['Price'].values)
    vasicek_simulation(data, params4, N_sim)
    print(params4)