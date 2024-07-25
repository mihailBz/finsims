import matplotlib.pyplot as plt
import numpy as np


def simulate_gbm(mu, sigma, S0, n, M, T):
    # calc each time step
    dt = T / n
    # simulation using numpy arrays
    St = np.exp(
        (mu - sigma**2 / 2) * dt
        + sigma * np.random.normal(0, np.sqrt(dt), size=(M, n)).T
    )
    # include array of 1's
    St = np.vstack([np.ones(M), St])
    # multiply through by S0 and return the cumulative product of elements along a given simulation path (axis=0).
    St = S0 * St.cumprod(axis=0)
    return St


def plot_gbm(St, n, M, T, S0, mu, sigma):
    time = np.linspace(0, T, n + 1)
    tt = np.full(shape=(M, n + 1), fill_value=time).T
    plt.plot(tt, St)
    plt.xlabel("Years $(t)$")
    plt.ylabel("Stock Price $(S_t)$")
    plt.title(
        "Realizations of Geometric Brownian Motion\n $dS_t = \mu S_t dt + \sigma S_t dW_t$\n $S_0 = {0}, \mu = {1}, \sigma = {2}$".format(
            S0, mu, sigma
        )
    )
    plt.show()


def estimate_sigma(series, dt):
    return np.sqrt((np.diff(series) ** 2).sum() / (len(series) * dt))


def estimate_log_mu(series, dt):
    return (series[-1] - series[0]) / (len(series) * dt)


def estimate_mu(log_mu, sigma):
    return log_mu + 0.5 * sigma**2


def estimate_parameters(series, T, n, ret_distribution=False):
    log_st = np.log(series).T
    estimated_mus = []
    estimated_sigmas = []
    for i, path in enumerate(log_st):
        estimated_sigma = estimate_sigma(path, dt=T / n)
        log_mu = estimate_log_mu(path, dt=T / n)
        estimated_mu = estimate_mu(log_mu, estimated_sigma)
        estimated_mus.append(estimated_mu)
        estimated_sigmas.append(estimated_sigma)
    estimated_sigmas = np.array(estimated_sigmas)
    estimated_mus = np.array(estimated_mus)
    if ret_distribution:
        return estimated_sigmas, estimated_mus
    else:
        return estimated_sigmas.mean(), estimated_mus.mean()
