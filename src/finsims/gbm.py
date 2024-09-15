import matplotlib.pyplot as plt
import numpy as np


def simulate_gbm(mu, sigma, n, M, dt, s0=None):
    St = np.exp(
        (mu - (sigma**2) / 2) * dt
        + sigma * np.sqrt(dt) * np.random.normal(0, 1, size=(M, n)).T
    )
    if s0 is not None:
        St = np.vstack([np.ones(M), St])
        St = s0 * St.cumprod(axis=0)
    else:
        St = St.cumprod(axis=0)

    return St


def plot_gbm(St, n, M, dt, mu, sigma, s0=None):
    time = np.linspace(0, n * dt, St.shape[0])
    tt = np.full(shape=(M, St.shape[0]), fill_value=time).T
    plt.plot(tt, St)
    plt.xlabel("Years $(t)$")
    plt.ylabel("Stock Price $(S_t)$")
    plt.title(
        f"Realizations of Geometric Brownian Motion\n"
        f"$dS_t = \\mu S_t dt + \\sigma S_t dW_t$; "
        f"$S_0 = {s0}, \\mu = {mu}, \\sigma = {sigma}$"
    )
    plt.show()


def estimate_sigma(series, dt):
    return np.sqrt((np.diff(series) ** 2).sum() / (len(series) * dt))


def estimate_log_mu(series, dt):
    return (series[-1] - series[0]) / (len(series) * dt)


def estimate_mle_log_mu(series, dt):
    tt = np.linspace(0, len(series) * dt, len(series))
    total = (1.0 / dt) * (tt**2).sum()
    return 1 / total * (1.0 / dt) * (tt * series).sum()


def estimate_mu(log_mu, sigma):
    return log_mu + 0.5 * sigma**2


def estimate_parameters(series, dt, ret_distribution=False, mle_estimator=False):
    log_st = np.log(series).T
    estimated_mus = []
    estimated_sigmas = []
    log_mu_estimator = estimate_mle_log_mu if mle_estimator else estimate_log_mu
    for i, path in enumerate(log_st):
        estimated_sigma = estimate_sigma(path, dt=dt)
        log_mu = log_mu_estimator(path, dt=dt)
        estimated_mu = estimate_mu(log_mu, estimated_sigma)
        estimated_mus.append(estimated_mu)
        estimated_sigmas.append(estimated_sigma)
    estimated_sigmas = np.array(estimated_sigmas)
    estimated_mus = np.array(estimated_mus)
    if ret_distribution:
        return estimated_sigmas, estimated_mus
    else:
        return estimated_sigmas.mean(), estimated_mus.mean()
