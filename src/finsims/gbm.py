import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct, idct


def cosine_transform(series):
    return dct(series, norm="ortho")


def inverse_cosine_transform(series):
    return idct(series, norm="ortho")


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


def estimate_mu(log_mu, sigma):
    return log_mu + 0.5 * sigma**2


def estimate_parameters(series, dt, ret_distribution=False):
    log_st = np.log(series).T
    estimated_mus = []
    estimated_sigmas = []
    for i, path in enumerate(log_st):
        estimated_sigma = estimate_sigma(path, dt=dt)
        log_mu = estimate_log_mu(path, dt=dt)
        estimated_mu = estimate_mu(log_mu, estimated_sigma)
        estimated_mus.append(estimated_mu)
        estimated_sigmas.append(estimated_sigma)
    estimated_sigmas = np.array(estimated_sigmas)
    estimated_mus = np.array(estimated_mus)
    if ret_distribution:
        return estimated_sigmas, estimated_mus
    else:
        return estimated_sigmas.mean(), estimated_mus.mean()


def main():
    mu = 0.05
    sigma = 0.3
    n = 100
    M = 5
    # T = 1
    dt = 1 / n
    St = simulate_gbm(mu, sigma, n, M, dt)
    print(St[0, :])
    plot_gbm(St, n, M, dt, mu, sigma)
    estimated_sigma, estimated_mu = estimate_parameters(St, dt)
    print(f"Estimated sigma: {estimated_sigma}, Estimated mu: {estimated_mu}")

    transformed = dct(St, norm="ortho")
    plot_gbm(transformed, n, M, dt, mu, sigma)
    recovered = idct(transformed, norm="ortho")
    plot_gbm(recovered, n, M, dt, mu, sigma)

    estimated_sigma, estimated_mu = estimate_parameters(recovered, dt)
    print(f"Estimated sigma: {estimated_sigma}, Estimated mu: {estimated_mu}")


if __name__ == "__main__":
    main()
