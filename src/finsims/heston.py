import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from gbm import estimate_parameters
from py_vollib_vectorized import vectorized_implied_volatility as implied_vol


def main():
    # Parameters
    # simulation dependent
    S0 = 100.0  # asset price
    T = 1.0  # time in years
    r = 0.02  # risk-free rate
    N = 252  # number of time steps in simulation
    M = 1000  # number of simulations
    # Heston dependent parameters
    kappa = 3  # rate of mean reversion of variance under risk-neutral dynamics
    theta = 0.20**2  # long-term mean of variance under risk-neutral dynamics
    v0 = 0.25**2  # initial variance under risk-neutral dynamics
    rho = 0.7  # correlation between returns and variances under risk-neutral dynamics
    sigma = 0.6  # volatility of volatility

    rho_p = 0.98
    rho_n = -0.98
    S_p, v_p = heston_model_sim(S0, v0, rho_p, kappa, theta, sigma, T, N, M, r)
    S_n, v_n = heston_model_sim(S0, v0, rho_n, kappa, theta, sigma, T, N, M, r)

    plot_heston(N, S_p, T, v_p)
    print(theta, v0)
    print(estimate_parameters(S_p, T, N))


def plot_heston(N, S_p, T, v_p):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    time = np.linspace(0, T, N + 1)
    ax1.plot(time, S_p)
    ax1.set_title("Heston Model Asset Prices")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Asset Prices")
    ax2.plot(time, v_p)
    ax2.set_title("Heston Model Variance Process")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Variance")
    plt.show()


def heston_model_sim(S0, v0, rho, kappa, theta, sigma, T, N, M, r):
    """
    Inputs:
     - S0, v0: initial parameters for asset and variance
     - rho   : correlation between asset returns and variance
     - kappa : rate of mean reversion in variance process
     - theta : long-term mean of variance process
     - sigma : vol of vol / volatility of variance process
     - T     : time of simulation
     - N     : number of time steps
     - M     : number of scenarios / simulations

    Outputs:
    - asset prices over time (numpy array)
    - variance over time (numpy array)
    """
    # initialise other parameters
    dt = T / N
    mu = np.array([0, 0])
    cov = np.array([[1, rho], [rho, 1]])
    # arrays for storing prices and variances
    S = np.full(shape=(N + 1, M), fill_value=S0)
    v = np.full(shape=(N + 1, M), fill_value=v0)
    # sampling correlated brownian motions under risk-neutral measure
    Z = np.random.multivariate_normal(mu, cov, (N, M))
    for i in range(1, N + 1):
        S[i] = S[i - 1] * np.exp(
            (r - 0.5 * v[i - 1]) * dt + np.sqrt(v[i - 1] * dt) * Z[i - 1, :, 0]
        )
        v[i] = np.maximum(
            v[i - 1]
            + kappa * (theta - v[i - 1]) * dt
            + sigma * np.sqrt(v[i - 1] * dt) * Z[i - 1, :, 1],
            0,
        )

    return S, v


if __name__ == "__main__":
    main()
