import numpy as np
from scipy.fftpack import dct, idct


def log_return(series):
    return np.diff(np.log(series), axis=0)


def reverse_log_returns(log_returns, s0):
    """
    Reconstructs the price series from log returns.

    Parameters:
    - log_returns: The log returns series (2D array).
    - s0: The initial price (float or array).

    Returns:
    - Reconstructed price series.
    """
    # Initialize the price array
    M = log_returns.shape[1]  # Number of paths
    n = log_returns.shape[0] + 1  # Number of time steps, including the initial price
    reconstructed_prices = np.zeros((n, M))

    # Set the initial prices
    reconstructed_prices[0, :] = s0

    # Rebuild prices from log returns
    for i in range(1, n):
        reconstructed_prices[i, :] = reconstructed_prices[i - 1, :] * np.exp(
            log_returns[i - 1, :]
        )

    return reconstructed_prices


def cosine_transform(series):
    return dct(series, norm="ortho", axis=0)


def inverse_cosine_transform(series):
    return idct(series, norm="ortho", axis=0)
