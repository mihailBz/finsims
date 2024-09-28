import numpy as np


def simulate_merton_jump_diffusion(mu, sigma, lamb, mu_j, sigma_j, n, M, dt, s0=None):
    """
    Simulate the Merton Jump Diffusion model.

    Parameters:
    - mu: Drift term of the diffusion component.
    - sigma: Volatility of the diffusion component.
    - lamb: Jump intensity (average number of jumps per unit time).
    - mu_j: Mean of the jump size distribution (log-normal jumps).
    - sigma_j: Standard deviation of the jump size distribution.
    - n: Number of time steps.
    - M: Number of simulation paths.
    - dt: Time step size.
    - s0: Initial asset price (defaults to 1 if None).

    Returns:
    - St: Simulated asset prices (array of shape (n+1, M)).
    """
    if s0 is None:
        s0 = 1

    # Precompute constants
    k = np.exp(mu_j + 0.5 * sigma_j**2) - 1  # Expected jump size
    drift = (mu - lamb * k - 0.5 * sigma**2) * dt

    # Initialize asset prices array
    St = np.zeros((n + 1, M))
    St[0] = s0

    for i in range(1, n + 1):
        # Simulate Poisson jumps
        N_t = np.random.poisson(lamb * dt, M)
        # Simulate jump sizes
        J_t = np.random.normal(mu_j, sigma_j, (M,)) * N_t
        # Simulate diffusion component
        diffusion = drift + sigma * np.sqrt(dt) * np.random.normal(0, 1, M)
        # Update asset prices
        St[i] = St[i - 1] * np.exp(diffusion + J_t)

    return St


def estimate_mjd_parameters(series, dt, threshold=3):
    """
    Estimate parameters of the Merton Jump Diffusion model from a price series.

    Parameters:
    - series: Time series of asset prices.
    - dt: Time step size.
    - threshold: Threshold (in standard deviations) to identify jumps.

    Returns:
    - params: Dictionary containing estimated parameters.
    """
    # Calculate log returns
    log_returns = np.diff(np.log(series))
    n = len(log_returns)

    # Calculate statistics of log returns
    mu_hat = np.mean(log_returns)
    sigma_hat = np.std(log_returns)

    # Identify jumps
    jump_indices = np.where(np.abs(log_returns - mu_hat) > threshold * sigma_hat)[0]
    no_jump_indices = np.where(np.abs(log_returns - mu_hat) <= threshold * sigma_hat)[0]

    # Estimate jump intensity (lambda)
    lamb_hat = len(jump_indices) / (n * dt)

    # Estimate jump sizes
    if len(jump_indices) > 0:
        jump_sizes = log_returns[jump_indices]
        mu_j_hat = np.mean(jump_sizes)
        sigma_j_hat = np.std(jump_sizes)
    else:
        mu_j_hat = 0
        sigma_j_hat = 0

    # Adjusted drift estimation
    # Remove jumps to estimate diffusion component
    diffusion_returns = log_returns[no_jump_indices]
    mu_diffusion_hat = np.mean(diffusion_returns) / dt
    sigma_diffusion_hat = np.std(diffusion_returns) / np.sqrt(dt)

    # Adjust drift for jump component
    k_hat = np.exp(mu_j_hat + 0.5 * sigma_j_hat**2) - 1
    mu_hat_adj = mu_diffusion_hat + lamb_hat * k_hat

    params = {
        "mu": mu_hat_adj,
        "sigma": sigma_diffusion_hat,
        "lamb": lamb_hat,
        "mu_j": None if len(jump_indices) == 0 else mu_j_hat,
        "sigma_j": None if len(jump_indices) <= 1 else sigma_j_hat,
    }

    return params
