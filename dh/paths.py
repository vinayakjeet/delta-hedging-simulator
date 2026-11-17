"""Stock path generators. All return an array of shape (n_paths, n_steps + 1)."""
import numpy as np


def gbm(S0, mu, sigma, T, n_steps, n_paths, seed=0):
    """Geometric Brownian Motion, exact log-normal discretisation."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    logret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    return S0 * np.exp(np.concatenate([np.zeros((n_paths, 1)), np.cumsum(logret, 1)], 1))


def jump_stress(S0, mu, sigma, T, n_steps, n_paths, seed=0, lam=3.0, jump_mu=-0.05,
                jump_sigma=0.08, sigma_high=None, p_switch=0.02):
    """Stress paths: Merton jump-diffusion plus a two-state volatility regime.

    lam: expected jumps per year; jumps are log-normal(jump_mu, jump_sigma).
    The volatility switches between sigma and sigma_high (default 2*sigma)
    with probability p_switch per step. Drift is jump-compensated so the
    expected return stays mu.
    """
    rng = np.random.default_rng(seed)
    sigma_high = 2 * sigma if sigma_high is None else sigma_high
    dt = T / n_steps
    regime = np.zeros(n_paths, dtype=bool)
    k = np.exp(jump_mu + 0.5 * jump_sigma**2) - 1          # mean jump size
    logS = np.full(n_paths, np.log(S0))
    out = np.empty((n_paths, n_steps + 1)); out[:, 0] = S0
    for t in range(n_steps):
        regime ^= rng.random(n_paths) < p_switch
        vol = np.where(regime, sigma_high, sigma)
        n_jumps = rng.poisson(lam * dt, n_paths)
        jumps = n_jumps * jump_mu + np.sqrt(n_jumps) * jump_sigma * rng.standard_normal(n_paths)
        logS += (mu - lam * k - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * rng.standard_normal(n_paths) + jumps
        out[:, t + 1] = np.exp(logS)
    return out
