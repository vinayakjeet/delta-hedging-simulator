"""Discrete-time hedging engine for a short European call.

The hedger sells one call for its Black-Scholes price, then at each of
n_steps rebalancing dates holds `position` shares, financing everything in a
cash account at rate r. Trading costs are cost_bps * |change in shares| * S.
At maturity the shares are sold and the option payoff is paid.
Terminal P&L = cash + shares * S_T - payoff (0 for a perfect hedge).
"""
import numpy as np

from . import bs


def run_hedge(paths, K, T, r, sigma, cost_bps=0.0, positions=None):
    """paths: (n_paths, n_steps+1). positions: optional (n_paths, n_steps)
    hedge ratios; defaults to Black-Scholes delta with volatility `sigma`."""
    n_paths, n1 = paths.shape
    n_steps = n1 - 1
    dt = T / n_steps
    times = np.arange(n_steps) * dt
    if positions is None:
        positions = bs.call_delta(paths[:, :-1], K, T - times, r, sigma)
    c = cost_bps / 1e4
    premium = bs.call_price(paths[0, 0], K, T, r, sigma)
    cash = np.full(n_paths, premium)
    prev = np.zeros(n_paths)
    costs = np.zeros(n_paths)
    for t in range(n_steps):
        trade = positions[:, t] - prev
        tc = c * np.abs(trade) * paths[:, t]
        cash = (cash - trade * paths[:, t] - tc) * np.exp(r * dt)
        costs += tc
        prev = positions[:, t]
    S_T = paths[:, -1]
    liquidation = c * np.abs(prev) * S_T
    payoff = np.maximum(S_T - K, 0.0)
    pnl = cash + prev * S_T - liquidation - payoff
    return {"pnl": pnl, "costs": costs + liquidation, "premium": premium}


def metrics(pnl, alpha=0.95):
    loss = -pnl
    var = np.quantile(loss, alpha)
    return {
        "mean_pnl": pnl.mean(),
        "pnl_std": pnl.std(),
        "hedging_rmse": np.sqrt((pnl**2).mean()),
        "VaR95": var,
        "CVaR95": loss[loss >= var].mean(),
        "worst": loss.max(),
    }
