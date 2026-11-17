import numpy as np

from dh import bs, hedge, paths

S0, K, T, r, sig = 100.0, 100.0, 1.0, 0.02, 0.2


def test_put_call_parity():
    c = bs.call_price(S0, K, T, r, sig)
    p = bs.put_price(S0, K, T, r, sig)
    assert np.isclose(c - p, S0 - K * np.exp(-r * T))


def test_monte_carlo_matches_black_scholes():
    P = paths.gbm(S0, r, sig, T, 1, 400_000, seed=1)
    disc = np.exp(-r * T) * np.maximum(P[:, -1] - K, 0)
    se = disc.std() / np.sqrt(len(disc))
    assert abs(disc.mean() - bs.call_price(S0, K, T, r, sig)) < 3 * se


def test_hedge_error_shrinks_with_rebalancing():
    stds = [hedge.metrics(hedge.run_hedge(paths.gbm(S0, r, sig, T, n, 20_000, seed=2), K, T, r, sig)["pnl"])["pnl_std"]
            for n in (25, 100, 400)]
    assert stds[0] > stds[1] > stds[2]
    assert stds[1] / stds[2] > 1.6  # error ~ 1/sqrt(n): 4x steps -> ~2x smaller


def test_costs_reduce_mean_pnl():
    P = paths.gbm(S0, r, sig, T, 50, 20_000, seed=3)
    a = hedge.run_hedge(P, K, T, r, sig, 0)["pnl"].mean()
    b = hedge.run_hedge(P, K, T, r, sig, 10)["pnl"].mean()
    assert b < a


def test_jump_paths_have_fatter_tails_and_worse_hedging():
    g = paths.gbm(S0, r, sig, T, 50, 20_000, seed=4)
    j = paths.jump_stress(S0, r, sig, T, 50, 20_000, seed=4)
    lr = lambda p: np.diff(np.log(p), axis=1).ravel()
    kurt = lambda x: ((x - x.mean())**4).mean() / x.var()**2
    assert kurt(lr(j)) > kurt(lr(g)) + 1
    cv = lambda p: hedge.metrics(hedge.run_hedge(p, K, T, r, sig)["pnl"])["CVaR95"]
    assert cv(j) > cv(g)
