"""Run every experiment and write results/summary.csv plus plots.

    python -m dh.run            # full run (10,000 paths per scenario)
    python -m dh.run --quick    # smaller, for a fast check
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import hedge, neural, paths

S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.02, 0.2


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=10_000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    n_paths = 2000 if a.quick else a.paths
    it = 60 if a.quick else 400
    out = Path(a.out); out.mkdir(exist_ok=True)
    t0 = time.time()
    rows = []

    # 1) Black-Scholes delta hedging: rebalancing frequency x costs x scenario
    for steps in (50, 100, 250):
        scen = {"GBM": paths.gbm(S0, R, SIGMA, T, steps, n_paths, seed=11),
                "jump_stress": paths.jump_stress(S0, R, SIGMA, T, steps, n_paths, seed=11)}
        for name, P in scen.items():
            for bps in (0, 5, 10):
                m = hedge.metrics(hedge.run_hedge(P, K, T, R, SIGMA, bps)["pnl"])
                rows.append({"hedger": "BS_delta", "scenario": name, "rebalances": steps, "cost_bps": bps, **m})
    print(f"[bs] done ({time.time() - t0:.0f}s)")

    # 2) Neural hedger (trained on GBM, 50 steps, 5 bps) and adversarial paths
    steps, bps = 50, 5
    sampler = lambda b, s: paths.gbm(S0, R, SIGMA, T, steps, b, seed=s)
    model = neural.train_hedger(sampler, K, T, R, SIGMA, bps, iters=it)
    print(f"[neural hedger] trained ({time.time() - t0:.0f}s)")
    adv = neural.train_adversary(S0, K, T, R, SIGMA, steps, bps, iters=it)
    print(f"[adversary] trained ({time.time() - t0:.0f}s)")
    scen = {"GBM": paths.gbm(S0, R, SIGMA, T, steps, n_paths, seed=99),
            "jump_stress": paths.jump_stress(S0, R, SIGMA, T, steps, n_paths, seed=99),
            "adversarial": neural.adversarial_paths(adv, S0, n_paths, steps, seed=99)}
    for name, P in scen.items():
        for hname, pos in {"BS_delta": None, "neural": neural.hedger_positions(model, P)}.items():
            m = hedge.metrics(hedge.run_hedge(P, K, T, R, SIGMA, bps, positions=pos)["pnl"])
            rows.append({"hedger": hname, "scenario": name, "rebalances": steps, "cost_bps": bps,
                         "head_to_head": True, **m})

    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    h2h = df[df.get("head_to_head", False) == True].drop(columns="head_to_head")
    h2h.to_csv(out / "head_to_head.csv", index=False)
    print(h2h.round(3).to_string(index=False))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bsr = df[(df.hedger == "BS_delta") & (df.cost_bps == 0) & df.head_to_head.isna()]
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, g in bsr.groupby("scenario"):
        ax.plot(g.rebalances, g.pnl_std, marker="o", label=name)
    ax.set_xscale("log"); ax.set_xlabel("rebalances"); ax.set_ylabel("P&L std"); ax.legend()
    ax.set_title("Hedging error vs rebalancing frequency (no costs)")
    fig.tight_layout(); fig.savefig(out / "error_vs_rebalancing.png", dpi=130)
    print(f"[done] {out}/ ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
