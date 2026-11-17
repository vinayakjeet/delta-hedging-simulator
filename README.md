# Delta Hedging & Adversarial Stock Path Simulator

This project sells a European call, hedges it in discrete time, and measures
how the hedge fails. It covers three kinds of market (normal GBM, jump and
volatility-regime stress, and paths a neural adversary builds to hurt the
hedger) and compares two hedgers: the classic Black-Scholes delta and a
PyTorch deep hedger trained to minimise CVaR.

The setup is an at-the-money call with S0 = K = 100, 1-year maturity, r = 2%
and σ = 20%. Every scenario uses 10,000 paths.

## Results (from `python -m dh.run`)

### 1. Black-Scholes delta hedging: P&L std vs rebalancing frequency

| Scenario | Cost (bps) | 50 rebalances | 100 | 250 |
|---|---|---|---|---|
| GBM | 0 | 0.962 | 0.692 | 0.443 |
| GBM | 10 | 0.977 | 0.714 | 0.492 |
| Jump stress | 0 | 4.351 | 4.155 | 3.836 |
| Jump stress | 10 | 4.419 | 4.266 | 4.036 |

- **Under GBM** the error falls roughly like 1/√n, as the theory predicts.
  Going from 50 to 250 rebalances cuts P&L std by 54%.
- **Under jumps** extra rebalancing barely helps. A jump is a gap that no
  continuous hedge can follow, and the error is about 4–9 times larger.
- **Costs scale with trading frequency.** At 10 bps the mean P&L is −0.33 with
  50 rebalances but −0.63 with 250. More rebalancing lowers the error but
  costs more, which is the core trade-off of discrete hedging.

### 2. Head to head: BS delta vs neural hedger (50 rebalances, 5 bps)

| Scenario | Hedger | Mean P&L | P&L std | CVaR 95% | Worst loss |
|---|---|---|---|---|---|
| GBM | BS delta | −0.170 | 0.993 | 2.510 | 8.18 |
| GBM | Neural | −0.154 | 1.142 | **2.456** | 5.63 |
| Jump stress | BS delta | −4.580 | 4.267 | 16.452 | 33.31 |
| Jump stress | Neural | −4.578 | 4.107 | **15.700** | 27.85 |
| Adversarial | BS delta | −5.234 | 2.664 | 11.649 | 18.30 |
| Adversarial | Neural | −4.952 | 2.358 | **10.695** | 19.54 |

- **CVaR:** the neural hedger had the lower CVaR in all three scenarios, 2%
  lower on GBM, 5% on jump stress and 8% on adversarial paths. That is the
  number it was trained to minimise.
- **Not a clean win:** on GBM its P&L std is higher than delta hedging's,
  because it trades some variance for a thinner loss tail. On adversarial
  paths its single worst path is also worse.
- **The adversary works:** it raises the delta hedger's CVaR from 2.5 to 11.6
  (4.6×). It does this mostly by shifting volatility and drift where the
  hedge is most exposed, and a penalty keeps it from moving far from the base
  model.
- **Training scope:** the neural hedger was trained only on GBM paths, so the
  jump and adversarial results test how well it generalises out of sample.

## How it works

| File | What it does |
|---|---|
| `dh/paths.py` | GBM (exact log-normal steps). Jump stress: Merton jump-diffusion with compensated drift and a two-state volatility regime (σ ↔ 2σ). |
| `dh/bs.py` | Black-Scholes call and put prices, delta and gamma. |
| `dh/hedge.py` | Hedging engine: sell the call at BS price, rebalance n times, proportional costs, cash account at r, liquidate at maturity. Reports mean P&L, std, RMSE, VaR, CVaR and worst loss. |
| `dh/neural.py` | Deep hedger: MLP(τ, log S/K, current position) → hedge ratio, trained on CVaR₉₅ via the Rockafellar–Uryasev form with costs inside the loss. Adversary: MLP(τ, log S/K) → volatility multiplier in [0.5σ, 1.5σ] and drift shift, trained to maximise the BS hedger's CVaR minus a plausibility penalty. |
| `dh/run.py` | Runs every experiment and writes `results/summary.csv`, `results/head_to_head.csv` and a plot. |

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q tests      # put-call parity, MC vs BS price, error ~ 1/sqrt(n), costs, jump tails
python -m dh.run               # about 80 seconds on a laptop CPU
```

## Next steps

- Train the deep hedger on a mix of scenarios, then test it on held-out
  regimes.
- Add a gamma-aware no-trade band (Whalley–Wilmott) as a third hedger for the
  cost trade-off.
- Train the hedger and the adversary in alternation (a min-max game) and
  track how the CVaR gap changes.
