"""PyTorch deep hedger and adversarial path generator.

Deep hedger (Buehler et al., 2019): an MLP maps (time to maturity,
log-moneyness, current position) to the next hedge ratio and is trained to
minimise CVaR_95 of the hedging loss *including transaction costs*, using
the Rockafellar-Uryasev form CVaR = min_w  w + E[(loss - w)+] / (1 - alpha).

Adversary: an MLP maps (time to maturity, log-moneyness) to a volatility
multiplier in [0.5, 1.5] and a drift shift, rescaling GBM increments to
MAXIMISE the Black-Scholes hedger's CVaR. A penalty on deviations from the
base model keeps the paths plausible.
"""
import numpy as np
import torch
from torch import nn

from . import bs


def _mlp(n_in, n_out, width=32):
    return nn.Sequential(nn.Linear(n_in, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, n_out))


def _torch_bs_delta(S, K, tau, r, sigma):
    tau = torch.clamp(tau, min=1e-6)
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * torch.sqrt(tau))
    return 0.5 * (1 + torch.erf(d1 / np.sqrt(2)))


def pnl_torch(S, positions, K, T, r, premium, cost):
    """Differentiable version of hedge.run_hedge. S: (B, n+1), positions: (B, n)."""
    n = positions.shape[1]
    dt = T / n
    growth = torch.exp(torch.tensor(r * dt))
    cash = torch.full((S.shape[0],), float(premium))
    prev = torch.zeros(S.shape[0])
    for t in range(n):
        trade = positions[:, t] - prev
        cash = (cash - trade * S[:, t] - cost * trade.abs() * S[:, t]) * growth
        prev = positions[:, t]
    ST = S[:, -1]
    return cash + prev * ST - cost * prev.abs() * ST - torch.relu(ST - K)


def cvar(loss, w, alpha=0.95):
    return w + torch.relu(loss - w).mean() / (1 - alpha)


class DeepHedger(nn.Module):
    def __init__(self, K, T):
        super().__init__()
        self.K, self.T = K, T
        self.net = _mlp(3, 1)

    def forward(self, S):
        n = S.shape[1] - 1
        pos, prev = [], torch.zeros(S.shape[0])
        for t in range(n):
            tau = torch.full_like(prev, self.T - t * self.T / n)
            x = torch.stack([tau, torch.log(S[:, t] / self.K), prev], 1)
            prev = torch.sigmoid(self.net(x)).squeeze(1) * 1.2 - 0.1  # positions in [-0.1, 1.1]
            pos.append(prev)
        return torch.stack(pos, 1)


def train_hedger(sample_paths, K, T, r, sigma, cost_bps, iters=400, batch=4096, seed=0):
    torch.manual_seed(seed)
    model = DeepHedger(K, T)
    w = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam(list(model.parameters()) + [w], lr=3e-3)
    premium = bs.call_price(sample_paths(1, 0)[0, 0], K, T, r, sigma)
    for i in range(iters):
        S = torch.tensor(sample_paths(batch, 1000 + i), dtype=torch.float32)
        loss = -pnl_torch(S, model(S), K, T, r, premium, cost_bps / 1e4)
        obj = cvar(loss, w)
        opt.zero_grad(); obj.backward(); opt.step()
    return model


def hedger_positions(model, paths):
    with torch.no_grad():
        return model(torch.tensor(paths, dtype=torch.float32)).numpy().astype(float)


class Adversary(nn.Module):
    def __init__(self, K, T, sigma, mu):
        super().__init__()
        self.K, self.T, self.sigma, self.mu = K, T, sigma, mu
        self.net = _mlp(2, 2, 16)

    def forward(self, S0, z):
        B, n = z.shape
        dt = self.T / n
        S = [torch.full((B,), float(S0))]
        devs = []
        for t in range(n):
            tau = torch.full((B,), self.T - t * dt)
            out = torch.tanh(self.net(torch.stack([tau, torch.log(S[-1] / self.K)], 1)))
            vol = self.sigma * (1 + 0.5 * out[:, 0])
            drift = self.mu + 0.2 * out[:, 1]
            S.append(S[-1] * torch.exp((drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z[:, t]))
            devs.append(out)
        return torch.stack(S, 1), torch.stack(devs, 1)


def train_adversary(S0, K, T, r, sigma, n_steps, cost_bps, iters=300, batch=4096, penalty=2.0, seed=0):
    torch.manual_seed(seed)
    adv = Adversary(K, T, sigma, r)
    w = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam(list(adv.parameters()) + [w], lr=3e-3)
    premium = bs.call_price(S0, K, T, r, sigma)
    times = torch.arange(n_steps) * T / n_steps
    for _ in range(iters):
        S, dev = adv(S0, torch.randn(batch, n_steps))
        pos = _torch_bs_delta(S[:, :-1], K, T - times, r, sigma)  # victim: BS delta hedger
        loss = -pnl_torch(S, pos, K, T, r, premium, cost_bps / 1e4)
        # w minimises the CVaR bound; the adversary maximises it minus a plausibility penalty
        obj_w = cvar(loss.detach(), w)
        obj_adv = -cvar(loss, w.detach()) + penalty * (dev**2).mean()
        opt.zero_grad(); (obj_w + obj_adv).backward(); opt.step()
    return adv


def adversarial_paths(adv, S0, n_paths, n_steps, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        S, _ = adv(S0, torch.randn(n_paths, n_steps, generator=g))
    return S.numpy().astype(float)
