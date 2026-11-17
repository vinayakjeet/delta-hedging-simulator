"""Black-Scholes pricing and Greeks for European options."""
import numpy as np
from scipy.stats import norm


def d1(S, K, tau, r, sigma):
    tau = np.maximum(tau, 1e-12)
    return (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))


def call_price(S, K, tau, r, sigma):
    a = d1(S, K, tau, r, sigma)
    b = a - sigma * np.sqrt(np.maximum(tau, 1e-12))
    return S * norm.cdf(a) - K * np.exp(-r * tau) * norm.cdf(b)


def put_price(S, K, tau, r, sigma):
    return call_price(S, K, tau, r, sigma) - S + K * np.exp(-r * tau)


def call_delta(S, K, tau, r, sigma):
    return norm.cdf(d1(S, K, tau, r, sigma))


def gamma(S, K, tau, r, sigma):
    tau = np.maximum(tau, 1e-12)
    return norm.pdf(d1(S, K, tau, r, sigma)) / (S * sigma * np.sqrt(tau))
