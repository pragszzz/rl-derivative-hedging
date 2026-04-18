import numpy as np
from scipy.stats import norm


def _validate_inputs(S, K, T, r, sigma):
    if S <= 0:
        raise ValueError("Stock price S must be positive.")
    if K <= 0:
        raise ValueError("Strike price K must be positive.")
    if T < 0:
        raise ValueError("Time to maturity T cannot be negative.")
    if sigma < 0:
        raise ValueError("Volatility sigma cannot be negative.")


def d1(S, K, T, r, sigma):
    _validate_inputs(S, K, T, r, sigma)
    if T == 0 or sigma == 0:
        return np.inf if S > K else -np.inf
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S, K, T, r, sigma):
    if T == 0 or sigma == 0:
        return np.inf if S > K else -np.inf
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def call_price(S, K, T, r, sigma):
    _validate_inputs(S, K, T, r, sigma)
    if T == 0:
        return max(S - K, 0.0)
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    return S * norm.cdf(d_1) - K * np.exp(-r * T) * norm.cdf(d_2)


def put_price(S, K, T, r, sigma):
    _validate_inputs(S, K, T, r, sigma)
    if T == 0:
        return max(K - S, 0.0)
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d_2) - S * norm.cdf(-d_1)