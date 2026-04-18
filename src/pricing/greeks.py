import numpy as np
from scipy.stats import norm
from src.pricing.black_scholes import d1, d2


def call_delta(S, K, T, r, sigma):
    if T == 0:
        return 1.0 if S > K else 0.0
    return norm.cdf(d1(S, K, T, r, sigma))


def put_delta(S, K, T, r, sigma):
    if T == 0:
        return -1.0 if S < K else 0.0
    return norm.cdf(d1(S, K, T, r, sigma)) - 1


def gamma(S, K, T, r, sigma):
    if T == 0 or sigma == 0:
        return 0.0
    d_1 = d1(S, K, T, r, sigma)
    return norm.pdf(d_1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    if T == 0:
        return 0.0
    d_1 = d1(S, K, T, r, sigma)
    return S * norm.pdf(d_1) * np.sqrt(T)


def theta_call(S, K, T, r, sigma):
    if T == 0:
        return 0.0
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    first_term = -(S * norm.pdf(d_1) * sigma) / (2 * np.sqrt(T))
    second_term = -r * K * np.exp(-r * T) * norm.cdf(d_2)
    return first_term + second_term