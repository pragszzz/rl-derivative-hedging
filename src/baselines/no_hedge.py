import numpy as np

from src.data.simulator import simulate_gbm_path
from src.pricing.black_scholes import call_price
from src.utils.config import (
    INITIAL_STOCK_PRICE,
    STRIKE_PRICE,
    RISK_FREE_RATE,
    VOLATILITY,
    MATURITY,
    N_STEPS,
)


def no_hedge_pnl():
    """
    Short 1 European call, never hedge with the stock.
    Return terminal PnL of the portfolio and the simulated path.
    """
    prices = simulate_gbm_path(
        S0=INITIAL_STOCK_PRICE,
        mu=RISK_FREE_RATE,
        sigma=VOLATILITY,
        T=MATURITY,
        n_steps=N_STEPS,
        random_seed=None,  # new random path each call
    )

    # Option is sold at t=0 for its Black–Scholes price
    option_premium = call_price(
        S=INITIAL_STOCK_PRICE,
        K=STRIKE_PRICE,
        T=MATURITY,
        r=RISK_FREE_RATE,
        sigma=VOLATILITY,
    )

    S_T = prices[-1]
    payoff = max(S_T - STRIKE_PRICE, 0.0)

    # Short call: initial cash = premium, terminal liability = payoff
    pnl = option_premium * np.exp(RISK_FREE_RATE * MATURITY) - payoff

    return float(pnl), prices