import numpy as np

from src.data.simulator import simulate_gbm_path
from src.pricing.black_scholes import call_price
from src.pricing.greeks import call_delta
from src.utils.config import (
    INITIAL_STOCK_PRICE,
    STRIKE_PRICE,
    RISK_FREE_RATE,
    VOLATILITY,
    MATURITY,
    N_STEPS,
    TRANSACTION_COST,
)


def delta_hedge_pnl():
    """
    Short 1 European call, hedge dynamically using Black–Scholes delta.
    Action: hold delta_t shares between steps t and t+1.
    Transaction cost is proportional to traded notional.
    Return terminal PnL, price path, and total transaction cost.
    """
    prices = simulate_gbm_path(
        S0=INITIAL_STOCK_PRICE,
        mu=RISK_FREE_RATE,
        sigma=VOLATILITY,
        T=MATURITY,
        n_steps=N_STEPS,
        random_seed=None,  # new random path each call
    )

    dt = MATURITY / N_STEPS

    # Option premium at t=0
    option_premium = call_price(
        S=INITIAL_STOCK_PRICE,
        K=STRIKE_PRICE,
        T=MATURITY,
        r=RISK_FREE_RATE,
        sigma=VOLATILITY,
    )

    # Start with cash from selling the option
    cash = option_premium
    stock_position = 0.0
    total_tc = 0.0

    # Rebalance delta hedge at each step
    for t in range(N_STEPS):
        S_t = prices[t]
        time_to_maturity = max(MATURITY - t * dt, 1e-8)

        # Target delta for short call position = -delta_call
        target_delta = -call_delta(
            S=S_t,
            K=STRIKE_PRICE,
            T=time_to_maturity,
            r=RISK_FREE_RATE,
            sigma=VOLATILITY,
        )

        trade = target_delta - stock_position  # shares to buy (>0) or sell (<0)

        # Transaction cost proportional to traded notional
        tc = TRANSACTION_COST * abs(trade) * S_t
        total_tc += tc

        # Update cash and position: buy/sell stock plus pay cost
        cash -= trade * S_t + tc
        stock_position = target_delta

    # At maturity: close stock position, pay option payoff
    S_T = prices[-1]
    payoff = max(S_T - STRIKE_PRICE, 0.0)

    # Close hedge
    cash += stock_position * S_T
    stock_position = 0.0

    # Short call liability
    cash -= payoff

    # Grow initial cash at risk‑free rate to be consistent with no‑hedge baseline
    # (optional; if you don’t do this above, remove it here for consistency)
    pnl = cash * np.exp(RISK_FREE_RATE * dt)  # tiny correction; or just pnl = cash

    return float(pnl), prices, float(total_tc)