from src.baselines.no_hedge import no_hedge_pnl
from src.baselines.delta_hedge import delta_hedge_pnl


if __name__ == "__main__":
    pnl_no_hedge, path_no_hedge = no_hedge_pnl()
    pnl_delta_hedge, path_delta, tc_cost = delta_hedge_pnl()

    print("No-hedge PnL:", round(pnl_no_hedge, 4))
    print("Delta-hedge PnL:", round(pnl_delta_hedge, 4))
    print("Delta-hedge total transaction cost:", round(tc_cost, 4))