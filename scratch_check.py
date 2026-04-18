from src.baselines.no_hedge import no_hedge_pnl
from src.agents.evaluate_ppo import evaluate_ppo_single_path

print("No-hedge PnL samples:")
nh_vals = [no_hedge_pnl()[0] for _ in range(10)]
print(nh_vals)

print("\nPPO PnL samples:")
ppo_vals = [evaluate_ppo_single_path()[0] for _ in range(10)]
print(ppo_vals)