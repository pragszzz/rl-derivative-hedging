import csv
import numpy as np
from pathlib import Path

from src.baselines.no_hedge import no_hedge_pnl
from src.baselines.delta_hedge import delta_hedge_pnl
from src.agents.evaluate_ppo import evaluate_ppo_single_path


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
CSV_PATH = RESULTS_DIR / "strategy_comparison_paths.csv"


def main():
    n_runs = 200  # increase for smoother histograms

    rows = []

    for _ in range(n_runs):
        # No hedge
        nh_pnl, _ = no_hedge_pnl()
        rows.append(["no_hedge", nh_pnl, 0.0])

        # Delta hedge
        dh_pnl, _, dh_tc = delta_hedge_pnl()
        rows.append(["delta_hedge", dh_pnl, dh_tc])

        # PPO hedge (evaluate on a single new path)
        ppo_pnl, ppo_tc = evaluate_ppo_single_path()
        rows.append(["ppo_hedge", ppo_pnl, ppo_tc])

    # Write CSV
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "pnl", "transaction_cost"])
        writer.writerows(rows)

    print(f"Saved path-level results to {CSV_PATH}")

    # Aggregate stats
    stats = {}
    for strat in ["no_hedge", "delta_hedge", "ppo_hedge"]:
        strat_pnls = [r[1] for r in rows if r[0] == strat]
        strat_tcs = [r[2] for r in rows if r[0] == strat]

        mean_pnl = float(np.mean(strat_pnls))
        std_pnl = float(np.std(strat_pnls))
        mean_tc = float(np.mean(strat_tcs))
        sharpe_like = mean_pnl / std_pnl if std_pnl > 0 else 0.0

        stats[strat] = (mean_pnl, std_pnl, mean_tc, sharpe_like)

    print("\n=== Strategy Comparison ===")
    for strat, (m, s, c, sh) in stats.items():
        print(
            f"{strat:11s}: mean PnL = {m:.4f}, std = {s:.4f}, "
            f"mean tc = {c:.4f}, mean/std = {sh:.4f}"
        )


if __name__ == "__main__":
    main()