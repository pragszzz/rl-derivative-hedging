import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")


def plot_pnl_histograms(csv_path=RESULTS_DIR / "strategy_comparison_paths.csv"):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 6))
    for strat, color in zip(
        ["no_hedge", "delta_hedge", "ppo_hedge"],
        ["tab:blue", "tab:orange", "tab:green"],
    ):
        subset = df[df["strategy"] == strat]["pnl"]
        plt.hist(
            subset,
            bins=30,
            alpha=0.5,
            density=True,
            label=f"{strat} (n={len(subset)})",
            color=color,
        )

    plt.xlabel("Terminal PnL")
    plt.ylabel("Density")
    plt.title("PnL Distributions of Hedging Strategies")
    plt.legend()
    plt.tight_layout()
    out_path = RESULTS_DIR / "pnl_histograms.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot_pnl_histograms()