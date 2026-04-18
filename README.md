# Derivative Hedging Using Reinforcement Learning

This project implements adaptive option hedging using reinforcement learning.

## Goal

Train a PPO-based RL agent to dynamically hedge an option position using the underlying asset and compare its performance with traditional delta hedging and a no‑hedge strategy.

## Project Structure

- `src/pricing/` – Black–Scholes pricing and Greeks
- `src/data/` – Simulation and data utilities (GBM paths, etc.)
- `src/env/` – `HedgingEnv` environment and portfolio logic (Gymnasium)
- `src/agents/` – RL training and evaluation scripts (PPO with Stable-Baselines3)
- `src/baselines/` – Traditional hedging strategies (no‑hedge, delta‑hedge)
- `src/evaluation/` – Metrics, comparison scripts, and plots
- `results/` – Generated figures and CSVs (e.g. PnL histograms)
- `notebooks/` – Optional exploratory notebooks

## Setup

Tested with Python 3.11.

```bash
git clone https://github.com/pragszzz/rl-derivative-hedging.git
cd rl-derivative-hedging

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

## How to Run

Train the PPO agent:

```bash
python -m src.agents.train_ppo
```

Run a quick sanity check comparing a few PnL samples:

```bash
python scratch_check.py
```

Run the full evaluation and generate plots:

```bash
python -m src.evaluation.compare_strategies
python -m src.evaluation.plots
```

The main PnL histogram is saved in `results/` (for example `results/pnl_histograms.png`).

## Methods

- Simulate the underlying asset price using geometric Brownian motion.
- Price a European call option and compute Greeks with the Black–Scholes model.
- Define a custom Gymnasium environment where:
  - the state includes time, underlying price, and position information;
  - the action is the hedge position in the underlying;
  - rewards are based on incremental PnL minus transaction costs.
- Train a PPO agent (Stable-Baselines3) to choose hedges at each time step.
- Compare against:
  - **No‑hedge**: hold the option to maturity.
  - **Delta‑hedge**: rebalance using Black–Scholes delta.

## Results (Summary)

In this simple GBM setting:

- The no‑hedge and delta‑hedge baselines achieve positive mean terminal PnL with non‑zero variance and transaction costs.
- The PPO agent converged to a stable but sub‑optimal hedging policy, producing almost constant negative terminal PnL across simulated paths.
- This shows that, under these assumptions and hyperparameters, PPO did **not** outperform classical hedging, highlighting the importance of reward design, model capacity, and environment complexity.


## Limitations and Future Work

- Tune PPO hyperparameters and network size to avoid degenerate constant policies.
- Extend the environment with stochastic volatility or multiple hedging instruments.
- Explore risk‑aware or distributionally robust RL formulations.
- Compare PPO with other continuous‑control algorithms (e.g. SAC).

## License

This project is licensed under the MIT License – see the `LICENSE` file for details.
