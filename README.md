# Derivative Hedging Using Reinforcement Learning

This project implements adaptive option hedging using reinforcement learning.

## Goal
Train an RL agent to dynamically hedge an option position and compare it with traditional delta hedging.

## Project Structure
- `src/pricing/` - Black-Scholes pricing and Greeks
- `src/data/` - Simulation and data loading
- `src/env/` - Hedging environment and portfolio logic
- `src/agents/` - RL training scripts
- `src/baselines/` - Traditional hedging strategies
- `src/evaluation/` - Metrics and plots