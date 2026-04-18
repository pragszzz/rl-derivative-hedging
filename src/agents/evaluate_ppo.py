import numpy as np
from stable_baselines3 import PPO

from src.env.hedging_env import HedgingEnv


def evaluate_ppo(n_episodes=50):
    model_path = "models/ppo_hedging_v2"   # <‑‑ use the v2 file
    print("Loading model from:", model_path)
    model = PPO.load(model_path)

    env = HedgingEnv()

    episode_pnls = []
    episode_costs = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_cost = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_cost += info["transaction_cost"]

        episode_pnls.append(info["portfolio_value"])
        episode_costs.append(total_cost)

    return (
        float(np.mean(episode_pnls)),
        float(np.std(episode_pnls)),
        float(np.mean(episode_costs)),
    )


def evaluate_ppo_single_path():
    model_path = "models/ppo_hedging_v2"   # <‑‑ same here
    print("Loading model from:", model_path)
    model = PPO.load(model_path)

    env = HedgingEnv()
    obs, info = env.reset()
    done = False
    total_cost = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_cost += info["transaction_cost"]

    final_value = info["portfolio_value"]
    return float(final_value), float(total_cost)


if __name__ == "__main__":
    mean_pnl, std_pnl, mean_cost = evaluate_ppo()
    print("PPO mean terminal portfolio value:", mean_pnl)
    print("PPO std terminal portfolio value:", std_pnl)
    print("PPO mean transaction cost:", mean_cost)