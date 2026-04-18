import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.env.hedging_env import HedgingEnv
from src.utils.config import MODEL_SAVE_PATH, RANDOM_SEED


def make_env():
    def _init():
        env = HedgingEnv()
        return env
    return _init


def main():
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    env = DummyVecEnv([make_env()])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=RANDOM_SEED,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        clip_range=0.2,
        n_epochs=10
    )

    total_timesteps = 100_000
    model.learn(total_timesteps=total_timesteps)

    save_path = os.path.join(MODEL_SAVE_PATH, "ppo_hedging_v2")
    model.save(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()