import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.data.simulator import simulate_gbm_path
from src.pricing.greeks import call_delta, gamma
from src.env.portfolio import HedgingPortfolio
from src.utils.config import (
    INITIAL_STOCK_PRICE,
    STRIKE_PRICE,
    RISK_FREE_RATE,
    VOLATILITY,
    MATURITY,
    N_STEPS,
    ACTION_LOW,
    ACTION_HIGH,
    TRANSACTION_COST,
    RANDOM_SEED
)


class HedgingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(HedgingEnv, self).__init__()

        self.S0 = INITIAL_STOCK_PRICE
        self.K = STRIKE_PRICE
        self.r = RISK_FREE_RATE
        self.sigma = VOLATILITY
        self.T = MATURITY
        self.n_steps = N_STEPS
        self.dt = self.T / self.n_steps
        self.random_seed = RANDOM_SEED

        self.portfolio = HedgingPortfolio(
            strike=self.K,
            r=self.r,
            sigma=self.sigma,
            transaction_cost=TRANSACTION_COST
        )

        self.action_space = spaces.Box(
            low=np.array([ACTION_LOW], dtype=np.float32),
            high=np.array([ACTION_HIGH], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),
            dtype=np.float32
        )

        self.current_step = 0
        self.stock_path = None
        self.done = False

    def _get_observation(self):
        stock_price = self.stock_path[self.current_step]
        time_to_maturity = max(self.T - self.current_step * self.dt, 1e-8)

        delta = call_delta(stock_price, self.K, time_to_maturity, self.r, self.sigma)
        gamma_value = gamma(stock_price, self.K, time_to_maturity, self.r, self.sigma)

        obs = np.array([
            stock_price,
            time_to_maturity,
            delta,
            gamma_value,
            self.portfolio.stock_position
        ], dtype=np.float32)

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.done = False

        self.stock_path = simulate_gbm_path(
    
            S0=self.S0,
            mu=self.r,
            sigma=self.sigma,
            T=self.T,
            n_steps=self.n_steps,
            random_seed=None  # different path every episode
        )

        self.portfolio.reset()

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        if self.done:
            raise ValueError("Episode is done. Call reset() to start a new episode.")

        stock_price = self.stock_path[self.current_step]
        time_to_maturity = max(self.T - self.current_step * self.dt, 1e-8)

        target_position = float(action[0])
        transaction_cost = self.portfolio.rebalance_hedge(stock_price, target_position)

        old_value = self.portfolio.portfolio_value(stock_price, time_to_maturity)

        self.current_step += 1

        if self.current_step >= self.n_steps:
            self.done = True

        next_stock_price = self.stock_path[self.current_step]
        next_time_to_maturity = max(self.T - self.current_step * self.dt, 0.0)
        new_value = self.portfolio.portfolio_value(next_stock_price, next_time_to_maturity)

        pnl = new_value - old_value

        # SIMPLE REWARD: incremental PnL minus transaction cost
        reward = pnl - transaction_cost

        observation = self._get_observation()
        info = {
            "pnl": pnl,
            "transaction_cost": transaction_cost,
            "portfolio_value": new_value
        }

        return observation, reward, self.done, False, info

    def render(self):
        stock_price = self.stock_path[self.current_step]
        print(f"Step: {self.current_step}, Stock Price: {stock_price:.4f}, Hedge Position: {self.portfolio.stock_position:.4f}")