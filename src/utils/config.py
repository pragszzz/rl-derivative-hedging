PROJECT_NAME = "Derivative Hedging Using Reinforcement Learning"

RANDOM_SEED = 42

# Market parameters
INITIAL_STOCK_PRICE = 100.0
STRIKE_PRICE = 100.0
RISK_FREE_RATE = 0.05
VOLATILITY = 0.2
MATURITY = 30 / 252  # 30 trading days in years

# Simulation parameters
N_STEPS = 30
N_EPISODES = 1000
DT = 1 / 252

# Transaction cost
TRANSACTION_COST = 0.001  # 0.1%

# RL settings
ACTION_LOW = -1.0
ACTION_HIGH = 1.0

# Paths
MODEL_SAVE_PATH = "models/"
RESULTS_PATH = "results/"