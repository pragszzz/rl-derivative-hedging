import numpy as np


def simulate_gbm_path(
    S0=100.0,
    mu=0.05,
    sigma=0.2,
    T=30/252,
    n_steps=30,
    random_seed=None
):
    if random_seed is not None:
        np.random.seed(random_seed)

    dt = T / n_steps
    prices = np.zeros(n_steps + 1)
    prices[0] = S0

    for t in range(1, n_steps + 1):
        z = np.random.normal()
        prices[t] = prices[t - 1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        )

    return prices


def simulate_multiple_paths(
    n_paths=100,
    S0=100.0,
    mu=0.05,
    sigma=0.2,
    T=30/252,
    n_steps=30,
    random_seed=42
):
    np.random.seed(random_seed)
    all_paths = []

    for _ in range(n_paths):
        path = simulate_gbm_path(
            S0=S0,
            mu=mu,
            sigma=sigma,
            T=T,
            n_steps=n_steps
        )
        all_paths.append(path)

    return np.array(all_paths)