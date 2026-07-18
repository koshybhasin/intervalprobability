"""
Markowitz Portfolio Optimizer
-------------------------------
Given a set of tickers, this script:
1. Downloads historical price data
2. Computes expected returns and the covariance matrix
3. Finds the Efficient Frontier
4. Finds the Max Sharpe Ratio portfolio and the Min Volatility portfolio
5. Plots everything

Run: python markowitz_optimizer.py
"""
import numpy as np
import pandas as pd
try:
    import yfinance as yf
except ImportError:
    print("yfinance is not installed. Please install it using 'pip install yfinance'.")
    raise
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "JNJ"]
START_DATE = "2019-01-01"
END_DATE = "2024-12-31"
RISK_FREE_RATE = 0.02  # annualized, for Sharpe ratio
NUM_RANDOM_PORTFOLIOS = 5000  # for visualizing the feasible region

# ---------------------------------------------------------
# 2. DATA
# ---------------------------------------------------------
def get_data(tickers, start, end):
    prices = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    prices = prices.dropna(how="all")
    return prices

def compute_returns_and_cov(prices):
    daily_returns = prices.pct_change().dropna()
    mean_daily_returns = daily_returns.mean()
    cov_daily = daily_returns.cov()

    # annualize (252 trading days)
    mean_annual_returns = mean_daily_returns * 252
    cov_annual = cov_daily * 252
    return mean_annual_returns, cov_annual

# ---------------------------------------------------------
# 3. PORTFOLIO MATH
# ---------------------------------------------------------
def portfolio_performance(weights, mean_returns, cov_matrix):
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return ret, vol

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(ret - risk_free_rate) / vol

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]

# ---------------------------------------------------------
# 4. OPTIMIZERS
# ---------------------------------------------------------
def optimize_max_sharpe(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))  # no short-selling
    init_guess = num_assets * [1.0 / num_assets]

    result = minimize(
        neg_sharpe_ratio, init_guess, args=args,
        method="SLSQP", bounds=bounds, constraints=constraints
    )
    return result

def optimize_min_volatility(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1.0 / num_assets]

    result = minimize(
        portfolio_volatility, init_guess, args=args,
        method="SLSQP", bounds=bounds, constraints=constraints
    )
    return result

def efficient_frontier(mean_returns, cov_matrix, return_targets):
    """For each target return, find the minimum-variance portfolio achieving it."""
    num_assets = len(mean_returns)
    frontier_vols = []

    for target in return_targets:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, target=target: portfolio_performance(w, mean_returns, cov_matrix)[0] - target}
        )
        bounds = tuple((0, 1) for _ in range(num_assets))
        init_guess = num_assets * [1.0 / num_assets]

        result = minimize(
            portfolio_volatility, init_guess, args=(mean_returns, cov_matrix),
            method="SLSQP", bounds=bounds, constraints=constraints
        )
        frontier_vols.append(result["fun"] if result.success else np.nan)

    return frontier_vols

# ---------------------------------------------------------
# 5. RANDOM PORTFOLIOS (for visualization / intuition)
# ---------------------------------------------------------
def random_portfolios(n, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3, n))
    num_assets = len(mean_returns)
    for i in range(n):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
        sharpe = (ret - risk_free_rate) / vol
        results[0, i] = vol
        results[1, i] = ret
        results[2, i] = sharpe
    return results

# ---------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------
def main():
    print(f"Downloading data for: {TICKERS}")
    prices = get_data(TICKERS, START_DATE, END_DATE)
    mean_returns, cov_matrix = compute_returns_and_cov(prices)

    print("\nAnnualized Expected Returns:")
    print(mean_returns.round(4))

    # Max Sharpe portfolio
    max_sharpe = optimize_max_sharpe(mean_returns, cov_matrix, RISK_FREE_RATE)
    ms_weights = max_sharpe.x
    ms_ret, ms_vol = portfolio_performance(ms_weights, mean_returns, cov_matrix)
    ms_sharpe = (ms_ret - RISK_FREE_RATE) / ms_vol

    print("\n=== Max Sharpe Ratio Portfolio ===")
    for t, w in zip(TICKERS, ms_weights):
        print(f"  {t}: {w:.2%}")
    print(f"  Expected Return: {ms_ret:.2%}")
    print(f"  Volatility:      {ms_vol:.2%}")
    print(f"  Sharpe Ratio:    {ms_sharpe:.2f}")

    # Min Volatility portfolio
    min_vol = optimize_min_volatility(mean_returns, cov_matrix)
    mv_weights = min_vol.x
    mv_ret, mv_vol = portfolio_performance(mv_weights, mean_returns, cov_matrix)

    print("\n=== Min Volatility Portfolio ===")
    for t, w in zip(TICKERS, mv_weights):
        print(f"  {t}: {w:.2%}")
    print(f"  Expected Return: {mv_ret:.2%}")
    print(f"  Volatility:      {mv_vol:.2%}")

    # Efficient frontier
    target_returns = np.linspace(mv_ret, mean_returns.max(), 50)
    frontier_vols = efficient_frontier(mean_returns, cov_matrix, target_returns)

    # Random portfolios for the cloud in the background
    random_results = random_portfolios(NUM_RANDOM_PORTFOLIOS, mean_returns, cov_matrix, RISK_FREE_RATE)

    # ---------------------------------------------------------
    # 7. PLOT
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 7))
    plt.scatter(random_results[0], random_results[1], c=random_results[2],
                cmap="viridis", alpha=0.4, s=10, label="Random portfolios")
    plt.colorbar(label="Sharpe Ratio")
    plt.plot(frontier_vols, target_returns, "r--", linewidth=2, label="Efficient Frontier")
    plt.scatter(ms_vol, ms_ret, marker="*", color="gold", s=400, edgecolors="black", label="Max Sharpe")
    plt.scatter(mv_vol, mv_ret, marker="*", color="cyan", s=400, edgecolors="black", label="Min Volatility")
    plt.xlabel("Volatility (Std. Dev.)")
    plt.ylabel("Expected Annual Return")
    plt.title("Markowitz Efficient Frontier")
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("\nSaved plot to efficient_frontier.png")

if __name__ == "__main__":
    main()