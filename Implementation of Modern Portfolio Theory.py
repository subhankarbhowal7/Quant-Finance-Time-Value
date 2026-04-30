import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as sco

# --- CONFIGURATION ---
stocks = ['AAPL', 'WMT', 'TSLA', 'GE', 'AMZN', 'DB']
start_date = '2001-01-01'
end_date = '2025-01-01'


# --- FUNCTIONS ---

def download_data(stocks):
    """Downloads adjusted close prices from Yahoo Finance."""
    data = yf.download(stocks, start=start_date, end=end_date, auto_adjust=True)['Close']
    return data

def show_data(data):
    data.plot(figsize=(10,5))
    plt.show()

def calculate_returns(data):
    """Calculates log returns for normalization."""
    return np.log(data / data.shift(1))
    return returns

def plot_daily_returns(returns):
    returns.plot(figsize=(10,5))
    plt.show()


def show_statistics(returns):
    """Prints annualized mean returns and covariance matrix."""
    print("\n--- Annualized Mean Returns ---")
    print(returns.mean() * 252)
    print("\n--- Annualized Covariance Matrix ---")
    print(returns.cov() * 252)


def initialize_weights():
    """Generates random weights that sum to 1."""
    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)
    return weights


def statistics(weights, returns):
    """Returns [Return, Volatility, Sharpe Ratio] for a given weight set."""
    weights = np.array(weights)
    p_ret = np.sum(returns.mean() * weights) * 252
    p_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return np.array([p_ret, p_vol, p_ret / p_vol])


def generate_portfolios(returns):
    """Monte Carlo simulation to generate random portfolios."""
    preturns = []
    pvolatilities = []
    for _ in range(10000):
        weights = initialize_weights()
        stats = statistics(weights, returns)
        preturns.append(stats[0])
        pvolatilities.append(stats[1])
    return np.array(preturns), np.array(pvolatilities)


# --- OPTIMIZATION ---

def min_func_sharpe(weights, returns):
    """Objective function for the optimizer (Negative Sharpe Ratio)."""
    return -statistics(weights, returns)[2]


def optimize_portfolio(weights, returns):
    """Finds the optimal weights using Scipy SLSQP."""
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(stocks)))
    return sco.minimize(fun=min_func_sharpe, x0=weights, args=returns,
                        method='SLSQP', bounds=bounds, constraints=constraints)


# --- PLOTTING ---

def show_optimal_portfolio(optimum, returns, preturns, pvolatilities):
    """Visualizes the Efficient Frontier and the Optimal Portfolio."""
    plt.figure(figsize=(10, 6))
    plt.scatter(pvolatilities, preturns, c=preturns / pvolatilities, marker='o', cmap='viridis')
    plt.grid(True)
    plt.xlabel('Expected Volatility (Risk)')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')

    # Mark the optimal portfolio with a green star
    opt_stats = statistics(optimum['x'], returns)
    plt.plot(opt_stats[1], opt_stats[0], 'g*', markersize=20.0, label='Optimal Portfolio')
    plt.legend()
    plt.title('Efficient Frontier Optimization')
    plt.show()


# --- EXECUTION ---

if __name__ == "__main__":
    # 1. Acquisition
    df = download_data(stocks)
    returns = calculate_returns(df).dropna()
    show_data(df)
    # show returns

    plot_daily_returns(returns)

    # 2. Analysis
    show_statistics(returns)

   
    # 3. Simulation
    print("\nRunning Monte Carlo Simulation...")
    preturns, pvolatilities = generate_portfolios(returns)

    # 4. Optimization
    initial_w = initialize_weights()
    optimum = optimize_portfolio(initial_w, returns)

    # 5. Final Output
    print("\n--- Optimization Results ---")
    print("Optimal Weights:", optimum['x'].round(3))
    final_stats = statistics(optimum['x'], returns)
    print(f"Exp. Return: {final_stats[0]:.3f}")
    print(f"Exp. Volatility: {final_stats[1]:.3f}")
    print(f"Max Sharpe Ratio: {final_stats[2]:.3f}")

    show_optimal_portfolio(optimum, returns, preturns, pvolatilities)
