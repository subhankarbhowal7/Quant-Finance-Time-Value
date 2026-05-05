import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

risk_free_rate = 0.05


def analyze_stocks(start_date, end_date, tickers, market_ticker='^GSPC'):
    # 1. Download all tickers + market at once
    all_tickers = tickers + [market_ticker]
    raw_data = yf.download(all_tickers, start=start_date, end=end_date)

    # 2. Handle MultiIndex and extract prices
    if 'Adj Close' in raw_data.columns.get_level_values(0):
        data = raw_data['Adj Close']
    else:
        data = raw_data['Close']

    # 3. Monthly Log Returns
    returns = np.log(data.resample('ME').last() / data.resample('ME').last().shift(1)).dropna()

    # 4. Loop through each stock to calculate metrics
    m_returns = returns[market_ticker]
    results = []

    for ticker in tickers:
        s_returns = returns[ticker]

        # Linear Regression: returns slope (Beta), intercept (Alpha), and r_value
        beta, alpha, r_value, p_value, std_err = stats.linregress(m_returns, s_returns)

        # Calculate R-squared (how much the market explains the stock)
        r_squared = r_value ** 2

        # Calculate Jensen's Alpha (Annualized)
        # Formula: Actual Return - [Rf + Beta * (Market Return - Rf)]
        annual_stock_ret = s_returns.mean() * 12
        annual_mkt_ret = m_returns.mean() * 12
        jensens_alpha = annual_stock_ret - (risk_free_rate + beta * (annual_mkt_ret - risk_free_rate))

        # Expected Return (CAPM)
        expected_ret = risk_free_rate + beta * (annual_mkt_ret - risk_free_rate)

        results.append({
            'Ticker': ticker,
            'Beta': round(beta, 4),
            'R-Squared': round(r_squared, 4),
            'Jensen Alpha': round(jensens_alpha, 4),
            'Expected Return': round(expected_ret, 4)
        })

    # 5. Display Results Table
    df_results = pd.DataFrame(results)
    print("\n--- CAPM Analysis Results ---")
    print(df_results)

    # 6. Comparative Plot
    plt.figure(figsize=(12, 6))
    for ticker in tickers:
        plt.scatter(m_returns, returns[ticker],
                    label=f"{ticker} (Beta: {df_results.loc[df_results['Ticker'] == ticker, 'Beta'].values[0]})",
                    alpha=0.5)

    plt.title('Market vs Stock Returns Comparison')
    plt.xlabel('Market Return (S&P 500)')
    plt.ylabel('Stock Returns')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # List of 5 different stocks
    my_stocks = ['IBM', 'AAPL', 'TSLA', 'MSFT', 'GOOGL']
    analyze_stocks('2015-01-01', '2025-01-01', my_stocks)
