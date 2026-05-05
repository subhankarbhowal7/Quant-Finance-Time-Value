import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

risk_free_rate = 0.05


def capm(start_date, end_date, ticker1, ticker2):
    # 1. Download data
    raw_data = yf.download([ticker1, ticker2], start=start_date, end=end_date)

    # 2. FIX: Handle the MultiIndex by selecting Adj Close or Close
    # This syntax works regardless of your yfinance version
    if 'Adj Close' in raw_data.columns.get_level_values(0):
        data = raw_data['Adj Close']
    else:
        data = raw_data['Close']

    # 3. Resample to Monthly (ME = Month End)
    monthly_prices = data.resample('ME').last()

    # 4. Calculate Log Returns
    returns = np.log(monthly_prices / monthly_prices.shift(1)).dropna()

    # 5. Extract specific returns
    s_returns = returns[ticker1]
    m_returns = returns[ticker2]

    # 6. Calculate Beta via regression
    beta, alpha = np.polyfit(m_returns, s_returns, deg=1)
    print(f"Beta for {ticker1}: {beta:.4f}")

    # 7. Plotting
    fig, axis = plt.subplots(1, figsize=(10, 6))
    axis.scatter(m_returns, s_returns, label="Monthly Returns", alpha=0.6)
    axis.plot(m_returns, beta * m_returns + alpha, color='red', label=f"CAPM Line (Beta={beta:.2f})")

    plt.title(f'CAPM Analysis: {ticker1} vs {ticker2}')
    plt.xlabel('Market Return ($R_m$)')
    plt.ylabel('Stock Return ($R_s$)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 8. Expected Return
    expected_return = risk_free_rate + beta * (m_returns.mean() * 12 - risk_free_rate)
    print(f"Annualized Expected Return: {expected_return:.2%}")


if __name__ == "__main__":
    capm('2010-01-01', '2025-01-01', 'IBM', '^GSPC')
