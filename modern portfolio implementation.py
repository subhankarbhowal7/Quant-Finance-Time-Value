import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# 1. Define your stock
stocks = ['WMT']

# 2. Use YYYY-MM-DD format
start_date = '2001-01-01'
end_date = '2025-01-01'

# 3. Download directly using yfinance (replaces pandas_datareader)
data = yf.download(stocks, start=start_date, end=end_date, auto_adjust=True)['Close']


# 4. Calculate returns
daily_returns = data.pct_change().dropna()

# 5. Plot
daily_returns.hist(bins=100)
plt.title('WMT Daily Returns')
plt.show()
