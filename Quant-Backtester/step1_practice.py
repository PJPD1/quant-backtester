import yfinance as yf   
import pandas as pd
import numpy as np
tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]  # List of stock tickers
start_date = "2023-01-01"  # Start date for historical data
end_date = "2026-01-01"  # End date for historical data
raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)  # Download historical data
close_prices = raw_data["Close"]  # Extract closing prices
print("--- CLOSING PRICES (LAST 5 DAYS) ---")
print(close_prices.tail(5))  # Display the last 5 days of closing prices
daily_returns = np.log(close_prices / close_prices.shift(1))  # Calculate daily returns using log returns
print("\n--- DAILY LOG RETURNS (LAST 5 DAYS) ---")
print(daily_returns.tail(5))  # Display the last 5 days of daily log returns