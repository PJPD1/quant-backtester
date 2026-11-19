import numpy as np
import pandas as pd
from data_loader import load_market_data

def compute_residual_momentum(close_df, bench_series, lookback=60):
  
    # 1.Calculate daily log returns for our 19 stocks and the NIFTY 50 benchmark
    stock_returns = np.log(close_df / close_df.shift(1))
    bench_returns = np.log(bench_series / bench_series.shift(1))

    # 2.Calculate Rolling 60-Day Covariance and Benchmark Variance 
    rolling_cov = stock_returns.rolling(window=lookback).cov(bench_returns)
    rolling_var = bench_returns.rolling(window=lookback).var()

    # 3. Calculate Rolling Beta for every stock: Beta = Cov(R_i, R_m) / Var(R_m)
    rolling_beta = rolling_cov.div(rolling_var, axis=0)

    # 4. Compute expected market-driven return: Beta_i * R_market 
    market_component = rolling_beta.mul(bench_returns, axis=0)

    # 5. Isolate pure stock-specific (residual) return: R_i - (Beta_i * R_market)
    residual_returns = stock_returns - market_component

    # 6. Calculate 60-day Residual Information Ratio (Mean / Volatility of residuals)
    resid_mean = residual_returns.rolling(window=lookback).mean()
    resid_std = residual_returns.rolling(window=lookback).std()
    raw_signal = resid_mean / resid_std

    # 7. Standardize across all 19 stocks each day (Cross-Sectional Z-Score)
    daily_cross_mean = raw_signal.mean(axis=1)
    daily_cross_std = raw_signal.std(axis=1)
    z_score_signal = (raw_signal.sub(daily_cross_mean, axis=0)).div(daily_cross_std, axis=0)

    return z_score_signal, rolling_beta 

if __name__ == "__main__":
    close_prices, open_prices, nifty_bench = load_market_data()
    signals, betas = compute_residual_momentum(close_prices, nifty_bench, lookback=60)
    print("\n--- SAMPLE ROLLING 60 DAYS BETAS (LAST 3 DAYS, FIRST 5 STOCKS) ---")
    print(betas.iloc[-3:, :5].round(2))

    print("\n--- SAMPLE ALPHA Z-SCORE SIGNALS (LAST 3 DAYS, FIRST 5 STOCKS) ---")
    print(signals.iloc[-3:, :5].round(2))

    print("\n--- SANITY CHECK: DAILY MEAN OF Z-SCORES (SHOULD BE 0.0) ---")
    print(signals.mean(axis=1).tail(3).round(6))
