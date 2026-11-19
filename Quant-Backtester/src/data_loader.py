import os
import yfinance as yf
import pandas as pd
import numpy as np

# 20 liquid NIFTY 50 stocks + 1 intentionally late IPO (LICI.NS) to test our filter
UNIVERSE_TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "LICI.NS", "HINDUNILVR.NS",
    "LT.NS", "BAJFINANCE.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "KOTAKBANK.NS", "TITAN.NS", "ONGC.NS", "NTPC.NS", "AXISBANK.NS",
    "TATAMOTORS.NS"]
BENCHMARK = "^NSEI"  # NIFTY 50 index

def load_market_data(start_date = "2019-01-01", end_date = "2026-11-01"):
    os.makedirs("data", exist_ok=True)  # Create a directory to store data if it doesn't exist
    close_file = "data/close_prices.parquet"
    open_file = "data/open_prices.parquet"
    benchmark_file = "data/benchmark.parquet"

    # 1. If we already downloaded the files earlier, load them instantly from disk!
    if os.path.exists(close_file) and os.path.exists(open_file) and os.path.exists(benchmark_file):
        print("Loading saved data from the 'data/' folder")
        close_df = pd.read_parquet(close_file)
        open_df = pd.read_parquet(open_file)
        bench_series = pd.read_parquet(benchmark_file) ["benchmark"]
        return close_df, open_df, bench_series

# 2. Otherwise, download fresh data from Yahoo Finance
    print("Downloading fresh data from Yahoo Finance")
    all_symbols = UNIVERSE_TICKERS + [BENCHMARK] 
    raw_data = yf.download(all_symbols, start=start_date, end=end_date, auto_adjust=True)  # Download historical data
    close_df = raw_data ["Close"].copy()  # Extract closing prices
    open_df = raw_data ["Open"].copy()  # Extract opening prices

# 3. Lock calendar strictly to days when NIFTY 50 (^NSEI) actually traded
    valid_trading_days = close_df[BENCHMARK].dropna().index
    close_df = close_df.loc[valid_trading_days]
    open_df = open_df.loc[valid_trading_days]

# Separate NIFTY 50 benchmark from our individual stocks
    bench_series = close_df[BENCHMARK].rename("benchmark")
    close_df = close_df.drop(columns=[BENCHMARK])
    open_df = open_df.drop(columns=[BENCHMARK])

# 4. Filter out stocks missing more than 10% of days (like late IPOs)
    missing_pct = close_df.isna().mean()
    good_tickers = missing_pct[missing_pct <= 0.10].index
    dropped = set(UNIVERSE_TICKERS) - set(good_tickers)
    print(f"Dropped tickers due to >10% missing history: {dropped}")
    close_df = close_df[good_tickers]
    open_df = open_df[good_tickers]

# 5. Forward-fill tiny 1-or-2 day gaps (never fill backwards!)
    close_df = close_df.ffill(limit=2)
    open_df = open_df.ffill(limit=2)

# 6. Save clean tables to the 'data/' folder
    close_df.to_parquet(close_file)
    open_df.to_parquet(open_file)
    bench_series.to_frame().to_parquet(benchmark_file)

    print(f"Success! Saved {close_df.shape[0]} days across {close_df.shape[1]} stocks")
    return close_df, open_df, bench_series

if __name__ == "__main__":
    close_df, open_df, nifty_bench = load_market_data()
    print("\n--- SAMPLE CLEANED CLOSING PRICES ---")
    print(close_df.tail(3))
    print("\n--- SAMPLE NIFTY 50 BENCHMARK ---")
    print(nifty_bench.tail(3)) 
    