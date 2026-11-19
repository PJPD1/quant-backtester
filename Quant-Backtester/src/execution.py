import numpy as np
import pandas as pd 
from data_loader import load_market_data
from signals import compute_residual_momentum
from portfolio import construct_neutral_weights

def run_backtest(weights_df, close_df, open_df, cost_bps=15.0):
# 1. Convert basis points (e.g., 15 bps) into a decimal (0.0015)
    cost_rate = cost_bps / 10000.0

    # 2. Shift weights forward by 1 day: Monday night's signal trades on Tuesday!
    held_weights = weights_df.shift(1).fillna(0.0)  # Fill any NaNs with 0.0    

    # 3. Calculate Open-to-Open returns (executing at 9:15 AM Open of Day t)
    # Return earned from Tuesday 9:15 AM Open to Wednesday 9:15 AM Open:
    open_to_open_returns = (open_df.shift(-1) - open_df) / open_df  # Shift -1 to get next day's open

    # For the very last row where tomorrow's Open doesn't exist yet, use Open-to-Close
    intraday_returns = (close_df - open_df) / open_df
    asset_returns = open_to_open_returns.fillna(intraday_returns).fillna(0.0)  # Fill NaNs with intraday returns for the last day

    # 4. Calculate Daily Gross Portfolio Return: sum(w_i * r_i) across all 19 stocks
    gross_daily_returns = (held_weights * asset_returns).sum(axis=1)

    # 5. Calculate Daily Turnover: sum(|w_{i,t} - w_{i,t-1}|)
    weights_changes = held_weights.diff().abs().fillna(0.0)  # Absolute change in weights
    daily_turnover = weights_changes.sum(axis=1)

    # 6. Deduct Indian Transaction Costs & Slippage
    daily_cost = daily_turnover * cost_rate
    net_daily_returns = gross_daily_returns - daily_cost

    # 7. Combine everything into a clean Daily Performance Table
    results_df = pd.DataFrame({
        "gross_return": gross_daily_returns,
        "turnover": daily_turnover,
        "transaction_cost": daily_cost,
        "net_return": net_daily_returns,
    })

    # 8. Build Cumulative Equity Curves (Growth of ₹100 invested)
    results_df["gross_equity"] = 100 * (1 + results_df["gross_return"]).cumprod()
    results_df["net_equity"] = 100 * (1 + results_df["net_return"]).cumprod()

    return results_df 

if __name__ == "__main__":
    close_prices, open_prices, nifty_bench = load_market_data()
    signals, betas = compute_residual_momentum(close_prices, nifty_bench, lookback=60)
    weights = construct_neutral_weights(signals, betas)
    backtest_results = run_backtest(weights, close_prices, open_prices, cost_bps=15.0)

    print("\n--- SAMPLE BACKTEST LEDGER (LAST 5 TRADING DAYS) ---")
    print(backtest_results.tail(5).round(4))

    print("\n--- AVERAGE DAILY TURNOVER ---")
    avg_turnover = backtest_results["turnover"].mean()
    print(f"{avg_turnover:.2%} of portfolio traded per year")
    