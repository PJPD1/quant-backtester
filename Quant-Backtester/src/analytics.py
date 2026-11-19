import numpy as np
import pandas as pd
import statsmodels.api as sm
from data_loader import load_market_data
from signals import compute_residual_momentum
from portfolio import construct_neutral_weights
from execution import run_backtest

def compute_tearsheet_metrics(backtest_results, benchmark_series):
# 1. Filter out the initial 120 warmup days where turnover and returns were 0
    active_mask = backtest_results["turnover"] > 0
    first_trade_date = backtest_results[active_mask].index[0]

    active_backtest = backtest_results.loc[first_trade_date:].copy()
    net_returns = active_backtest["net_return"]

    # 2. Annualized Return and Annualized Volatility (252 trading days/year)
    annualized_return = net_returns.mean() * 252
    annualized_volatility = net_returns.std() * np.sqrt(252)

    # 3. Annualized Sharpe Ratio
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0.0

    # 4. Annualized Sortino Ratio (penalizes only negative return days)
    downside_returns = net_returns[net_returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(252)
    sortino_ratio = annualized_return / downside_volatility if downside_volatility != 0 else 0.0

    # 5. Maximum Drawdown (Worst peak-to-trough percentage drop)
    equity_curve = (1.0 + net_returns).cumprod()
    running_peak = equity_curve.cummax()
    drawdown = (equity_curve - running_peak) / running_peak
    max_drawdown = drawdown.min()

    # 6. Align NIFTY 50 Benchmark Simple Returns to our active trading days
    bench_returns = benchmark_series.pct_change().loc[first_trade_date:].fillna(0.0)

    # 7. Run Ordinary Least Squares (OLS) Regression: R_port = Alpha + Beta * R_nifty
    X = sm.add_constant(bench_returns)
    ols_model = sm.OLS(net_returns, X).fit()
    
    daily_alpha = ols_model.params["const"]
    annualised_alpha = daily_alpha * 252
    alpha_tstat = ols_model.tvalues["const"]
    realised_beta = ols_model.params["benchmark"]

    # 8. Package all metrics into a clean dictionary
    metrics = {
        "Start date" : str(first_trade_date.date()),
        "End date" : str (active_backtest.index[-1].date()),
        "Annualized Net Return" : f"{annualized_return:.2%}",
        "Annualized Volatility": f"{annualized_volatility:.2%}",
        "Sharpe Ratio": round(sharpe_ratio, 2),
        "Sortino Ratio": round(sortino_ratio, 2),
        "Max Drawdown": f"{max_drawdown:.2%}",
        "Avg Daily Turnover": f"{active_backtest['turnover'].mean():.2%}",
        "Realized NIFTY Beta": round(realised_beta, 4),
        "Annualized Alpha": f"{annualised_alpha:.2%}",
        "Alpha t-stat": round(alpha_tstat, 2),
    }

    return pd.Series(metrics, name="Strategy Tear-Sheet")

if __name__ ==  "__main__":
    close_prices, open_prices, nifty_bench = load_market_data()
    signals, betas = compute_residual_momentum(close_prices, nifty_bench, lookback=60)
    weights = construct_neutral_weights(signals, betas)
    backtest_results = run_backtest(weights, close_prices, open_prices, cost_bps=15.0)

    tearsheet = compute_tearsheet_metrics(backtest_results, nifty_bench)
    print("\n==============================================")
    print("   INSTITUTIONAL QUANT STRATEGY TEAR-SHEET    ")
    print("==============================================")
    print(tearsheet.to_string())
    print("==============================================")