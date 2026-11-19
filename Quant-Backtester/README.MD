# Factor-Neutral Statistical Arbitrage Backtester (NSE Large-Caps)

A modular, vectorized quantitative equity backtesting engine built in Python to research cross-sectional residual return dynamics, enforce Rupee- and Beta-neutrality via linear algebra projection, and evaluate strategy performance under realistic $t+1$ execution lags and Indian transaction costs.

## Architecture & Pipeline

* **`src/data_loader.py`**: Downloads and caches NSE equity and NIFTY 50 (`^NSEI`) OHLC data in Parquet format, locks trading dates to official benchmark sessions, filters out tickers exceeding a 10% missing-history threshold (handling late IPOs and corporate actions/demergers), and applies bounded forward-filling.
* **`src/signals.py`**: Computes rolling 60-day market betas ($\beta_i$), isolates residual returns ($\epsilon_{i,t} = R_{i,t} - \beta_i R_{m,t}$), forms rolling residual information ratios, and standardizes scores cross-sectionally each day.
* **`src/portfolio.py`**: Constructs Rupee-neutral ($\sum w_i = 0$) and Beta-neutral ($\sum w_i \beta_i = 0$) weights via vector projection and normalizes gross exposure to `1.0` (`50%` long / `50%` short).
* **`src/execution.py`**: Enforces a strict $t+1$ execution lag with open-to-open holding returns to prevent look-ahead bias, tracks daily portfolio turnover, and deducts configurable per-turnover transaction costs (modeling STT, exchange/brokerage charges, and slippage in basis points).
* **`src/analytics.py`**: Computes annualized net return, annualized volatility, Sharpe ratio, Sortino ratio, maximum drawdown, realized NIFTY 50 beta, annualized OLS alpha, and alpha $t$-statistics via `statsmodels`.

## Setup & Usage

Install dependencies:

```bash
python -m pip install pandas numpy yfinance pyarrow statsmodels