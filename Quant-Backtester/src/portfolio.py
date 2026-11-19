import pandas as pd
import numpy as np
from data_loader import load_market_data
from signals import compute_residual_momentum

def construct_neutral_weights(signals_df, betas_df):
    # 1. Keep only dates where both signals and betas exist (after the 60-day warmup)
    valid_mask = signals_df.notna() & betas_df.notna()
    clean_signals = signals_df.where(valid_mask)
    clean_betas = betas_df.where(valid_mask)

    # 2. De-mean signals across each row so sum of weights = 0 (Rupee-Neutral
    row_means = clean_signals.mean(axis=1)
    weights = clean_signals.sub(row_means, axis=0)

    # 3. Center the betas across each row (Beta_i - Mean_Beta)
    beta_demeaned = clean_betas.sub(clean_betas.mean(axis=1), axis=0)

    # 4. Calculate current portfolio beta exposure on each day: sum(w_i * beta_demeaned_i)
    port_beta_num = (weights * beta_demeaned).sum(axis=1)
    beta_norm_sq = (beta_demeaned ** 2).sum(axis=1)

    # 5. Subtract the beta projection so net portfolio beta becomes exactly 0.0!
    projection_factor = port_beta_num.div(beta_norm_sq.replace(0, np.nan))  # Avoid division by zero
    beta_adjustment = beta_demeaned.mul(projection_factor, axis=0)
    neutral_weights = weights - beta_adjustment

    # 6. Scale gross leverage so |Longs| + |Shorts| = 1.0 (50% Long, 50% Short)
    gross_exposure = neutral_weights.abs().sum(axis=1)
    final_weights = neutral_weights.div(gross_exposure.replace(0, np.nan), axis=0)  # Avoid division by zero
    return final_weights.fillna(0.0)  # Fill any remaining NaNs with 0.0

if __name__ == "__main__":
    close_prices, open_prices, nifty_bench = load_market_data()
    signals, betas = compute_residual_momentum(close_prices, nifty_bench, lookback=60)
    weights = construct_neutral_weights(signals, betas)

    print("\n--- SAMPLE PORTFOLIO WEIGHTS (LAST 3 DAYS, FIRST 5 STOCKS) ---")
    print(weights.iloc[-3:, :5].round(4))

    # Verification 1: Net Rupee Exposure (Sum of Longs + Shorts should be 0.0)
    net_rupee_exposure = weights.sum(axis=1).tail(3).round(6)
    print("\n--- CHECK 1:NET RUPEE EXPOSURE (SHOULD BE 0.0) ---")
    print(net_rupee_exposure)

    # Verification 2: Gross Leverage (|Longs| + |Shorts| should be 1.0)
    gross_leverage = weights.abs().sum(axis=1).tail(3).round(6)
    print("\n--- CHECK 2:GROSS LEVERAGE (SHOULD BE 1.0) ---")
    print(gross_leverage)

    # Verification 3: Net Portfolio Beta (Sum of w_i * Beta_i should be 0.0!)
    net_portfolio_beta = (weights * betas).sum(axis=1).tail(3).round(6)
    print("\n--- CHECK 3:NET PORTFOLIO BETA TO NIFTY 50(SHOULD BE 0.0) ---")
    print(net_portfolio_beta)
    