"""
src/features.py
FreshBasket Consumer Products - FMCG Demand Forecasting
Leakage-Free Feature Engineering Engine for Weekly FMCG Demand Modeling.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple


def engineer_features(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a comprehensive, leakage-safe tabular feature set for FMCG demand forecasting.
    All temporal lags and rolling statistics are calculated with strict historical shifts (shift(1)).
    """
    print("[FeatureEngine] Engineering temporal, promotional, calendar, and structural features...")
    df = panel_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["series_id", "date"]).reset_index(drop=True)

    # 1. Calendar & Cyclical Fourier Features
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month.astype(int)
    df["quarter"] = df["date"].dt.quarter.astype(int)
    df["year"] = df["date"].dt.year.astype(int)
    df["is_month_start"] = (df["date"].dt.day <= 7).astype(int)
    df["is_month_end"] = (df["date"].dt.day >= 24).astype(int)
    df["is_december_peak"] = (df["month"] == 12).astype(int)

    # Cyclical transformations
    df["sin_week_52"] = np.sin(2 * np.pi * df["week_of_year"] / 52.1775)
    df["cos_week_52"] = np.cos(2 * np.pi * df["week_of_year"] / 52.1775)
    df["sin_month_12"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["cos_month_12"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # 2. Promotional Features
    df["has_promo"] = (df["promo_count"] > 0).astype(int)
    df["promo_intensity"] = (df["promo_days"] / 7.0).clip(0, 1)

    grouped = df.groupby("series_id", group_keys=False)
    df["promo_lag_1"] = grouped["promo_count"].shift(1).fillna(0)
    df["promo_lead_1"] = grouped["promo_count"].shift(-1).fillna(0)  # Exogenous planned future promos

    # Shifted sales series for computing historical statistics
    df["sales_shifted_1"] = grouped["unit_sales"].shift(1).fillna(0)
    grouped_shifted = df.groupby("series_id", group_keys=False)["sales_shifted_1"]

    # 3. Leakage-Safe Lag Features
    df["sales_lag_1"] = df["sales_shifted_1"]
    df["sales_lag_2"] = grouped["unit_sales"].shift(2).fillna(0)
    df["sales_lag_3"] = grouped["unit_sales"].shift(3).fillna(0)
    df["sales_lag_4"] = grouped["unit_sales"].shift(4).fillna(0)
    df["sales_lag_13"] = grouped["unit_sales"].shift(13).fillna(0)
    df["sales_lag_52"] = grouped["unit_sales"].shift(52).fillna(0)

    # 4. Rolling Window Statistical Aggregations
    # 4-week window
    df["rolling_mean_4w"] = grouped_shifted.transform(lambda s: s.rolling(window=4, min_periods=1).mean()).fillna(0)
    df["rolling_std_4w"] = grouped_shifted.transform(lambda s: s.rolling(window=4, min_periods=1).std()).fillna(0)
    df["rolling_min_4w"] = grouped_shifted.transform(lambda s: s.rolling(window=4, min_periods=1).min()).fillna(0)
    df["rolling_max_4w"] = grouped_shifted.transform(lambda s: s.rolling(window=4, min_periods=1).max()).fillna(0)

    # 13-week window (Quarterly momentum)
    df["rolling_mean_13w"] = grouped_shifted.transform(lambda s: s.rolling(window=13, min_periods=1).mean()).fillna(0)
    df["rolling_std_13w"] = grouped_shifted.transform(lambda s: s.rolling(window=13, min_periods=1).std()).fillna(0)
    df["rolling_max_13w"] = grouped_shifted.transform(lambda s: s.rolling(window=13, min_periods=1).max()).fillna(0)

    # 52-week window (Annual baseline)
    df["rolling_mean_52w"] = grouped_shifted.transform(lambda s: s.rolling(window=52, min_periods=1).mean()).fillna(0)

    # Exponentially weighted moving average (Recency-weighted baseline)
    df["ewm_mean_4w"] = grouped_shifted.transform(lambda s: s.ewm(span=4, min_periods=1).mean()).fillna(0)

    # Promotional rolling intensity
    grouped_promo_shifted = df.groupby("series_id", group_keys=False)["promo_lag_1"]
    df["promo_rolling_mean_4w"] = grouped_promo_shifted.transform(lambda s: s.rolling(window=4, min_periods=1).mean()).fillna(0)
    df["promo_rolling_mean_13w"] = grouped_promo_shifted.transform(lambda s: s.rolling(window=13, min_periods=1).mean()).fillna(0)

    # Ratios and Momentum Signals
    df["sales_ratio_lag1_rolling4"] = np.where(
        df["rolling_mean_4w"] > 0,
        df["sales_lag_1"] / (df["rolling_mean_4w"] + 1e-3),
        1.0
    ).clip(0, 10)

    df["sales_momentum_lag1_lag2"] = np.where(
        df["sales_lag_2"] > 0,
        df["sales_lag_1"] / (df["sales_lag_2"] + 1e-3),
        1.0
    ).clip(0, 10)

    # 5. Macro & External Drivers
    df["oil_price_lag_1"] = df.groupby("market_id", group_keys=False)["oil_price"].shift(1).bfill().ffill()
    df["oil_price_rolling_mean_4w"] = df.groupby("market_id", group_keys=False)["oil_price"].transform(
        lambda s: s.shift(1).rolling(window=4, min_periods=1).mean()
    ).bfill().ffill()
    df["oil_price_momentum"] = (df["oil_price"] / (df["oil_price_lag_1"] + 1e-5)).clip(0.5, 2.0)

    # 6. Holiday & Event Flags
    df["is_holiday_week"] = (df["holiday_days"] > 0).astype(int)
    df["is_national_holiday_week"] = (df["national_holiday_days"] > 0).astype(int)

    # 7. Series Lifecycle & Cold-Start Dynamics
    df["series_launch_week"] = pd.to_datetime(df["series_launch_week"])
    df["weeks_since_launch"] = ((df["date"] - df["series_launch_week"]).dt.days // 7).clip(lower=0).fillna(0)

    # Drop temporary column
    df.drop(columns=["sales_shifted_1"], inplace=True)

    # Categorical Type Conversions for LightGBM / XGBoost
    cat_cols = ["market_id", "sku_id", "city", "state", "store_type", "store_cluster", "series_id"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print(f"[FeatureEngine] Completed feature engineering. Dataset shape: {df.shape}")
    return df


def get_feature_columns() -> Tuple[List[str], List[str]]:
    """
    Returns lists of numerical and categorical feature names used for model training.
    """
    categorical_features = ["market_id", "sku_id", "city", "state", "store_type", "store_cluster"]
    
    numerical_features = [
        # Calendar & Seasonality
        "week_of_year", "month", "quarter", "year", "is_month_start", "is_month_end",
        "is_december_peak", "sin_week_52", "cos_week_52", "sin_month_12", "cos_month_12",
        # Promotional
        "promo_count", "promo_days", "promo_intensity", "has_promo",
        "promo_lag_1", "promo_lead_1", "promo_rolling_mean_4w", "promo_rolling_mean_13w",
        # Autoregressive Lags & Rolling
        "sales_lag_1", "sales_lag_2", "sales_lag_3", "sales_lag_4", "sales_lag_13", "sales_lag_52",
        "rolling_mean_4w", "rolling_std_4w", "rolling_min_4w", "rolling_max_4w",
        "rolling_mean_13w", "rolling_std_13w", "rolling_max_13w",
        "rolling_mean_52w", "ewm_mean_4w",
        "sales_ratio_lag1_rolling4", "sales_momentum_lag1_lag2",
        # Macro & Holidays
        "oil_price", "oil_price_lag_1", "oil_price_rolling_mean_4w", "oil_price_momentum",
        "holiday_days", "national_holiday_days", "event_days", "total_holiday_events",
        "is_holiday_week", "is_national_holiday_week",
        # Lifecycle
        "is_pre_launch", "weeks_since_launch"
    ]
    
    return numerical_features, categorical_features


if __name__ == "__main__":
    panel = pd.read_parquet("data/processed/weekly_fmcg_dataset.parquet")
    feat_df = engineer_features(panel)
    output_path = "data/processed/modeling_features.parquet"
    feat_df.to_parquet(output_path, index=False)
    print(f"[FeatureEngine] Saved full feature dataset to {output_path}")
