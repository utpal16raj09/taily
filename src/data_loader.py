"""
src/data_loader.py
FreshBasket Consumer Products - FMCG Demand Forecasting
Data Ingestion, Cleaning, and Weekly Panel Resampling Engine.
"""

import os
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd


def load_raw_data(data_dir: str = "data/raw") -> Dict[str, pd.DataFrame]:
    """
    Load all raw FMCG tabular datasets with strict data types.
    """
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    stores_path = os.path.join(data_dir, "stores.csv")
    oil_path = os.path.join(data_dir, "oil.csv")
    holidays_path = os.path.join(data_dir, "holidays_events.csv")
    transactions_path = os.path.join(data_dir, "transactions.csv")

    print(f"[DataLoader] Ingesting raw datasets from: {data_dir}")
    train_df = pd.read_csv(train_path, parse_dates=["date"])
    stores_df = pd.read_csv(stores_path)
    oil_df = pd.read_csv(oil_path, parse_dates=["date"])
    holidays_df = pd.read_csv(holidays_path, parse_dates=["date"])
    transactions_df = pd.read_csv(transactions_path, parse_dates=["date"]) if os.path.exists(transactions_path) else None
    test_df = pd.read_csv(test_path, parse_dates=["date"]) if os.path.exists(test_path) else None

    return {
        "train": train_df,
        "test": test_df,
        "stores": stores_df,
        "oil": oil_df,
        "holidays": holidays_df,
        "transactions": transactions_df,
    }


def clean_oil_data(oil_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean oil price data by establishing full daily calendar and interpolating missing values.
    """
    oil_clean = oil_df.copy().sort_values("date").drop_duplicates(subset=["date"])
    # Create complete date range
    min_date, max_date = oil_clean["date"].min(), oil_clean["date"].max()
    full_dates = pd.date_range(start=min_date, end=max_date, freq="D", name="date")
    oil_clean = oil_clean.set_index("date").reindex(full_dates)
    # Interpolate linearly, then forward/back fill edges
    oil_clean["dcoilwtico"] = oil_clean["dcoilwtico"].interpolate(method="linear").bfill().ffill()
    oil_clean = oil_clean.reset_index()
    oil_clean.rename(columns={"index": "date", "dcoilwtico": "oil_price"}, inplace=True)
    return oil_clean


def process_holidays(holidays_df: pd.DataFrame, stores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process holidays and events, distinguishing transferred, national, regional, and local holidays.
    """
    df = holidays_df.copy()
    # If transferred is True, it's not celebrated on that date (moved to Transfer date)
    df = df[df["transferred"] == False]

    # Flag earthquake emergency period (April 16, 2016 onwards for ~6 weeks)
    df["is_earthquake_event"] = df["description"].str.contains("Terremoto", case=False, na=False).astype(int)

    # Aggregate daily holiday flags
    daily_holidays = df.groupby("date").agg(
        has_holiday=("type", lambda x: int(len(x) > 0)),
        has_national_holiday=("locale", lambda x: int("National" in x.values)),
        has_event=("type", lambda x: int("Event" in x.values or "Additional" in x.values or "Bridge" in x.values)),
        holiday_count=("type", "count")
    ).reset_index()

    return daily_holidays


def build_weekly_panel(
    raw_data: Dict[str, pd.DataFrame],
    trim_partial_weeks: bool = True
) -> pd.DataFrame:
    """
    Resample and aggregate daily FMCG transactions into a consistent, leakage-safe weekly panel
    across all (market_id, sku_id) combinations.
    """
    train_df = raw_data["train"].copy()
    stores_df = raw_data["stores"].copy()
    oil_clean = clean_oil_data(raw_data["oil"])
    holidays_clean = process_holidays(raw_data["holidays"], stores_df)
    transactions_df = raw_data.get("transactions", None)

    # Standardize column naming
    train_df.rename(columns={"store_nbr": "market_id", "family": "sku_id"}, inplace=True)
    stores_df.rename(columns={"store_nbr": "market_id"}, inplace=True)

    # Map dates to week starts (Monday)
    train_df["week_start"] = train_df["date"].dt.to_period("W-SUN").dt.start_time
    oil_clean["week_start"] = oil_clean["date"].dt.to_period("W-SUN").dt.start_time
    holidays_clean["week_start"] = holidays_clean["date"].dt.to_period("W-SUN").dt.start_time

    # Aggregate weekly sales & promotions per market & sku
    weekly_sales = train_df.groupby(["week_start", "market_id", "sku_id"]).agg(
        unit_sales=("sales", "sum"),
        promo_count=("onpromotion", "sum"),
        promo_days=("onpromotion", lambda x: (x > 0).sum()),
        days_recorded=("date", "count")
    ).reset_index()

    # Aggregate weekly oil
    weekly_oil = oil_clean.groupby("week_start").agg(
        oil_price=("oil_price", "mean"),
        oil_price_std=("oil_price", "std")
    ).reset_index().fillna(0)

    # Aggregate weekly holidays
    weekly_holidays = holidays_clean.groupby("week_start").agg(
        holiday_days=("has_holiday", "sum"),
        national_holiday_days=("has_national_holiday", "sum"),
        event_days=("has_event", "sum"),
        total_holiday_events=("holiday_count", "sum")
    ).reset_index()

    # Aggregate weekly store transactions if available
    if transactions_df is not None:
        tx = transactions_df.copy()
        tx.rename(columns={"store_nbr": "market_id"}, inplace=True)
        tx["week_start"] = tx["date"].dt.to_period("W-SUN").dt.start_time
        weekly_tx = tx.groupby(["week_start", "market_id"]).agg(
            total_transactions=("transactions", "sum"),
            avg_daily_transactions=("transactions", "mean")
        ).reset_index()
    else:
        weekly_tx = None

    # Merge complete panel
    # 1. Build complete cartesian grid across all weeks, markets, skus
    all_weeks = weekly_sales["week_start"].drop_duplicates().sort_values()
    if trim_partial_weeks:
        # Check if first or last week has fewer than 7 days in train set
        week_day_counts = train_df.groupby("week_start")["date"].nunique()
        valid_weeks = week_day_counts[week_day_counts == 7].index
        # Include if at least 6 days or valid
        all_weeks = all_weeks[all_weeks.isin(valid_weeks)]

    all_markets = stores_df["market_id"].unique()
    all_skus = train_df["sku_id"].unique()

    grid = pd.MultiIndex.from_product(
        [all_weeks, all_markets, all_skus],
        names=["week_start", "market_id", "sku_id"]
    ).to_frame().reset_index(drop=True)

    # Merge sales data onto grid
    panel = pd.merge(grid, weekly_sales, on=["week_start", "market_id", "sku_id"], how="left")
    panel["unit_sales"] = panel["unit_sales"].fillna(0.0)
    panel["promo_count"] = panel["promo_count"].fillna(0)
    panel["promo_days"] = panel["promo_days"].fillna(0)
    panel["promo_intensity"] = panel["promo_days"] / 7.0

    # Merge store metadata
    panel = pd.merge(panel, stores_df, on="market_id", how="left")

    # Merge weekly oil
    panel = pd.merge(panel, weekly_oil, on="week_start", how="left")

    # Merge weekly holidays
    panel = pd.merge(panel, weekly_holidays, on="week_start", how="left")
    panel["holiday_days"] = panel["holiday_days"].fillna(0)
    panel["national_holiday_days"] = panel["national_holiday_days"].fillna(0)
    panel["event_days"] = panel["event_days"].fillna(0)
    panel["total_holiday_events"] = panel["total_holiday_events"].fillna(0)

    # Merge transactions if available
    if weekly_tx is not None:
        panel = pd.merge(panel, weekly_tx, on=["week_start", "market_id", "how"], how="left") if "how" in panel else pd.merge(panel, weekly_tx, on=["week_start", "market_id"], how="left")
        panel["total_transactions"] = panel["total_transactions"].fillna(0)

    # Add series key
    panel["series_id"] = panel["market_id"].astype(str) + "__" + panel["sku_id"].astype(str)

    # Identify series lifecycle / cold start: first non-zero sales week
    first_sales = panel[panel["unit_sales"] > 0].groupby("series_id")["week_start"].min().reset_index()
    first_sales.rename(columns={"week_start": "series_launch_week"}, inplace=True)
    panel = pd.merge(panel, first_sales, on="series_id", how="left")
    panel["is_pre_launch"] = (panel["week_start"] < panel["series_launch_week"]).astype(int)

    # Standardize column naming
    panel.rename(columns={
        "week_start": "date",
        "type": "store_type",
        "cluster": "store_cluster"
    }, inplace=True)

    panel = panel.sort_values(["series_id", "date"]).reset_index(drop=True)
    print(f"[DataLoader] Successfully built weekly panel with {len(panel):,} rows across {panel['series_id'].nunique()} series.")
    print(f"[DataLoader] Date span: {panel['date'].min().strftime('%Y-%m-%d')} to {panel['date'].max().strftime('%Y-%m-%d')} ({panel['date'].nunique()} weeks).")

    return panel


if __name__ == "__main__":
    raw = load_raw_data("data/raw")
    panel = build_weekly_panel(raw)
    output_path = "data/processed/weekly_fmcg_dataset.parquet"
    panel.to_parquet(output_path, index=False)
    print(f"[DataLoader] Saved weekly panel to {output_path}")
