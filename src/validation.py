"""
src/validation.py
FreshBasket Consumer Products - FMCG Demand Forecasting
Expanding-Window Rolling-Origin Time-Series Cross-Validation Engine.
"""

from typing import List, Tuple, Dict
import pandas as pd
import numpy as np


class TimeSeriesCrossValidator:
    """
    Leakage-safe expanding window time-series cross-validator designed for FMCG quarterly planning (13-week horizon).
    """

    def __init__(self, horizon_weeks: int = 13, n_splits: int = 2):
        self.horizon_weeks = horizon_weeks
        self.n_splits = n_splits

    def split(self, df: pd.DataFrame) -> List[Dict[str, pd.DataFrame]]:
        """
        Generate (train_df, val_df) splits based on unique ordered weekly dates.
        """
        dates = sorted(df["date"].unique())
        total_weeks = len(dates)
        splits = []

        for fold_idx in range(self.n_splits, 0, -1):
            val_end_idx = total_weeks - (fold_idx - 1) * self.horizon_weeks
            val_start_idx = val_end_idx - self.horizon_weeks
            train_end_idx = val_start_idx

            train_dates = dates[:train_end_idx]
            val_dates = dates[val_start_idx:val_end_idx]

            train_mask = df["date"].isin(train_dates)
            val_mask = df["date"].isin(val_dates)

            # Filter out pre-launch periods for cleaner training signals
            train_split = df[train_mask & (df["is_pre_launch"] == 0)].copy()
            val_split = df[val_mask].copy()

            split_meta = {
                "fold": self.n_splits - fold_idx + 1,
                "train_start": pd.to_datetime(train_dates[0]).strftime("%Y-%m-%d"),
                "train_end": pd.to_datetime(train_dates[-1]).strftime("%Y-%m-%d"),
                "train_weeks": len(train_dates),
                "val_start": pd.to_datetime(val_dates[0]).strftime("%Y-%m-%d"),
                "val_end": pd.to_datetime(val_dates[-1]).strftime("%Y-%m-%d"),
                "val_weeks": len(val_dates),
                "train_df": train_split,
                "val_df": val_split
            }
            splits.append(split_meta)

            print(f"[Validation] Fold {split_meta['fold']}: Train {split_meta['train_start']} to {split_meta['train_end']} ({split_meta['train_weeks']} wks) -> Val {split_meta['val_start']} to {split_meta['val_end']} ({split_meta['val_weeks']} wks)")

        return splits
