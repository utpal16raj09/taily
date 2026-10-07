"""
src/evaluate.py
FreshBasket Consumer Products - FMCG Demand Forecasting
Evaluation Metrics, Forecast Accuracy, Bias, and Multi-Level Segment Diagnostics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_train_history: np.ndarray = None) -> Dict[str, float]:
    """
    Compute comprehensive FMCG forecasting metrics:
    - WMAPE: Weighted Mean Absolute Percentage Error (sum(|y - y_hat|) / sum(y))
    - RMSE: Root Mean Squared Error
    - MAE: Mean Absolute Error
    - Forecast Bias: (sum(y_hat - y) / sum(y)) * 100%
    - RMSSE: Root Mean Squared Scaled Error (if historical train series provided)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 0, None)  # Sales cannot be negative

    sum_true = np.sum(y_true)
    sum_pred = np.sum(y_pred)
    abs_errors = np.abs(y_true - y_pred)
    sq_errors = (y_true - y_pred) ** 2

    # WMAPE (Industry Standard)
    wmape = (np.sum(abs_errors) / (sum_true + 1e-6)) * 100.0

    # MAE & RMSE
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(sq_errors))

    # Bias percentage
    bias_pct = ((sum_pred - sum_true) / (sum_true + 1e-6)) * 100.0

    # RMSSE (M5 competition standard)
    if y_train_history is not None and len(y_train_history) > 1:
        naive_sq_diff = np.mean((y_train_history[1:] - y_train_history[:-1]) ** 2)
        rmsse = np.sqrt(np.mean(sq_errors) / (naive_sq_diff + 1e-6))
    else:
        rmsse = np.nan

    return {
        "WMAPE (%)": round(float(wmape), 2),
        "RMSE": round(float(rmse), 2),
        "MAE": round(float(mae), 2),
        "Bias (%)": round(float(bias_pct), 2),
        "RMSSE": round(float(rmsse), 3) if not np.isnan(rmsse) else None
    }


def compute_segment_breakdown(
    df: pd.DataFrame,
    actual_col: str = "unit_sales",
    pred_col: str = "forecast",
    group_cols: list = None
) -> pd.DataFrame:
    """
    Compute segment-level forecasting performance breakdown across Categories, Store Types, Clusters, etc.
    """
    if group_cols is None:
        group_cols = ["sku_id"]

    results = []
    for keys, group in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        
        m = compute_metrics(group[actual_col].values, group[pred_col].values)
        row = dict(zip(group_cols, keys))
        row.update({
            "Total Actual Sales": round(group[actual_col].sum(), 1),
            "Total Forecast Sales": round(group[pred_col].sum(), 1),
            "WMAPE (%)": m["WMAPE (%)"],
            "RMSE": m["RMSE"],
            "MAE": m["MAE"],
            "Bias (%)": m["Bias (%)"],
            "Record Count": len(group)
        })
        results.append(row)

    return pd.DataFrame(results).sort_values("Total Actual Sales", ascending=False)
