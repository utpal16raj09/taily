"""
src/ablation.py
FreshBasket Consumer Products - FMCG Demand Forecasting
Ablation / Feature Impact Study Engine.
"""

import os
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from src.features import get_feature_columns
from src.evaluate import compute_metrics


def run_ablation_study(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: str = "outputs/reports",
    fig_dir: str = "outputs/figures"
) -> pd.DataFrame:
    """
    Evaluate the marginal predictive contribution of each feature group by training models with specific feature groups removed.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    num_cols, cat_cols = get_feature_columns()
    all_features = [c for c in num_cols + cat_cols if c in train_df.columns]

    promo_features = [
        "promo_count", "promo_days", "promo_intensity", "has_promo",
        "promo_lag_1", "promo_lead_1", "promo_rolling_mean_4w", "promo_rolling_mean_13w"
    ]
    macro_features = [
        "oil_price", "oil_price_lag_1", "oil_price_rolling_mean_4w", "oil_price_momentum"
    ]
    calendar_holiday_features = [
        "week_of_year", "month", "quarter", "year", "is_month_start", "is_month_end",
        "is_december_peak", "sin_week_52", "cos_week_52", "sin_month_12", "cos_month_12",
        "holiday_days", "national_holiday_days", "event_days", "total_holiday_events",
        "is_holiday_week", "is_national_holiday_week"
    ]
    lag_rolling_features = [
        "sales_lag_1", "sales_lag_2", "sales_lag_3", "sales_lag_4", "sales_lag_13", "sales_lag_52",
        "rolling_mean_4w", "rolling_std_4w", "rolling_min_4w", "rolling_max_4w",
        "rolling_mean_13w", "rolling_std_13w", "rolling_max_13w",
        "rolling_mean_52w", "ewm_mean_4w", "sales_ratio_lag1_rolling4", "sales_momentum_lag1_lag2"
    ]

    configurations = {
        "Full Feature Model (Benchmark)": all_features,
        "Ablation: Without Promotions": [c for c in all_features if c not in promo_features],
        "Ablation: Without Macro / Oil": [c for c in all_features if c not in macro_features],
        "Ablation: Without Calendar & Holidays": [c for c in all_features if c not in calendar_holiday_features],
        "Ablation: Without Historical Lags & Rolling": [c for c in all_features if c not in lag_rolling_features]
    }

    results = []
    y_train = train_df["unit_sales"].values
    y_val = val_df["unit_sales"].values

    print("[Ablation] Running feature ablation experiments...")
    for name, feat_subset in configurations.items():
        sub_cats = [c for c in cat_cols if c in feat_subset]
        X_tr = train_df[feat_subset].copy()
        X_va = val_df[feat_subset].copy()

        for col in sub_cats:
            X_tr[col] = X_tr[col].astype("category")
            X_va[col] = X_va[col].astype("category")

        model = lgb.LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.15,
            metric="mae",
            learning_rate=0.05,
            num_leaves=63,
            n_estimators=450,
            verbosity=-1,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_tr, y_train)
        preds = np.clip(model.predict(X_va), 0, None)
        metrics = compute_metrics(y_val, preds)

        results.append({
            "Experiment": name,
            "Features Count": len(feat_subset),
            "WMAPE (%)": metrics["WMAPE (%)"],
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "Bias (%)": metrics["Bias (%)"]
        })
        print(f"  -> {name} (N={len(feat_subset)}): WMAPE = {metrics['WMAPE (%)']}%, RMSE = {metrics['RMSE']}")

    ablation_df = pd.DataFrame(results)
    benchmark_wmape = ablation_df.loc[ablation_df["Experiment"] == "Full Feature Model (Benchmark)", "WMAPE (%)"].values[0]
    ablation_df["WMAPE Degradation (Δ%)"] = (ablation_df["WMAPE (%)"] - benchmark_wmape).round(2)

    report_path = os.path.join(output_dir, "ablation_study_results.csv")
    ablation_df.to_csv(report_path, index=False)
    print(f"[Ablation] Saved ablation results to {report_path}")

    # Visualization
    plt.figure(figsize=(11, 5))
    chart_df = ablation_df.sort_values("WMAPE (%)", ascending=True)
    sns.barplot(
        data=chart_df,
        x="WMAPE (%)",
        y="Experiment",
        palette="rocket_r"
    )
    plt.title("Ablation Study: Forecast Error (WMAPE %) Across Feature Group Exclusions", pad=12, fontweight="bold")
    plt.xlabel("Validation WMAPE (%) [Lower is Better]")
    plt.ylabel("")
    plt.axvline(benchmark_wmape, color="green", linestyle="--", label=f"Benchmark ({benchmark_wmape}%)")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "ablation_study_metrics.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[Ablation] Saved ablation chart to {fig_path}")

    return ablation_df
