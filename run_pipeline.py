"""
run_pipeline.py
FreshBasket Consumer Products - FMCG Demand Forecasting
Master End-to-End Orchestrator Pipeline.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_raw_data, build_weekly_panel
from src.eda_analyzer import run_data_quality_audit, generate_eda_visualizations
from src.features import engineer_features, get_feature_columns
from src.validation import TimeSeriesCrossValidator
from src.evaluate import compute_metrics, compute_segment_breakdown
from src.models import (
    SeasonalNaiveForecaster,
    MovingAverageForecaster,
    LightGBMForecaster,
    XGBoostForecaster,
    HybridEnsembleForecaster,
    QuantileLightGBMForecaster
)
from src.explainability import compute_shap_explanations
from src.ablation import run_ablation_study
from src.scenario_simulator import run_commercial_scenarios


def generate_future_13w_template(last_history_df: pd.DataFrame, n_weeks: int = 13) -> pd.DataFrame:
    """
    Construct the future 13-week panel template for all 1,782 SKU-market combinations.
    Carries forward store metadata, planned calendar/holidays, recent lag states, and projected macro signals.
    """
    print(f"[Pipeline] Generating future {n_weeks}-week operational forecasting template...")
    last_date = last_history_df["date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=n_weeks, freq="W-MON")

    all_series = last_history_df[["series_id", "market_id", "sku_id", "city", "state", "store_type", "store_cluster", "series_launch_week"]].drop_duplicates()

    records = []
    for dt in future_dates:
        temp = all_series.copy()
        temp["date"] = dt
        records.append(temp)

    future_df = pd.concat(records, ignore_index=True)

    # Estimate future promotional activity based on historical category promotional density
    recent_promo_rates = last_history_df[last_history_df["date"] >= (last_date - pd.Timedelta(weeks=13))].groupby("series_id").agg(
        promo_count=("promo_count", "mean"),
        promo_days=("promo_days", "mean")
    ).reset_index()

    future_df = pd.merge(future_df, recent_promo_rates, on="series_id", how="left")
    future_df["promo_count"] = future_df["promo_count"].fillna(0)
    future_df["promo_days"] = future_df["promo_days"].fillna(0)
    future_df["promo_intensity"] = (future_df["promo_days"] / 7.0).clip(0, 1)
    future_df["unit_sales"] = 0.0  # Placeholder target

    # Macro & Calendar
    future_df["oil_price"] = last_history_df["oil_price"].iloc[-1]
    future_df["holiday_days"] = 0
    future_df["national_holiday_days"] = 0
    future_df["event_days"] = 0
    future_df["total_holiday_events"] = 0
    future_df["is_pre_launch"] = 0

    # Combine with tail of historical panel to compute rolling features smoothly
    combined = pd.concat([last_history_df, future_df], ignore_index=True)
    combined_featured = engineer_features(combined)

    # Extract future partition
    future_features = combined_featured[combined_featured["date"].isin(future_dates)].copy()
    return future_features


def main():
    start_time = time.time()
    print("=" * 80)
    print(" FreshBasket Consumer Products - FMCG Demand Forecasting Pipeline")
    print("=" * 80)

    # 1. Ingestion & Data Preparation
    raw_data = load_raw_data("data/raw")
    panel = build_weekly_panel(raw_data)
    os.makedirs("data/processed", exist_ok=True)
    panel.to_parquet("data/processed/weekly_fmcg_dataset.parquet", index=False)

    # 2. EDA & Data Quality Audit
    print("\n--- Step 2: Data Quality Audit & EDA ---")
    audit_df = run_data_quality_audit(panel)
    generate_eda_visualizations(panel)

    # 3. Feature Engineering
    print("\n--- Step 3: Feature Engineering ---")
    featured_df = engineer_features(panel)
    featured_df.to_parquet("data/processed/modeling_features.parquet", index=False)

    # 4. Time-Series Cross-Validation
    print("\n--- Step 4: Time-Series Cross-Validation ---")
    validator = TimeSeriesCrossValidator(horizon_weeks=13, n_splits=2)
    splits = validator.split(featured_df)

    models_to_test = {
        "Seasonal Naive Baseline": SeasonalNaiveForecaster(),
        "Moving Average Baseline": MovingAverageForecaster(),
        "Global LightGBM": LightGBMForecaster(),
        "Global XGBoost": XGBoostForecaster(),
        "Hybrid Ensemble": HybridEnsembleForecaster()
    }

    cv_results = []
    val_predictions = []

    for split in splits:
        fold = split["fold"]
        tr_df = split["train_df"]
        va_df = split["val_df"]
        print(f"\n[CV] Evaluating Fold {fold} ({split['val_start']} to {split['val_end']})...")

        for m_name, model in models_to_test.items():
            print(f"  Training {m_name}...")
            if hasattr(model, "fit"):
                if m_name in ["Global LightGBM", "Global XGBoost"]:
                    model.fit(tr_df, va_df)
                else:
                    model.fit(tr_df)

            preds = model.predict(va_df)
            metrics = compute_metrics(va_df["unit_sales"].values, preds)

            cv_results.append({
                "Fold": fold,
                "Model": m_name,
                "Val Horizon": f"{split['val_start']} to {split['val_end']}",
                "WMAPE (%)": metrics["WMAPE (%)"],
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "Bias (%)": metrics["Bias (%)"]
            })

            # Save fold 2 LightGBM predictions for deep breakdown
            if fold == 2 and m_name == "Global LightGBM":
                va_copy = va_df[["date", "market_id", "sku_id", "store_type", "store_cluster", "unit_sales", "promo_count"]].copy()
                va_copy["forecast"] = preds
                val_predictions.append(va_copy)

    cv_summary_df = pd.DataFrame(cv_results)
    agg_cv = cv_summary_df.groupby("Model").agg({
        "WMAPE (%)": "mean",
        "RMSE": "mean",
        "MAE": "mean",
        "Bias (%)": "mean"
    }).reset_index().sort_values("WMAPE (%)")

    print("\n--- Model Cross-Validation Benchmark Summary ---")
    print(agg_cv.to_string(index=False))
    cv_summary_df.to_csv("outputs/reports/model_comparison_results.csv", index=False)

    # Plot Model Comparison
    plt.figure(figsize=(10, 5))
    sns.barplot(data=agg_cv, x="WMAPE (%)", y="Model", palette="viridis")
    plt.title("Cross-Validation Performance: Mean 13-Week Horizon WMAPE (%) [Lower is Better]", pad=12, fontweight="bold")
    plt.xlabel("Validation WMAPE (%)")
    plt.tight_layout()
    plt.savefig("outputs/figures/model_comparison_cv.png", dpi=300)
    plt.close()

    # 5. Segment Breakdown for Best Model
    if val_predictions:
        best_val = val_predictions[0]
        segment_df = compute_segment_breakdown(best_val, actual_col="unit_sales", pred_col="forecast", group_cols=["sku_id"])
        segment_df.to_csv("outputs/reports/sku_market_metrics_breakdown.csv", index=False)

        # Plot Forecast vs Actual for Top Categories
        plt.figure(figsize=(14, 6))
        top_cats = best_val.groupby("sku_id")["unit_sales"].sum().nlargest(4).index
        val_top = best_val[best_val["sku_id"].isin(top_cats)].groupby(["date", "sku_id"]).agg(
            actual=("unit_sales", "sum"),
            pred=("forecast", "sum")
        ).reset_index()

        for cat in top_cats:
            sub = val_top[val_top["sku_id"] == cat]
            plt.plot(sub["date"], sub["actual"], marker="o", label=f"Actual: {cat}", lw=2)
            plt.plot(sub["date"], sub["pred"], marker="x", linestyle="--", label=f"Forecast: {cat}", lw=2)

        plt.title("13-Week Validation Forecast vs Actual Demand for Top FMCG Categories", pad=12, fontweight="bold")
        plt.xlabel("Validation Week")
        plt.ylabel("Total Category Sales (Units)")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig("outputs/figures/cv_forecast_vs_actual.png", dpi=300)
        plt.close()

    # 6. Final Production Model Training & Future 13-Week Forecasting
    print("\n--- Step 6: Final Production Model Training & Future 13-Week Forecasting ---")
    active_history = featured_df[featured_df["is_pre_launch"] == 0].copy()
    
    prod_lgbm = LightGBMForecaster(n_estimators=750, learning_rate=0.035)
    prod_lgbm.fit(active_history)

    prod_quantile = QuantileLightGBMForecaster(quantiles=[0.10, 0.50, 0.90], n_estimators=450)
    prod_quantile.fit(active_history)

    future_13w_template = generate_future_13w_template(featured_df, n_weeks=13)
    future_13w_template.to_parquet("data/processed/future_13w_template.parquet", index=False)

    # Generate Point Forecast & Quantile Intervals
    point_forecast = prod_lgbm.predict(future_13w_template)
    intervals = prod_quantile.predict_intervals(future_13w_template)

    future_forecasts = future_13w_template[["date", "market_id", "sku_id", "city", "state", "store_type", "store_cluster"]].copy()
    future_forecasts["forecast"] = np.round(point_forecast, 2)
    future_forecasts["p10_lower_bound"] = np.round(intervals["p10_lower"], 2)
    future_forecasts["p50_median"] = np.round(intervals["p50_median"], 2)
    future_forecasts["p90_upper_bound"] = np.round(intervals["p90_upper"], 2)
    future_forecasts["uncertainty_range"] = np.round(intervals["uncertainty_spread"], 2)

    # Save standard & interval forecast files
    os.makedirs("outputs/forecasts", exist_ok=True)
    future_forecasts.to_csv("outputs/forecasts/forecast_next_13_weeks_with_intervals.csv", index=False)
    
    simple_forecast = future_forecasts[["date", "market_id", "sku_id", "forecast", "p10_lower_bound", "p90_upper_bound", "uncertainty_range"]]
    simple_forecast.to_csv("outputs/forecasts/forecast_next_13_weeks.csv", index=False)
    print(f"[Pipeline] Saved 13-week forecasts ({len(simple_forecast):,} rows) to outputs/forecasts/forecast_next_13_weeks.csv")

    # Plot 13-Week Future Fan Chart
    plt.figure(figsize=(14, 6))
    weekly_future = future_forecasts.groupby("date").agg(
        total_forecast=("forecast", "sum"),
        total_p10=("p10_lower_bound", "sum"),
        total_p90=("p90_upper_bound", "sum")
    ).reset_index()

    history_tail = panel.groupby("date")["unit_sales"].sum().reset_index().tail(26)

    plt.plot(history_tail["date"], history_tail["unit_sales"] / 1e6, label="Historical Actual Sales", color="#1f77b4", lw=2.5)
    plt.plot(weekly_future["date"], weekly_future["total_forecast"] / 1e6, label="FreshBasket 13-Week Forecast", color="#2ca02c", lw=2.5, linestyle="--", marker="o")
    plt.fill_between(
        weekly_future["date"],
        weekly_future["total_p10"] / 1e6,
        weekly_future["total_p90"] / 1e6,
        color="#2ca02c",
        alpha=0.25,
        label="80% Prediction Interval (P10 - P90)"
    )

    plt.title("FreshBasket Consumer Products: 13-Week Ahead Total Network Demand Forecast with Uncertainty Bounds", pad=12, fontweight="bold")
    plt.xlabel("Timeline (Weeks)")
    plt.ylabel("Total Demand (Million Units)")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig("outputs/figures/forecast_13w_fan_chart.png", dpi=300)
    plt.close()

    # 7. Explainability & Driver Attribution (SHAP)
    print("\n--- Step 7: SHAP Explainability & Driver Attribution ---")
    sample_for_shap = active_history.sample(n=min(3000, len(active_history)), random_state=42)
    compute_shap_explanations(prod_lgbm, sample_for_shap)

    # 8. Ablation / Feature-Impact Study
    print("\n--- Step 8: Ablation Study ---")
    val_fold = splits[-1]
    run_ablation_study(val_fold["train_df"], val_fold["val_df"])

    # 9. Commercial Scenario Simulation
    print("\n--- Step 9: Commercial Scenario Simulator ---")
    run_commercial_scenarios(prod_lgbm, future_13w_template)

    elapsed = time.time() - start_time
    print(f"\n[Pipeline] Completed end-to-end execution in {elapsed:.1f}s ({elapsed/60:.2f} mins).")
    print("=" * 80)


if __name__ == "__main__":
    main()
