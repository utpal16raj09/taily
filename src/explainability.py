"""
src/explainability.py
FreshBasket Consumer Products - FMCG Demand Forecasting
SHAP Tree Explainability, Demand Driver Attribution, and Waterfall Decomposition Engine.
"""

import os
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap


def compute_shap_explanations(
    lgbm_model,
    sample_df: pd.DataFrame,
    output_dir: str = "outputs/figures",
    report_dir: str = "outputs/reports"
) -> Tuple[np.ndarray, shap.Explanation]:
    """
    Compute SHAP values using TreeExplainer for the trained LightGBM model.
    Generates summary plot, feature importance ranking, and saves attribution table.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    print("[Explainability] Computing SHAP TreeExplainer values...")
    feature_cols = lgbm_model.feature_cols
    cat_cols = lgbm_model.cat_cols

    X_sample = sample_df[feature_cols].copy()
    for col in cat_cols:
        X_sample[col] = X_sample[col].astype("category")

    explainer = shap.TreeExplainer(lgbm_model.model)
    shap_values = explainer.shap_values(X_sample)

    # Global Mean |SHAP| values
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    shap_importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Mean_Absolute_SHAP": mean_abs_shap
    }).sort_values("Mean_Absolute_SHAP", ascending=False)

    report_path = os.path.join(report_dir, "driver_attribution_summary.csv")
    shap_importance_df.to_csv(report_path, index=False)
    print(f"[Explainability] Saved SHAP driver attribution to {report_path}")

    # Plot 1: SHAP Global Feature Importance Bar Chart
    plt.figure(figsize=(10, 8))
    top_feats = shap_importance_df.head(15)
    sns.barplot(
        data=top_feats,
        y="Feature",
        x="Mean_Absolute_SHAP",
        palette="crest"
    )
    plt.title("Top 15 FMCG Demand Drivers (Mean Absolute SHAP Value)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Mean |SHAP Value| (Impact on Weekly Unit Sales Forecast)")
    plt.ylabel("Feature")
    plt.tight_layout()
    summary_path = os.path.join(output_dir, "shap_summary_plot.png")
    plt.savefig(summary_path, dpi=300)
    plt.close()
    print(f"[Explainability] Saved SHAP summary plot to {summary_path}")

    # Plot 2: Demand Decomposition Case Study (High Promo vs Baseline)
    # Select a high promo instance and low promo instance
    plt.figure(figsize=(12, 5))
    high_promo_idx = sample_df["promo_count"].argmax()
    
    # Feature contributions for this instance
    inst_contribs = pd.DataFrame({
        "Feature": feature_cols,
        "SHAP_Value": shap_values[high_promo_idx]
    }).sort_values("SHAP_Value", key=abs, ascending=False).head(10)

    colors = ["#2ca02c" if v > 0 else "#d62728" for v in inst_contribs["SHAP_Value"]]
    plt.barh(inst_contribs["Feature"], inst_contribs["SHAP_Value"], color=colors)
    plt.axvline(0, color="black", linestyle="--", alpha=0.7)
    plt.title(f"Demand Driver Attribution for Peak Promo Week ({sample_df.iloc[high_promo_idx]['sku_id']}, Store {sample_df.iloc[high_promo_idx]['market_id']})", fontweight="bold")
    plt.xlabel("SHAP Impact on Forecast Units (+ / -)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    case_path = os.path.join(output_dir, "shap_waterfall_case_study.png")
    plt.savefig(case_path, dpi=300)
    plt.close()
    print(f"[Explainability] Saved SHAP waterfall case study to {case_path}")

    return shap_values, shap_importance_df
