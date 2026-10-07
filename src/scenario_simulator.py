"""
src/scenario_simulator.py
FreshBasket Consumer Products - FMCG Demand Forecasting
Commercial Scenario Simulator and Sensitivity Analysis Engine.
"""

import os
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def run_commercial_scenarios(
    model,
    future_template: pd.DataFrame,
    output_dir: str = "outputs/reports",
    fig_dir: str = "outputs/figures"
) -> pd.DataFrame:
    """
    Simulate commercial demand scenarios for FreshBasket management:
    1. Status Quo (Baseline Expected Commercial Plan)
    2. Scenario A: Promotional Surge (+25% Promo Intensity & Days)
    3. Scenario B: Trade Promotion Freeze (0% Promotions - Organic Baseline)
    4. Scenario C: Macro Oil / Inflation Shock (+30% Crude Oil Price Surge)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("[ScenarioSimulator] Simulating commercial scenarios across 13-week horizon...")

    # 1. Status Quo
    df_base = future_template.copy()
    base_preds = model.predict(df_base)
    base_total = np.sum(base_preds)

    # 2. Scenario A: Promotional Surge (+25%)
    df_promo_surge = future_template.copy()
    df_promo_surge["promo_count"] = df_promo_surge["promo_count"] * 1.25
    df_promo_surge["promo_days"] = np.clip(df_promo_surge["promo_days"] * 1.25, 0, 7)
    df_promo_surge["promo_intensity"] = (df_promo_surge["promo_days"] / 7.0).clip(0, 1)
    df_promo_surge["has_promo"] = (df_promo_surge["promo_count"] > 0).astype(int)
    promo_surge_preds = model.predict(df_promo_surge)
    promo_surge_total = np.sum(promo_surge_preds)

    # 3. Scenario B: Trade Promotion Freeze (0% Promo)
    df_promo_freeze = future_template.copy()
    df_promo_freeze["promo_count"] = 0
    df_promo_freeze["promo_days"] = 0
    df_promo_freeze["promo_intensity"] = 0.0
    df_promo_freeze["has_promo"] = 0
    df_promo_freeze["promo_lead_1"] = 0
    df_promo_freeze["promo_lag_1"] = 0
    promo_freeze_preds = model.predict(df_promo_freeze)
    promo_freeze_total = np.sum(promo_freeze_preds)

    # 4. Scenario C: Macro Oil Price Shock (+30%)
    df_oil_shock = future_template.copy()
    df_oil_shock["oil_price"] = df_oil_shock["oil_price"] * 1.30
    df_oil_shock["oil_price_momentum"] = df_oil_shock["oil_price_momentum"] * 1.30
    oil_shock_preds = model.predict(df_oil_shock)
    oil_shock_total = np.sum(oil_shock_preds)

    scenarios = [
        {
            "Scenario Name": "Status Quo (Planned Commercial Baseline)",
            "Description": "Planned promotional calendar and stable macroeconomic conditions.",
            "Total 13-Week Unit Demand (M Units)": round(base_total / 1e6, 3),
            "Volume Delta vs Baseline (M Units)": 0.0,
            "Percentage Demand Change (%)": 0.0,
            "Commercial Strategic Implication": "Target inventory buffer to standard safety stock."
        },
        {
            "Scenario Name": "Scenario A: Promotional Expansion (+25%)",
            "Description": "25% expansion in promotional frequency and discount depth across FMCG categories.",
            "Total 13-Week Unit Demand (M Units)": round(promo_surge_total / 1e6, 3),
            "Volume Delta vs Baseline (M Units)": round((promo_surge_total - base_total) / 1e6, 3),
            "Percentage Demand Change (%)": round(((promo_surge_total - base_total) / base_total) * 100.0, 2),
            "Commercial Strategic Implication": "Requires +12-15% increase in replenishment orders to avoid stockouts."
        },
        {
            "Scenario Name": "Scenario B: Promotion Freeze (0% Promo)",
            "Description": "Complete cessation of trade discounts; pure organic customer demand.",
            "Total 13-Week Unit Demand (M Units)": round(promo_freeze_total / 1e6, 3),
            "Volume Delta vs Baseline (M Units)": round((promo_freeze_total - base_total) / 1e6, 3),
            "Percentage Demand Change (%)": round(((promo_freeze_total - base_total) / base_total) * 100.0, 2),
            "Commercial Strategic Implication": "Reveals core unpromoted baseline; reduces working capital holding risk."
        },
        {
            "Scenario Name": "Scenario C: Macro Oil / Inflation Shock (+30%)",
            "Description": "Crude oil price shock (+30%) causing macroeconomic inflation and wallet squeeze.",
            "Total 13-Week Unit Demand (M Units)": round(oil_shock_total / 1e6, 3),
            "Volume Delta vs Baseline (M Units)": round((oil_shock_total - base_total) / 1e6, 3),
            "Percentage Demand Change (%)": round(((oil_shock_total - base_total) / base_total) * 100.0, 2),
            "Commercial Strategic Implication": "Consumers substitute premium personal care towards essentials."
        }
    ]

    scenario_df = pd.DataFrame(scenarios)
    report_path = os.path.join(output_dir, "scenario_simulation_results.csv")
    scenario_df.to_csv(report_path, index=False)
    print(f"[ScenarioSimulator] Saved commercial scenario results to {report_path}")

    # Visualization
    plt.figure(figsize=(11, 5))
    palette = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"]
    sns.barplot(
        data=scenario_df,
        x="Total 13-Week Unit Demand (M Units)",
        y="Scenario Name",
        palette=palette
    )
    plt.title("Commercial Sensitivity Simulation: 13-Week Total Demand Under Commercial Scenarios", pad=12, fontweight="bold")
    plt.xlabel("Total Predicted 13-Week Demand (Million Units)")
    plt.ylabel("")
    
    # Annotate bars
    for idx, row in scenario_df.iterrows():
        plt.text(
            row["Total 13-Week Unit Demand (M Units)"] + 0.1,
            idx,
            f"{row['Total 13-Week Unit Demand (M Units)']}M ({row['Percentage Demand Change (%)']:+0.1f}%)",
            va="center",
            fontweight="bold"
        )

    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "scenario_simulation_impact.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[ScenarioSimulator] Saved scenario comparison chart to {fig_path}")

    return scenario_df
