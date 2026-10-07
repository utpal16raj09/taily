"""
src/eda_analyzer.py
FreshBasket Consumer Products - FMCG Demand Forecasting
Automated Exploratory Data Analysis, Data Quality Auditing, and Visualization Engine.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def run_data_quality_audit(panel: pd.DataFrame, output_dir: str = "outputs/reports") -> pd.DataFrame:
    """
    Perform comprehensive data quality and integrity checks on the FMCG weekly panel.
    """
    os.makedirs(output_dir, exist_ok=True)
    total_records = len(panel)
    unique_series = panel["series_id"].nunique()
    unique_markets = panel["market_id"].nunique()
    unique_skus = panel["sku_id"].nunique()
    unique_weeks = panel["date"].nunique()

    # Metrics
    zero_sales_count = (panel["unit_sales"] == 0).sum()
    zero_sales_pct = (zero_sales_count / total_records) * 100.0
    missing_values = panel.isna().sum().to_dict()
    duplicate_rows = panel.duplicated(subset=["date", "market_id", "sku_id"]).sum()

    # Pre-launch / inactive records
    pre_launch_count = panel["is_pre_launch"].sum()
    pre_launch_pct = (pre_launch_count / total_records) * 100.0

    # Promo incidence
    promo_weeks_count = (panel["promo_count"] > 0).sum()
    promo_weeks_pct = (promo_weeks_count / total_records) * 100.0

    audit_records = [
        {"Metric": "Total Weekly Records", "Value": f"{total_records:,}"},
        {"Metric": "Unique Stores / Markets", "Value": str(unique_markets)},
        {"Metric": "Unique SKU Categories", "Value": str(unique_skus)},
        {"Metric": "Total Series (Store x SKU)", "Value": f"{unique_series:,}"},
        {"Metric": "Time Horizon (Weeks)", "Value": f"{unique_weeks} ({panel['date'].min().strftime('%Y-%m-%d')} to {panel['date'].max().strftime('%Y-%m-%d')})"},
        {"Metric": "Duplicate Records (date x store x sku)", "Value": str(duplicate_rows)},
        {"Metric": "Zero Sales Records", "Value": f"{zero_sales_count:,} ({zero_sales_pct:.2f}%)"},
        {"Metric": "Pre-Launch / Inactive Records", "Value": f"{pre_launch_count:,} ({pre_launch_pct:.2f}%)"},
        {"Metric": "Active Promotional Weeks", "Value": f"{promo_weeks_count:,} ({promo_weeks_pct:.2f}%)"},
        {"Metric": "Missing Target Values (unit_sales)", "Value": str(panel["unit_sales"].isna().sum())},
        {"Metric": "Missing Oil Prices", "Value": str(panel["oil_price"].isna().sum())},
        {"Metric": "Missing Holiday Data", "Value": str(panel["holiday_days"].isna().sum())},
    ]

    audit_df = pd.DataFrame(audit_records)
    audit_path = os.path.join(output_dir, "data_quality_summary.csv")
    audit_df.to_csv(audit_path, index=False)
    print(f"[EDA] Data quality audit saved to {audit_path}")
    print(audit_df.to_string(index=False))
    return audit_df


def generate_eda_visualizations(panel: pd.DataFrame, output_dir: str = "outputs/figures") -> None:
    """
    Generate professional publication-quality EDA charts for reporting.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "font.sans-serif": "DejaVu Sans",
        "font.family": "sans-serif",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 15,
        "figure.titleweight": "bold",
        "figure.dpi": 300,
    })

    # 1. Macro Sales Trend & Oil Price Relationship
    fig, ax1 = plt.subplots(figsize=(14, 6))
    weekly_total = panel.groupby("date").agg(
        total_sales=("unit_sales", "sum"),
        avg_oil=("oil_price", "mean"),
        total_promo=("promo_count", "sum")
    ).reset_index()

    color_sales = "#1f77b4"
    color_oil = "#d62728"

    ax1.plot(weekly_total["date"], weekly_total["total_sales"] / 1e6, color=color_sales, lw=2.2, label="Total Weekly FMCG Sales (M Units)")
    ax1.set_xlabel("Timeline (Weeks)")
    ax1.set_ylabel("Weekly Sales (Million Units)", color=color_sales, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_sales)

    # Earthquake marker
    ax1.axvline(pd.to_datetime("2016-04-16"), color="#ff7f0e", linestyle="--", alpha=0.8, lw=1.8, label="Ecuador Earthquake (Apr 2016)")

    ax2 = ax1.twinx()
    ax2.plot(weekly_total["date"], weekly_total["avg_oil"], color=color_oil, lw=1.8, linestyle=":", label="WTI Crude Oil ($/bbl)")
    ax2.set_ylabel("WTI Oil Price ($)", color=color_oil, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_oil)
    ax2.grid(False)

    plt.title("FreshBasket Demand History: Aggregated Weekly FMCG Sales vs. Macro Oil Price (2013-2017)", pad=15)
    fig.tight_layout()
    trend_path = os.path.join(output_dir, "eda_sales_trend.png")
    fig.savefig(trend_path)
    plt.close()
    print(f"[EDA] Saved sales trend plot to {trend_path}")

    # 2. Promotional Elasticity: Sales Distribution with vs. without Promotions
    fig, ax = plt.subplots(figsize=(12, 6))
    panel_active = panel[panel["is_pre_launch"] == 0].copy()
    top_categories = panel_active.groupby("sku_id")["unit_sales"].sum().nlargest(10).index

    cat_promo = panel_active[panel_active["sku_id"].isin(top_categories)].copy()
    cat_promo["Promo_Status"] = np.where(cat_promo["promo_count"] > 0, "On Promotion", "No Promotion")
    
    # Calculate average sales by category and promo status
    promo_lift = cat_promo.groupby(["sku_id", "Promo_Status"])["unit_sales"].mean().reset_index()

    sns.barplot(
        data=promo_lift,
        x="sku_id",
        y="unit_sales",
        hue="Promo_Status",
        palette={"On Promotion": "#2ca02c", "No Promotion": "#7f7f7f"},
        ax=ax
    )
    ax.set_title("Promotional Demand Uplift Across Top 10 FMCG Categories (Average Weekly Unit Sales)", pad=15)
    ax.set_xlabel("SKU Category")
    ax.set_ylabel("Average Weekly Sales Per Store (Units)")
    plt.xticks(rotation=40, ha="right")
    plt.legend(title="Promotion Status", frameon=True)
    fig.tight_layout()
    promo_path = os.path.join(output_dir, "eda_promo_elasticity.png")
    fig.savefig(promo_path)
    plt.close()
    print(f"[EDA] Saved promo elasticity plot to {promo_path}")

    # 3. Seasonality Heatmap: Month vs Category Sales
    fig, ax = plt.subplots(figsize=(14, 8))
    panel_active["month_name"] = panel_active["date"].dt.strftime("%b")
    panel_active["month_num"] = panel_active["date"].dt.month

    monthly_cat = panel_active.groupby(["sku_id", "month_num", "month_name"])["unit_sales"].mean().reset_index()
    monthly_cat = monthly_cat.sort_values("month_num")
    pivot_seasonality = monthly_cat.pivot(index="sku_id", columns="month_name", values="unit_sales")
    
    # Normalize by category mean for visual index (1.0 = average month)
    pivot_norm = pivot_seasonality.div(pivot_seasonality.mean(axis=1), axis=0)
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot_norm = pivot_norm.reindex(columns=[m for m in month_order if m in pivot_norm.columns])

    sns.heatmap(
        pivot_norm,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Seasonal Index (1.0 = Mean Demand)"},
        ax=ax,
        linewidths=0.5
    )
    ax.set_title("FMCG Category Seasonality Matrix (Normalized Monthly Demand Index)", pad=15)
    ax.set_xlabel("Month")
    ax.set_ylabel("SKU Category")
    fig.tight_layout()
    season_path = os.path.join(output_dir, "eda_seasonality_heatmap.png")
    fig.savefig(season_path)
    plt.close()
    print(f"[EDA] Saved seasonality heatmap to {season_path}")

    # 4. Store Clustering & Type Performance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    store_type_perf = panel_active.groupby("store_type")["unit_sales"].agg(["mean", "sum"]).reset_index()
    sns.barplot(data=store_type_perf, x="store_type", y="mean", palette="viridis", ax=ax1)
    ax1.set_title("Mean Weekly Sales per Store by Store Type", pad=12)
    ax1.set_xlabel("Store Type")
    ax1.set_ylabel("Mean Weekly Unit Sales")

    cluster_perf = panel_active.groupby("store_cluster")["unit_sales"].mean().reset_index()
    cluster_perf = cluster_perf.sort_values("unit_sales", ascending=False)
    sns.barplot(data=cluster_perf, x="store_cluster", y="unit_sales", color="#3b528b", ax=ax2, order=cluster_perf["store_cluster"])
    ax2.set_title("Store Cluster Demand Hierarchy", pad=12)
    ax2.set_xlabel("Store Cluster ID")
    ax2.set_ylabel("Mean Weekly Unit Sales")
    
    fig.tight_layout()
    store_path = os.path.join(output_dir, "eda_store_clustering.png")
    fig.savefig(store_path)
    plt.close()
    print(f"[EDA] Saved store clustering chart to {store_path}")


if __name__ == "__main__":
    panel_df = pd.read_parquet("data/processed/weekly_fmcg_dataset.parquet")
    run_data_quality_audit(panel_df)
    generate_eda_visualizations(panel_df)
