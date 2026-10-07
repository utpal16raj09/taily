"""
generate_notebooks.py
Generate clean, executable Jupyter Notebooks for EDA, Modeling, and Explainability.
"""

import nbformat as nbf
import os

os.makedirs("notebooks", exist_ok=True)

# ----------------------------------------------------
# 1. Notebook 01: EDA & Data Quality
# ----------------------------------------------------
nb1 = nbf.v4.new_notebook()
nb1.cells = [
    nbf.v4.new_markdown_cell("""# FreshBasket Consumer Products - FMCG Demand Forecasting
## Part 1: Exploratory Data Analysis & Automated Data Quality Audit

**Author:** Data Science Intern Team  
**Business Context:** FreshBasket Consumer Products plans inventory and trade promotions across 54 retail markets and 33 product categories. This notebook performs data hygiene, missingness detection, anomaly diagnosis, promotional elasticity analysis, and seasonality profiling.
"""),
    nbf.v4.new_code_cell("""import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_raw_data, build_weekly_panel
from src.eda_analyzer import run_data_quality_audit, generate_eda_visualizations

# Load Raw Data & Resample into Weekly Panel
raw_data = load_raw_data("data/raw")
panel = build_weekly_panel(raw_data)
panel.head()
"""),
    nbf.v4.new_markdown_cell("""### 1. Data Quality & Integrity Audit
We assess missing dates, zero sales distribution, pre-launch series, and promotional frequencies across the 1,782 series panel.
"""),
    nbf.v4.new_code_cell("""audit_df = run_data_quality_audit(panel)
audit_df
"""),
    nbf.v4.new_markdown_cell("""### 2. Macroeconomic & Seasonal Trends
Examine the aggregate FMCG sales volume alongside WTI crude oil movements and external shocks (e.g. 2016 Ecuador Earthquake).
"""),
    nbf.v4.new_code_cell("""# Visualizations are generated and stored in outputs/figures/
from IPython.display import Image, display

display(Image(filename="outputs/figures/eda_sales_trend.png"))
display(Image(filename="outputs/figures/eda_promo_elasticity.png"))
display(Image(filename="outputs/figures/eda_seasonality_heatmap.png"))
display(Image(filename="outputs/figures/eda_store_clustering.png"))
""")
]

with open("notebooks/01_eda_and_data_quality.ipynb", "w") as f:
    nbf.write(nb1, f)
print("Saved notebooks/01_eda_and_data_quality.ipynb")

# ----------------------------------------------------
# 2. Notebook 02: Modeling & Cross-Validation
# ----------------------------------------------------
nb2 = nbf.v4.new_notebook()
nb2.cells = [
    nbf.v4.new_markdown_cell("""# FreshBasket Consumer Products - FMCG Demand Forecasting
## Part 2: Feature Engineering, Time-Series CV & Model Benchmarking

This notebook constructs leakage-safe temporal, promotional, calendar, and structural features, and runs expanding-window rolling-origin cross-validation (13-week horizon).
"""),
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.features import engineer_features, get_feature_columns
from src.validation import TimeSeriesCrossValidator
from src.evaluate import compute_metrics, compute_segment_breakdown
from src.models import (
    SeasonalNaiveForecaster,
    MovingAverageForecaster,
    LightGBMForecaster,
    XGBoostForecaster,
    HybridEnsembleForecaster
)

# Load engineered dataset
feat_df = pd.read_parquet("data/processed/modeling_features.parquet")
print("Features shape:", feat_df.shape)
feat_df.head()
"""),
    nbf.v4.new_markdown_cell("""### 1. Cross-Validation Results Summary
Evaluating Seasonal Naive Baseline, Moving Average, Global LightGBM, Global XGBoost, and Hybrid Ensemble over 13-week planning horizons.
"""),
    nbf.v4.new_code_cell("""cv_results = pd.read_csv("outputs/reports/model_comparison_results.csv")
cv_results
"""),
    nbf.v4.new_code_cell("""from IPython.display import Image, display
display(Image(filename="outputs/figures/model_comparison_cv.png"))
display(Image(filename="outputs/figures/cv_forecast_vs_actual.png"))
"""),
    nbf.v4.new_markdown_cell("""### 2. SKU-Level & Market Breakdown
Reviewing granular forecasting accuracy (WMAPE, RMSE, Bias) across product categories.
"""),
    nbf.v4.new_code_cell("""sku_breakdown = pd.read_csv("outputs/reports/sku_market_metrics_breakdown.csv")
sku_breakdown.head(10)
""")
]

with open("notebooks/02_model_development_and_validation.ipynb", "w") as f:
    nbf.write(nb2, f)
print("Saved notebooks/02_model_development_and_validation.ipynb")

# ----------------------------------------------------
# 3. Notebook 03: Explainability, Uncertainty & Scenarios
# ----------------------------------------------------
nb3 = nbf.v4.new_notebook()
nb3.cells = [
    nbf.v4.new_markdown_cell("""# FreshBasket Consumer Products - FMCG Demand Forecasting
## Part 3: SHAP Driver Attribution, Feature Ablation & Commercial Scenario Simulator

This notebook delivers business interpretation through Tree SHAP attributions, feature group ablation studies, probabilistic uncertainty bands, and commercial what-if simulations.
"""),
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
from IPython.display import Image, display

# 1. Driver Attribution Summary
shap_summary = pd.read_csv("outputs/reports/driver_attribution_summary.csv")
shap_summary.head(10)
"""),
    nbf.v4.new_code_cell("""display(Image(filename="outputs/figures/shap_summary_plot.png"))
display(Image(filename="outputs/figures/shap_waterfall_case_study.png"))
"""),
    nbf.v4.new_markdown_cell("""### 2. Feature Group Ablation Study
Quantifying the marginal accuracy contribution of Promotions, Macro/Oil, Calendar/Holidays, and Autoregressive Lags.
"""),
    nbf.v4.new_code_cell("""ablation_df = pd.read_csv("outputs/reports/ablation_study_results.csv")
display(ablation_df)
display(Image(filename="outputs/figures/ablation_study_metrics.png"))
"""),
    nbf.v4.new_markdown_cell("""### 3. Commercial Sensitivity & What-If Simulations
Simulating 13-week demand under Promotional Expansion (+25%), Promotion Freeze (0%), and Macro Oil Price Shock (+30%).
"""),
    nbf.v4.new_code_cell("""scenario_df = pd.read_csv("outputs/reports/scenario_simulation_results.csv")
display(scenario_df)
display(Image(filename="outputs/figures/scenario_simulation_impact.png"))
display(Image(filename="outputs/figures/forecast_13w_fan_chart.png"))
""")
]

with open("notebooks/03_explainability_and_scenarios.ipynb", "w") as f:
    nbf.write(nb3, f)
print("Saved notebooks/03_explainability_and_scenarios.ipynb")
