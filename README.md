# FreshBasket Consumer Products: FMCG Demand Forecasting Solution

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7.0-green.svg)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-TreeExplainer-red.svg)](https://shap.readthedocs.io/)

A production-grade, reproducible Machine Learning demand forecasting system engineered for **FreshBasket Consumer Products**. The solution predicts weekly unit sales across **1,782 SKU–market combinations** (54 retail stores $\times$ 33 product categories) for a **13-week planning horizon**, quantifies uncertainty through multi-quantile prediction intervals, attributes demand drivers via Tree SHAP, and stress-tests commercial plans with a scenario simulator.

---

## Key Performance Highlights

| Metric | Baseline (Seasonal Naive) | Moving Average | **Global LightGBM (Ours)** | **Hybrid Ensemble (Ours)** | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **13-Week WMAPE** | 22.05% | 31.31% | **8.75%** | **8.86%** | **-60.3% Error Reduction** |
| **RMSE** | 2,774.19 | 3,192.02 | **1,111.77** | **1,082.52** | **-61.0% Variance Reduction** |
| **MAE** | 741.50 | 1,052.81 | **294.38** | **297.73** | **-60.3% Error Reduction** |
| **Forecast Bias** | +15.16% | +29.84% | **-1.16%** | **+0.75%** | **Unbiased (< 1.2%)** |

---

## Repository Structure

```
├── data/
│   ├── raw/                           # Raw input datasets (train, test, stores, oil, holidays, transactions)
│   └── processed/                     # Processed weekly panel, modeling features, and future templates
│       ├── weekly_fmcg_dataset.parquet
│       ├── modeling_features.parquet
│       └── future_13w_template.parquet
├── src/                               # Modular Python source code
│   ├── __init__.py
│   ├── data_loader.py                 # Data ingestion, schema validation, weekly resampling
│   ├── eda_analyzer.py                # Automated data quality audit, anomaly detection & plotting
│   ├── features.py                    # Leakage-free temporal, promotional, calendar & macro features
│   ├── validation.py                  # Expanding-window rolling-origin cross-validation engine
│   ├── models.py                      # Statistical, LightGBM, XGBoost, Quantile & Ensemble models
│   ├── evaluate.py                    # WMAPE, RMSE, MAE, Bias & multi-level segment evaluation
│   ├── explainability.py              # SHAP TreeExplainer, driver attribution & waterfall decomposition
│   ├── ablation.py                    # Feature group ablation experiment engine
│   └── scenario_simulator.py          # Commercial what-if sensitivity simulator
├── notebooks/                         # Interactive Jupyter Notebooks
│   ├── 01_eda_and_data_quality.ipynb
│   ├── 02_model_development_and_validation.ipynb
│   └── 03_explainability_and_scenarios.ipynb
├── outputs/
│   ├── figures/                       # High-resolution (300 DPI) publication-quality charts
│   │   ├── eda_sales_trend.png
│   │   ├── eda_promo_elasticity.png
│   │   ├── eda_seasonality_heatmap.png
│   │   ├── eda_store_clustering.png
│   │   ├── model_comparison_cv.png
│   │   ├── cv_forecast_vs_actual.png
│   │   ├── forecast_13w_fan_chart.png
│   │   ├── shap_summary_plot.png
│   │   ├── shap_waterfall_case_study.png
│   │   ├── ablation_study_metrics.png
│   │   └── scenario_simulation_impact.png
│   ├── reports/                       # Quantitative evaluation summaries and CSV reports
│   │   ├── data_quality_summary.csv
│   │   ├── model_comparison_results.csv
│   │   ├── sku_market_metrics_breakdown.csv
│   │   ├── driver_attribution_summary.csv
│   │   ├── ablation_study_results.csv
│   │   └── scenario_simulation_results.csv
│   └── forecasts/                     # 13-week future operational forecasts
│       ├── forecast_next_13_weeks.csv
│       └── forecast_next_13_weeks_with_intervals.csv
├── run_pipeline.py                    # Master CLI pipeline orchestrator
├── generate_notebooks.py              # Notebook generation script
├── PROJECT_REPORT.md                  # Comprehensive Technical & Executive Assessment Report
├── PRESENTATION_DECK.md               # 10-15 min Executive Presentation Deck
├── requirements.txt                   # Locked production dependencies
└── README.md                          # Project documentation and reproducibility guide
```

---

## Environment Setup & Installation

### Option 1: Using `uv` (Fastest)
```bash
# 1. Create a virtual environment with Python 3.11
uv venv .venv --python 3.11

# 2. Activate environment
source .venv/bin/activate

# 3. Install required dependencies
uv pip install -r requirements.txt
```

### Option 2: Using standard `pip`
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Step-by-Step Execution

### 1. Run Complete End-to-End Pipeline
To execute data ingestion, quality audit, feature engineering, time-series cross-validation, production model training, uncertainty quantification, SHAP explainability, ablation study, and commercial scenario simulation in a single command:

```bash
python run_pipeline.py
```
*(Total runtime: ~2.5 to 3.0 minutes on standard multi-core CPU).*

### 2. Run Individual Modules Separately
```bash
# Ingest raw data and build weekly panel
python src/data_loader.py

# Run Data Quality Audit & Generate EDA Figures
python src/eda_analyzer.py

# Generate 62 Leakage-Free Features
python src/features.py
```

### 3. Launch Interactive Jupyter Notebooks
```bash
jupyter lab notebooks/
```
* `01_eda_and_data_quality.ipynb`: Visual EDA, missingness audit, promotional uplift matrix.
* `02_model_development_and_validation.ipynb`: Time-series CV, benchmark tournament, category breakdowns.
* `03_explainability_and_scenarios.ipynb`: SHAP attributions, feature ablation charts, scenario simulator.

---

## Core Modeling Methodology

### 1. Leakage-Free Validation Strategy
* **Expanding-Window Rolling-Origin Time-Series Splits**: Simulates real-world quarterly forecasting without temporal lookahead bias.
* **Horizon**: Exactly 13 weeks out-of-time per evaluation fold.
* **Lag Discipline**: All lag, rolling mean, rolling standard deviation, and momentum signals are computed strictly on $t-1$ or earlier shifted series.

### 2. Global Machine Learning Architecture
* **Global Multi-Series LightGBM Regressor**: Pools all 1,782 SKU–market series into a unified gradient boosting model, allowing cross-series learning of promotional elasticity, calendar seasonality, and macroeconomic responses.
* **Loss Function**: Tweedie distribution loss ($\rho=1.15$) to robustly accommodate zero-inflated, right-skewed FMCG unit distributions.
* **Quantile Loss**: Separate models trained at $\alpha \in \{0.10, 0.50, 0.90\}$ for calibrated 80% prediction intervals.

### 3. Commercial Sensitivity Simulator
Simulates demand changes under 4 operational scenarios:
1. **Status Quo (Planned Baseline)**: Standard promotional and macroeconomic plan.
2. **Promotional Expansion (+25%)**: Evaluates capacity requirements for aggressive trade marketing pushes (+11.8% volume increase).
3. **Trade Promotion Freeze (0%)**: Uncovers pure organic customer demand (-17.6% volume drop).
4. **Macro Oil Shock (+30%)**: Simulates fuel/inflation pressure (+2.3% essential basket volume shift).

---

## Deliverables Summary

* [PROJECT_REPORT.md](file:///home/utpal/assignment/PROJECT_REPORT.md): In-depth technical & business report.
* [PRESENTATION_DECK.md](file:///home/utpal/assignment/PRESENTATION_DECK.md): 10-15 minute slide deck for executive presentation.
* [forecast_next_13_weeks.csv](file:///home/utpal/assignment/outputs/forecasts/forecast_next_13_weeks.csv): 23,166 forecast rows with P10, P50, P90 uncertainty bounds.
* [model_comparison_results.csv](file:///home/utpal/assignment/outputs/reports/model_comparison_results.csv): Cross-validation benchmark metrics.
* [ablation_study_results.csv](file:///home/utpal/assignment/outputs/reports/ablation_study_results.csv): Quantitative feature group impact rankings.
* [scenario_simulation_results.csv](file:///home/utpal/assignment/outputs/reports/scenario_simulation_results.csv): Demand predictions across commercial scenarios.

---
*Author: Data Science Intern Team for FreshBasket Consumer Products.*
