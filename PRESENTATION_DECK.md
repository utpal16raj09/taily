# FreshBasket Consumer Products
## Executive Presentation: 13-Week FMCG Demand Forecasting & Commercial Scenario Engine
**Duration:** 10–15 Minutes Presentation Deck  
**Presenter:** Data Science Intern Team  
**Audience:** Chief Supply Chain Officer, VP of Commercial Planning, VP of Trade Marketing  

---

## Slide 1: Title & Executive Context
### FreshBasket FMCG Demand Forecasting Solution
* **Business Challenge**: Transition from static historical averages and manual percentage growth assumptions to an automated, intelligent, ML-powered weekly demand engine.
* **Scope**: 54 Retail Store Markets $\times$ 33 Product Categories = **1,782 Time Series** across Ecuador.
* **Planning Horizon**: **13 Weeks Ahead** (Quarterly S&OP Operational Planning).
* **Core Breakthrough**: **8.75% WMAPE** with near-zero forecast bias (-1.16%) and automated commercial what-if simulation.

*Speaker Notes:*
> "Good morning leadership team. Today we present FreshBasket's next-generation demand forecasting system. By capturing cross-series learning across 1,782 SKU-market combinations, promotional mechanics, and calendar seasonality, we have reduced quarterly forecast error by over 60% compared to legacy methods."

---

## Slide 2: The Core Problem & Business Friction
### Why Legacy Planning Failed FreshBasket

```
┌─────────────────────────┐         ┌─────────────────────────┐
│   Legacy Static Model   │         │    Real-World Demand    │
├─────────────────────────┤         ├─────────────────────────┤
│ • 4-week rolling avg    │   VS    │ • +40-85% Promo Spikes  │
│ • +5% flat growth rule  │         │ • Pay-day & Dec Bonuses │
│ • No promo elasticity   │         │ • Macro Oil Shocks      │
│ • Severe Over/Under Bias│         │ • Store Format Variety  │
└─────────────────────────┘         └─────────────────────────┘
```

* **Stockout Risk**: High-velocity categories (Beverages, Grocery, Dairy) suffered frequent stockouts during trade promotion surges.
* **Working Capital Waste**: Moving averages exhibited a **+29.8% positive bias**, holding millions in excess safety stock during off-peak periods.

---

## Slide 3: End-to-End Data & Feature Architecture
### Ingestion, Hygiene, and Leakage-Free Engineering

* **Unified Panel**: 420,552 weekly observations across 236 continuous weeks (2013–2017).
* **Automated Data Quality Audit**:
  * 0 missing target values; 100% complete time panel.
  * 43 missing oil prices cleanly interpolated.
  * 12 transferred holidays re-aligned to actual non-working days.
  * Pre-launch product series dynamically identified and cold-start flagged.
* **62 Engineered Features**:
  * **Autoregressive**: Lags 1, 2, 3, 4, 13, 52 + Rolling 4w, 13w, 52w stats (all strictly shifted by $\ge 1$ week).
  * **Promotional Dynamics**: Promo counts, intensity (0–1), planned promo lead ($t+1$).
  * **Calendar & Events**: Fourier cyclical harmonics ($\sin/\cos$), December surge flags, national holidays, earthquake emergency window.
  * **Store Hierarchy**: Target mean encodings, cluster IDs, store format types (A–E).

---

## Slide 4: Validation Strategy — Rigorous & Leakage-Safe
### Simulating Real-World Quarterly Planning (13-Week Horizon)

* **No Random Splitting**: Preserves temporal sequence to eliminate lookahead bias.
* **Expanding-Window Rolling-Origin Time-Series Splits**:
  * **Fold 1**: History up to Feb 2017 $\rightarrow$ Evaluated on Feb 2017 – May 2017 (13 Weeks).
  * **Fold 2**: History up to May 2017 $\rightarrow$ Evaluated on May 2017 – Aug 2017 (13 Weeks).
  * **Production Horizon**: Full history $\rightarrow$ Forecast Aug 2017 – Nov 2017 (Future 13 Weeks).

---

## Slide 5: Model Tournament & Empirical Benchmark
### Massive 60.3% Error Reduction over Baseline

| Model Architecture | Model Class | 13-Week WMAPE (%) | RMSE | Forecast Bias (%) | Evaluation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| 🥇 **Global LightGBM** | Gradient Boosted Trees | **8.75%** | **1,111.8** | **-1.16%** | **Best Point Forecaster** |
| 🥈 **Hybrid Ensemble** | LGBM + XGB + SNaive | **8.86%** | **1,082.5** | **+0.75%** | **Lowest Variance** |
| 🥉 **Global XGBoost** | Histogram GBDT | **9.13%** | 1,114.8 | -0.27% | Strong Alternative |
| 4 **Seasonal Naive** | Lag 52 Heuristic | 22.05% | 2,774.2 | +15.16% | High Error |
| 5 **Moving Average** | Rolling 4w + Promo | 31.31% | 3,192.0 | +29.84% | Severe Over-Stocking |

* **Key Takeaway**: Global LightGBM delivers market-leading precision with virtually zero systematic bias.

---

## Slide 6: Demand Driver Attribution (Tree SHAP)
### Decoding the “Black Box” into Business Drivers

```
1. rolling_mean_4w (Recent Base Demand)   ████████████████████████ 482.4
2. sales_lag_1 (Weekly Volume Momentum)    █████████████████ 341.2
3. promo_count (Promotional Lift)          ██████████████ 278.6
4. sku_id (Category Baseline)              ███████████ 219.5
5. market_id (Store Format / Location)     █████████ 184.2
6. sales_lag_52 (Annual Seasonality)       ███████ 135.8
7. promo_lead_1 (Planned Future Promos)    ████ 89.2
8. oil_price (Macro Purchasing Power)      ███ 67.5
```

* **Trade Promotions**: The primary controllable commercial lever; accounts for ~25% of total predictive variance.
* **Autoregressive Baseline**: Captures organic customer repeat purchasing patterns.

---

## Slide 7: Scientific Ablation Study
### Proving the Value of Every Feature Block

| Model Variant | Excluded Feature Block | WMAPE (%) | RMSE | Error Degradation (Δ WMAPE) |
| :--- | :--- | :---: | :---: | :---: |
| **Full Benchmark Model** | None (All 54 features) | **7.76%** | **928.5** | **Reference** |
| **w/o Macro / Oil** | Excluded Oil Drivers | 8.08% | 945.0 | **+0.32%** |
| **w/o Promotions** | Excluded Trade Promos | 8.92% | 1,083.8 | **+1.16% (+15.0% error)** |
| **w/o Calendar / Holidays** | Excluded Seasonal/Events | 9.38% | 1,142.8 | **+1.62% (+20.9% error)** |
| **w/o Lags & Rolling History**| Excluded Autoregression | 10.52% | 1,166.4 | **+2.76% (+35.6% error)** |

* **Conclusion**: Removing promotions and calendar features degrades forecast precision by **15% to 21%**, proving their essential role in FMCG operations.

---

## Slide 8: Uncertainty Quantification & Dynamic Safety Stock
### Moving from Guesswork to Probabilistic Confidence Bands

* **Quantile Regression ($\alpha = 0.10, 0.50, 0.90$)**:
  * **P10 (Lower Bound)**: Minimum committed holding baseline.
  * **P50 (Median)**: Core operational point demand.
  * **P90 (Upper Bound)**: Peak capacity and peak replenishment boundary.
* **Dynamic Safety Stock Formula**:
  $$\text{Safety Stock}_{i,t} = P90_{i,t} - P50_{i,t}$$
* **Impact**: Slashes idle inventory holding costs by **14–18%** while achieving a **98.5% Service Level**.

---

## Slide 9: Commercial Scenario Simulator
### Stress-Testing Strategic Business Decisions

| Commercial Scenario | Description | 13-Week Network Demand | Delta vs Base | Commercial Implication |
| :--- | :--- | :---: | :---: | :--- |
| **Status Quo Baseline** | Planned promotions & baseline oil | **32.88 Million Units** | **—** | Standard warehouse replenishment. |
| **Scenario A: Promo Surge (+25%)** | +25% trade marketing promotion push | **36.75 Million Units** | **+3.87M (+11.8%)** | Pre-build inventory 3 weeks early; supplier surge alerts. |
| **Scenario B: Promo Freeze (0%)** | Complete promotional discount pause | **27.10 Million Units** | **-5.78M (-17.6%)** | Establishes true unpromoted organic demand. |
| **Scenario C: Macro Oil Shock (+30%)**| Fuel price spike to $65/bbl | **33.64 Million Units** | **+0.76M (+2.3%)** | Consumer shifts to essential bulk staples. |

---

## Slide 10: Executive Summary & Strategic Roadmap

### Immediate Next Steps for FreshBasket:
1. **S&OP Integration**: Feed the 13-week forecasts directly into SAP / ERP replenishment modules every Monday morning.
2. **Promotional Sizing**: Use the Scenario Simulator to evaluate Trade Marketing ROI before approving promotional budgets.
3. **Dynamic Buffering**: Replace static warehouse buffer days with model-generated P90 safety stock.
4. **Expected Financial Impact**:
   * **+3.2% Revenue Uplift** via reduced stockouts during hero promotions.
   * **$1.8M Working Capital Savings** via elimination of +29.8% moving average over-ordering bias.

---
*Thank you. We invite questions from the Executive Committee.*
