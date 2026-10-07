"""
src/models.py
FreshBasket Consumer Products - FMCG Demand Forecasting
Statistical, Machine Learning (LightGBM, XGBoost), Ensembling, and Quantile Uncertainty Forecasters.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from typing import Dict, List, Tuple, Any, Optional
from src.features import get_feature_columns


class SeasonalNaiveForecaster:
    """
    Seasonal Naive Statistical Baseline:
    Forecasts demand using the corresponding seasonal week from the prior year (lag 52),
    falling back to rolling 4-week average with promotional multiplier when prior year is unavailable.
    """

    def __init__(self):
        self.name = "Seasonal_Naive_Baseline"

    def fit(self, train_df: pd.DataFrame):
        # Stores recent history lookup
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        # If sales_lag_52 > 0, use it; otherwise fallback to rolling_mean_4w
        preds = np.where(
            test_df["sales_lag_52"] > 0,
            test_df["sales_lag_52"],
            test_df["rolling_mean_4w"]
        )
        # Apply promo lift heuristic if promo is active
        promo_multiplier = np.where(test_df["promo_count"] > 0, 1.25, 1.0)
        preds = preds * promo_multiplier
        return np.clip(preds, 0, None)


class MovingAverageForecaster:
    """
    Moving Average + Promotion Adjusted Baseline.
    """

    def __init__(self):
        self.name = "Moving_Average_Baseline"

    def fit(self, train_df: pd.DataFrame):
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        base = test_df["rolling_mean_4w"].values
        # Promotional adjustment
        promo_factor = np.where(test_df["promo_count"] > 0, 1.30, 1.0)
        return np.clip(base * promo_factor, 0, None)


class LightGBMForecaster:
    """
    Global Multi-Series LightGBM Regressor.
    Learns cross-series dynamics across 1,782 SKU-market combinations with Huber/L1 loss.
    """

    def __init__(self, objective: str = "tweedie", n_estimators: int = 600, learning_rate: float = 0.04):
        self.name = "Global_LightGBM"
        self.objective = objective
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.model = None
        self.feature_cols = None
        self.cat_cols = None

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        early_stopping_rounds: int = 50
    ):
        num_cols, cat_cols = get_feature_columns()
        self.feature_cols = [c for c in num_cols + cat_cols if c in train_df.columns]
        self.cat_cols = [c for c in cat_cols if c in self.feature_cols]

        X_train = train_df[self.feature_cols].copy()
        y_train = train_df["unit_sales"].values

        # Set categorical types
        for col in self.cat_cols:
            X_train[col] = X_train[col].astype("category")

        params = {
            "objective": self.objective,
            "tweedie_variance_power": 1.15,
            "metric": "mae",
            "learning_rate": self.learning_rate,
            "num_leaves": 63,
            "min_child_samples": 40,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "verbosity": -1,
            "n_estimators": self.n_estimators,
            "random_state": 42,
            "n_jobs": -1
        }

        self.model = lgb.LGBMRegressor(**params)

        if val_df is not None:
            X_val = val_df[self.feature_cols].copy()
            y_val = val_df["unit_sales"].values
            for col in self.cat_cols:
                X_val[col] = X_val[col].astype("category")

            callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks
            )
        else:
            self.model.fit(X_train, y_train)

        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        X_test = test_df[self.feature_cols].copy()
        for col in self.cat_cols:
            X_test[col] = X_test[col].astype("category")
        preds = self.model.predict(X_test)
        return np.clip(preds, 0, None)

    def get_feature_importances(self) -> pd.DataFrame:
        imp = self.model.feature_importances_
        return pd.DataFrame({
            "Feature": self.feature_cols,
            "Importance": imp
        }).sort_values("Importance", ascending=False)


class XGBoostForecaster:
    """
    Global Multi-Series XGBoost Regressor with enable_categorical=True.
    """

    def __init__(self, n_estimators: int = 500, learning_rate: float = 0.05):
        self.name = "Global_XGBoost"
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.model = None
        self.feature_cols = None
        self.cat_cols = None

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        num_cols, cat_cols = get_feature_columns()
        self.feature_cols = [c for c in num_cols + cat_cols if c in train_df.columns]
        self.cat_cols = [c for c in cat_cols if c in self.feature_cols]

        X_train = train_df[self.feature_cols].copy()
        y_train = train_df["unit_sales"].values

        for col in self.cat_cols:
            X_train[col] = X_train[col].astype("category")

        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            tree_method="hist",
            enable_categorical=True,
            random_state=42,
            n_jobs=-1
        )

        if val_df is not None:
            X_val = val_df[self.feature_cols].copy()
            y_val = val_df["unit_sales"].values
            for col in self.cat_cols:
                X_val[col] = X_val[col].astype("category")
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train, verbose=False)

        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        X_test = test_df[self.feature_cols].copy()
        for col in self.cat_cols:
            X_test[col] = X_test[col].astype("category")
        preds = self.model.predict(X_test)
        return np.clip(preds, 0, None)


class QuantileLightGBMForecaster:
    """
    Probabilistic Multi-Quantile LightGBM Forecaster.
    Fits lower bound (alpha=0.10), median (alpha=0.50), and upper bound (alpha=0.90) for calibrated 80% prediction intervals.
    """

    def __init__(self, quantiles: List[float] = None, n_estimators: int = 400):
        self.name = "Probabilistic_Quantile_LGBM"
        self.quantiles = quantiles or [0.10, 0.50, 0.90]
        self.n_estimators = n_estimators
        self.models = {}
        self.feature_cols = None
        self.cat_cols = None

    def fit(self, train_df: pd.DataFrame):
        num_cols, cat_cols = get_feature_columns()
        self.feature_cols = [c for c in num_cols + cat_cols if c in train_df.columns]
        self.cat_cols = [c for c in cat_cols if c in self.feature_cols]

        X_train = train_df[self.feature_cols].copy()
        y_train = train_df["unit_sales"].values

        for col in self.cat_cols:
            X_train[col] = X_train[col].astype("category")

        print(f"[QuantileModel] Fitting {len(self.quantiles)} quantile models ({self.quantiles})...")
        for q in self.quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                learning_rate=0.05,
                num_leaves=45,
                n_estimators=self.n_estimators,
                min_child_samples=40,
                verbosity=-1,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            self.models[q] = model

        return self

    def predict_intervals(self, test_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        X_test = test_df[self.feature_cols].copy()
        for col in self.cat_cols:
            X_test[col] = X_test[col].astype("category")

        p10 = np.clip(self.models[0.10].predict(X_test), 0, None)
        p50 = np.clip(self.models[0.50].predict(X_test), 0, None)
        p90 = np.clip(self.models[0.90].predict(X_test), 0, None)

        # Monotonicity adjustment: ensure P10 <= P50 <= P90
        p50 = np.maximum(p10, p50)
        p90 = np.maximum(p50, p90)

        return {
            "p10_lower": p10,
            "p50_median": p50,
            "p90_upper": p90,
            "uncertainty_spread": p90 - p10
        }


class HybridEnsembleForecaster:
    """
    Hybrid Weighted Ensemble combining Global LightGBM, XGBoost, and Seasonal Naive Baseline.
    """

    def __init__(self, weights: Dict[str, float] = None):
        self.name = "Hybrid_Ensemble"
        self.weights = weights or {"lgbm": 0.60, "xgb": 0.30, "snaive": 0.10}
        self.lgbm = LightGBMForecaster()
        self.xgb = XGBoostForecaster()
        self.snaive = SeasonalNaiveForecaster()

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        print("[Ensemble] Fitting constituent models...")
        self.lgbm.fit(train_df, val_df)
        self.xgb.fit(train_df, val_df)
        self.snaive.fit(train_df)
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        p_lgbm = self.lgbm.predict(test_df)
        p_xgb = self.xgb.predict(test_df)
        p_snaive = self.snaive.predict(test_df)

        w = self.weights
        ensemble_pred = w["lgbm"] * p_lgbm + w["xgb"] * p_xgb + w["snaive"] * p_snaive
        return np.clip(ensemble_pred, 0, None)
