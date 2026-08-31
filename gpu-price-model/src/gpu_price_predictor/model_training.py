"""
Model Training Module
=====================

This module encapsulates all machine learning model operations:
- Building and configuring preprocessors (numeric/categorical transformers)
- Creating candidate models (Linear Regression, XGBoost)
- Hyperparameter tuning
- Model evaluation and diagnostics
- Feature importance analysis

Separates ML logic from data pipeline concerns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        "XGBoost is required for training. Install dependencies from requirements.txt."
    ) from exc

from gpu_price_predictor.pipeline import (
    CATEGORICAL_SPEC_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_SPEC_COLUMNS,
    TARGET_COLUMN,
    evaluate_predictions,
    build_segment_metrics,
)


RANDOM_STATE = 42
TARGET_STRATEGIES = ("raw", "log1p")


def build_linear_preprocessor() -> ColumnTransformer:
    """
    Build preprocessing pipeline for Linear Regression.
    Median-imputes and scales numerical features; one-hot encodes categoricals.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_SPEC_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_SPEC_COLUMNS),
        ]
    ).set_output(transform="default")


def build_xgboost_preprocessor() -> ColumnTransformer:
    """
    Build preprocessing pipeline for XGBoost.
    Tree-based estimators do not require standard scaling for numeric features.
    """
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_SPEC_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_SPEC_COLUMNS),
        ]
    ).set_output(transform="default")


def build_candidates() -> dict[str, Pipeline]:
    """Create candidate baseline models for training comparison."""
    return {
        "linear_regression": Pipeline(
            steps=[
                ("preprocessor", build_linear_preprocessor()),
                ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("preprocessor", build_xgboost_preprocessor()),
                (
                    "model",
                    XGBRegressor(
                        objective="reg:squarederror",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def transform_target(values: np.ndarray, strategy: str) -> np.ndarray:
    if strategy == "raw":
        return values
    if strategy == "log1p":
        return np.log1p(values)
    raise ValueError(f"Unsupported target strategy: {strategy}")


def inverse_transform_predictions(values: np.ndarray, strategy: str) -> np.ndarray:
    if strategy == "raw":
        return values
    if strategy == "log1p":
        return np.expm1(values)
    raise ValueError(f"Unsupported target strategy: {strategy}")


def neg_mae_on_original_scale(strategy: str):
    """
    Scorer computing MAE on original price scale after inverting target transform.
    """
    def scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        restored_true = inverse_transform_predictions(np.asarray(y_true), strategy)
        restored_pred = inverse_transform_predictions(np.asarray(y_pred), strategy)
        return -mean_absolute_error(restored_true, restored_pred)

    return make_scorer(scorer, greater_is_better=True)


def tune_xgboost(
    candidate: Pipeline,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    strategy: str,
) -> tuple[Pipeline, dict[str, float | int]]:
    """Grid-search hyperparameter tuning for XGBoost."""
    grid = GridSearchCV(
        estimator=candidate,
        param_grid={
            "model__max_depth": [4, 6],
            "model__learning_rate": [0.05, 0.1],
            "model__n_estimators": [200, 350],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        },
        cv=5,
        scoring=neg_mae_on_original_scale(strategy),
        n_jobs=1,
        refit=True,
    )
    grid.fit(x_train, y_train)
    best_params = {
        key.replace("model__", ""): value
        for key, value in grid.best_params_.items()
    }
    best_params["best_cv_mae_lkr"] = round(float(-grid.best_score_), 2)
    return grid.best_estimator_, best_params


def fit_and_evaluate_model(
    name: str,
    candidate: Pipeline,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    test_df: pd.DataFrame,
) -> dict[str, object]:
    """
    Train and evaluate model across both raw and log target transformations.
    """
    strategy_results: dict[str, dict[str, object]] = {}

    for strategy in TARGET_STRATEGIES:
        tuned_candidate = clone(candidate)
        y_train_transformed = transform_target(y_train, strategy)

        tuning_details: dict[str, float | int] = {}
        if name == "xgboost":
            tuned_candidate, tuning_details = tune_xgboost(
                tuned_candidate,
                x_train,
                y_train_transformed,
                strategy,
            )
        else:
            tuned_candidate.fit(x_train, y_train_transformed)

        predictions = inverse_transform_predictions(
            np.asarray(tuned_candidate.predict(x_test), dtype=float),
            strategy,
        )
        predictions = np.maximum(predictions, 0)

        try:
            cv_scores = -cross_val_score(
                clone(tuned_candidate),
                x_train,
                y_train_transformed,
                cv=5,
                scoring=neg_mae_on_original_scale(strategy),
                n_jobs=1,
            )
            cv_summary = {
                "cv_mae_mean": round(float(cv_scores.mean()), 2),
                "cv_mae_std": round(float(cv_scores.std()), 2),
            }
        except Exception as exc:
            cv_summary = {"status": f"cv_failed: {exc}"}

        strategy_results[strategy] = {
            "pipeline": tuned_candidate,
            "metrics": evaluate_predictions(y_test, predictions),
            "segment_metrics": build_segment_metrics(test_df, y_test, predictions),
            "cv_summary": cv_summary,
            "predictions": predictions.tolist(),
            "tuning_details": tuning_details,
        }

    # Prefer raw scale unless log1p yields a distinct (>100 LKR) MAE reduction
    raw_metrics = strategy_results["raw"]["metrics"]
    log_metrics = strategy_results["log1p"]["metrics"]
    preferred_strategy = "raw"
    if log_metrics["mae_lkr"] + 100 < raw_metrics["mae_lkr"]:
        preferred_strategy = "log1p"

    selected = strategy_results[preferred_strategy]
    return {
        "selected_strategy": preferred_strategy,
        "selected_pipeline": selected["pipeline"],
        "selected_metrics": selected["metrics"],
        "selected_segment_metrics": selected["segment_metrics"],
        "selected_cv_summary": selected["cv_summary"],
        "selected_predictions": selected["predictions"],
        "target_strategy_comparison": {
            strategy: {
                "metrics": result["metrics"],
                "cv_summary": result["cv_summary"],
                "tuning_details": result["tuning_details"],
            }
            for strategy, result in strategy_results.items()
        },
    }


def extract_transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Retrieve output feature names from the column transformer step."""
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    return preprocessor.get_feature_names_out().tolist()


def build_linear_diagnostics(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compute coefficients, condition number, and high correlation pairs."""
    transformed_feature_names = extract_transformed_feature_names(pipeline)
    transformed_train = pipeline.named_steps["preprocessor"].transform(x_train)
    transformed_train = np.asarray(transformed_train, dtype=float)

    coefficients = pd.DataFrame(
        {
            "feature": transformed_feature_names,
            "coefficient": pipeline.named_steps["model"].coef_,
            "abs_coefficient": np.abs(pipeline.named_steps["model"].coef_),
        }
    ).sort_values("abs_coefficient", ascending=False)

    correlation_matrix = x_train[NUMERIC_SPEC_COLUMNS].corr(numeric_only=True)
    high_corr_pairs: list[dict[str, object]] = []
    for idx, left_name in enumerate(correlation_matrix.columns):
        for right_name in correlation_matrix.columns[idx + 1 :]:
            corr_value = float(correlation_matrix.loc[left_name, right_name])
            if abs(corr_value) >= 0.9:
                high_corr_pairs.append(
                    {
                        "left": left_name,
                        "right": right_name,
                        "correlation": round(corr_value, 4),
                    }
                )

    diagnostics = {
        "condition_number": round(float(np.linalg.cond(transformed_train)), 2),
        "high_numeric_correlations": high_corr_pairs,
    }
    return coefficients, diagnostics


def build_xgboost_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Extract feature importance weights from fitted XGBoost model."""
    transformed_feature_names = extract_transformed_feature_names(pipeline)
    importances = pipeline.named_steps["model"].feature_importances_
    return pd.DataFrame(
        {
            "feature": transformed_feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)
