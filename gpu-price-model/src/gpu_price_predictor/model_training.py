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


# ============================================================================
# SECTION 1: PREPROCESSOR BUILDERS
# ============================================================================

def build_linear_preprocessor() -> ColumnTransformer:
    """
    Build preprocessing pipeline for Linear Regression.

    Includes:
    - Numeric: Median imputation + StandardScaler
    - Categorical: Constant imputation ("Unknown") + OneHotEncoder

    Returns:
        ColumnTransformer: Configured preprocessor for linear models
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

    Includes:
    - Numeric: Median imputation (no scaling needed for tree models)
    - Categorical: Constant imputation ("Unknown") + OneHotEncoder

    Returns:
        ColumnTransformer: Configured preprocessor for gradient boosting
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


# ============================================================================
# SECTION 2: MODEL CANDIDATES
# ============================================================================

def build_candidates() -> dict[str, Pipeline]:
    """
    Create all candidate models to compare during training.

    Returns:
        dict[str, Pipeline]: Named models ready for evaluation
            - "linear_regression": Simple baseline for interpretability
            - "xgboost": Complex model for best performance
    """
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


# ============================================================================
# SECTION 3: TARGET TRANSFORMATION
# ============================================================================

def transform_target(values: np.ndarray, strategy: str) -> np.ndarray:
    """
    Apply target transformation strategy.

    Args:
        values: Raw target values (prices in LKR)
        strategy: "raw" (no transformation) or "log1p" (log(1 + x))

    Returns:
        np.ndarray: Transformed values

    Raises:
        ValueError: If strategy is unsupported
    """
    if strategy == "raw":
        return values
    if strategy == "log1p":
        return np.log1p(values)
    raise ValueError(f"Unsupported target strategy: {strategy}")


def inverse_transform_predictions(values: np.ndarray, strategy: str) -> np.ndarray:
    """
    Reverse the target transformation for predictions.

    Args:
        values: Transformed predictions
        strategy: "raw" or "log1p" (must match transform_target)

    Returns:
        np.ndarray: Predictions on original scale (prices in LKR)

    Raises:
        ValueError: If strategy is unsupported
    """
    if strategy == "raw":
        return values
    if strategy == "log1p":
        return np.expm1(values)
    raise ValueError(f"Unsupported target strategy: {strategy}")


# ============================================================================
# SECTION 4: HYPERPARAMETER TUNING
# ============================================================================

def neg_mae_on_original_scale(strategy: str):
    """
    Create a custom scorer for cross-validation on original price scale.

    Converts transformed predictions back to original scale before
    computing Mean Absolute Error.

    Args:
        strategy: Target transformation strategy ("raw" or "log1p")

    Returns:
        make_scorer object: Scorer function for GridSearchCV
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
    """
    Hyperparameter tuning for XGBoost using GridSearchCV.

    Searches combinations of:
    - max_depth: Tree depth
    - learning_rate: Gradient descent rate
    - n_estimators: Number of boosting rounds
    - subsample: Row sampling ratio
    - colsample_bytree: Column sampling ratio

    Args:
        candidate: XGBoost pipeline to tune
        x_train: Training features
        y_train: Training targets (possibly transformed)
        strategy: Target transformation strategy ("raw" or "log1p")

    Returns:
        tuple containing:
        - Pipeline: Best fitted model
        - dict: Best parameters and cross-validation MAE
    """
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


# ============================================================================
# SECTION 5: MODEL EVALUATION & FITTING
# ============================================================================

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
    Train, evaluate, and compare both target transformation strategies.

    For each strategy:
    1. Transform training targets
    2. Tune hyperparameters (XGBoost only)
    3. Generate predictions on test set
    4. Inverse transform to original scale
    5. Compute metrics and diagnostics

    Finally, selects the best strategy (prefers "raw" unless "log1p" wins clearly).

    Args:
        name: Model identifier ("linear_regression" or "xgboost")
        candidate: ML pipeline to train
        x_train: Training features
        y_train: Raw training targets (prices in LKR)
        x_test: Test features
        y_test: Raw test targets (prices in LKR)
        test_df: Full test dataframe (needed for segment metrics)

    Returns:
        dict: Comprehensive evaluation results including:
        - selected_strategy: Best transformation strategy
        - selected_pipeline: Best fitted model
        - selected_metrics: Test set performance
        - selected_segment_metrics: Performance per segment
        - selected_cv_summary: Cross-validation statistics
        - target_strategy_comparison: Both strategies' results
    """
    strategy_results: dict[str, dict[str, object]] = {}

    # Train for each target transformation strategy
    for strategy in TARGET_STRATEGIES:
        tuned_candidate = clone(candidate)
        y_train_transformed = transform_target(y_train, strategy)

        # Hyperparameter tuning for XGBoost, direct fit for Linear Regression
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

        # Generate predictions and inverse transform
        predictions = inverse_transform_predictions(
            np.asarray(tuned_candidate.predict(x_test), dtype=float),
            strategy,
        )
        # Ensure non-negative prices
        predictions = np.maximum(predictions, 0)

        # Cross-validation evaluation
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

    # Select best strategy: prefer "raw" unless "log1p" wins clearly
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


# ============================================================================
# SECTION 6: DIAGNOSTICS & FEATURE IMPORTANCE
# ============================================================================

def extract_transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """
    Extract final feature names after preprocessing.

    After one-hot encoding, categorical features expand. This retrieves
    the exact final feature names for coefficient/importance matching.

    Args:
        pipeline: Trained ML pipeline

    Returns:
        list[str]: Feature names after all transformations
    """
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    return preprocessor.get_feature_names_out().tolist()


def build_linear_diagnostics(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Generate diagnostics for Linear Regression model.

    Analyzes:
    1. Feature coefficients (magnitude, direction)
    2. Multicollinearity (high correlations between features)
    3. Condition number (numerical stability indicator)

    Args:
        pipeline: Trained Linear Regression pipeline
        x_train: Training features (needed for correlation analysis)

    Returns:
        tuple containing:
        - pd.DataFrame: Coefficients sorted by absolute value
        - dict: Diagnostics (condition number, high correlations)
    """
    transformed_feature_names = extract_transformed_feature_names(pipeline)
    transformed_train = pipeline.named_steps["preprocessor"].transform(x_train)
    transformed_train = np.asarray(transformed_train, dtype=float)

    # Feature coefficients
    coefficients = pd.DataFrame(
        {
            "feature": transformed_feature_names,
            "coefficient": pipeline.named_steps["model"].coef_,
            "abs_coefficient": np.abs(pipeline.named_steps["model"].coef_),
        }
    ).sort_values("abs_coefficient", ascending=False)

    # Multicollinearity check
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
    """
    Extract feature importance scores from XGBoost model.

    Shows which features contribute most to prediction decisions.

    Args:
        pipeline: Trained XGBoost pipeline

    Returns:
        pd.DataFrame: Features sorted by importance (descending)
    """
    transformed_feature_names = extract_transformed_feature_names(pipeline)
    importances = pipeline.named_steps["model"].feature_importances_
    return pd.DataFrame(
        {
            "feature": transformed_feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)
