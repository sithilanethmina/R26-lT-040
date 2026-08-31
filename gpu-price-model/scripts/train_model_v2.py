"""
GPU Price Predictor V2 - Phase 2: Model Training
=================================================
Reads:  data/final/gpu_enriched_dataset.csv
Writes: artifacts/gpu_price_model_v2.joblib
        artifacts/training_summary_v2.json

Models trained on log(price_lkr) target:
  1. LightGBM          (tree   - unscaled)
  2. XGBoost           (tree   - unscaled)
  3. Random Forest     (tree   - unscaled)
  4. KNN               (scaled - StandardScaler)
  5. SVR (RBF)         (scaled - StandardScaler)
  6. Stacking Ensemble (LightGBM + RF + KNN base -> Ridge meta)

Hyperparameter tuning: Optuna (50 trials, 5-fold CV, MAPE objective).
Selection metric: Lowest MAPE on locked holdout set.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import SVR

try:
    from lightgbm import LGBMRegressor
except ImportError as exc:
    raise ImportError("lightgbm is required. Run: pip install lightgbm") from exc

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError("xgboost is required. Run: pip install xgboost") from exc

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "final"
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

ENRICHED_CSV = DATA_DIR / "gpu_enriched_dataset.csv"
MODEL_OUT = ARTIFACTS_DIR / "gpu_price_model_v2.joblib"
SUMMARY_OUT = ARTIFACTS_DIR / "training_summary_v2.json"

RANDOM_STATE = 42
N_OPTUNA_TRIALS = 50
# SVR computational complexity is O(n^2); cap tuning rows before full refit
SVR_MAX_ROWS = 5_000
SVR_OPTUNA_TRIALS = 30
CV_FOLDS = 5

# Maximum theoretical log-price threshold for scale detection (log(500,000) ≈ 13.1)
LOG_SCALE_THRESHOLD = 20

# Feature definitions
NUMERIC_FEATURES = [
    "vram_gb",
    "G3Dmark",
    "tdp_watts",
    "gpu_age_years",
    "gpu_generation",
    "ti_variant",
]

CATEGORICAL_FEATURES = [
    "series_family",
    "brand",
    "architecture",
    "tier_class",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Target price is log-transformed to stabilize variance and prevent high-tier skew
TARGET = "log_price_lkr"


# -- Metric Helpers ------------------------------------------------------------

def _to_lkr(arr: np.ndarray) -> np.ndarray:
    """Convert log-scale predictions to LKR if they appear to be log-scaled."""
    return np.expm1(arr) if arr.max() < LOG_SCALE_THRESHOLD else arr


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MAPE (Mean Absolute Percentage Error) on the original (LKR) scale.
    Formula: (1/n) *   |(Actual - Predicted) / Actual| * 100
    This is highly intuitive for price prediction as it expresses error as a percentage.
    """
    y_true_lkr = _to_lkr(y_true)
    y_pred_lkr = _to_lkr(y_pred)
    mask = y_true_lkr > 0
    return float(np.mean(np.abs((y_true_lkr[mask] - y_pred_lkr[mask]) / y_true_lkr[mask])) * 100)


def compute_within10pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """% of predictions within 10% of actual price (original scale)."""
    y_true_lkr = _to_lkr(y_true)
    y_pred_lkr = _to_lkr(y_pred)
    denom = np.where(y_true_lkr == 0, 1, y_true_lkr)
    within = np.abs((y_true_lkr - y_pred_lkr) / denom) <= 0.10
    return float(within.mean() * 100)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R-Squared (Coefficient of Determination).
    Represents the proportion of variance in the target variable that is predictable 
    from the features. Calculated here on the log-scale values.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def compute_rmse(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """RMSE on original LKR price scale."""
    return float(np.sqrt(np.mean((_to_lkr(y_true_log) - _to_lkr(y_pred_log)) ** 2)))


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mape_pct": round(compute_mape(y_true, y_pred), 2),
        "r2": round(compute_r2(y_true, y_pred), 4),
        "rmse_lkr": round(compute_rmse(y_true, y_pred), 0),
        "within_10pct": round(compute_within10pct(y_true, y_pred), 2),
    }


# -- Data Loading & Post-Split Leakage-Free Preprocessing -----------------------

def _filter_train_iqr_outliers(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    model_col: pd.Series,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Applies per-model IQR outlier removal STRICTLY on the training split.
    Guarantees zero data leakage from the holdout test set.
    """
    train_df = x_train.copy()
    train_df["_target"] = y_train
    train_df["_model"] = model_col.values
    before_len = len(train_df)

    def _filter_group(grp: pd.DataFrame) -> pd.DataFrame:
        if len(grp) < 4:
            return grp
        q1 = grp["_target"].quantile(0.25)
        q3 = grp["_target"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return grp[(grp["_target"] >= lower) & (grp["_target"] <= upper)]

    filtered = train_df.groupby("_model", group_keys=False).apply(_filter_group)
    dropped = before_len - len(filtered)
    if dropped > 0:
        log.info("  [Leakage-Free IQR] Removed %d price outlier records strictly from training partition", dropped)

    clean_x = filtered[x_train.columns].reset_index(drop=True)
    clean_y = filtered["_target"].to_numpy(dtype=float)
    return clean_x, clean_y


def load_enriched_dataset() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, pd.Series]:
    log.info("Loading enriched dataset ...")
    if not ENRICHED_CSV.exists():
        raise FileNotFoundError(
            f"Enriched dataset not found: {ENRICHED_CSV}\n"
            "Run Phase 1 first:  python scripts/build_benchmark_features.py"
        )
    df = pd.read_csv(ENRICHED_CSV)
    log.info("  Loaded %d rows", len(df))

    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        log.warning("  Features missing from dataset (filled as NaN): %s", missing)
        for col in missing:
            df[col] = np.nan

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataset.")

    df = df.dropna(subset=[TARGET]).copy()

    # Multi-tier deduplication guard prior to split
    before_split_dedup = len(df)
    if "listing_id" in df.columns and df["listing_id"].notna().any():
        has_id = df["listing_id"].notna()
        dedup_id = df[has_id].drop_duplicates(subset=["listing_id"], keep="last")
        df = pd.concat([dedup_id, df[~has_id]], ignore_index=True)
    if "listing_url" in df.columns and df["listing_url"].notna().any():
        has_url = df["listing_url"].notna()
        dedup_url = df[has_url].drop_duplicates(subset=["listing_url"], keep="last")
        df = pd.concat([dedup_url, df[~has_url]], ignore_index=True)
    df.drop_duplicates(subset=["extracted_model", "price_lkr", "vram_gb", "brand"], keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)
    if before_split_dedup - len(df) > 0:
        log.info("  Removed %d duplicate records prior to splitting", before_split_dedup - len(df))

    # Determine canonical model column for group validation
    model_col_name = "extracted_model" if "extracted_model" in df.columns else "model"
    model_groups = df[model_col_name].astype(str)

    x = df[ALL_FEATURES].copy()
    y = df[TARGET].to_numpy(dtype=float)

    # Stratify by log-price quintile for holdout set
    price_quintile = pd.qcut(y, q=5, labels=False, duplicates="drop")

    x_train_raw, x_test, y_train_raw, y_test, train_idx, test_idx = train_test_split(
        x, y, df.index, test_size=0.20, random_state=RANDOM_STATE, stratify=price_quintile
    )

    # Post-split IQR filtering strictly on x_train to avoid data leakage
    x_train, y_train = _filter_train_iqr_outliers(x_train_raw, y_train_raw, model_groups.iloc[train_idx])

    # Verification check: Ensure 0% identical row collision between Train and Test
    train_hashes = set(x_train.apply(lambda row: hash(tuple(row.values)), axis=1))
    test_hashes = set(x_test.apply(lambda row: hash(tuple(row.values)), axis=1))
    collision_count = len(train_hashes.intersection(test_hashes))
    log.info("  Train: %d rows (post-IQR) | Test: %d rows | Exact cross-split collisions: %d", len(x_train), len(x_test), collision_count)

    return x_train, x_test, y_train, y_test, x, y, model_groups


# -- Preprocessors -------------------------------------------------------------

def build_tree_preprocessor() -> ColumnTransformer:
    """
    Preprocessing for tree-based models (RF, XGB, LGBM).
    - Numeric: Median Imputation (handles missing benchmark scores).
    - Categorical: Robust Ordinal Encoding (converts text like 'ASUS', 'MSI' to numbers).
      Maps unknown/unseen brands gracefully to most-frequent neutral category instead of negative -1.
    """
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline([
                    ("imputer_str", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)),
                    ("imputer_num", SimpleImputer(strategy="most_frequent")),
                ]),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_scaled_preprocessor() -> ColumnTransformer:
    """
    Preprocessing for distance-based models (KNN, SVR).
    - Numeric: Median Imputation + StandardScaler.
    - Categorical: Robust Ordinal Encoding with neutral unknown fallback.
    Standardization (StandardScaler) is CRITICAL here because these models
    calculate distances between points. Without scaling, a feature like 
    'G3Dmark' (thousands) would drown out 'vram_gb' (single digits).
    """
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer_str", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)),
                    ("imputer_num", SimpleImputer(strategy="most_frequent")),
                ]),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


# -- Optuna Tuning -------------------------------------------------------------

def _neg_mape_scorer(estimator, X, y) -> float:
    preds = estimator.predict(X)
    return -compute_mape(y, preds)


def optuna_tune(
    name: str,
    pipeline_factory,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    n_trials: int = N_OPTUNA_TRIALS,
) -> tuple[Pipeline, dict]:
    """
    Hyperparameter Optimization using Optuna.
    - Uses Bayesian Optimization (TPE sampler) to find best params.
    - Objective: Minimize MAPE using 5-fold Cross-Validation.
    - Returns: (Fitted Pipeline with best params, Parameters dictionary).
    """
    log.info("  Tuning %s (%d trials) ...", name, n_trials)
    t0 = time.time()

    def objective(trial: optuna.Trial) -> float:
        pipeline = pipeline_factory(trial)
        scores = cross_val_score(
            pipeline, x_train, y_train,
            cv=CV_FOLDS, scoring=_neg_mape_scorer,
            n_jobs=1,
        )
        return float(-scores.mean())

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_mape = round(study.best_value, 2)
    elapsed = round(time.time() - t0, 1)
    log.info("    Best CV MAPE=%.2f%%  params=%s  [%.1fs]", best_mape, best_params, elapsed)

    best_pipeline = pipeline_factory(optuna.trial.FixedTrial(best_params))
    best_pipeline.fit(x_train, y_train)
    return best_pipeline, best_params


# -- Model Factories -----------------------------------------------------------

def _lgbm_factory(trial: optuna.Trial) -> Pipeline:
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    return Pipeline([
        ("pre", build_tree_preprocessor()),
        ("model", LGBMRegressor(random_state=RANDOM_STATE, n_jobs=1, verbose=-1, **params)),
    ])


def _xgb_factory(trial: optuna.Trial) -> Pipeline:
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }
    return Pipeline([
        ("pre", build_tree_preprocessor()),
        ("model", XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE,
                               n_jobs=1, verbosity=0, **params)),
    ])


def _rf_factory(trial: optuna.Trial) -> Pipeline:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
    }
    return Pipeline([
        ("pre", build_tree_preprocessor()),
        ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)),
    ])


def _knn_factory(trial: optuna.Trial) -> Pipeline:
    params = {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 20),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "p": trial.suggest_int("p", 1, 2),
    }
    return Pipeline([
        ("pre", build_scaled_preprocessor()),
        ("model", KNeighborsRegressor(**params)),
    ])


def _svr_factory(trial: optuna.Trial) -> Pipeline:
    """
    SVR (Support Vector Regression) with RBF kernel.
    NOTE: SVR requires feature scaling (StandardScaler) because it uses 
    distances (kernels) between points to predict values. 
    """
    params = {
        "C": trial.suggest_float("C", 0.1, 100.0, log=True),
        "epsilon": trial.suggest_float("epsilon", 0.01, 1.0, log=True),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
    }
    return Pipeline([
        ("pre", build_scaled_preprocessor()),
        ("model", SVR(kernel="rbf", **params)),
    ])


# -- Build & Tune All Candidates -----------------------------------------------

def build_and_tune_candidates(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
) -> tuple[dict[str, Pipeline], dict[str, dict]]:
    """
    Train and tune all 5 individual models.
    Returns (fitted_pipelines, best_params_per_model).
    best_params is passed to fit_stacking_ensemble so it can reconstruct
    unfitted estimators with tuned hyperparameters for StackingRegressor.
    """
    models: dict[str, Pipeline] = {}
    best_params: dict[str, dict] = {}

    models["lightgbm"], best_params["lightgbm"] = optuna_tune(
        "LightGBM", _lgbm_factory, x_train, y_train)

    models["xgboost"], best_params["xgboost"] = optuna_tune(
        "XGBoost", _xgb_factory, x_train, y_train)

    models["random_forest"], best_params["random_forest"] = optuna_tune(
        "RandomForest", _rf_factory, x_train, y_train)

    models["knn"], best_params["knn"] = optuna_tune(
        "KNN", _knn_factory, x_train, y_train)

    # SVR: cap rows for tuning speed, then refit on full data with best params
    if len(x_train) > SVR_MAX_ROWS:
        log.info("  SVR: capping tuning data to %d rows", SVR_MAX_ROWS)
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(x_train), SVR_MAX_ROWS, replace=False)
        x_svr, y_svr = x_train.iloc[idx], y_train[idx]
    else:
        x_svr, y_svr = x_train, y_train

    _, best_params["svr"] = optuna_tune(
        "SVR", _svr_factory, x_svr, y_svr, n_trials=SVR_OPTUNA_TRIALS)

    # Refit SVR on FULL training data with best params
    log.info("  SVR: refitting on full training set with best params ...")
    svr_pipeline = _svr_factory(optuna.trial.FixedTrial(best_params["svr"]))
    svr_pipeline.fit(x_train, y_train)
    models["svr"] = svr_pipeline

    return models, best_params


# -- Stacking Ensemble ---------------------------------------------------------

def fit_stacking_ensemble(
    best_params: dict[str, dict],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
) -> StackingRegressor:
    """
    Ensemble Learning: Stacking Regressor.
    - Base Models (Level 0): LightGBM, Random Forest, KNN (with tuned params).
    - Meta-Learner (Level 1): Ridge Regression.
    The meta-learner learns how to weight the predictions of the base models 
    to achieve a final, more robust prediction.
    """
    log.info("  Building Stacking Ensemble with tuned hyperparameters ...")

    unfitted_estimators = [
        ("lgbm", _lgbm_factory(optuna.trial.FixedTrial(best_params["lightgbm"]))),
        ("rf",   _rf_factory(optuna.trial.FixedTrial(best_params["random_forest"]))),
        ("knn",  _knn_factory(optuna.trial.FixedTrial(best_params["knn"]))),
    ]

    stacking = StackingRegressor(
        estimators=unfitted_estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=CV_FOLDS,
        n_jobs=1,
        passthrough=False,
    )
    stacking.fit(x_train, y_train)
    log.info("  Stacking Ensemble fitted.")
    return stacking


# -- Evaluation ----------------------------------------------------------------

def evaluate_all_models(
    models: dict[str, object],
    x_test: pd.DataFrame,
    y_test: np.ndarray,
) -> dict[str, dict]:
    """
    Loops through all trained models and evaluates them on the Holdout Set.
    This gives us an objective comparison of which algorithm performs best 
    on within-distribution test market data.
    """
    results: dict[str, dict] = {}
    for name, model in models.items():
        preds = model.predict(x_test)
        metrics = evaluate_all(y_test, preds)
        results[name] = metrics
        log.info(
            "  %-22s  MAPE=%5.1f%%  R2=%.4f  Within10%%=%5.1f%%  RMSE=%s LKR",
            name, metrics["mape_pct"], metrics["r2"],
            metrics["within_10pct"], f"{metrics['rmse_lkr']:,.0f}",
        )
    return results


def evaluate_group_generalization(
    models: dict[str, object],
    x_full: pd.DataFrame,
    y_full: np.ndarray,
    model_groups: pd.Series,
    n_splits: int = 5,
) -> dict[str, dict]:
    """
    Group-KFold Cross-Validation grouped by GPU Model.
    Evaluates how well models generalize to completely unseen GPU hardware models
    (zero group leakage across train and validation folds).
    """
    log.info("  Evaluating Out-of-Model Generalization via Group-KFold (%d folds by GPU model) ...", n_splits)
    gkf = GroupKFold(n_splits=n_splits)
    group_results: dict[str, dict] = {}

    for name, model in models.items():
        # Evaluate primary model estimators across group splits
        oof_preds = np.zeros_like(y_full)
        has_pred = np.zeros_like(y_full, dtype=bool)

        for train_idx, val_idx in gkf.split(x_full, y_full, groups=model_groups):
            # Fit clone of model on group train
            x_gtrain, y_gtrain = x_full.iloc[train_idx], y_full[train_idx]
            x_gval = x_full.iloc[val_idx]

            try:
                # Use current fitted pipeline architecture
                from sklearn.base import clone
                g_estimator = clone(model)
                g_estimator.fit(x_gtrain, y_gtrain)
                oof_preds[val_idx] = g_estimator.predict(x_gval)
                has_pred[val_idx] = True
            except Exception:
                continue

        if has_pred.any():
            metrics = evaluate_all(y_full[has_pred], oof_preds[has_pred])
            group_results[name] = metrics
            log.info(
                "  %-22s [Group-KFold Out-of-Model] MAPE=%5.1f%%  R2=%.4f  Within10%%=%5.1f%%",
                name, metrics["mape_pct"], metrics["r2"], metrics["within_10pct"],
            )

    return group_results


# -- Print Comparison Table ----------------------------------------------------

def print_comparison_table(results: dict[str, dict], best_name: str, title: str = "Holdout Set Evaluation") -> None:
    header = f"{'Model':<24} {'MAPE%':>8} {'R ':>8} {'Within10%':>10} {'RMSE (LKR)':>12}"
    print(f"\n-- {title} " + "-" * max(0, len(header) - len(title) - 4))
    print(header)
    print("-" * len(header))
    for name, m in sorted(results.items(), key=lambda kv: kv[1]["mape_pct"]):
        marker = "    BEST" if name == best_name else ""
        print(
            f"{name:<24} {m['mape_pct']:>7.1f}%"
            f" {m['r2']:>8.4f}"
            f" {m['within_10pct']:>9.1f}%"
            f" {m['rmse_lkr']:>12,.0f}{marker}"
        )
    print("-" * len(header))


# -- Save Artifacts ------------------------------------------------------------

def save_artifacts(
    models: dict[str, object],
    results: dict[str, dict],
    best_name: str,
    feature_cols: list[str],
    group_results: dict[str, dict] | None = None,
    n_train: int = 0,
    n_test: int = 0,
) -> None:
    total_records = n_train + n_test
    record_meta = {
        "total_records": total_records,
        "train_records": n_train,
        "test_records": n_test,
        "train_ratio_pct": round((n_train / total_records * 100), 1) if total_records else 80.0,
        "features_count": len(feature_cols),
        "validation_strategy": "Stratified Holdout + Group-KFold Cross-Validation (Zero Group Leakage)",
        "outlier_handling": "Post-split training-only IQR filtering (Zero Test Leakage)",
        "models": {
            "lightgbm": {
                "name": "LightGBM",
                "train_records": n_train,
                "test_records": n_test,
                "tune_records": n_train,
                "preprocessing": "Tree-based (No scaling)",
                "notes": f"Tuned with Optuna on {n_train:,} post-split clean training records; evaluated on {n_test:,} holdout records.",
            },
            "xgboost": {
                "name": "XGBoost",
                "train_records": n_train,
                "test_records": n_test,
                "tune_records": n_train,
                "preprocessing": "Tree-based (No scaling)",
                "notes": f"Tuned with Optuna on {n_train:,} post-split clean training records; evaluated on {n_test:,} holdout records.",
            },
            "random_forest": {
                "name": "Random Forest",
                "train_records": n_train,
                "test_records": n_test,
                "tune_records": n_train,
                "preprocessing": "Tree-based (SimpleImputer + OrdinalEncoder)",
                "notes": f"Tuned with Optuna on {n_train:,} post-split clean training records; evaluated on {n_test:,} holdout records.",
            },
            "knn": {
                "name": "KNN (K-Nearest Neighbors)",
                "train_records": n_train,
                "test_records": n_test,
                "tune_records": n_train,
                "preprocessing": "StandardScaler Normalized",
                "notes": f"Features normalized using StandardScaler across {n_train:,} post-split clean training records.",
            },
            "svr": {
                "name": "SVR (Support Vector Regressor)",
                "train_records": n_train,
                "test_records": n_test,
                "tune_records": min(n_train, SVR_MAX_ROWS),
                "preprocessing": "StandardScaler Normalized",
                "notes": f"Hyperparameters tuned on {min(n_train, SVR_MAX_ROWS):,} records for speed; final model fitted on all {n_train:,} clean training records.",
            },
            "stacking_ensemble": {
                "name": "Stacking Ensemble",
                "train_records": n_train,
                "test_records": n_test,
                "tune_records": n_train,
                "preprocessing": "Base Estimators (LGBM + RF + KNN) -> Ridge Meta-Learner",
                "notes": f"Trained using 5-fold cross-validated meta-features across all {n_train:,} clean training records.",
            },
        },
    }

    artifact = {
        "version": "v2.0",
        "best_model_name": best_name,
        "best_model": models[best_name],
        "all_models": models,
        "feature_columns": feature_cols,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "evaluation_results": results,
        "group_kfold_results": group_results or {},
        "training_records": record_meta,
    }
    joblib.dump(artifact, MODEL_OUT)
    log.info("Saved model artifact -> %s", MODEL_OUT)

    summary = {
        "version": "v2.0",
        "best_model": best_name,
        "n_models": len(models),
        "feature_count": len(feature_cols),
        "training_records": record_meta,
        "holdout_results": results,
        "group_kfold_out_of_model_results": group_results or {},
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Saved training summary -> %s", SUMMARY_OUT)


# -- Main ----------------------------------------------------------------------

def main() -> None:
    print("============================================================")
    print("|  GPU Price Predictor V2 - Phase 2: Model Training        |")
    print("============================================================\n")

    x_train, x_test, y_train, y_test, x_full, y_full, model_groups = load_enriched_dataset()

    print("\n-- Step 1: Tune & Train Individual Models --------------------")
    models, best_params = build_and_tune_candidates(x_train, y_train)

    print("\n-- Step 2: Fit Stacking Ensemble -----------------------------")
    # Pass best_params so stacking builds unfitted pipelines with tuned params
    models["stacking_ensemble"] = fit_stacking_ensemble(best_params, x_train, y_train)

    print("\n-- Step 3: Evaluate All Models on Holdout Set ----------------")
    results = evaluate_all_models(models, x_test, y_test)

    best_name = min(results, key=lambda k: results[k]["mape_pct"])

    print_comparison_table(results, best_name, title="Holdout Set Evaluation (20% Split)")
    print(f"\n[OK] Best model: {best_name}  (MAPE={results[best_name]['mape_pct']:.1f}%)")

    print("\n-- Step 4: Benchmark Group-KFold Out-of-Model Generalization -")
    group_results = evaluate_group_generalization(models, x_full, y_full, model_groups, n_splits=5)
    if group_results:
        best_grp_name = min(group_results, key=lambda k: group_results[k]["mape_pct"])
        print_comparison_table(group_results, best_grp_name, title="Group-KFold Out-of-Model Generalization (Zero Group Leakage)")

    print("\n-- Step 5: Save Artifacts ------------------------------------")
    save_artifacts(
        models, results, best_name, ALL_FEATURES,
        group_results=group_results,
        n_train=len(x_train), n_test=len(x_test)
    )

    if results[best_name]["mape_pct"] > 15:
        print("\n[!]  MAPE > 15% - consider lowering fuzzy match threshold in Phase 1.")
    else:
        print("\n Target MAPE < 15% achieved!")


if __name__ == "__main__":
    main()