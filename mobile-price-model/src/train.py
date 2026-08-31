"""
Training pipeline for mobile phone price prediction.

Key improvements over the original:
- NO data leakage: all imputation/encoding happens inside the sklearn Pipeline
  and is fitted ONLY on training data.
- Cross-validation for reliable metrics.
- Hyperparameter tuning via RandomizedSearchCV.
- Multiple model comparison (RF, XGBoost, GradientBoosting, optionally LightGBM).
- TargetEncoder for high-cardinality categoricals (brand, model) instead of
  OneHotEncoder with 800+ sparse columns.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    CATEGORICAL_FEATURES,
    CLEANED_DATA_FILE,
    CV_FOLDS,
    FAIR_PRICE_CSV,
    FAIR_PRICE_GROUP_COLUMNS,
    FAIR_PRICE_JSON,
    FEATURE_COLUMNS,
    MIN_ROWS_REQUIRED,
    MODEL_DIR,
    NUMERIC_FEATURES,
    OUTPUT_COLUMNS,
    OUTPUTS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    TRAINING_CONDITION,
    RAW_DATA_FILE,
)
from .data_preprocessing import load_data, preprocess_data, _clean_text
from .evaluate import (
    build_comparison_table,
    evaluate_model,
    print_final_summary,
    recommend_best_models,
    save_evaluation_results,
)
from .feature_engineering import add_engineered_features

logger = logging.getLogger(__name__)


# ── Encoder selection ────────────────────────────────────────────────────────

def _get_target_encoder():
    """Try sklearn TargetEncoder (>=1.3) with continuous target type, fall back to OrdinalEncoder."""
    try:
        from sklearn.preprocessing import TargetEncoder
        return TargetEncoder(smooth="auto", target_type="continuous", cv=5, random_state=RANDOM_STATE)
    except (ImportError, TypeError):
        try:
            from sklearn.preprocessing import TargetEncoder
            return TargetEncoder(smooth="auto", random_state=RANDOM_STATE)
        except ImportError:
            from sklearn.preprocessing import OrdinalEncoder
            logger.warning("TargetEncoder not available. Using OrdinalEncoder.")
            return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)


def build_preprocessor() -> ColumnTransformer:
    """
    Build an sklearn ColumnTransformer that is fitted inside the Pipeline.

    This ensures NO data leakage: imputers/encoders learn only from training data.
    """
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", _get_target_encoder()),
    ])

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, CATEGORICAL_FEATURES),
            ("num", num_pipe, NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


# ── GPU Detection ────────────────────────────────────────────────────────────

def check_gpu_availability() -> Tuple[bool, str]:
    """Check if CUDA GPU is available for accelerated training."""
    try:
        import xgboost as xgb
        test_model = xgb.XGBRegressor(device="cuda", n_estimators=1)
        test_model.fit(np.array([[1.0, 2.0]]), np.array([1.0]))
        return True, "cuda"
    except Exception:
        return False, "cpu"


# ── Model definitions ────────────────────────────────────────────────────────

def _get_model_configs() -> list[Dict[str, Any]]:
    """Return model configurations with hyperparameter search spaces and GPU support."""
    has_gpu, gpu_device = check_gpu_availability()
    if has_gpu:
        logger.info("GPU Acceleration: ENABLED (NVIDIA CUDA)")
    else:
        logger.info("GPU Acceleration: DISABLED (Running on CPU)")

    configs: list[Dict[str, Any]] = [
        {
            "name": "Random Forest",
            "estimator": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            "param_grid": {
                "model__n_estimators": [200, 300, 400],
                "model__max_depth": [15, 20, 25, 30, None],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", 0.5, 0.8, None],
            },
        },
    ]

    # LightGBM (fast gradient booster)
    try:
        from lightgbm import LGBMRegressor
        configs.append({
            "name": "LightGBM",
            "estimator": LGBMRegressor(
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
            ),
            "param_grid": {
                "model__n_estimators": [300, 400, 500, 600],
                "model__learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
                "model__max_depth": [-1, 6, 8, 10],
                "model__num_leaves": [20, 31, 50, 70],
                "model__subsample": [0.7, 0.8, 0.9, 1.0],
                "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9],
            },
        })
    except ImportError:
        logger.info("LightGBM not installed. Skipping.")

    # XGBoost (GPU CUDA accelerated when available)
    try:
        from xgboost import XGBRegressor
        configs.append({
            "name": "XGBoost",
            "estimator": XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                device="cuda" if has_gpu else "cpu",
                tree_method="hist",
                eval_metric="rmse",
            ),
            "param_grid": {
                "model__n_estimators": [300, 400, 500, 600],
                "model__learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
                "model__max_depth": [4, 5, 6, 7, 8],
                "model__subsample": [0.7, 0.8, 0.9, 1.0],
                "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9],
                "model__min_child_weight": [1, 2, 3, 5],
            },
        })
    except ImportError:
        logger.warning("XGBoost not installed. Skipping.")

    # CatBoost (GPU accelerated when available)
    try:
        from catboost import CatBoostRegressor
        configs.append({
            "name": "CatBoost",
            "estimator": CatBoostRegressor(
                random_seed=RANDOM_STATE,
                task_type="GPU" if has_gpu else "CPU",
                verbose=0,
            ),
            "param_grid": {
                "model__iterations": [300, 500, 700],
                "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
                "model__depth": [4, 6, 8],
                "model__l2_leaf_reg": [1, 3, 5, 7],
            },
        })
    except ImportError:
        logger.info("CatBoost not installed. Skipping.")

    return configs


# ── Training ─────────────────────────────────────────────────────────────────

def train_single_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_config: Dict[str, Any],
    phone_type: str,
    total_records: int,
    tune: bool = True,
    n_iter: int = 15,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train one model with optional hyperparameter tuning.

    Returns the fitted Pipeline and evaluation metrics.
    """
    name = model_config["name"]
    logger.info("Training %s for %s phones...", name, phone_type)

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", model_config["estimator"]),
    ])

    if tune and model_config.get("param_grid"):
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=model_config["param_grid"],
            n_iter=n_iter,
            cv=CV_FOLDS,
            scoring="neg_mean_absolute_error",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        logger.info("%s best params: %s", name, search.best_params_)
    else:
        pipeline.fit(X_train, y_train)

    # Cross-validation on training data for robust metrics
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=CV_FOLDS, scoring="neg_mean_absolute_error", n_jobs=1,
    )
    cv_mae = -cv_scores.mean()
    logger.info("%s %s-fold CV MAE: LKR %s (±%s)",
                name, CV_FOLDS, f"{cv_mae:,.2f}", f"{cv_scores.std():,.2f}")

    # Test set evaluation
    metrics = evaluate_model(
        pipeline, X_test, y_test, name, phone_type,
        total_records, len(y_train),
    )
    metrics["cv_mae"] = float(cv_mae)

    return pipeline, metrics


def train_all_models(
    iphone_df: pd.DataFrame,
    android_df: pd.DataFrame,
    tune: bool = True,
    dedicated_only: bool = True,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Pipeline]]:
    """Train dedicated winning models (CatBoost for iPhone, XGBoost for Android) or all models."""
    results: Dict[str, Dict[str, Any]] = {}
    skipped: Dict[str, Dict[str, Any]] = {}
    trained: Dict[str, Pipeline] = {}

    all_configs = _get_model_configs()
    config_by_name = {c["name"].lower().replace(" ", "_"): c for c in all_configs}

    target_assignments = {
        "iphone": ["catboost", "xgboost", "random_forest", "lightgbm"] if not dedicated_only else ["catboost"],
        "android": ["xgboost", "lightgbm", "random_forest", "catboost"] if not dedicated_only else ["xgboost"],
    }

    for phone_type, dataset in [("iphone", iphone_df), ("android", android_df)]:
        if len(dataset) < MIN_ROWS_REQUIRED:
            logger.warning("Skipping %s: only %s rows (need %s).",
                           phone_type, len(dataset), MIN_ROWS_REQUIRED)
            continue

        X = dataset[FEATURE_COLUMNS].copy()
        y = dataset[TARGET_COLUMN].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        )

        desired_names = target_assignments.get(phone_type, ["xgboost"])
        selected_configs = []
        for name in desired_names:
            if name in config_by_name:
                selected_configs.append(config_by_name[name])
                if dedicated_only:
                    break

        if not selected_configs:
            selected_configs = all_configs[:1]

        for config in selected_configs:
            key = f"{config['name'].lower().replace(' ', '_')}_{phone_type}"
            try:
                pipeline, metrics = train_single_model(
                    X_train, y_train, X_test, y_test,
                    config, phone_type, len(dataset), tune=tune,
                )
                results[key] = metrics
                trained[key] = pipeline
            except Exception as exc:
                logger.error("Failed training %s for %s: %s", config["name"], phone_type, exc)
                skipped[key] = {"reason": str(exc)}

    return results, skipped, trained


# ── Fair price generation ────────────────────────────────────────────────────

def _round_lkr(v: Any) -> Optional[float]:
    if v is None or pd.isna(v):
        return None
    return round(float(v), 2)


def build_fair_price_predictions(
    cleaned_df: pd.DataFrame,
    trained_models: Dict[str, Pipeline],
    results: Dict[str, Dict[str, Any]],
    recommendations: Dict[str, Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """Generate fair price predictions for each model/storage category using batch prediction."""
    records: list[Dict[str, Any]] = []

    for pt in ["iphone", "android"]:
        pt_df = cleaned_df[cleaned_df["phone_type"] == pt]
        if pt_df.empty:
            continue

        rec = recommendations.get(pt, {})
        rec_name = rec.get("recommended_model", "XGBoost")
        model_key = f"{rec_name.lower().replace(' ', '_')}_{pt}"

        if model_key not in trained_models:
            fallback = [k for k in trained_models if k.endswith(f"_{pt}")]
            if not fallback:
                continue
            model_key = fallback[0]

        model = trained_models[model_key]
        mae = results.get(model_key, {}).get("mae")

        group_records = []
        batch_inputs = []

        for cat_vals, gdf in pt_df.groupby(FAIR_PRICE_GROUP_COLUMNS, dropna=False, sort=True):
            cat = dict(zip(FAIR_PRICE_GROUP_COLUMNS, cat_vals))
            prices = gdf[TARGET_COLUMN]
            brand = str(cat["brand"])
            model_name = str(cat["model"])
            storage = float(cat["storage_gb"]) if not pd.isna(cat["storage_gb"]) else None

            rep = {col: gdf[col].mode().iloc[0] if not gdf[col].mode().empty else "Unknown"
                   for col in ["brand", "model"]}
            rep["condition"] = TRAINING_CONDITION
            rep["currency"] = "LKR"
            for col in NUMERIC_FEATURES:
                if col in gdf.columns and not gdf[col].dropna().empty:
                    v = pd.to_numeric(gdf[col], errors="coerce").median()
                    rep[col] = float(v) if not pd.isna(v) else 0.0
                else:
                    rep[col] = 0.0

            batch_inputs.append(rep)
            group_records.append({
                "phone_type": pt,
                "brand": brand,
                "model": model_name,
                "storage_gb": _round_lkr(storage),
                "category_key": f"{brand} {model_name} | {int(storage) if storage else '?'}GB",
                "sample_count": len(gdf),
                "confidence": "high" if len(gdf) >= 20 else "medium" if len(gdf) >= 8
                              else "low" if len(gdf) >= 3 else "very_low",
                "observed_median_lkr": _round_lkr(prices.median()),
                "observed_q1_lkr": _round_lkr(prices.quantile(0.25)),
                "observed_q3_lkr": _round_lkr(prices.quantile(0.75)),
                "observed_min_lkr": _round_lkr(prices.min()),
                "observed_max_lkr": _round_lkr(prices.max()),
                "model_used": results.get(model_key, {}).get("model_name", "Unknown"),
                "model_mae_lkr": _round_lkr(mae),
            })

        if batch_inputs:
            batch_df = pd.DataFrame(batch_inputs)[FEATURE_COLUMNS]
            preds = np.maximum(0.0, model.predict(batch_df))

            for i, rec_item in enumerate(group_records):
                pred = float(preds[i])
                rec_item["fair_price_lkr"] = _round_lkr(pred)
                rec_item["fair_price_range_low_lkr"] = _round_lkr(max(0, pred - mae)) if mae else None
                rec_item["fair_price_range_high_lkr"] = _round_lkr(pred + mae) if mae else None
                records.append(rec_item)

    records.sort(key=lambda r: (r["phone_type"], r["brand"], r["model"],
                                 r["storage_gb"] or -1))
    logger.info("Built fair-price predictions for %s categories.", f"{len(records):,}")
    return records


def save_fair_prices(records: list[Dict[str, Any]]) -> Dict[str, str]:
    """Save fair-price predictions as JSON and CSV."""
    FAIR_PRICE_JSON.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target": TARGET_COLUMN,
        "grouping": FAIR_PRICE_GROUP_COLUMNS,
        "records": records,
    }
    with FAIR_PRICE_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    pd.DataFrame(records).to_csv(FAIR_PRICE_CSV, index=False, encoding="utf-8")

    logger.info("Saved fair prices: %s, %s", FAIR_PRICE_JSON, FAIR_PRICE_CSV)
    return {"json": str(FAIR_PRICE_JSON), "csv": str(FAIR_PRICE_CSV)}


def save_metadata_and_lookup(cleaned_df: pd.DataFrame) -> None:
    """Save verified supported mobile_brand_model_lookup.json and mobile_metadata.json."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    lookup: dict[str, list[str]] = {}
    for brand, gdf in cleaned_df.groupby("brand"):
        models = sorted([str(m) for m in gdf["model"].dropna().unique() if str(m).strip()])
        if models:
            lookup[str(brand)] = models

    lookup_file = OUTPUTS_DIR / "mobile_brand_model_lookup.json"
    with lookup_file.open("w", encoding="utf-8") as f:
        json.dump(lookup, f, indent=2, ensure_ascii=False)

    metadata_file = OUTPUTS_DIR / "mobile_metadata.json"
    metadata = {
        "brands": sorted(list(lookup.keys())),
        "total_records": len(cleaned_df),
        "total_models": sum(len(m) for m in lookup.values()),
        "generated_at": datetime.now().isoformat(timespec="seconds")
    }
    with metadata_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        
    logger.info("Saved lookup & metadata: %s, %s", lookup_file, metadata_file)


# ── Save / delete helpers ────────────────────────────────────────────────────

def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model: %s", path)


def delete_old_models() -> None:
    if MODEL_DIR.exists():
        for p in MODEL_DIR.glob("*.pkl"):
            p.unlink()
            logger.info("Deleted old model: %s", p)


# ── Main entry point ────────────────────────────────────────────────────────

def main() -> None:
    """Full pipeline: preprocess → engineer features → train → evaluate → save."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("Starting mobile phone price model training pipeline.")

    try:
        # 1. Load and preprocess
        raw_df = load_data(RAW_DATA_FILE)
        cleaned_df = preprocess_data(raw_df)

        # 2. Feature engineering
        cleaned_df = add_engineered_features(cleaned_df)

        # 3. Save cleaned dataset
        CLEANED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_json(CLEANED_DATA_FILE, orient="records", indent=2, force_ascii=False)
        logger.info("Saved ML-ready dataset: %s (%s records)",
                     CLEANED_DATA_FILE, f"{len(cleaned_df):,}")

        # 4. Split by phone type
        iphone_df = cleaned_df[cleaned_df["phone_type"] == "iphone"].copy()
        android_df = cleaned_df[cleaned_df["phone_type"] == "android"].copy()
        logger.info("iPhone: %s | Android: %s", f"{len(iphone_df):,}", f"{len(android_df):,}")

        # 5. Train all models
        delete_old_models()
        results, skipped, trained = train_all_models(iphone_df, android_df, tune=True)

        # 6. Save best models
        comparison = build_comparison_table(results)
        recommendations = recommend_best_models(comparison)

        for pt in ["iphone", "android"]:
            rec = recommendations.get(pt)
            if rec:
                key = f"{rec['recommended_model'].lower().replace(' ', '_')}_{pt}"
                if key in trained:
                    save_model(trained[key], MODEL_DIR / f"best_{pt}_model.pkl")

        # 7. Fair price predictions
        fair_records = build_fair_price_predictions(
            cleaned_df, trained, results, recommendations)
        fair_files = save_fair_prices(fair_records)

        # 8. Save evaluation results and lookup metadata
        save_evaluation_results(results, skipped, recommendations, fair_files)
        save_metadata_and_lookup(cleaned_df)
        print_final_summary(len(iphone_df), len(android_df), comparison, recommendations)

        logger.info("Pipeline completed successfully.")

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
