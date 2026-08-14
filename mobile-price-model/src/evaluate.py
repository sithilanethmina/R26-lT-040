"""
Model evaluation: metrics computation, comparison tables, and reporting.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline

from .config import EVALUATION_FILE, FEATURE_COLUMNS, TARGET_COLUMN, TRAINING_CONDITION

logger = logging.getLogger(__name__)


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    phone_type: str,
    total_records: int,
    train_records: int,
) -> Dict[str, Any]:
    """Compute MAE, RMSE, R², MAPE for a trained model on held-out test data."""
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = r2_score(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds) * 100

    metrics = {
        "model_name": model_name,
        "phone_type": phone_type,
        "total_records": int(total_records),
        "train_records": int(train_records),
        "test_records": int(len(y_test)),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2_score": float(r2),
        "mape_percent": float(mape),
    }

    sep = "-" * 78
    logger.info(sep)
    logger.info("Evaluation | %s | %s", phone_type.upper(), model_name)
    logger.info("Records    | total=%s train=%s test=%s",
                total_records, train_records, len(y_test))
    logger.info("MAE        | LKR %s", f"{mae:,.2f}")
    logger.info("RMSE       | LKR %s", f"{rmse:,.2f}")
    logger.info("R²         | %.4f", r2)
    logger.info("MAPE       | %.2f%%", mape)
    logger.info(sep)

    return metrics


def build_comparison_table(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Create a model comparison table sorted by phone type and MAE."""
    if not results:
        return pd.DataFrame()
    table = pd.DataFrame(results.values())
    cols = ["phone_type", "model_name", "total_records", "test_records",
            "mae", "rmse", "r2_score", "mape_percent"]
    table = table[[c for c in cols if c in table.columns]]
    return table.sort_values(["phone_type", "mae"]).reset_index(drop=True)


def recommend_best_models(table: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Select the best model per phone type using combined MAE + R² ranking."""
    recs: Dict[str, Dict[str, Any]] = {}
    if table.empty:
        return recs
    for pt, group in table.groupby("phone_type"):
        g = group.copy()
        g["mae_rank"] = g["mae"].rank(method="min", ascending=True)
        g["r2_rank"] = g["r2_score"].rank(method="min", ascending=False)
        g["combined"] = g["mae_rank"] + g["r2_rank"]
        best = g.sort_values(["combined", "mae"]).iloc[0]
        best_mae = g.sort_values("mae").iloc[0]
        best_r2 = g.sort_values("r2_score", ascending=False).iloc[0]
        recs[str(pt)] = {
            "recommended_model": str(best["model_name"]),
            "recommended_model_mae": float(best["mae"]),
            "recommended_model_r2_score": float(best["r2_score"]),
            "lowest_mae_model": str(best_mae["model_name"]),
            "lowest_mae": float(best_mae["mae"]),
            "highest_r2_model": str(best_r2["model_name"]),
            "highest_r2_score": float(best_r2["r2_score"]),
        }
    return recs


def save_evaluation_results(
    results: Dict[str, Dict[str, Any]],
    skipped: Dict[str, Dict[str, Any]],
    recommendations: Dict[str, Dict[str, Any]],
    fair_price_files: Optional[Dict[str, str]] = None,
    output_path: Path = EVALUATION_FILE,
) -> None:
    """Persist evaluation metrics, recommendations, and fair-price file paths."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target": TARGET_COLUMN,
        "features": FEATURE_COLUMNS,
        "training_condition_filter": TRAINING_CONDITION,
        "results": results,
        "skipped_models": skipped,
        "recommendations": recommendations,
        "fair_price_output_files": fair_price_files or {},
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Saved evaluation results: %s", output_path)


def print_final_summary(
    iphone_count: int,
    android_count: int,
    table: pd.DataFrame,
    recs: Dict[str, Dict[str, Any]],
) -> None:
    """Print the final summary to the console."""
    logger.info("=" * 78)
    logger.info("iPhone records: %s | Android records: %s", f"{iphone_count:,}", f"{android_count:,}")
    if not table.empty:
        display = table.copy()
        display["mae"] = display["mae"].map(lambda v: f"{v:,.2f}")
        display["rmse"] = display["rmse"].map(lambda v: f"{v:,.2f}")
        display["r2_score"] = display["r2_score"].map(lambda v: f"{v:.4f}")
        display["mape_percent"] = display["mape_percent"].map(lambda v: f"{v:.2f}%")
        logger.info("\n%s", display.to_string(index=False))
    for pt, rec in recs.items():
        logger.info(
            "Recommended for %s: %s (MAE=LKR %s, R²=%.4f)",
            pt, rec["recommended_model"],
            f"{rec['recommended_model_mae']:,.2f}", rec["recommended_model_r2_score"],
        )
    logger.info("=" * 78)
