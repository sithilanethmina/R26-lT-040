"""
Prediction module — shared by the Streamlit app and CLI.

Loads the trained model pipeline and applies exactly the same preprocessing
as training (guaranteed by the sklearn Pipeline).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from .config import EVALUATION_FILE, FEATURE_COLUMNS, MODEL_DIR

logger = logging.getLogger(__name__)


def load_model(phone_type: str) -> Pipeline:
    """Load the best trained model for the given phone type."""
    path = MODEL_DIR / f"best_{phone_type}_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def load_evaluation_data() -> Dict[str, Any]:
    """Load saved evaluation metrics."""
    if not EVALUATION_FILE.exists():
        return {}
    with EVALUATION_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_model_info(phone_type: str) -> Dict[str, Any]:
    """Get the recommended model name, metrics, and file path."""
    eval_data = load_evaluation_data()
    rec = eval_data.get("recommendations", {}).get(phone_type, {})
    model_name = rec.get("recommended_model", "Unknown")
    model_key = f"{model_name.lower().replace(' ', '_')}_{phone_type}"
    metrics = eval_data.get("results", {}).get(model_key, {})
    return {
        "model_name": model_name,
        "model_path": str(MODEL_DIR / f"best_{phone_type}_model.pkl"),
        "mae": metrics.get("mae", 0.0),
        "r2_score": metrics.get("r2_score", 0.0),
        "mape_percent": metrics.get("mape_percent", 0.0),
        "cv_mae": metrics.get("cv_mae"),
    }


def predict_price(
    phone_type: str,
    brand: str,
    model: str,
    storage_gb: float,
    ram_gb: float,
    warranty_days: float = 0.0,
    battery_health_percent: float = 90.0,
    dual_sim: int = 1,
    has_5g: int = 0,
    has_esim: int = 0,
    model_tier: int = 5,
    brand_tier: int = 2,
    phone_age_years: float = 3.0,
    is_flagship: int = 0,
) -> Dict[str, Any]:
    """
    Predict the fair price for a phone.

    Returns a dict with predicted_price, range_low, range_high, and model_info.
    """
    pipeline = load_model(phone_type)
    info = get_model_info(phone_type)

    input_df = pd.DataFrame([{
        "brand": brand,
        "model": model,
        "storage_gb": float(storage_gb),
        "ram_gb": float(ram_gb),
        "warranty_days": float(warranty_days),
        "battery_health_percent": float(battery_health_percent),
        "dual_sim": int(dual_sim),
        "has_5g": int(has_5g),
        "has_esim": int(has_esim),
        "model_tier": int(model_tier),
        "brand_tier": int(brand_tier),
        "phone_age_years": float(phone_age_years),
        "is_flagship": int(is_flagship),
    }], columns=FEATURE_COLUMNS)

    predicted = max(0.0, float(pipeline.predict(input_df)[0]))
    mae = float(info.get("mae", 0.0))

    return {
        "predicted_price": predicted,
        "range_low": max(0.0, predicted - mae),
        "range_high": predicted + mae,
        "model_info": info,
    }
