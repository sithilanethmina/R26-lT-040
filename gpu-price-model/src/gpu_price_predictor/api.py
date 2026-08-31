from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path

# Add src to path to import app correctly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gpu_price_predictor.app import load_artifact, load_enriched, predict_all
from gpu_price_predictor.pipeline import (
    apply_condition_adjustment,
    calculate_fair_market_range,
    get_fairness_verdict,
    get_model_sample_count,
    normalize_model,
)

app = FastAPI(title="GPU Price Predictor API")

# Allow CORS requests from browser extensions and local web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "gpu_price_predictor"}

artifact, artifact_ver = load_artifact()
enriched = load_enriched()

@app.get("/metadata")
def get_metadata():
    if enriched is None:
        raise HTTPException(status_code=500, detail="Enriched dataset not loaded.")
    
    model_col = "extracted_model" if "extracted_model" in enriched.columns else "model"
    unique_models = sorted(enriched[model_col].dropna().unique().tolist())
    unique_brands = sorted(enriched["brand"].dropna().unique().tolist()) if "brand" in enriched.columns else []

    models_data = []
    for model_name in unique_models:
        mask = enriched[model_col] == model_name
        grp = enriched[mask]
        vrams = sorted([float(x) for x in grp["vram_gb"].dropna().unique().tolist()]) if "vram_gb" in grp.columns else [4.0]
        brands = sorted([str(x) for x in grp["brand"].dropna().unique().tolist()]) if "brand" in grp.columns else []
        default_vram = float(grp["vram_gb"].dropna().median()) if "vram_gb" in grp.columns and not grp["vram_gb"].dropna().empty else 4.0

        manuf = "NVIDIA"
        m_upper = model_name.upper()
        if any(p in m_upper for p in ["RX", "R9", "R7", "HD", "RADEON"]):
            manuf = "AMD"
        elif any(p in m_upper for p in ["ARC", "INTEL"]):
            manuf = "Intel"

        models_data.append({
            "model": model_name,
            "normalized": normalize_model(model_name),
            "manufacturer": manuf,
            "default_vram": default_vram,
            "valid_vrams": vrams if vrams else [default_vram],
            "brands": brands if brands else unique_brands
        })

    return {
        "models": models_data,
        "brands": ["Any"] + unique_brands,
        "manufacturers": ["Any", "NVIDIA", "AMD", "Intel"]
    }

class PredictRequest(BaseModel):
    model: str
    vram_gb: Optional[float] = None
    brand: Optional[str] = "Any"
    manufacturer: Optional[str] = "Any"
    stock: Optional[str] = "In Stock"
    listed_price: Optional[float] = None
    description: Optional[str] = None
    is_shop: Optional[bool] = False

@app.post("/predict")
def predict(request: PredictRequest):
    if artifact is None:
        raise HTTPException(status_code=500, detail="Model artifact not found.")

    vram = request.vram_gb if request.vram_gb is not None else 8.0
    brand = request.brand if request.brand else "Any"
    
    try:
        # Retrieve direct log-scale predictions from model estimators to avoid log-transform rounding errors
        predictions_log = predict_all(
            artifact=artifact,
            model_name=request.model,
            vram=vram,
            brand=brand,
            enriched=enriched,
            return_log=True
        )
        
        predictions = predict_all(
            artifact=artifact,
            model_name=request.model,
            vram=vram,
            brand=brand,
            enriched=enriched,
            return_log=False
        )
        
        if not predictions_log or not predictions:
            raise HTTPException(status_code=500, detail="Failed to generate prediction.")
            
        best_name = artifact.get("best_model_name", "")
        
        sorted_preds_log = sorted(predictions_log.items(), key=lambda kv: kv[1])
        base_log_price = predictions_log.get(best_name, sorted_preds_log[0][1])

        sorted_preds_lkr = sorted(predictions.items(), key=lambda kv: kv[1])
        base_best_price = predictions.get(best_name, sorted_preds_lkr[0][1])

        sample_count = get_model_sample_count(request.model, enriched)

        # Safety Guard 1: Newly Released Generation Restriction (RTX 50-series / Blackwell)
        import re
        if bool(re.search(r'\b(rtx\s*50\d{2}|rx\s*90\d{2})\b', str(request.model).lower())):
            return {
                "status": "generation_restricted",
                "can_predict": False,
                "predicted_price": None,
                "price": None,
                "base_specs_price": None,
                "condition_adjusted_price": None,
                "condition_adjustment_pct": 0,
                "condition_delta_lkr": 0,
                "condition_tags": [],
                "applied_condition_factors": [],
                "best_model_used": best_name,
                "lower_price": None,
                "upper_price": None,
                "fair_market_range": None,
                "evaluation": {
                    "verdict": "Newly Released Generation",
                    "badge_color": "warning",
                    "badge_text": "New Architecture",
                    "badge_class": "warning",
                    "fairness_score": None,
                    "description": f"The {request.model} belongs to a newly released hardware generation. Secondary market pricing has not yet stabilized in Sri Lanka, so automatic price valuation is restricted to ensure accuracy.",
                    "message": f"The {request.model} belongs to a newly released hardware generation. Secondary market pricing has not yet stabilized in Sri Lanka, so automatic price valuation is restricted to ensure accuracy."
                },
                "metadata": {
                    "model_name": request.model,
                    "vram_gb": vram,
                    "brand": brand,
                    "tier": "unknown",
                    "limited_data_warning": True,
                    "sample_count": sample_count,
                    "reason": "new_generation_restriction"
                },
                "all_predictions": {}
            }

        # Safety Guard 2: Check for minimum sample size threshold (N >= 3)
        MIN_SAMPLE_THRESHOLD = 3
        if sample_count < MIN_SAMPLE_THRESHOLD:
            return {
                "status": "insufficient_data",
                "can_predict": False,
                "predicted_price": None,
                "price": None,
                "base_specs_price": None,
                "condition_adjusted_price": None,
                "condition_adjustment_pct": 0,
                "condition_delta_lkr": 0,
                "condition_tags": [],
                "applied_condition_factors": [],
                "best_model_used": best_name,
                "lower_price": None,
                "upper_price": None,
                "fair_market_range": None,
                "evaluation": {
                    "verdict": "Insufficient Market Data",
                    "badge_color": "warning",
                    "badge_text": "Insufficient Data",
                    "badge_class": "warning",
                    "fairness_score": None,
                    "description": f"Market listings for {request.model} are currently limited in Sri Lanka. Automatic price valuation is unavailable to ensure accuracy.",
                    "message": f"Market listings for {request.model} are currently limited in Sri Lanka. Automatic price valuation is unavailable to ensure accuracy."
                },
                "metadata": {
                    "model_name": request.model,
                    "vram_gb": vram,
                    "brand": brand,
                    "tier": "unknown",
                    "limited_data_warning": True,
                    "sample_count": sample_count,
                    "min_required_samples": MIN_SAMPLE_THRESHOLD,
                    "reason": "sample_count_below_minimum_threshold"
                },
                "all_predictions": {}
            }

        # Hedonic adjustment for listing text factors (warranty duration, defect penalty, verified shop markup)
        condition_adj = apply_condition_adjustment(
            predicted_log_price=base_log_price,
            description=request.description,
            is_shop=bool(request.is_shop),
        )

        effective_log_price = condition_adj["adjusted_log_price"]
        final_adjusted_price = condition_adj["adjusted_price_lkr"]

        calibration_data = artifact.get("conformal_calibration", None)
        
        range_info = calculate_fair_market_range(
            predicted_log_price=effective_log_price,
            sample_count=sample_count,
            calibration_data=calibration_data,
            confidence_level="90%"
        )

        verdict_info = get_fairness_verdict(
            listed_price=request.listed_price or 0.0,
            lower_bound=range_info["lower_price_lkr"],
            upper_bound=range_info["upper_price_lkr"]
        )
        
        return {
            "predicted_price": final_adjusted_price,
            "base_specs_price": float(round(base_best_price, -2)),
            "condition_adjusted_price": final_adjusted_price,
            "condition_adjustment_pct": condition_adj["condition_multiplier_pct"],
            "condition_delta_lkr": condition_adj["condition_delta_lkr"],
            "condition_tags": condition_adj["condition_tags"],
            "applied_condition_factors": condition_adj["applied_factors"],
            "best_model_used": best_name,
            "lower_price": range_info["lower_price_lkr"],
            "upper_price": range_info["upper_price_lkr"],
            "fair_market_range": {
                "lower_price_lkr": range_info["lower_price_lkr"],
                "upper_price_lkr": range_info["upper_price_lkr"],
                "currency": "LKR",
                "coverage_confidence": range_info["confidence_level"]
            },
            "evaluation": verdict_info,
            "metadata": {
                "model_name": request.model,
                "vram_gb": vram,
                "brand": brand,
                "tier": range_info["tier"],
                "limited_data_warning": range_info["limited_data_warning"],
                "sample_count": sample_count
            },
            "all_predictions": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def import_np():
    import numpy as np
    return np


if __name__ == "__main__":
    uvicorn.run("gpu_price_predictor.api:app", host="0.0.0.0", port=8001, reload=True)
