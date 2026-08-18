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
    calculate_fair_market_range,
    get_fairness_verdict,
    get_model_sample_count,
    normalize_model,
)

app = FastAPI(title="GPU Price Predictor API")

# Allow CORS for browser extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "gpu_price_predictor"}

# Load globals
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

@app.post("/predict")
def predict(request: PredictRequest):
    if artifact is None:
        raise HTTPException(status_code=500, detail="Model artifact not found.")

    vram = request.vram_gb if request.vram_gb is not None else 8.0
    brand = request.brand if request.brand else "Any"
    
    try:
        predictions = predict_all(
            artifact=artifact,
            model_name=request.model,
            vram=vram,
            brand=brand,
            enriched=enriched
        )
        
        if not predictions:
            raise HTTPException(status_code=500, detail="Failed to generate prediction.")
            
        best_name = artifact.get("best_model_name", "")
        
        # Sort to get a fallback if best_name is not in predictions
        sorted_preds = sorted(predictions.items(), key=lambda kv: kv[1])
        best_price = predictions.get(best_name, sorted_preds[0][1])

        # Sample count lookup from enriched dataset for data penalty adjustment
        sample_count = get_model_sample_count(request.model, enriched)

        # Conformal prediction log-space calculation
        predicted_log_price = float(import_np().log1p(best_price))
        calibration_data = artifact.get("conformal_calibration", None)
        
        range_info = calculate_fair_market_range(
            predicted_log_price=predicted_log_price,
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
            "predicted_price": best_price,
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
