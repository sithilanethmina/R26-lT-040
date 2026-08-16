"""
FastAPI Prediction Server
Vehicle Price Predictor — Sri Lanka Market
Serves:
  GET  /metadata/cars       → brand/model lookup for Cars
  GET  /metadata/suv        → brand/model lookup for SUVs
  POST /api/predict         → Car price prediction (with optional NLP scoring)
  POST /api/predict/suv     → SUV price prediction (with optional NLP scoring)
  GET  /health              → server liveness check

Run:  uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

import json
import re
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
CAR_MODEL_DIR   = BASE_DIR / "models"  / "combined"
CAR_OUT_DIR     = BASE_DIR / "outputs" / "combined"

SUV_BASE_DIR    = BASE_DIR.parent / "SUV_Price_Prediction"
SUV_MODEL_DIR   = SUV_BASE_DIR / "models"  / "suv"
SUV_OUT_DIR     = SUV_BASE_DIR / "outputs" / "suv"

# ─── Feature config (must match training scripts exactly) ─────────────────────
CAR_CATEGORICAL  = ["brand", "model", "variant", "fuel_type", "transmission"]
CAR_NUMERIC      = ["model_year", "mileage_km", "vehicle_age"]
CAR_FEATURES     = CAR_CATEGORICAL + CAR_NUMERIC

SUV_CATEGORICAL  = ["brand", "model", "variant", "fuel_type", "transmission"]
SUV_NUMERIC      = ["model_year", "mileage_km", "vehicle_age", "engine_cc"]
SUV_FEATURES     = SUV_CATEGORICAL + SUV_NUMERIC

REFERENCE_YEAR = 2026

# ─── Load Car Artifacts ───────────────────────────────────────────────────────
print("Loading Car model artifacts …")
try:
    car_model    = joblib.load(CAR_MODEL_DIR / "best_model.pkl")
    car_encoder  = joblib.load(CAR_MODEL_DIR / "ordinal_encoder.pkl")
    car_model_type = type(car_model).__name__
    print(f"  [OK] Car model  : {car_model_type}")
    print(f"  [OK] Car encoder: OrdinalEncoder")
except FileNotFoundError as exc:
    raise RuntimeError(
        f"Car model artifact not found: {exc}. "
        "Run train_combined_model.py first."
    ) from exc

CAR_IS_CATBOOST = "CatBoost" in car_model_type

# ─── Load SUV Artifacts ───────────────────────────────────────────────────────
print("Loading SUV model artifacts …")
try:
    suv_model    = joblib.load(SUV_MODEL_DIR / "best_suv_model.pkl")
    suv_encoder  = joblib.load(SUV_MODEL_DIR / "suv_ordinal_encoder.pkl")
    suv_model_type = type(suv_model).__name__
    print(f"  [OK] SUV model  : {suv_model_type}")
    print(f"  [OK] SUV encoder: OrdinalEncoder")
except FileNotFoundError as exc:
    raise RuntimeError(
        f"SUV model artifact not found: {exc}. "
        "Run train_suv_model.py first."
    ) from exc

SUV_IS_CATBOOST = "CatBoost" in suv_model_type

# ─── Load NLP Configs ─────────────────────────────────────────────────────────
print("Loading NLP configs …")
with open(CAR_OUT_DIR / "nlp_config.json", encoding="utf-8") as f:
    CAR_NLP_CONFIG = json.load(f)

with open(SUV_OUT_DIR / "suv_nlp_config.json", encoding="utf-8") as f:
    SUV_NLP_CONFIG = json.load(f)

print("  [OK] NLP configs loaded")

# ─── Load Brand/Model Lookups ────────────────────────────────────────────────
print("Loading brand/model lookup tables …")
car_lookup_df = pd.read_csv(CAR_OUT_DIR / "brand_model_lookup.csv")
suv_lookup_df = pd.read_csv(SUV_OUT_DIR / "suv_brand_model_lookup.csv")
print(f"  [OK] Car lookup : {len(car_lookup_df)} entries")
print(f"  [OK] SUV lookup : {len(suv_lookup_df)} entries")


def _build_metadata(df: pd.DataFrame) -> dict:
    """Convert a brand_model_lookup DataFrame into {brands, models} dict."""
    brands = sorted(df["brand"].unique().tolist())
    models: dict[str, list[str]] = {}
    for brand, group in df.groupby("brand"):
        models[brand] = sorted(group["model"].unique().tolist())
    return {"brands": brands, "models": models}


CAR_METADATA = _build_metadata(car_lookup_df)
SUV_METADATA = _build_metadata(suv_lookup_df)

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Vehicle Price Predictor API",
    description="ML-powered vehicle price prediction for the Sri Lanka market. "
                "Supports Cars and SUVs with optional NLP listing-quality scoring.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── NLP Scoring Helper ───────────────────────────────────────────────────────
def _score_description(description: str, nlp_config: dict) -> dict:
    """
    Score a free-text vehicle description against positive/negative signal lists.
    Returns nlp_score (int), nlp_signals (list of matched labels), and
    nlp_verdict (str: 'Fairly Priced ✅', 'Review Carefully ⚠️', 'Caution 🔴').
    """
    if not description:
        return {"nlp_score": 0, "nlp_signals": [], "nlp_verdict": None}

    text = description.lower()
    total_nlp_points = 0
    matched_labels: list[str] = []

    for signal in nlp_config.get("positive_signals", {}).values():
        if any(kw in text for kw in signal["keywords"]):
            total_nlp_points += signal["points"]
            matched_labels.append(signal["label"])

    for signal in nlp_config.get("negative_signals", {}).values():
        if any(kw in text for kw in signal["keywords"]):
            total_nlp_points += signal["points"]  # already negative
            matched_labels.append(f"⚠ {signal['label']}")

    scoring = nlp_config.get("scoring", {})
    nlp_max        = scoring.get("nlp_max", 30)
    fairly_priced  = scoring.get("fairly_priced_min", 65)
    review_min     = scoring.get("review_min", 45)
    base_max       = scoring.get("base_max", 70)

    # Clamp NLP portion
    nlp_clamped = max(-nlp_max, min(nlp_max, total_nlp_points))

    # Combine with a neutral base score of 50 (no extra model signals available at runtime)
    base_score = 50
    final_score = min(scoring.get("final_max", 100), max(0, base_score + nlp_clamped))

    if final_score >= fairly_priced:
        verdict = "Fairly Priced ✅"
    elif final_score >= review_min:
        verdict = "Review Carefully ⚠️"
    else:
        verdict = "Caution 🔴"

    return {
        "nlp_score":   final_score,
        "nlp_signals": matched_labels,
        "nlp_verdict": verdict,
    }


# ─── Feature Engineering ──────────────────────────────────────────────────────
def _extract_engine_cc(text: str) -> float:
    """Best-effort engine CC extraction from variant/model string (Car fallback)."""
    m = re.search(r"(\d{3,4})\s*cc", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"\b(\d\.\d)\s*(?:l|liter)?\b", text, re.IGNORECASE)
    if m:
        litre_map = {"1.0": 1000, "1.2": 1200, "1.3": 1300,
                     "1.5": 1500, "1.8": 1800, "2.0": 2000,
                     "2.4": 2400, "3.0": 3000, "3.5": 3500}
        return float(litre_map.get(m.group(1), -1))
    return -1.0   # sentinel used in training


def _get_confidence(brand: str, model_name: str, lookup_df: pd.DataFrame) -> str:
    """Look up market-data confidence level for a brand/model pair."""
    row = lookup_df[
        (lookup_df["brand"].str.lower() == brand.lower()) &
        (lookup_df["model"].str.lower() == model_name.lower())
    ]
    if row.empty:
        return "Unknown"
    raw = str(row.iloc[0]["confidence"])
    # Confidence column may be "High", "Medium", or "Low — insufficient …"
    if raw.startswith("High"):
        return "High"
    if raw.startswith("Medium"):
        return "Medium"
    return "Low"


def _map_suv_generations(brand: str, model: str, year: int, variant: str) -> str:
    """Assign generation tags to the variant column."""
    v = str(variant).strip()
    b = str(brand).strip().lower()
    m = str(model).strip().lower()
    y = int(year)
    
    # 1. Force mapping for older RAV4s regardless of what was typed
    if b == "toyota" and m == "rav4":
        if y <= 2000:
            return "1st Gen"
        elif 2001 <= y <= 2005:
            return "2nd Gen"
        elif v.lower() == "standard" and 2006 <= y <= 2012:
            return "3rd Gen"
        elif v.lower() == "standard" and 2013 <= y <= 2018:
            return "4th Gen"
        elif v.lower() == "standard":
            return "5th Gen"
        
    # 2. For custom variants on newer models, return the variant as is
    if v.lower() != "standard":
        return variant
        
    # 3. Handle Montero / Pajero / CR-V standard variants
    if b == "mitsubishi" and m in ["montero", "pajero"]:
        if y <= 1999:
            return "2nd Gen"
        elif 2000 <= y <= 2006:
            return "3rd Gen"
        else:
            return "4th Gen"
        
    elif b == "honda" and m in ["cr-v", "crv"]:
        if y <= 2001:
            return "1st Gen"
        elif 2002 <= y <= 2006:
            return "2nd Gen"
        elif 2007 <= y <= 2011:
            return "3rd Gen"
        else:
            return "4th Gen"
        
    return variant

# ─── Request / Response Schemas ───────────────────────────────────────────────
class PredictRequest(BaseModel):
    brand:        str            = Field(...,        examples=["Toyota"])
    model:        str            = Field(...,        examples=["Vitz"])
    variant:      str            = Field("Standard", examples=["KSP90"])
    model_year:   int            = Field(...,        ge=1990, le=2026, examples=[2018])
    mileage_km:   Optional[float] = Field(None,     ge=0, le=500_000, examples=[55000])
    fuel_type:    str            = Field(...,        examples=["Petrol"])
    transmission: str            = Field(...,        examples=["Automatic"])
    description:  Optional[str]  = Field(None,       examples=["One owner, accident free, full option"])

    @field_validator("variant")
    @classmethod
    def default_variant(cls, v: str) -> str:
        return v.strip() if v.strip() else "Standard"


class SUVPredictRequest(BaseModel):
    brand:        str            = Field(...,        examples=["Toyota"])
    model:        str            = Field(...,        examples=["Raize"])
    variant:      str            = Field("Standard", examples=["GLS"])
    model_year:   int            = Field(...,        ge=1990, le=2026, examples=[2022])
    mileage_km:   Optional[float] = Field(None,     ge=0, le=500_000, examples=[45000])
    fuel_type:    str            = Field(...,        examples=["Hybrid"])
    transmission: str            = Field(...,        examples=["Automatic"])
    engine_cc:    int            = Field(...,        ge=600, le=8000,  examples=[1490])
    description:  Optional[str]  = Field(None,       examples=["4WD, one owner, service records"])

    @field_validator("variant")
    @classmethod
    def default_variant(cls, v: str) -> str:
        return v.strip() if v.strip() else "Standard"


class PredictResponse(BaseModel):
    predicted_price:      int
    model_used:           str
    vehicle_age:          int
    mileage_per_year:     float
    used_mileage_km:      float
    is_mileage_estimated: bool
    confidence:           str
    nlp_score:            Optional[int]
    nlp_signals:          Optional[list[str]]
    nlp_verdict:          Optional[str]


# ─── Metadata Endpoints ───────────────────────────────────────────────────────
@app.get("/metadata/cars")
def metadata_cars():
    """Return the brand → [models] lookup for the Car prediction model."""
    return CAR_METADATA


@app.get("/metadata/suv")
def metadata_suv():
    """Return the brand → [models] lookup for the SUV prediction model."""
    return SUV_METADATA


# ─── Car Prediction Endpoint ──────────────────────────────────────────────────
@app.post("/api/predict", response_model=PredictResponse)
def predict_car(req: PredictRequest):
    vehicle_age = max(REFERENCE_YEAR - req.model_year, 1)

    if req.mileage_km is None or req.mileage_km <= 0:
        effective_mileage    = float(vehicle_age * 12000)
        is_estimated_mileage = True
    else:
        effective_mileage    = req.mileage_km
        is_estimated_mileage = False

    mileage_per_year = round(effective_mileage / vehicle_age, 2)
    engine_cc        = _extract_engine_cc(f"{req.variant} {req.model}")

    row = {
        "brand":            req.brand,
        "model":            req.model,
        "variant":          req.variant,
        "fuel_type":        req.fuel_type,
        "transmission":     req.transmission,
        "model_year":       req.model_year,
        "mileage_km":       effective_mileage,
        "vehicle_age":      vehicle_age,
    }
    df = pd.DataFrame([row])[CAR_FEATURES]

    try:
        if CAR_IS_CATBOOST:
            log_price = car_model.predict(df)[0]
        else:
            df_enc = df.copy()
            df_enc[CAR_CATEGORICAL] = car_encoder.transform(
                df[CAR_CATEGORICAL].astype(str)
            )
            log_price = car_model.predict(df_enc)[0]

        predicted_price = int(round(np.expm1(log_price)))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    confidence = _get_confidence(req.brand, req.model, car_lookup_df)
    nlp        = _score_description(req.description or "", CAR_NLP_CONFIG)

    return PredictResponse(
        predicted_price=predicted_price,
        model_used=car_model_type,
        vehicle_age=vehicle_age,
        mileage_per_year=mileage_per_year,
        used_mileage_km=effective_mileage,
        is_mileage_estimated=is_estimated_mileage,
        confidence=confidence,
        nlp_score=nlp["nlp_score"] if req.description else None,
        nlp_signals=nlp["nlp_signals"] if req.description else None,
        nlp_verdict=nlp["nlp_verdict"] if req.description else None,
    )


# ─── SUV Prediction Endpoint ──────────────────────────────────────────────────
@app.post("/api/predict/suv", response_model=PredictResponse)
def predict_suv(req: SUVPredictRequest):
    vehicle_age = max(REFERENCE_YEAR - req.model_year, 1)

    if req.mileage_km is None or req.mileage_km <= 0:
        effective_mileage    = float(vehicle_age * 15000)   # SUVs avg higher annual km
        is_estimated_mileage = True
    else:
        effective_mileage    = req.mileage_km
        is_estimated_mileage = False

    mileage_per_year = round(effective_mileage / vehicle_age, 2)
    variant = _map_suv_generations(req.brand, req.model, req.model_year, req.variant)

    row = {
        "brand":        req.brand,
        "model":        req.model,
        "variant":      variant,
        "fuel_type":    req.fuel_type,
        "transmission": req.transmission,
        "model_year":   req.model_year,
        "mileage_km":   effective_mileage,
        "vehicle_age":  vehicle_age,
        "engine_cc":    float(req.engine_cc),
    }
    df = pd.DataFrame([row])[SUV_FEATURES]

    try:
        if SUV_IS_CATBOOST:
            log_price = suv_model.predict(df)[0]
        else:
            df_enc = df.copy()
            df_enc[SUV_CATEGORICAL] = suv_encoder.transform(
                df[SUV_CATEGORICAL].astype(str)
            )
            log_price = suv_model.predict(df_enc)[0]

        predicted_price = int(round(np.expm1(log_price)))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SUV prediction failed: {exc}") from exc

    confidence = _get_confidence(req.brand, req.model, suv_lookup_df)
    nlp        = _score_description(req.description or "", SUV_NLP_CONFIG)

    return PredictResponse(
        predicted_price=predicted_price,
        model_used=suv_model_type,
        vehicle_age=vehicle_age,
        mileage_per_year=mileage_per_year,
        used_mileage_km=effective_mileage,
        is_mileage_estimated=is_estimated_mileage,
        confidence=confidence,
        nlp_score=nlp["nlp_score"] if req.description else None,
        nlp_signals=nlp["nlp_signals"] if req.description else None,
        nlp_verdict=nlp["nlp_verdict"] if req.description else None,
    )


# ─── Health Check ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "car_model": car_model_type,
        "suv_model": suv_model_type,
    }


# ─── Dev Entry Point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
