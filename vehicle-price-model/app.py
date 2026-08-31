"""
FastAPI Prediction Server
Vehicle Price Predictor — Sri Lanka Market
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
BASE_DIR        = Path(__file__).resolve().parent
CAR_BASE_DIR    = BASE_DIR / "Car_price_Prediction"
CAR_MODEL_DIR   = CAR_BASE_DIR / "models"  / "combined"
CAR_OUT_DIR     = CAR_BASE_DIR / "outputs" / "combined"

SUV_BASE_DIR    = BASE_DIR / "SUV_Price_Prediction"
SUV_MODEL_DIR   = SUV_BASE_DIR / "models"  / "suv"
SUV_OUT_DIR     = SUV_BASE_DIR / "outputs" / "suv"

VAN_BASE_DIR    = BASE_DIR / "VAN_Price_Prediction"
VAN_MODEL_DIR   = VAN_BASE_DIR / "models" / "van"
VAN_OUT_DIR     = VAN_BASE_DIR / "outputs" / "van"

# ─── Feature config ───────────────────────────────────────────────────────────
CAR_CATEGORICAL  = ["brand", "model", "variant", "fuel_type", "transmission"]
CAR_NUMERIC      = ["model_year", "mileage_km", "vehicle_age"]
CAR_FEATURES     = CAR_CATEGORICAL + CAR_NUMERIC

SUV_CATEGORICAL  = ["brand", "model", "variant", "fuel_type", "transmission"]
SUV_NUMERIC      = ["model_year", "mileage_km", "vehicle_age", "engine_cc"]
SUV_FEATURES     = SUV_CATEGORICAL + SUV_NUMERIC

VAN_CATEGORICAL  = ["brand", "model", "variant", "fuel_type", "transmission", "engine_code"]
VAN_NUMERIC      = ["model_year", "mileage_km", "vehicle_age", "engine_cc"]
VAN_FEATURES     = VAN_CATEGORICAL + VAN_NUMERIC

REFERENCE_YEAR = 2026

# ─── Load Artifacts ───────────────────────────────────────────────────────────
print("Loading Car model artifacts …")
try:
    car_model      = joblib.load(CAR_MODEL_DIR / "best_model.pkl")
    car_encoder    = joblib.load(CAR_MODEL_DIR / "ordinal_encoder.pkl")
    car_model_type = type(car_model).__name__
except FileNotFoundError as exc:
    raise RuntimeError(f"Car model artifact not found: {exc}") from exc

CAR_IS_CATBOOST = "CatBoost" in car_model_type

print("Loading SUV model artifacts …")
try:
    suv_model      = joblib.load(SUV_MODEL_DIR / "best_suv_model.pkl")
    suv_encoder    = joblib.load(SUV_MODEL_DIR / "suv_ordinal_encoder.pkl")
    suv_model_type = type(suv_model).__name__
except FileNotFoundError as exc:
    raise RuntimeError(f"SUV model artifact not found: {exc}") from exc

SUV_IS_CATBOOST = "CatBoost" in suv_model_type

print("Loading Van model artifacts …")
try:
    van_model      = joblib.load(VAN_MODEL_DIR / "best_van_model.pkl")
    van_encoder    = joblib.load(VAN_MODEL_DIR / "van_ordinal_encoder.pkl")
    van_model_type = type(van_model).__name__
except FileNotFoundError as exc:
    van_model      = None
    van_encoder    = None
    van_model_type = None

VAN_IS_CATBOOST = "CatBoost" in van_model_type if van_model_type else False

# ─── Load Lookups & NLP Configs ───────────────────────────────────────────────
with open(CAR_OUT_DIR / "nlp_config.json", encoding="utf-8") as f:
    CAR_NLP_CONFIG = json.load(f)
with open(SUV_OUT_DIR / "suv_nlp_config.json", encoding="utf-8") as f:
    SUV_NLP_CONFIG = json.load(f)
try:
    with open(VAN_OUT_DIR / "van_nlp_config.json", encoding="utf-8") as f:
        VAN_NLP_CONFIG = json.load(f)
except FileNotFoundError:
    VAN_NLP_CONFIG = {}

car_lookup_df = pd.read_csv(CAR_OUT_DIR / "brand_model_lookup.csv")
suv_lookup_df = pd.read_csv(SUV_OUT_DIR / "suv_brand_model_lookup.csv")
try:
    van_lookup_df = pd.read_csv(VAN_OUT_DIR / "van_brand_model_lookup.csv")
except FileNotFoundError:
    van_lookup_df = pd.DataFrame(columns=["brand", "model", "confidence"])

def _build_metadata(df: pd.DataFrame) -> dict:
    brands = sorted(df["brand"].unique().tolist())
    models: dict[str, list[str]] = {}
    for brand, group in df.groupby("brand"):
        models[brand] = sorted(group["model"].unique().tolist())
    return {"brands": brands, "models": models}

CAR_METADATA = _build_metadata(car_lookup_df)
SUV_METADATA = _build_metadata(suv_lookup_df)
VAN_METADATA = _build_metadata(van_lookup_df)

app = FastAPI(title="Vehicle Price Predictor API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ─── NORMALIZERS ──────────────────────────────────────────────────────────────
def normalize_car_model(raw_model: str, brand: str = "") -> str:
    if not isinstance(raw_model, str): 
        return "Unknown"
    
    m = raw_model.strip().lower()
    
    if "wagon r" in m or "wagonr" in m:
        if "stingray" in m: return "Wagon R Stingray"
        if "fz" in m: return "Wagon R FZ"
        if "fx" in m: return "Wagon R FX"
        return "Wagon R"
    if "alto k10" in m: return "Alto K10"
    if "alto" in m: return "Alto"
    if "celerio" in m: return "Celerio"
    if "swift" in m: return "Swift"
    if "hustler" in m: return "Hustler"
    if "maruti" in m: return "Maruti"
    if "spacia" in m: return "Spacia"
    
    if "kelisa" in m: return "Kelisa"
    if "bezza" in m: return "Bezza"
    if "kenari" in m: return "Kenari"
    if "viva elite" in m: return "Viva Elite"
    if "axia" in m or "axeya" in m: return "Axia"
    
    if "aqua" in m: return "Aqua"
    if "prius" in m: return "Prius"
    if "vitz" in m: return "Vitz"
    if "premio" in m: return "Premio"
    if "axio" in m: return "Axio"
    if "carina" in m: return "Carina"
    if "allion" in m: return "Allion"
    if "vios" in m: return "Vios"
    if "passo" in m: return "Passo"
    if "121" in m and ("corolla" in m or brand.lower() == "toyota"): return "Corolla 121"
    if "141" in m and ("corolla" in m or brand.lower() == "toyota"): return "Corolla 141"
    if "110" in m and ("corolla" in m or brand.lower() == "toyota"): return "110"
    
    if "fit gp1" in m: return "Fit GP1"
    if "fit gp5" in m: return "FIT GP5"
    if "civic fd3" in m: return "Civic FD3"
    if "civic fd4" in m: return "Civic FD4"
    if "civic fd1" in m: return "Civic FD1"
    if "civic es8" in m: return "Civic ES8"
    if "civic es5" in m: return "Civic ES5"
    if "grace" in m: return "Honda Grace"
    if "insight" in m: return "Insight"
    if "civic" in m: return "Civic"
    
    if "panda cross" in m: return "Panda Cross"
    if "panda" in m: return "Panda"
    if "mx7" in m: return "MX7"
    if "emgrand" in m: return "Emgrand"
    
    if "accent" in m: return "Accent"
    if "sonata" in m: return "Sonata"
    if "eon" in m: return "Eon"
    if "elantra" in m: return "Elantra"
    
    if any(x in m for x in ["axela", "mazda 3", "mazda 6", "mazda3", "mazda6"]): return "Axela"
    if m == "3" or m == "6" or m.startswith("3 ") or m.startswith("6 "): return "Axela"
    if "demio" in m: return "Demio"
    if "familia" in m: return "Familia"
    
    if "cs3" in m: return "CS3"
    if "cs1" in m: return "CS1"
    if "cs2" in m: return "CS2"
    
    if "fb13" in m: return "FB13"
    if "fb14" in m: return "FB14"
    if "fb15" in m: return "FB15"
    if "n16" in m: return "N16"
    if "n17" in m: return "N17"
    if "leaf" in m: return "Leaf"
    if "cefiro" in m: return "Cefiro"
    if "march" in m:
        if "k10" in m: return "March K10"
        if "k11" in m: return "March K11"
        if "k12" in m or "ak12" in m: return "March K12"
        return "March"
    if "tiida" in m: return "Tiida"
    
    if "mira" in m: return "Mira"

    if "indica" in m: return "Indica"
    if "nano" in m: return "Nano"
    if "indigo" in m: return "Indigo"

    if "a1" in m: return "A1"
    if "a3" in m: return "A3"
    if "a4" in m: return "A4"
    if "a5" in m: return "A5"
    if "a6" in m: return "A6"

    if "318i" in m: return "318i"
    if "520d" in m: return "520D"
    if "320d" in m: return "320D"
    if "730ld" in m: return "730Ld"
    if "530e" in m: return "530e"
    if "523i" in m: return "523i"
    if "mini cooper" in m: return "Mini Cooper"
    if "i8" in m: return "I8"

    if "c180" in m: return "C180"
    if "e200" in m: return "E200"
    if "a200" in m: return "A200"
    if "e300" in m: return "E300"
    if "slk200" in m or "slk 200" in m: return "SLK200"
    if "e220" in m: return "E220"
    if "s350" in m: return "S350"
    if "e180" in m: return "E180"
    if "e240" in m: return "E240"
    if "c200" in m: return "C200"
    if "cla 200" in m or "cla200" in m: return "CLA 200"
    if "c250" in m: return "C250"
    if "w210" in m: return "W210"
    if "e350" in m: return "E350"
    
    # Return Unknown for any explicitly unmapped Car model
    return "Unknown"


def normalize_suv_model(raw_model: str, engine_cc: float, year: int) -> str:
    if not isinstance(raw_model, str): 
        return "Unknown"
    
    m = raw_model.strip().lower()

    if re.search(r'\bq2\b', m): return "Q2"
    if re.search(r'\bq3\b', m): return "Q3"
    if re.search(r'\bq5\b', m): return "Q5"
    if re.search(r'\bq7\b', m): return "Q7"

    if re.search(r'\bx1\b', m): return "X1"
    if re.search(r'\bx2\b', m): return "X2"
    if re.search(r'\bx3\b', m): return "X3"
    if re.search(r'\bx5\b', m): return "X5"
    if "zs" in m: return "ZS"

    if "outlander" in m: return "Outlander"
    if "pajero io" in m or "gdi io" in m or "pajero dgi" in m: return "Pajero Io"
    if "eclipse cross" in m: return "Eclipse Cross"
    if "asx" in m: return "ASX"
    if "montero sport" in m: return "Montero Sport"
    if "montero" in m:
        if year and year <= 2006:
            return "Montero 3rd gen"
        elif year and year >= 2007:
            return "Montero 4th gen"
        return "Montero"

    if "box prado" in m or "bj75" in m: return "Box Prado"
    if "xtrail" in m or "x-trail" in m: return "X-Trail"
    if "chr" in m or "c-hr" in m: return "CHR"
    if "rav4" in m: return "RAV4"
    if "raize" in m: return "Raize"
    if any(k in m for k in ["vitara", "escudo"]): return "Vitara"
    if "xbee" in m: return "Xbee"
    if "fronx" in m: return "Fronx"
    if "s cross" in m or "scross" in m: return "S Cross"
    if "jimny" in m: return "Jimny"
    if "vezel" in m: return "Vezel"
    if "crv" in m or "cr-v" in m: return "CRV"
    if "tucson" in m: return "Tucson"
    if "sorento" in m: return "Sorento"
    if "rexton" in m: return "Rexton"
    if "kyron" in m: return "Kyron"
    if "actyon" in m: return "Actyon"
    if "korando" in m: return "Korando"
    if "kuv" in m and "100" in m: return "KUV 100"
    if "3008" in m: return "3008"
    if "5008" in m: return "5008"
    if "2008" in m: return "2008"
        
    if "land cruiser" in m or "prado" in m or "v8" in m:
        if "sahara" in m or "v8" in m or "zx" in m: return "V8"
        if engine_cc and engine_cc > 4000: return "V8"
        if "land cruiser" in m: return "Prado"
        return "Prado"
        
    # Return Unknown for any explicitly unmapped SUV model
    return "Unknown"


def normalize_van_model(raw_model: str) -> str:
    if not isinstance(raw_model, str): 
        return "Unknown"
    
    m = raw_model.strip().lower()

    if "townace" in m or "town ace" in m: return "Townace"
    if "liteace" in m or "lite ace" in m: return "Liteace"
    if "dolphin" in m: return "Dolphin"
    if "kdh" in m: return "KDH"
    if "voxy" in m: return "Voxy"

    if "every" in m: return "Every"

    if "caravan e25" in m or "e25" in m: return "E25"
    if "caravan" in m: return "Caravan"
    if "serena" in m: return "Serena"
    if "vanette" in m: return "Vanette"

    if "bongo" in m: return "Bongo"
    if "brawny" in m: return "Brawny"

    if "fargo" in m: return "Fargo"

    if "hijet" in m: return "Hijet"
        
    # Return Unknown for any explicitly unmapped Van model
    return "Unknown"


def normalize_variant(clean_model_name: str, raw_variant: str) -> str:
    if not isinstance(raw_variant, str): 
        return "Standard"
        
    v = raw_variant.strip().lower()
    
    if not v or v in ["", "-", "none", "n/a", "unknown", "null"]:
        return "Standard"
        
    standard_only_models = [
        "Alto", "Alto K10", "Wagon R", "Wagon R FZ", "Wagon R FX", "Wagon R Stingray",
        "Celerio", "Swift", "Hustler", "Maruti", "Spacia",
        "Aqua", "Prius", "Vitz", "Premio", "Axio", "Carina", "Allion", "Vios", "Passo", "Corolla 121", "Corolla 141", "110",
        "Kelisa", "Bezza", "Kenari", "Viva Elite", "Axia",
        "Fit GP1", "FIT GP5", "Civic FD3", "Civic FD4", "Civic FD1", "Civic ES8", "Civic ES5", "Honda Grace", "Insight", "Civic",
        "Panda", "Panda Cross", "MX7", "Emgrand",
        "Accent", "Sonata", "Eon", "Elantra",
        "Axela", "Demio", "Familia",
        "CS3", "CS1", "CS2",
        "FB13", "FB14", "FB15", "N16", "N17", "Leaf", "Cefiro", "March K10", "March K11", "March K12", "Tiida",
        "Mira",
        "Indica", "Nano", "Indigo",
        "A1", "A3", "A4", "A5", "A6", "Q2", "Q3", "Q5", "Q7",
        "318i", "520D", "320D", "730Ld", "530e", "523i", "Mini Cooper", "I8",
        "C180", "E200", "A200", "E300", "SLK200", "E220", "S350", "E180", "E240", "C200", "CLA 200", "C250", "W210", "E350",
        "X1", "X2", "X3", "X5", "ZS",
        "Outlander", "Pajero Io", "Eclipse Cross", "ASX", "Montero Sport", "Montero 3rd gen", "Montero 4th gen",
        # Vans
        "Townace", "Liteace", "Dolphin", "KDH", "Voxy", "Every", "E25", "Caravan", "Serena", "Vanette", "Bongo", "Brawny", "Fargo", "Hijet"
    ]
    if clean_model_name in standard_only_models:
        return "Standard"
        
    junk_words = ["model", "car", "used", "unregistered", "brand new"]
    if any(junk in v for junk in junk_words) or v.isdigit():
        return "Standard"
        
    return raw_variant.strip().title()


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def _score_description(description: str, nlp_config: dict) -> dict:
    if not description: 
        return {"nlp_score": 0, "nlp_signals": [], "nlp_verdict": None}
    
    text = description.lower()
    total_nlp_points = 0
    matched_labels = []
    has_fatal_issue = False

    # 1. Process Negative Signals (Identify fatal issues first)
    for signal in nlp_config.get("negative_signals", {}).values():
        if any(kw in text for kw in signal["keywords"]):
            total_nlp_points += signal["points"]
            matched_labels.append(f"⚠️ {signal['label']}")
            if signal["points"] <= -10:
                has_fatal_issue = True

    # 2. Process Positive Signals
    for signal in nlp_config.get("positive_signals", {}).values():
        if any(kw in text for kw in signal["keywords"]):
            if has_fatal_issue:
                matched_labels.append(f"(Ignored: {signal['label']})")
            else:
                total_nlp_points += signal["points"]
                matched_labels.append(signal["label"])

    scoring = nlp_config.get("scoring", {})
    
    # 3. Apply Fatal Override Logic
    if has_fatal_issue:
        final_score = 20
        verdict = "High Risk 🔴"
    else:
        final_score = min(scoring.get("final_max", 100), max(0, 50 + max(-30, min(30, total_nlp_points))))
        if final_score >= scoring.get("fairly_priced_min", 65): 
            verdict = "Fairly Priced ✅"
        elif final_score >= scoring.get("review_min", 45): 
            verdict = "Review Carefully ⚠️"
        else: 
            verdict = "Caution 🔴"

    return {"nlp_score": final_score, "nlp_signals": matched_labels, "nlp_verdict": verdict}


def _extract_engine_cc(text: str) -> float:
    m = re.search(r"(\d{3,4})\s*cc", text, re.IGNORECASE)
    if m: 
        return float(m.group(1))
    
    m = re.search(r"\b(\d\.\d)\s*(?:l|liter)?\b", text, re.IGNORECASE)
    if m: 
        return float({"1.0": 1000, "1.2": 1200, "1.5": 1500, "1.8": 1800, "2.0": 2000}.get(m.group(1), -1))
        
    return -1.0


def _get_confidence(brand: str, model_name: str, lookup_df: pd.DataFrame) -> str:
    row = lookup_df[(lookup_df["brand"].str.lower() == brand.lower()) & (lookup_df["model"].str.lower() == model_name.lower())]
    if row.empty: 
        return "Unknown"
    
    raw = str(row.iloc[0]["confidence"])
    if raw.startswith("High"): return "High"
    if raw.startswith("Medium"): return "Medium"
    
    return "Low"


def _map_van_engine_code(brand, model, fuel_type, engine_cc):
    if brand.lower() != "toyota" or fuel_type.lower() != "diesel" or model.lower() != "dolphin": 
        return "NA"
    if pd.isna(engine_cc) or engine_cc < 0: 
        return "NA"
    if 2300 <= engine_cc <= 2600: return "2L"
    if 2600 <= engine_cc <= 2850: return "3L"
    if 2850 <= engine_cc <= 3150: return "5L"
    return "Other"


def _map_suv_generations(brand, model, year, variant):
    v = str(variant).strip()
    b = str(brand).strip().lower()
    m = str(model).strip().lower()
    y = int(year)
    
    if b == "toyota" and m == "rav4":
        if y <= 2000: return "1st Gen"
        elif 2001 <= 2005: return "2nd Gen"
        elif v.lower() == "standard" and 2006 <= y <= 2012: return "3rd Gen"
        elif v.lower() == "standard" and 2013 <= y <= 2018: return "4th Gen"
        elif v.lower() == "standard": return "5th Gen"
        
    return variant


class PredictRequest(BaseModel):
    brand: str
    model: str
    variant: str = "Standard"
    model_year: int
    mileage_km: Optional[float] = None
    fuel_type: str
    transmission: str
    description: Optional[str] = None
    
    @field_validator("variant")
    @classmethod
    def default_variant(cls, v: str) -> str: 
        return v.strip() if v.strip() else "Standard"


class SUVPredictRequest(PredictRequest):
    engine_cc: int


class VanPredictRequest(PredictRequest):
    engine_cc: Optional[float] = None


class PredictResponse(BaseModel):
    predicted_price: int
    model_used: str
    vehicle_age: int
    mileage_per_year: float
    used_mileage_km: float
    is_mileage_estimated: bool
    confidence: str
    nlp_score: Optional[int]
    nlp_signals: Optional[list[str]]
    nlp_verdict: Optional[str]

# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/metadata/cars")
def metadata_cars(): return CAR_METADATA

@app.get("/metadata/suv")
def metadata_suv(): return SUV_METADATA

@app.get("/metadata/van")
def metadata_van(): return VAN_METADATA

@app.post("/api/predict", response_model=PredictResponse)
@app.post("/predict", response_model=PredictResponse)
def predict_car(req: PredictRequest):
    vehicle_age = max(REFERENCE_YEAR - req.model_year, 1)
    effective_mileage = req.mileage_km if req.mileage_km and req.mileage_km > 0 else float(vehicle_age * 12000)
    
    clean_model_name = normalize_car_model(req.model, req.brand)
    clean_variant = normalize_variant(clean_model_name, req.variant)
    
    print("\n" + "="*50)
    print(f"[CAR NORMALIZER] Raw Model: '{req.model}' -> Cleaned: '{clean_model_name}'")
    print(f"[CAR NORMALIZER] Raw Variant: '{req.variant}' -> Cleaned: '{clean_variant}'")
    print("="*50 + "\n", flush=True)

    # Intercept unknown models before ML Prediction
    if clean_model_name == "Unknown":
        return PredictResponse(
            predicted_price=0, 
            model_used=car_model_type, 
            vehicle_age=vehicle_age,
            mileage_per_year=round(effective_mileage / vehicle_age, 2), 
            used_mileage_km=effective_mileage,
            is_mileage_estimated=(req.mileage_km is None or req.mileage_km <= 0),
            confidence="Unknown", 
            nlp_score=0, 
            nlp_signals=[], 
            nlp_verdict=None
        )

    row = {
        "brand": req.brand, 
        "model": clean_model_name, 
        "variant": clean_variant, 
        "fuel_type": req.fuel_type, 
        "transmission": req.transmission,
        "model_year": req.model_year, 
        "mileage_km": effective_mileage, 
        "vehicle_age": vehicle_age
    }
    df = pd.DataFrame([row])[CAR_FEATURES]
    
    if not CAR_IS_CATBOOST:
        df[CAR_CATEGORICAL] = car_encoder.transform(df[CAR_CATEGORICAL].astype(str))
    
    log_price = car_model.predict(df)[0]
    nlp = _score_description(req.description or "", CAR_NLP_CONFIG)
    
    return PredictResponse(
        predicted_price=int(round(np.expm1(log_price))), 
        model_used=car_model_type, 
        vehicle_age=vehicle_age,
        mileage_per_year=round(effective_mileage / vehicle_age, 2), 
        used_mileage_km=effective_mileage,
        is_mileage_estimated=(req.mileage_km is None or req.mileage_km <= 0),
        confidence=_get_confidence(req.brand, clean_model_name, car_lookup_df),
        **nlp
    )


@app.post("/api/predict/suv", response_model=PredictResponse)
@app.post("/predict/suv", response_model=PredictResponse)
def predict_suv(req: SUVPredictRequest):
    vehicle_age = max(REFERENCE_YEAR - req.model_year, 1)
    effective_mileage = req.mileage_km if req.mileage_km and req.mileage_km > 0 else float(vehicle_age * 15000)
    
    clean_model_name = normalize_suv_model(req.model, float(req.engine_cc), req.model_year)
    clean_variant = normalize_variant(clean_model_name, req.variant)
    
    final_variant = _map_suv_generations(req.brand, clean_model_name, req.model_year, clean_variant)
    
    print("\n" + "="*50)
    print(f"[SUV NORMALIZER] Raw Model: '{req.model}' -> Cleaned: '{clean_model_name}'")
    print(f"[SUV NORMALIZER] Raw Variant: '{req.variant}' -> Cleaned: '{final_variant}'")
    print("="*50 + "\n", flush=True)

    # Intercept unknown models before ML Prediction
    if clean_model_name == "Unknown":
        return PredictResponse(
            predicted_price=0, 
            model_used=suv_model_type, 
            vehicle_age=vehicle_age,
            mileage_per_year=round(effective_mileage / vehicle_age, 2), 
            used_mileage_km=effective_mileage,
            is_mileage_estimated=(req.mileage_km is None or req.mileage_km <= 0),
            confidence="Unknown", 
            nlp_score=0, 
            nlp_signals=[], 
            nlp_verdict=None
        )

    row = {
        "brand": req.brand, 
        "model": clean_model_name, 
        "variant": final_variant,
        "fuel_type": req.fuel_type, 
        "transmission": req.transmission, 
        "model_year": req.model_year,
        "mileage_km": effective_mileage, 
        "vehicle_age": vehicle_age, 
        "engine_cc": float(req.engine_cc)
    }
    df = pd.DataFrame([row])[SUV_FEATURES]
    
    if not SUV_IS_CATBOOST:
        df[SUV_CATEGORICAL] = suv_encoder.transform(df[SUV_CATEGORICAL].astype(str))

    log_price = suv_model.predict(df)[0]
    nlp = _score_description(req.description or "", SUV_NLP_CONFIG)
    
    return PredictResponse(
        predicted_price=int(round(np.expm1(log_price))), 
        model_used=suv_model_type, 
        vehicle_age=vehicle_age,
        mileage_per_year=round(effective_mileage / vehicle_age, 2), 
        used_mileage_km=effective_mileage,
        is_mileage_estimated=(req.mileage_km is None or req.mileage_km <= 0),
        confidence=_get_confidence(req.brand, clean_model_name, suv_lookup_df),
        **nlp
    )


@app.post("/api/predict/van", response_model=PredictResponse)
@app.post("/predict/van", response_model=PredictResponse)
def predict_van(req: VanPredictRequest):
    if not van_model: 
        raise HTTPException(status_code=503, detail="Van model not loaded.")
        
    vehicle_age = max(REFERENCE_YEAR - req.model_year, 1)
    effective_mileage = req.mileage_km if req.mileage_km and req.mileage_km > 0 else float(vehicle_age * 15000)
    
    clean_model_name = normalize_van_model(req.model)
    clean_variant = normalize_variant(clean_model_name, req.variant)
    
    engine_cc_extracted = req.engine_cc if req.engine_cc else _extract_engine_cc(f"{req.variant} {req.model}")

    if clean_model_name == "KDH":
        if "diesel" in req.fuel_type.lower():
            engine_cc_extracted = 3000.0
        elif "petrol" in req.fuel_type.lower():
            engine_cc_extracted = 2000.0

    print("\n" + "="*50)
    print(f"[VAN NORMALIZER] Raw Model: '{req.model}' -> Cleaned: '{clean_model_name}'")
    print(f"[VAN NORMALIZER] Raw Variant: '{req.variant}' -> Cleaned: '{clean_variant}'")
    print(f"[VAN NORMALIZER] Final Engine CC: {engine_cc_extracted}")
    print("="*50 + "\n", flush=True)

    # Intercept unknown models before ML Prediction
    if clean_model_name == "Unknown":
        return PredictResponse(
            predicted_price=0, 
            model_used=van_model_type, 
            vehicle_age=vehicle_age,
            mileage_per_year=round(effective_mileage / vehicle_age, 2), 
            used_mileage_km=effective_mileage,
            is_mileage_estimated=(req.mileage_km is None or req.mileage_km <= 0),
            confidence="Unknown", 
            nlp_score=0, 
            nlp_signals=[], 
            nlp_verdict=None
        )

    row = {
        "brand": req.brand, 
        "model": clean_model_name, 
        "variant": clean_variant, 
        "fuel_type": req.fuel_type,
        "transmission": req.transmission, 
        "engine_code": _map_van_engine_code(req.brand, clean_model_name, req.fuel_type, engine_cc_extracted),
        "model_year": req.model_year, 
        "mileage_km": effective_mileage, 
        "vehicle_age": vehicle_age,
        "engine_cc": float(engine_cc_extracted) if engine_cc_extracted > 0 else -1.0
    }
    df = pd.DataFrame([row])[VAN_FEATURES]
    
    if not VAN_IS_CATBOOST:
        df[VAN_CATEGORICAL] = van_encoder.transform(df[VAN_CATEGORICAL].astype(str))

    log_price = van_model.predict(df)[0]
    nlp = _score_description(req.description or "", VAN_NLP_CONFIG)
    
    return PredictResponse(
        predicted_price=int(round(np.expm1(log_price))), 
        model_used=van_model_type, 
        vehicle_age=vehicle_age,
        mileage_per_year=round(effective_mileage / vehicle_age, 2), 
        used_mileage_km=effective_mileage,
        is_mileage_estimated=(req.mileage_km is None or req.mileage_km <= 0),
        confidence=_get_confidence(req.brand, clean_model_name, van_lookup_df),
        **nlp
    )


@app.get("/health")
def health():
    return {
        "status": "ok", 
        "car_model": car_model_type, 
        "suv_model": suv_model_type, 
        "van_model": van_model_type
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8003, reload=True)