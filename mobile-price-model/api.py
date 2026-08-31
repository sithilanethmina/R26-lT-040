from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import json
import sys
import re
from pathlib import Path
import uvicorn
from typing import Optional, Dict, Any

app = FastAPI(title="Mobile Price Predictor API")

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data_preprocessing import standardize_brand
from src.feature_engineering import (
    compute_brand_tier,
    compute_model_tier,
    compute_phone_age,
    compute_is_flagship
)
from src.phone_specs import (
    IPHONE_RAM_GB_BY_NORMALIZED_MODEL,
    get_android_valid_ram,
    snap_ram_to_nearest_valid,
    get_iphone_capabilities,
    get_android_capabilities,
    normalized_model_key
)
from src.llm_extractor import extract_mobile_specs_llm

EVALUATION_FILE = BASE_DIR / "outputs" / "model_evaluation_results.json"
MODEL_DIR = BASE_DIR / "models"

# Global cache for models
models_cache = {}
evaluation_data = {}

def load_evaluation_data() -> dict:
    if not EVALUATION_FILE.exists():
        return {}
    with EVALUATION_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)

def get_recommended_model_info(phone_type: str, eval_data: dict) -> tuple:
    recommendation = eval_data.get("recommendations", {}).get(phone_type, {})
    model_name = str(recommendation.get("recommended_model", "CatBoost"))
    # The actual saved models are named best_{phone_type}_model.pkl
    model_path = MODEL_DIR / f"best_{phone_type}_model.pkl"
    model_key = f"{model_name.lower().replace(' ', '_')}_{phone_type}"
    metrics = eval_data.get("results", {}).get(model_key, {})
    return model_name, model_path, metrics

PROCESSED_DATA_FILE = BASE_DIR / "data" / "processed" / "ikman_mobile_phones_ml_ready.json"
LOOKUP_FILE = BASE_DIR / "outputs" / "mobile_brand_model_lookup.json"
supported_brand_models = {}

def load_brand_model_lookup() -> dict:
    """Build supported brand-model catalog directly from the ML-ready dataset."""
    if PROCESSED_DATA_FILE.exists():
        try:
            with PROCESSED_DATA_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
            lookup: dict[str, set[str]] = {}
            for item in data:
                b = item.get("brand")
                m = item.get("model")
                if b and m and str(m).strip():
                    b_str = str(b).strip()
                    m_str = str(m).strip()
                    if b_str not in lookup:
                        lookup[b_str] = set()
                    lookup[b_str].add(m_str)
            return {b: sorted(list(models)) for b, models in sorted(lookup.items())}
        except Exception as e:
            print(f"Warning: Failed to load from {PROCESSED_DATA_FILE}: {e}")

    if LOOKUP_FILE.exists():
        with LOOKUP_FILE.open("r", encoding="utf-8") as file:
            return json.load(file)
    return {}

def normalize_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def find_matched_model(brand: str, model: str) -> tuple[Optional[str], Optional[str]]:
    """Match brand and model against verified supported database."""
    if not supported_brand_models:
        return brand, model
        
    b_match = None
    b_norm = normalize_key(brand) if brand else ""
    
    # 1. Exact match on provided brand
    if b_norm:
        for b in supported_brand_models:
            if normalize_key(b) == b_norm:
                b_match = b
                break
                
    # 2. Check if a known brand is present in the brand/model string
    if not b_match:
        combined = f"{brand} {model}".lower()
        for b in ["Apple", "Samsung", "Xiaomi", "Google", "OnePlus", "Huawei", "Vivo", "Oppo", "Realme", "Nokia", "Sony", "Motorola", "Honor", "Nothing", "Infinix", "Tecno"]:
            if re.search(rf"\b{re.escape(b.lower())}\b", combined) or (b == "Apple" and "iphone" in combined):
                if b in supported_brand_models:
                    b_match = b
                    break

    # 3. Fallback: Search all models across brands preferring longest match
    if not b_match:
        all_candidates = []
        for b, models in supported_brand_models.items():
            for m in models:
                if len(m) >= 3 and normalize_key(m) in normalize_key(model):
                    all_candidates.append((len(m), b, m))
        if all_candidates:
            all_candidates.sort(key=lambda x: -x[0])
            b_match = all_candidates[0][1]

    if not b_match:
        return None, None
        
    models_list = supported_brand_models[b_match]
    m_clean = re.sub(r"(?i)\b(\d{1,4}\s*(?:gb|tb)|\d+\s*gb\s*ram|\d+\s*mah|5g|4g|lte|dual\s*sim|esim|factory\s*unlocked|brand\s*new|sealed|box|used|like\s*new|mint|awesome\s*\w+|graphite|silver|gold|midnight|starlight|blue|green|red|purple|black|white|yellow|coral|titanium|gray|grey|space\s*gray|pacific\s*blue|sierra\s*blue|alpine\s*green|deep\s*purple|natural\s*titanium|desert\s*titanium|ll/a|zp/a|za/a|ch/a|vn/a|my/a|x/a|b/a|ah/a|ae/a|j/a|kh/a|th/a|hn/a|ta/a|fb/a)\b", "", model)
    m_clean = re.sub(r"\s+", " ", m_clean).strip()
    m_norm = normalize_key(m_clean)

    exact_matches = [cand for cand in models_list if normalize_key(cand) == m_norm]
    if exact_matches:
        return b_match, exact_matches[0]

    raw_norm = normalize_key(model)
    raw_exact = [cand for cand in models_list if normalize_key(cand) == raw_norm]
    if raw_exact:
        return b_match, raw_exact[0]

    candidates = [
        cand for cand in models_list 
        if normalize_key(cand) in m_norm or normalize_key(cand) in raw_norm
    ]
    if candidates:
        best_cand = max(candidates, key=lambda cand: (len(normalize_key(cand)), -abs(len(normalize_key(cand)) - len(m_norm))))
        return b_match, best_cand

    return b_match, None

@app.on_event("startup")
def startup_event():
    global evaluation_data, supported_brand_models
    evaluation_data = load_evaluation_data()
    supported_brand_models = load_brand_model_lookup()
    
    # Pre-load recommended models
    for phone_type in ["android", "iphone"]:
        try:
            name, path, metrics = get_recommended_model_info(phone_type, evaluation_data)
            if path.exists():
                models_cache[phone_type] = joblib.load(str(path))
                print(f"Loaded {name} for {phone_type} from {path}")
            else:
                print(f"Warning: Model file not found at {path}")
        except Exception as e:
            print(f"Error loading model for {phone_type}: {e}")

class PredictRequest(BaseModel):
    phone_type: Optional[str] = None
    brand: Optional[str] = ""
    model: Optional[str] = ""
    title: Optional[str] = None
    description: Optional[str] = None
    raw_text: Optional[str] = None
    storage_gb: Optional[float] = None
    ram_gb: Optional[float] = None
    warranty_days: Optional[float] = None
    battery_health_percent: Optional[float] = None
    dual_sim: Optional[bool] = None
    has_5g: Optional[bool] = None
    has_esim: Optional[bool] = None
    model_tier: Optional[int] = None
    brand_tier: Optional[int] = None
    phone_age_years: Optional[float] = None
    is_flagship: Optional[int] = None

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "mobile_price_predictor"}

@app.get("/metadata")
def get_metadata():
    if not supported_brand_models:
        lookup = load_brand_model_lookup()
    else:
        lookup = supported_brand_models
    return {
        "brands": sorted(list(lookup.keys())),
        "models": lookup
    }

@app.post("/predict")
def predict(request: PredictRequest):
    # 1. Optional LLM Extraction Layer (Gemini Flash Lite)
    llm_extracted = None
    extraction_source = "DOM / Direct Client"

    # If unstructured text is present, invoke LLM extraction
    has_unstructured_input = bool(request.title or request.description or request.raw_text)

    if has_unstructured_input:
        llm_extracted = extract_mobile_specs_llm(
            title=request.title or request.model or "",
            description=request.description or "",
            raw_text=request.raw_text or ""
        )
        if llm_extracted:
            extraction_source = "Gemini Flash Lite (LLM)"
            if llm_extracted.get("brand") and (not request.brand or request.brand in ["Apple", "Samsung", "Xiaomi", "Google"]):
                request.brand = llm_extracted["brand"]
            if llm_extracted.get("model"):
                request.model = llm_extracted["model"]
            if llm_extracted.get("storage_gb") is not None:
                request.storage_gb = float(llm_extracted["storage_gb"])
            if llm_extracted.get("ram_gb") is not None:
                request.ram_gb = float(llm_extracted["ram_gb"])
            if request.battery_health_percent is None and llm_extracted.get("battery_health_percent") is not None:
                request.battery_health_percent = float(llm_extracted["battery_health_percent"])
            if (request.warranty_days is None or request.warranty_days == 0) and llm_extracted.get("warranty_days") is not None:
                request.warranty_days = float(llm_extracted["warranty_days"])

    # 2. Standardize brand casing (e.g. "VIVO" -> "Vivo", "APPLE" -> "Apple")
    clean_brand = standardize_brand(request.brand or "")
    clean_model = (request.model or "").strip()

    # Fallback to title/text if model/brand was not explicitly specified
    if not clean_model and request.title:
        clean_model = request.title.strip()
    if not clean_brand and clean_model:
        for b in ["Apple", "Samsung", "Xiaomi", "Google", "OnePlus", "Huawei", "Vivo", "Oppo", "Realme", "Nokia", "Sony", "Motorola"]:
            if b.lower() in clean_model.lower() or (b == "Apple" and "iphone" in clean_model.lower()):
                clean_brand = b
                break

    # 3. Strict model validation: Ensure the model is in the supported models dataset
    matched_brand, matched_model = find_matched_model(clean_brand, clean_model)
    if not matched_brand or not matched_model:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' ({clean_brand}) is not supported. FairPriceLK only provides valuation for verified mobile phone models in its dataset."
        )

    clean_brand = matched_brand
    clean_model = matched_model
    model_key = normalized_model_key(clean_model)

    # 4. Determine and validate phone_type
    req_phone_type = (request.phone_type or "").lower().strip()
    if clean_brand == "Apple" or "iphone" in clean_model.lower() or "iphone" in clean_brand.lower():
        phone_type = "iphone"
        clean_brand = "Apple"
    elif req_phone_type in ["android", "iphone"]:
        if req_phone_type == "iphone" and clean_brand not in ["Apple", "Unknown"]:
            phone_type = "android"
        else:
            phone_type = req_phone_type
    else:
        phone_type = "android"
        
    if phone_type not in models_cache:
        # Attempt to load it on demand
        name, path, metrics = get_recommended_model_info(phone_type, evaluation_data)
        if path.exists():
            models_cache[phone_type] = joblib.load(str(path))
        else:
            raise HTTPException(status_code=500, detail=f"Model for {phone_type} not found at {path}.")

    model_pipeline = models_cache[phone_type]
    
    STANDARD_STORAGE_OPTIONS = [16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]

    # 5. Resolve specs & enforce phone_specs.py as the absolute Single Source of Truth
    
    # ── Hardware Capabilities (5G, eSIM, Dual SIM) ── (100% phone_specs.py authoritative)
    if phone_type == "iphone":
        caps = get_iphone_capabilities(model_key)
    else:
        caps = get_android_capabilities(clean_brand, clean_model)

    resolved_dual_sim = int(caps.get("dual_sim", 1))
    resolved_5g = int(caps.get("has_5g", 0))
    resolved_esim = int(caps.get("has_esim", 0))
    caps_source = "Canonical Hardware Truth [phone_specs.py]"

    # ── RAM Resolution ── (phone_specs.py authoritative)
    if phone_type == "iphone":
        # iPhones have fixed hardware RAM by model (cannot be modified)
        canonical_iphone_ram = IPHONE_RAM_GB_BY_NORMALIZED_MODEL.get(model_key, 4.0)
        resolved_ram = float(canonical_iphone_ram)
        if request.ram_gb and float(request.ram_gb) != resolved_ram:
            ram_source = f"Overridden to Canonical {resolved_ram:.0f}GB [phone_specs.py] (Seller claimed {request.ram_gb}GB)"
        else:
            ram_source = "Canonical Hardware Truth [phone_specs.py]"
    else:
        android_rams = get_android_valid_ram(clean_brand, clean_model)
        if android_rams:
            if request.ram_gb and float(request.ram_gb) in android_rams:
                resolved_ram = float(request.ram_gb)
                ram_source = f"Verified Variant ({resolved_ram:.0f}GB)"
            elif request.ram_gb and float(request.ram_gb) > 0:
                snapped_ram = snap_ram_to_nearest_valid(float(request.ram_gb), android_rams)
                resolved_ram = float(snapped_ram)
                ram_source = f"Snapped to Valid Variant {resolved_ram:.0f}GB [phone_specs.py] (Seller claimed {request.ram_gb}GB)"
            else:
                resolved_ram = float(android_rams[0])
                ram_source = "Primary Valid Variant [phone_specs.py]"
        else:
            if request.ram_gb and 1.0 <= float(request.ram_gb) <= 16.0:
                resolved_ram = float(request.ram_gb)
                ram_source = "Client / LLM Extracted"
            else:
                resolved_ram = 6.0
                ram_source = "Default Standard (6 GB)"

    # ── Storage Resolution ── (sanitized against standard smartphone capacities)
    if request.storage_gb and float(request.storage_gb) >= 16.0:
        raw_storage = float(request.storage_gb)
        resolved_storage = min(STANDARD_STORAGE_OPTIONS, key=lambda s: abs(s - raw_storage))
        storage_source = f"Standard Tier ({resolved_storage:.0f} GB)"
    else:
        resolved_storage = 128.0
        storage_source = "Default (128 GB)"

    # ── Warranty Resolution ──
    resolved_warranty = float(request.warranty_days) if request.warranty_days is not None and request.warranty_days >= 0 else 0.0

    # ── Battery Health Resolution ── (iPhone: sanitize [50, 100], Android: nan)
    if phone_type == "iphone" and request.battery_health_percent is not None and 50 <= float(request.battery_health_percent) <= 100:
        resolved_battery = float(request.battery_health_percent)
    else:
        resolved_battery = np.nan

    # ── Derived Features ── (100% computed from canonical model, no client overrides)
    row_dict = {"brand": clean_brand, "model": clean_model}
    resolved_model_tier = int(compute_model_tier(row_dict))
    resolved_brand_tier = int(compute_brand_tier(clean_brand))
    resolved_phone_age = float(compute_phone_age(row_dict))
    resolved_is_flagship = int(compute_is_flagship(row_dict))
    tiers_source = "Auto-Computed [feature_engineering.py]"

    # ── Detailed Terminal Logging ─────────
    caps_source = "Client Input" if (request.has_5g is not None and request.has_esim is not None) else "Auto-Enriched [phone_specs.py]"
    tiers_source = "Client Input" if (request.model_tier is not None and request.brand_tier is not None) else "Auto-Computed [feature_engineering.py]"

    print("\n" + "=" * 70)
    print("[MOBILE PREDICTION REQUEST RECEIVED]")
    print("-" * 70)
    print(">> EXTRACTION PIPELINE:")
    print(f"   * Information Source:    {extraction_source}")
    if llm_extracted:
        print(f"   * LLM Extracted Model:   '{llm_extracted.get('model')}' ({llm_extracted.get('brand')})")
        if llm_extracted.get("is_installment_trap"):
            print("   * [WARNING] Installment / Lease down-payment trap detected by LLM!")

    print("\n>> INCOMING EXTENSION / CLIENT PAYLOAD:")
    print(f"   * Brand:                 '{request.brand}'")
    print(f"   * Raw Model:             '{request.model}'")
    print(f"   * Storage (GB):          {request.storage_gb} ({'Explicit' if request.storage_gb else 'Missing / Auto-fill'})")
    print(f"   * RAM (GB):              {request.ram_gb} ({'Explicit' if request.ram_gb else 'Missing / Auto-fill'})")
    print(f"   * Battery Health:        {request.battery_health_percent}%" if request.battery_health_percent is not None else "   * Battery Health:        None (Missing / Unspecified)")
    print(f"   * Warranty:              {request.warranty_days} days" if request.warranty_days is not None else "   * Warranty:              0 days (Default / Unspecified)")
    print(f"   * 5G / eSIM / Dual SIM:  5G={request.has_5g}, eSIM={request.has_esim}, DualSIM={request.dual_sim}")

    print("\n>> BACKEND SINGLE SOURCE OF TRUTH (SSOT) RESOLUTION:")
    print(f"   -> Matched Catalog Model: '{clean_model}' ({clean_brand})")
    print(f"   -> Model Architecture:    {phone_type.upper()} ({'CatBoost' if phone_type == 'iphone' else 'XGBoost'} Pipeline)")
    print(f"   -> Resolved RAM:          {resolved_ram} GB  <-- {ram_source}")
    print(f"   -> Resolved Storage:      {resolved_storage} GB  <-- {storage_source}")
    print(f"   -> Hardware 5G:           {bool(resolved_5g)}  <-- {caps_source}")
    print(f"   -> Hardware eSIM:         {bool(resolved_esim)}  <-- {caps_source}")
    print(f"   -> Hardware Dual SIM:     {bool(resolved_dual_sim)}  <-- {caps_source}")
    print(f"   -> Model Tier:            Tier {resolved_model_tier}/10  <-- {tiers_source}")
    print(f"   -> Brand Tier:            Tier {resolved_brand_tier}/3  <-- {tiers_source}")
    print(f"   -> Phone Age:             {resolved_phone_age} years  <-- Auto-Computed [phone_specs.py]")
    print(f"   -> Is Flagship:           {bool(resolved_is_flagship)}  <-- {tiers_source}")

    # 6. Build input DataFrame matching FEATURE_COLUMNS
    input_data = {
        "brand": clean_brand,
        "model": clean_model,
        "storage_gb": resolved_storage,
        "ram_gb": resolved_ram,
        "warranty_days": resolved_warranty,
        "battery_health_percent": resolved_battery,
        "dual_sim": resolved_dual_sim,
        "has_5g": resolved_5g,
        "has_esim": resolved_esim,
        "model_tier": resolved_model_tier,
        "brand_tier": resolved_brand_tier,
        "phone_age_years": resolved_phone_age,
        "is_flagship": resolved_is_flagship,
    }
    
    df = pd.DataFrame([input_data])
    
    try:
        predicted_price = float(model_pipeline.predict(df)[0])
        predicted_price = max(0.0, predicted_price)
        
        # Compute fair market range using the model's MAE
        _, _, metrics = get_recommended_model_info(phone_type, evaluation_data)
        mae = float(metrics.get("mae", predicted_price * 0.1))
        
        # Sanitize input data to ensure JSON compliance (replace NaN with None)
        json_safe_inputs = {
            k: (None if isinstance(v, float) and np.isnan(v) else v)
            for k, v in input_data.items()
        }

        print("\n>> VALUATION RESULT:")
        print(f"   * Predicted Fair Price:  Rs. {predicted_price:,.2f}")
        print(f"   * Fair Market Range:     Rs. {max(0.0, predicted_price - mae):,.2f} - Rs. {predicted_price + mae:,.2f}")
        print("=" * 70 + "\n")

        return {
            "predicted_price": predicted_price,
            "fair_market_range": {
                "lower_price_lkr": max(0.0, predicted_price - mae),
                "upper_price_lkr": predicted_price + mae
            },
            "phone_type": phone_type,
            "matched_model": clean_model,
            "inputs": json_safe_inputs,
            "enriched_specs": {
                "brand": clean_brand,
                "model": clean_model,
                "storage_gb": resolved_storage,
                "ram_gb": resolved_ram,
                "warranty_days": resolved_warranty,
                "battery_health_percent": None if np.isnan(resolved_battery) else resolved_battery,
                "dual_sim": bool(resolved_dual_sim),
                "has_5g": bool(resolved_5g),
                "has_esim": bool(resolved_esim),
                "model_tier": resolved_model_tier,
                "brand_tier": resolved_brand_tier,
                "phone_age_years": resolved_phone_age,
                "is_flagship": bool(resolved_is_flagship)
            }
        }
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8002, reload=True)