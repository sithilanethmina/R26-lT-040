"""
Multi-Brand Van Price Prediction — Training Script
===================================================
Workspace : Model_training/
Script    : Model_training/Van_Price_Prediction/train_van_model.py
Dataset   : Model_training/Cleaned_Van_Brands/

Key differences from the SUV model (train_suv_model.py):
  1. NEW FEATURE — engine_code (Sri Lanka Toyota Dolphin market convention)
     For the Toyota Dolphin specifically, the SAME ~2.8L-class diesel block
     was sold under two different factory codes, and the local market
     prices them differently even at near-identical cc:
        "3L"  ≈ 2600–2850 cc  → cheaper   (older/base block)
        "5L"  ≈ 2850–3150 cc  → pricier   (newer/stronger block)
        "2L"  ≈ 2300–2600 cc  → older/smaller diesel code
     engine_cc alone (raw number) does NOT capture this — a 2800cc "3L" and
     a 2980cc "5L" are only 180cc apart numerically but the market treats
     them as different tiers. So we derive `engine_code` as its own
     categorical feature — Toyota + Dolphin + Diesel ONLY. Every other
     model (Hiace, Townace, KDH, Noah, LH-series...) and every other
     brand gets "NA", since this specific market convention was confirmed
     for Dolphin only and should not be assumed to generalize to other
     Toyota van models.

  2. engine_cc is EXTREMELY noisy in this dataset (scraped listings)
     Raw values include garbage like 1, 2, 3, 5, 56, 160, 12345, 28000,
     29000, 555888 cc — clearly typos/extraction errors, not real engines.
     Valid van engine range: 550cc (kei vans: Every/Hijet/Minicab) to
     3500cc (largest diesel Hiace/Caravan/Delica). Anything outside that
     is set to NaN and imputed, same pattern as the SUV script.

  3. No brand exclusion by record count
     Every brand in the folder (Toyota, Nissan, Suzuki, Mitsubishi, Mazda,
     Daihatsu, Isuzu) has ≥55 records. The van dataset is much smaller than
     the SUV dataset (~4.1K vs tens of thousands), so we keep all 7 brands
     rather than dropping any — dropping Isuzu (55) or Mazda (191) would
     cost more in coverage than it gains in per-brand reliability.

  4. Price ceiling is 90M LKR (vs 200M for SUVs, 100M for cars)
     Van listings top out far lower than SUVs — no van in this dataset
     exceeds ~83M LKR.

  5. Price floor is 500K LKR
     Below that, listings in this dataset are consistently garbage/
     placeholder prices (11,111 / 12,345 / 123,456 / "Unknown" models),
     not real cheap vans.

  6. `condition` and `vehicle_type` columns dropped
     Both are constant across every record ("Registered (Used)" / "Van")
     in this dataset — zero predictive signal, just noise for the encoder.

  7. Transmission typo fix uses the column MODE, not a hardcoded value
     A few rows have "1950" as the transmission (data entry error bleeding
     in from a year field). Instead of hardcoding a replacement, we use
     whichever of Manual/Automatic is more common in the cleaned data.

  8. Monotone constraints
     model_year: +1 (newer → pricier)
     mileage_km: -1 (more km → cheaper)
     vehicle_age: -1 (older → cheaper)
     engine_cc:   +1 (bigger engine → pricier)
     engine_code: 0  (categorical — no forced direction; XGBoost learns
                      3L vs 5L vs 2L pricing from the data itself)

Models trained (4):
  - CatBoost         (handles categoricals natively)
  - XGBRegressor     (production model — monotone constraints)
  - Gradient Boosting
  - Random Forest

Production model: XGBRegressor
  Reason: Monotone constraints guarantee:
    newer year     → always higher price
    more mileage   → always lower price
    older age      → always lower price
    bigger engine  → always higher price
"""

import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import joblib

warnings.filterwarnings("ignore")

# ─── Path Setup ───────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR.parent / "Cleaned_Van_Brands"
MODEL_DIR  = BASE_DIR / "models" / "van"
OUTPUT_DIR = BASE_DIR / "outputs" / "van"

RANDOM_STATE = 42

# ─── Brand Config ─────────────────────────────────────────────────────────────
# All brands present in the folder are trainable — smallest (Isuzu) still
# has 55 records, and the overall dataset is small enough that we keep
# everything rather than dropping brands like the SUV script does.
TRAINABLE_BRANDS = [
    "Toyota", "Nissan", "Suzuki", "Mitsubishi",
    "Mazda", "Daihatsu", "Isuzu",
]

# ─── Data Bounds ──────────────────────────────────────────────────────────────
PRICE_MIN         = 500_000      # below this = garbage/placeholder listings
PRICE_MAX         = 90_000_000   # covers the most expensive vans in-market
YEAR_MIN          = 1980
YEAR_MAX          = 2026         # 2025/2026 kept — unregistered/new stock
MILEAGE_MAX       = 1_000_000
ENGINE_CC_MIN     = 550          # smallest van engine (kei vans: Every/Hijet)
ENGINE_CC_MAX     = 3_500        # largest (Hiace/Caravan/Delica diesel)
MIN_COMBO_RECORDS = 2            # min records per brand+model+year

# Toyota diesel engine-code bands (Sri Lanka market convention — see
# module docstring point 1). Boundaries chosen from the actual cc
# clustering in the data (peaks around 2770cc and 3000cc).
ENGINE_CODE_BANDS = [
    # (fuel_type restriction, cc_low, cc_high, code_label)
    (2300, 2600, "2L"),
    (2600, 2850, "3L"),
    (2850, 3150, "5L"),
]

# ─── Features ─────────────────────────────────────────────────────────────────
# engine_code is van-specific: captures the "same displacement class, two
# different factory codes, two different price tiers" effect that raw
# engine_cc misses.
CATEGORICAL_FEATURES = ["brand", "model", "variant", "fuel_type",
                         "transmission", "engine_code"]
NUMERIC_FEATURES     = ["model_year", "mileage_km", "vehicle_age", "engine_cc"]
ALL_FEATURES         = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET               = "price"

# Monotone constraints — order must match ALL_FEATURES exactly:
# brand=0, model=0, variant=0, fuel_type=0, transmission=0, engine_code=0,
# model_year=+1, mileage_km=-1, vehicle_age=-1, engine_cc=+1
MONOTONE_CONSTRAINTS = (0, 0, 0, 0, 0, 0, 1, -1, -1, 1)


# ══════════════════════════════════════════════════════════════
#  NLP SCORING LAYER — Sri Lanka Van Market Keywords
# ══════════════════════════════════════════════════════════════

NLP_POSITIVE = {
    "one_owner": {
        "keywords": ["one owner", "1 owner", "1st owner", "single owner",
                     "first owner", "lady owner", "lady driven"],
        "points": 5, "label": "Single Owner",
    },
    "accident_free": {
        "keywords": ["accident free", "no accident", "no accidents",
                     "no crash", "mint condition", "never been in"],
        "points": 5, "label": "Accident Free",
    },
    "low_mileage": {
        "keywords": ["low mileage", "low km", "low kilo", "less mileage",
                     "very low mileage"],
        "points": 4, "label": "Low Mileage Stated",
    },
    "service_records": {
        "keywords": ["service records", "service history", "full service",
                     "maintained", "company maintained", "agent maintained",
                     "dealer maintained"],
        "points": 4, "label": "Service Records",
    },
    "full_option": {
        "keywords": ["full option", "fully loaded", "full options",
                     "all options", "top spec", "fully optioned"],
        "points": 3, "label": "Full Option",
    },
    "original_condition": {
        "keywords": ["original paint", "original condition",
                     "factory paint", "unmodified", "genuine"],
        "points": 3, "label": "Original Condition",
    },
    "good_engine": {
        "keywords": ["engine in good condition", "smooth engine",
                     "no smoke", "no leaks", "well maintained engine"],
        "points": 3, "label": "Engine Confirmed Good",
    },
    "new_unregistered": {
        "keywords": ["brand new", "unregistered", "zero km", "0 km",
                     "showroom", "recondition", "reconditioned"],
        "points": 2, "label": "New/Unregistered",
    },
    "high_roof": {
        "keywords": ["high roof", "super long", "super gl", "grand cabin"],
        "points": 2, "label": "High-Roof/Long Variant",
    },
}

NLP_NEGATIVE = {
    "urgent_sale": {
        "keywords": ["urgent", "urgent sale", "quick sale",
                     "must sell", "need to sell", "asap"],
        "points": -4, "label": "Urgent Sale (Suspicious)",
    },
    "accident_damage": {
        "keywords": ["accident", "collision damage", "front damage",
                     "rear damage", "accident repaired", "had an accident"],
        "points": -8, "label": "Accident History",
    },
    "engine_issues": {
        "keywords": ["engine issue", "engine problem", "engine repair",
                     "gearbox issue", "gearbox problem", "needs repair",
                     "not working", "transmission problem",
                     "smoke issue", "oil leak"],
        "points": -10, "label": "Engine/Mechanical Issues",
    },
    "reconstructed": {
        "keywords": ["reconstructed", "re-con", "salvage", "written off"],
        "points": -8, "label": "Reconstructed/Salvage",
    },
    "body_damage": {
        "keywords": ["body damage", "panel damage", "dents",
                     "rust", "scratches", "bumper damage"],
        "points": -5, "label": "Body/Cosmetic Damage",
    },
    "high_usage_flag": {
        "keywords": ["high mileage", "used heavily", "fleet vehicle",
                     "school van", "hire van", "rental", "lease"],
        "points": -5, "label": "High Usage/Fleet/Lease",
    },
}


def extract_nlp_signals(text: str) -> dict:
    if not text or pd.isna(text):
        return {"signals": [], "nlp_score": 0}
    text_lower = str(text).lower()
    detected, total_pts = [], 0
    for key, cfg in NLP_POSITIVE.items():
        for kw in cfg["keywords"]:
            if kw in text_lower:
                detected.append({"type": "positive", "key": key,
                                  "label": cfg["label"], "points": cfg["points"]})
                total_pts += cfg["points"]
                break
    for key, cfg in NLP_NEGATIVE.items():
        for kw in cfg["keywords"]:
            if kw in text_lower:
                detected.append({"type": "negative", "key": key,
                                  "label": cfg["label"], "points": cfg["points"]})
                total_pts += cfg["points"]
                break
    total_pts = max(-20, min(30, total_pts))
    return {"signals": detected, "nlp_score": total_pts}


def compute_base_score(listing_price: float, predicted_price: float) -> int:
    if predicted_price <= 0:
        return 35
    dev = ((listing_price - predicted_price) / predicted_price) * 100
    if   dev <= -30: return 25
    elif dev <= -15: return 55
    elif dev <=  -5: return 65
    elif dev <=   5: return 70   # fairly priced
    elif dev <=  15: return 50
    elif dev <=  30: return 30
    else:            return 10


def compute_fair_score(listing_price, predicted_price, nlp_score) -> dict:
    base  = compute_base_score(listing_price, predicted_price)
    final = min(100, max(0, base + nlp_score))
    dev   = ((listing_price - predicted_price) / predicted_price) * 100
    if   final >= 65:  label = "Fairly Priced ✅"
    elif final >= 45:  label = "Review Carefully ⚠️"
    elif dev   < -25:  label = "Suspiciously Underpriced 🔵"
    else:              label = "Overpriced ❌"
    return {"base_score": base, "nlp_score": nlp_score,
            "final_score": final, "label": label,
            "deviation_pct": round(dev, 2)}


def get_confidence(record_count: int) -> str:
    if   record_count >= 50: return "High"
    elif record_count >= 15: return "Medium"
    else:                    return "Low — insufficient van market data"


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def ensure_dirs():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred) if len(y_true) >= 2 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2_Score": r2}


def map_engine_code(row) -> str:
    """
    Sri Lanka Toyota Dolphin market convention:
    the ~2.8L-class diesel block was sold under different factory codes
    (2L / 3L / 5L) at slightly different displacements, and the market
    prices them as distinct tiers rather than as one continuous cc curve.
    Confirmed against this specific user's domain knowledge and applies
    ONLY to Toyota Dolphin + Diesel — other Toyota van models (Hiace,
    Townace, KDH, Noah, etc.) are NOT covered by this convention and get
    "NA", same as every other brand/model.
    """
    if row.get("brand") != "Toyota" or row.get("fuel_type") != "Diesel":
        return "NA"
    if str(row.get("model", "")).strip().lower() != "dolphin":
        return "NA"
    cc = row.get("engine_cc")
    if pd.isna(cc):
        return "NA"
    for low, high, label in ENGINE_CODE_BANDS:
        if low <= cc <= high:
            return label
    return "Other"


# ══════════════════════════════════════════════════════════════
#  YEAR-WISE MARKET SUMMARY
# ══════════════════════════════════════════════════════════════

def generate_yearwise_summary(df: pd.DataFrame, output_path: Path):
    summary = df.groupby(
        ["brand", "model", "variant", "fuel_type", "model_year"]
    )["price"].agg(
        mean="mean", median="median",
        min="min", max="max",
        observed_count="count"
    ).reset_index()
    for col in ["mean", "median", "min", "max"]:
        summary[col] = summary[col].round(0).astype(int)
    summary = summary.sort_values(
        ["brand", "model", "variant", "fuel_type", "model_year"]
    )
    summary.to_csv(output_path, index=False)
    print(f"  Saved year-wise van market summary → {output_path.name}")
    return summary


# ══════════════════════════════════════════════════════════════
#  DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════════

def load_and_clean_data() -> pd.DataFrame:
    print(f"Loading Van datasets from: {DATA_DIR} ...")

    if not DATA_DIR.exists():
        print(f"ERROR: Data directory not found: {DATA_DIR}")
        sys.exit(1)

    dfs = []
    for brand in TRAINABLE_BRANDS:
        fpath = DATA_DIR / f"{brand}.json"
        if not fpath.exists():
            print(f"  [SKIP] {brand}.json not found")
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            d = json.load(f)
        df = pd.DataFrame(d)
        df["brand"] = brand
        dfs.append(df)
        print(f"  Loaded  {brand:<18} {len(df):>5} records")

    if not dfs:
        print("ERROR: No data files loaded.")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total raw records: {len(combined):,}")

    # Drop constant, non-predictive columns if present (condition,
    # vehicle_type are always the same value in this dataset)
    for col in ["condition", "vehicle_type"]:
        if col in combined.columns:
            combined = combined.drop(columns=[col])

    # ── Step 1: Numeric conversion ──
    for col in ["price", "model_year", "mileage_km", "engine_cc"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined = combined.dropna(subset=["price", "model_year"]).copy()

    # ── Step 2: Price filter (500K – 90M LKR) ──
    before = len(combined)
    combined = combined[
        (combined["price"] >= PRICE_MIN) &
        (combined["price"] <= PRICE_MAX)
    ].copy()
    print(f"  Price filter removed     : {before - len(combined):,} records")

    # ── Step 3: Year filter (1980 – 2026) ──
    # 2025/2026 KEPT — unregistered/new stock, not data errors
    before = len(combined)
    combined = combined[
        (combined["model_year"] >= YEAR_MIN) &
        (combined["model_year"] <= YEAR_MAX)
    ].copy()
    combined["model_year"] = combined["model_year"].astype(int)
    print(f"  Year filter removed      : {before - len(combined):,} records")

    # ── Step 4: Mileage filter ──
    before = len(combined)
    combined = combined[
        combined["mileage_km"].isna() |
        (combined["mileage_km"] <= MILEAGE_MAX)
    ].copy()
    # Zero mileage → NaN (unregistered — will be imputed)
    combined.loc[combined["mileage_km"] == 0, "mileage_km"] = np.nan
    print(f"  Mileage outliers removed : {before - len(combined):,} records")

    # ── Step 5: Fix bad transmission ──
    # A few rows have "1950" (a year value that leaked into this field).
    # Replace with whichever of Manual/Automatic is more common overall,
    # rather than hardcoding a guess.
    combined["transmission"] = combined["transmission"].astype(str).str.strip()
    valid_mask = combined["transmission"].isin(["Automatic", "Manual"])
    fallback_transmission = (
        combined.loc[valid_mask, "transmission"].mode().iloc[0]
        if valid_mask.any() else "Manual"
    )
    combined.loc[~valid_mask, "transmission"] = fallback_transmission
    print(f"  Bad transmission fixed   : {(~valid_mask).sum()} records "
          f"→ '{fallback_transmission}' (column mode)")

    # ── Step 6: Fuel type cleanup ──
    combined["fuel_type"] = combined["fuel_type"].fillna("Diesel").str.strip()
    # Consolidate rare label variants
    combined["fuel_type"] = combined["fuel_type"].replace({
        "Kick / Electric": "Electric",
    })

    # ── Step 7: Engine CC cleaning ──
    # Van engine_cc is extremely noisy (scraped listings) — values like
    # 1, 2, 3, 5, 56, 160, 12345, 28000, 555888 are typos/extraction bugs,
    # not real engines. Valid range: ENGINE_CC_MIN to ENGINE_CC_MAX.
    invalid_cc = (
        combined["engine_cc"].notna() &
        ~combined["engine_cc"].between(ENGINE_CC_MIN, ENGINE_CC_MAX)
    )
    combined.loc[invalid_cc, "engine_cc"] = np.nan
    print(f"  Invalid engine_cc set NaN: {invalid_cc.sum()} records")

    # ── Step 8: Engine CC imputation ──
    # Impute with brand+model median (more accurate than year median)
    model_cc_median  = combined.groupby(["brand", "model"])["engine_cc"].median()
    brand_cc_median  = combined.groupby("brand")["engine_cc"].median()
    overall_cc_med   = combined["engine_cc"].median()
    missing_cc_count = combined["engine_cc"].isna().sum()

    def impute_cc(row):
        if pd.notna(row["engine_cc"]):
            return row["engine_cc"]
        key = (row["brand"], row["model"])
        if key in model_cc_median and pd.notna(model_cc_median[key]):
            return model_cc_median[key]
        if row["brand"] in brand_cc_median and pd.notna(brand_cc_median[row["brand"]]):
            return brand_cc_median[row["brand"]]
        return overall_cc_med

    combined["engine_cc"] = combined.apply(impute_cc, axis=1)
    print(f"  Engine CC imputed        : {missing_cc_count:,} records (model median)")

    # ── Step 9: Mileage imputation ──
    # Use brand+model+year median for vans (more specific than year-only)
    missing_mil = combined["mileage_km"].isna().sum()
    model_yr_mil = combined.groupby(
        ["brand", "model", "model_year"]
    )["mileage_km"].median()
    yr_mil_med   = combined.groupby("model_year")["mileage_km"].median()
    overall_mil  = combined["mileage_km"].median()

    def impute_mileage(row):
        if pd.notna(row["mileage_km"]):
            return row["mileage_km"]
        key3 = (row["brand"], row["model"], row["model_year"])
        if key3 in model_yr_mil and pd.notna(model_yr_mil[key3]):
            return model_yr_mil[key3]
        return yr_mil_med.get(row["model_year"], overall_mil)

    combined["mileage_km"] = combined.apply(impute_mileage, axis=1)
    print(f"  Mileage imputed          : {missing_mil:,} records (model+year median)")

    # ── Step 10: Minimum combo filter ──
    # Remove brand+model+year combos with < MIN_COMBO_RECORDS
    # Prevents rare combos from teaching the model wrong patterns
    before = len(combined)
    combo_counts = combined.groupby(
        ["brand", "model", "model_year"]
    )["price"].transform("count")
    combined = combined[combo_counts >= MIN_COMBO_RECORDS].copy()
    print(f"  Min combo filter removed : {before - len(combined):,} "
          f"(combos with <{MIN_COMBO_RECORDS} records)")

    # ── Step 11: Categoricals ──
    combined["model"]   = combined["model"].fillna("Unknown").str.strip()
    combined["variant"] = (
        combined.get("variant", pd.Series("Standard", index=combined.index))
        .fillna("Standard").replace("", "Standard").str.strip()
    )

    # ── Step 12: engine_code — Toyota diesel 2L/3L/5L market convention ──
    combined["engine_code"] = combined.apply(map_engine_code, axis=1)

    # ── Step 13: Feature engineering ──
    combined["vehicle_age"] = (2026 - combined["model_year"]).clip(lower=1)

    # ── Step 14: NLP text ──
    combined["title_raw"] = combined.get(
        "title_raw", pd.Series("", index=combined.index)
    ).fillna("")

    # ── Step 15: Record count for confidence ──
    combined["_model_count"] = combined.groupby(
        ["brand", "model"]
    )["price"].transform("count")

    print(f"\n  Final clean records: {len(combined):,}")
    return combined


# ══════════════════════════════════════════════════════════════
#  DATA SUMMARY
# ══════════════════════════════════════════════════════════════

def print_data_summary(df: pd.DataFrame):
    print(f"\n{'='*70}")
    print("VAN DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"  Total records : {len(df):,}")
    print(f"  Brands        : {df['brand'].nunique()}")
    print(f"  Models        : {df['model'].nunique()}")
    print(f"  Year range    : {int(df['model_year'].min())} – "
          f"{int(df['model_year'].max())}")
    print(f"  Price range   : {df['price'].min()/1e6:.1f}M – "
          f"{df['price'].max()/1e6:.1f}M LKR")
    print(f"  Engine CC     : {int(df['engine_cc'].min())} – "
          f"{int(df['engine_cc'].max())} cc")

    print(f"\n  {'Brand':<20} {'Records':>7}  {'Mean':>8}  {'Median':>9}")
    print(f"  {'─'*50}")
    for brand, grp in df.groupby("brand"):
        print(f"  {brand:<20} {len(grp):>7}  "
              f"{grp['price'].mean()/1e6:>7.1f}M  "
              f"{grp['price'].median()/1e6:>8.1f}M")

    print(f"\n  Engine code price impact (why engine_code is a feature):")
    ec = df[df["engine_code"] != "NA"].groupby("engine_code")["price"].agg(
        ["count", "mean", "median"]
    )
    for code_name, r in ec.sort_values("mean", ascending=False).iterrows():
        print(f"    {code_name:<8} N={int(r['count']):>5}  "
              f"mean={r['mean']/1e6:>5.2f}M  median={r['median']/1e6:>5.2f}M")

    print(f"\n  Fuel type price impact:")
    ft = df.groupby("fuel_type")["price"].agg(["count", "mean", "median"])
    for ft_name, r in ft.sort_values("mean", ascending=False).iterrows():
        print(f"    {ft_name:<16} N={int(r['count']):>5}  "
              f"mean={r['mean']/1e6:>5.1f}M  median={r['median']/1e6:>5.1f}M")


# ══════════════════════════════════════════════════════════════
#  MODEL TRAINING
# ══════════════════════════════════════════════════════════════

def train_all_models(df: pd.DataFrame):
    print(f"\n{'='*70}")
    print("TRAINING — 4 MODEL PIPELINE")
    print(f"{'='*70}")
    print(f"  Features: {ALL_FEATURES}")
    print(f"  Target  : log1p(price)")
    print()

    X = df[ALL_FEATURES].copy()
    y = np.log1p(df[TARGET])

    # OrdinalEncoder for XGBoost / Gradient Boosting / Random Forest
    encoder   = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )
    X_encoded = X.copy()
    X_encoded[CATEGORICAL_FEATURES] = encoder.fit_transform(
        X[CATEGORICAL_FEATURES].astype(str)
    )

    # CatBoost uses original string categoricals
    cat_indices = [ALL_FEATURES.index(c) for c in CATEGORICAL_FEATURES]

    # Stratified split by brand
    try:
        (X_cat_tr, X_cat_te,
         X_enc_tr, X_enc_te,
         y_tr, y_te) = train_test_split(
            X, X_encoded, y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=df["brand"]
        )
    except ValueError:
        print("  Stratified split failed → normal split")
        (X_cat_tr, X_cat_te,
         X_enc_tr, X_enc_te,
         y_tr, y_te) = train_test_split(
            X, X_encoded, y,
            test_size=0.2,
            random_state=RANDOM_STATE
        )

    print(f"  Train: {len(X_cat_tr):,}   Test: {len(X_cat_te):,}")
    print()

    models = {
        "CatBoost": CatBoostRegressor(
            iterations=600,
            learning_rate=0.04,
            depth=7,
            l2_leaf_reg=3,
            loss_function="RMSE",
            cat_features=cat_indices,
            random_seed=RANDOM_STATE,
            verbose=False,
        ),
        # Production model — monotone constraints:
        # model_year: +1 (newer = pricier)
        # mileage_km: -1 (more km = cheaper)
        # vehicle_age: -1 (older = cheaper)
        # engine_cc: +1 (bigger engine = pricier)
        # engine_code: 0 (categorical — Dolphin 2L/3L/5L tiering learned freely)
        "XGBRegressor": XGBRegressor(
            n_estimators=400,
            learning_rate=0.04,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            verbosity=0,
            monotone_constraints=MONOTONE_CONSTRAINTS,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.04,
            max_depth=6,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=18,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results, trained = [], {}

    for name, model in models.items():
        print(f"  Training {name} ...")
        if name == "CatBoost":
            model.fit(X_cat_tr, y_tr)
            y_pred = model.predict(X_cat_te)
        else:
            model.fit(X_enc_tr, y_tr)
            y_pred = model.predict(X_enc_te)

        metrics = compute_metrics(
            np.expm1(y_te), np.expm1(y_pred)
        )
        results.append({"Model": name, **metrics})
        trained[name] = model
        print(f"    MAE: {metrics['MAE']/1e6:>6.3f}M  "
              f"RMSE: {metrics['RMSE']/1e6:>6.3f}M  "
              f"R²: {metrics['R2_Score']:.4f}")

    results_df = pd.DataFrame(results).sort_values("MAE")

    # Honest reporting
    best_by_mae   = results_df.iloc[0]["Model"]
    production    = "XGBRegressor"
    xgb_row       = results_df[results_df["Model"] == production].iloc[0]

    print(f"\n  Best by MAE         : {best_by_mae} "
          f"(MAE={results_df.iloc[0]['MAE']/1e6:.3f}M)")
    print(f"  Production model    : {production} "
          f"(MAE={xgb_row['MAE']/1e6:.3f}M, R²={xgb_row['R2_Score']:.4f})")
    print(f"  Reason              : Monotone constraints ensure logical "
          f"van pricing (year↑ price↑, mileage↑ price↓, engine_cc↑ price↑)")

    # Build prediction display
    prod_model      = trained[production]
    final_preds_log = prod_model.predict(X_enc_te)

    display_df = X_cat_te.copy()
    display_df["Actual_Price_LKR"]    = np.expm1(y_te).round(0).values
    display_df["Predicted_Price_LKR"] = np.expm1(final_preds_log).round(0)
    display_df["Difference_LKR"]      = (
        display_df["Predicted_Price_LKR"] -
        display_df["Actual_Price_LKR"]
    )
    display_df["Error_Margin_%"] = (
        abs(display_df["Difference_LKR"]) /
        display_df["Actual_Price_LKR"] * 100
    ).round(2)

    # Confidence level
    model_count_map = df.groupby(["brand", "model"])["price"].count().to_dict()
    display_df["Confidence"] = display_df.apply(
        lambda r: get_confidence(
            model_count_map.get((r["brand"], r["model"]), 0)
        ), axis=1
    )

    return (results_df, trained, encoder, prod_model, production,
            display_df, X_cat_te, X_enc_te, y_te,
            X_cat_tr, X_enc_tr, y_tr, cat_indices)


# ══════════════════════════════════════════════════════════════
#  PRINT TABLES
# ══════════════════════════════════════════════════════════════

def print_model_comparison(results_df, production):
    print(f"\n{'='*70}")
    print("OVERALL MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Model':<22} {'MAE (M)':>9}  {'RMSE (M)':>10}  {'R²':>8}  Note")
    print(f"  {'─'*65}")
    for _, r in results_df.iterrows():
        note = ""
        if r["Model"] == results_df.iloc[0]["Model"]:
            note = "← Best MAE"
        if r["Model"] == production:
            note = (note + "  ← Production").strip()
        print(f"  {r['Model']:<22} {r['MAE']/1e6:>8.3f}M  "
              f"{r['RMSE']/1e6:>9.3f}M  "
              f"{r['R2_Score']:>8.4f}  {note}")


def print_per_brand_table(display_df, production):
    print(f"\n{'─'*70}")
    print(f"PER-BRAND PERFORMANCE  ({production})")
    print(f"  {'Brand':<20} {'N':>5}  {'MAE (M)':>9}  {'R²':>8}  Confidence")
    print(f"  {'─'*60}")
    for brand in sorted(display_df["brand"].unique()):
        sub = display_df[display_df["brand"] == brand]
        if len(sub) < 2:
            continue
        mae = mean_absolute_error(
            sub["Actual_Price_LKR"], sub["Predicted_Price_LKR"]
        )
        r2 = r2_score(
            sub["Actual_Price_LKR"], sub["Predicted_Price_LKR"]
        )
        conf = sub["Confidence"].iloc[0]
        print(f"  {brand:<20} {len(sub):>5}  {mae/1e6:>8.3f}M  "
              f"{r2:>8.4f}  {conf}")


def print_sample_predictions(display_df):
    print(f"\n{'─'*95}")
    print("SAMPLE PREDICTIONS  (10 random van listings)")
    print(f"  {'Brand':<13} {'Model':<14} {'Year':>5} {'Fuel':<8} "
          f"{'Actual (M)':>10} {'Pred (M)':>10} {'Err%':>6}  Confidence")
    print(f"  {'─'*88}")
    sample = display_df.sample(min(10, len(display_df)), random_state=RANDOM_STATE)
    for _, r in sample.iterrows():
        flag = " ⚠️" if r["Error_Margin_%"] > 15 else ""
        print(f"  {str(r['brand']):<13} {str(r['model'])[:14]:<14} "
              f"{int(r['model_year']):>5} {str(r['fuel_type'])[:7]:<8} "
              f"{r['Actual_Price_LKR']/1e6:>9.2f}M "
              f"{r['Predicted_Price_LKR']/1e6:>9.2f}M "
              f"{r['Error_Margin_%']:>6.1f}%  {r['Confidence']}{flag}")
    print(f"  {'─'*88}")
    high_err = display_df[display_df["Error_Margin_%"] > 15]
    print(f"\n  Predictions >15% error : {len(high_err)} / {len(display_df)} "
          f"({len(high_err)/len(display_df)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════
#  CROSS VALIDATION
# ══════════════════════════════════════════════════════════════

def run_cross_validation(df):
    print(f"\n{'─'*70}")
    print("5-FOLD CROSS VALIDATION  (XGBRegressor — Production Model)")

    X = df[ALL_FEATURES].copy()
    y = np.log1p(df[TARGET])

    encoder   = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    )
    X_encoded = X.copy()
    X_encoded[CATEGORICAL_FEATURES] = encoder.fit_transform(
        X[CATEGORICAL_FEATURES].astype(str)
    )

    kf      = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rows = []

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X_encoded), 1):
        X_tr, X_te = X_encoded.iloc[tr_idx], X_encoded.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        m = XGBRegressor(
            n_estimators=400, learning_rate=0.04, max_depth=7,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=3, gamma=0.1,
            objective="reg:squarederror",
            random_state=RANDOM_STATE, verbosity=0,
            monotone_constraints=MONOTONE_CONSTRAINTS,
        )
        m.fit(X_tr, y_tr)
        metrics = compute_metrics(
            np.expm1(y_te), np.expm1(m.predict(X_te))
        )
        cv_rows.append({"Fold": fold, **metrics})
        print(f"  Fold {fold}: MAE={metrics['MAE']/1e6:>6.3f}M  "
              f"RMSE={metrics['RMSE']/1e6:>6.3f}M  "
              f"R²={metrics['R2_Score']:.4f}")

    cv_df = pd.DataFrame(cv_rows)
    print(f"\n  CV Mean MAE  : {cv_df['MAE'].mean()/1e6:>8.3f}M  "
          f"(±{cv_df['MAE'].std()/1e6:.3f}M)")
    print(f"  CV Mean RMSE : {cv_df['RMSE'].mean()/1e6:>8.3f}M  "
          f"(±{cv_df['RMSE'].std()/1e6:.3f}M)")
    print(f"  CV Mean R²   : {cv_df['R2_Score'].mean():>8.4f}  "
          f"(±{cv_df['R2_Score'].std():.4f})")
    return cv_df


# ══════════════════════════════════════════════════════════════
#  NLP DEMO
# ══════════════════════════════════════════════════════════════

def demonstrate_nlp(df, prod_model, encoder):
    print(f"\n{'='*70}")
    print("NLP SCORING LAYER — Van Demonstration")
    print(f"{'='*70}")

    sample = df[df["title_raw"].str.len() > 5].sample(
        min(5, len(df)), random_state=RANDOM_STATE
    )
    for _, row in sample.iterrows():
        inp     = pd.DataFrame([{f: row[f] for f in ALL_FEATURES}])
        inp_enc = inp.copy()
        inp_enc[CATEGORICAL_FEATURES] = encoder.transform(
            inp[CATEGORICAL_FEATURES].astype(str)
        )
        predicted     = float(np.expm1(prod_model.predict(inp_enc)[0]))
        listing_price = float(row["price"])
        nlp_result    = extract_nlp_signals(row["title_raw"])
        fair_result   = compute_fair_score(
            listing_price, predicted, nlp_result["nlp_score"]
        )
        print(f"\n  {row['brand']} {row['model']} "
              f"{int(row['model_year'])} {row['fuel_type']} "
              f"({int(row['engine_cc'])}cc, {row['engine_code']})")
        print(f"  Title       : {str(row['title_raw'])[:70]}")
        print(f"  Listing     : Rs. {listing_price/1e6:.2f}M")
        print(f"  Predicted   : Rs. {predicted/1e6:.2f}M")
        print(f"  Deviation   : {fair_result['deviation_pct']:>+.1f}%")
        print(f"  Base Score  : {fair_result['base_score']:>3} / 70")
        for sig in nlp_result["signals"]:
            sign = "+" if sig["points"] > 0 else ""
            print(f"  NLP         : {sig['label']:<32} {sign}{sig['points']} pts")
        print(f"  FINAL SCORE : {fair_result['final_score']:>3} / 100  "
              f"→  {fair_result['label']}")


# ══════════════════════════════════════════════════════════════
#  SAVE ARTIFACTS
# ══════════════════════════════════════════════════════════════

def save_artifacts(results_df, trained, encoder, prod_model,
                   display_df, cv_df, df_clean):
    ensure_dirs()

    # All 4 models
    for name, model in trained.items():
        fname = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, MODEL_DIR / fname)

    # Production model + encoder
    joblib.dump(prod_model, MODEL_DIR / "best_van_model.pkl")
    joblib.dump(encoder,    MODEL_DIR / "van_ordinal_encoder.pkl")

    # CSVs
    results_df.to_csv(OUTPUT_DIR / "van_model_comparison.csv",        index=False)
    display_df.to_csv(OUTPUT_DIR / "van_test_predictions.csv",         index=False)
    cv_df.to_csv(     OUTPUT_DIR / "van_cross_validation_results.csv", index=False)

    # NLP config for API
    nlp_config = {
        "vehicle_type": "Van",
        "positive_signals": {
            k: {"keywords": v["keywords"],
                "points"  : v["points"],
                "label"   : v["label"]}
            for k, v in NLP_POSITIVE.items()
        },
        "negative_signals": {
            k: {"keywords": v["keywords"],
                "points"  : v["points"],
                "label"   : v["label"]}
            for k, v in NLP_NEGATIVE.items()
        },
        "scoring": {
            "base_max"         : 70,
            "nlp_max"          : 30,
            "final_max"        : 100,
            "fairly_priced_min": 65,
            "review_min"       : 45,
        },
        "engine_code_bands": {
            label: {"cc_low": low, "cc_high": high}
            for low, high, label in ENGINE_CODE_BANDS
        },
    }
    with open(OUTPUT_DIR / "van_nlp_config.json", "w") as f:
        json.dump(nlp_config, f, indent=2)

    # Brand-model lookup (for browser extension dropdowns)
    lookup = df_clean.groupby(["brand", "model"]).agg(
        record_count=("price", "count"),
        mean_price=("price", "mean"),
        min_year=("model_year", "min"),
        max_year=("model_year", "max"),
        dominant_fuel=("fuel_type", lambda x: x.mode()[0]),
        mean_engine_cc=("engine_cc", "mean"),
    ).reset_index()
    lookup["confidence"]     = lookup["record_count"].apply(get_confidence)
    lookup["mean_price"]     = lookup["mean_price"].round(0).astype(int)
    lookup["mean_engine_cc"] = lookup["mean_engine_cc"].round(0).astype(int)
    lookup = lookup.sort_values(
        ["brand", "record_count"], ascending=[True, False]
    )
    lookup.to_csv(OUTPUT_DIR / "van_brand_model_lookup.csv", index=False)

    # Year-wise market summary
    generate_yearwise_summary(
        df_clean, OUTPUT_DIR / "van_yearwise_summary.csv"
    )

    print(f"\n  models/van/best_van_model.pkl")
    print(f"  models/van/van_ordinal_encoder.pkl")
    print(f"  models/van/[catboost/xgbregressor/gradient_boosting/random_forest].pkl")
    print(f"  outputs/van/van_model_comparison.csv")
    print(f"  outputs/van/van_test_predictions.csv")
    print(f"  outputs/van/van_cross_validation_results.csv")
    print(f"  outputs/van/van_nlp_config.json")
    print(f"  outputs/van/van_brand_model_lookup.csv")
    print(f"  outputs/van/van_yearwise_summary.csv")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    ensure_dirs()
    df = load_and_clean_data()
    print_data_summary(df)

    (results_df, trained, encoder, prod_model, production,
     display_df, X_cat_te, X_enc_te, y_te,
     X_cat_tr, X_enc_tr, y_tr,
     cat_indices) = train_all_models(df)

    print_model_comparison(results_df, production)
    print_per_brand_table(display_df, production)
    print_sample_predictions(display_df)

    cv_df = run_cross_validation(df)

    demonstrate_nlp(df, prod_model, encoder)

    print(f"\n{'='*70}")
    print("SAVING ALL ARTIFACTS")
    print(f"{'='*70}")
    save_artifacts(results_df, trained, encoder, prod_model,
                   display_df, cv_df, df)

    xgb_row = results_df[results_df["Model"] == "XGBRegressor"].iloc[0]
    print(f"\n{'='*70}")
    print("VAN TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Models trained  : 4 (CatBoost, XGBoost, Gradient Boosting, "
          f"Random Forest)")
    print(f"  Production model: XGBRegressor  "
          f"MAE={xgb_row['MAE']/1e6:.3f}M  R²={xgb_row['R2_Score']:.4f}")
    print(f"  CV Mean R²      : {cv_df['R2_Score'].mean():.4f}")
    print(f"  Total records   : {len(df):,} across "
          f"{df['brand'].nunique()} van brands")


if __name__ == "__main__":
    main()