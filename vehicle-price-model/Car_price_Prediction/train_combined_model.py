"""
Multi-Brand Vehicle Price Prediction — Combined Model
======================================================
Workspace : Model_training/
Script    : Model_training/Car_price_Prediction/train_combined_model.py
Dataset   : Model_training/Cleaned_Car_Brands/

Changes from previous version:
  1. Added Gradient Boosting (now 4 models: CatBoost, XGBoost, GB, RF)
  2. Minimum 3 records per brand+model+year combo filter
     → Removes noise from rare combos (Daihatsu Passo 2017 = 1 record)
  3. Confidence level added to predictions (High/Medium/Low)
  4. engine_cc dropped from features (regex extraction unreliable)
  5. Honest model reporting — best by MAE reported separately
     from XGBoost selected for production (monotone constraints)
  6. Cross validation added (5-fold)
  7. NLP keyword scoring layer added with Fatal Override logic
"""

import os
import sys
import json
import warnings
import re
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
DATA_DIR   = BASE_DIR.parent / "Cleaned_Car_Brands"
MODEL_DIR  = BASE_DIR / "models" / "combined"
OUTPUT_DIR = BASE_DIR / "outputs" / "combined"

RANDOM_STATE = 42

# ─── Config ───────────────────────────────────────────────────────────────────
TRAINABLE_BRANDS = [
    "Toyota", "Suzuki", "Nissan", "Honda", "Mitsubishi",
    "Daihatsu", "Micro", "Mazda", "Mercedes-Benz", "Perodua",
    "Hyundai", "BMW", "Audi", "Tata", "Kia", "Peugeot",
    "Ford", "Renault",
]

PRICE_MIN        = 500_000
PRICE_MAX        = 100_000_000
YEAR_MIN         = 1990
YEAR_MAX         = 2026
MILEAGE_MAX      = 500_000
MIN_COMBO_RECORDS = 3   # minimum records per brand+model+year combo

# engine_cc removed — regex extraction from title is unreliable
# 90%+ records get -1 (unknown), adding noise not signal
CATEGORICAL_FEATURES = ["brand", "model", "variant", "fuel_type", "transmission"]
NUMERIC_FEATURES     = ["model_year", "mileage_km", "vehicle_age"]
ALL_FEATURES         = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET               = "price"

# Monotone constraints for XGBoost
# Order must match ALL_FEATURES exactly:
# brand=0, model=0, variant=0, fuel=0, trans=0, year=+1, mileage=-1, age=-1
MONOTONE_CONSTRAINTS = (0, 0, 0, 0, 0, 1, -1, -1)


# ══════════════════════════════════════════════════════════════
#  NLP SCORING LAYER — Sri Lanka Car Market Keywords
# ══════════════════════════════════════════════════════════════

NLP_POSITIVE = {
    "verified_ownership": {
        "keywords": [
            "one owner", "1 owner", "1st owner", "single owner", "first owner",
            "පළමු අයිතිකරු", "එකම අයිතිකරු"
        ],
        "points": 5,
        "label": "Single Owner Verified",
    },
    "clear_original_paperwork": {
        "keywords": [
            "original book", "clear documents", "clear papers", "clear title",
            "ඔරිජිනල් පොත", "නිරවුල් ලියකියවිලි", "පැහැදිලි ලියකියවිලි", "ලියකියවිලි සම්පූර්ණයි"
        ],
        "points": 4,
        "label": "Original Book & Clear Papers",
    },
    "accident_free_original": {
        "keywords": [
            "accident free", "no accident", "no accidents", "no crash", "never been in",
            "original paint", "factory paint", "unmodified", "genuine", "original condition",
            "අනතුරක් වී නොමැත", "ඔරිජිනල් පේන්ට්", "හැප්පී නොමැත", "සුපිරිම තත්වයෙන්", "ඔරිජිනල් බොඩි"
        ],
        "points": 5,
        "label": "Accident Free & Original Body",
    },
    "full_option_and_features": {
        "keywords": [
            "full option", "fully loaded", "full options", "all options", "top spec", 
            "push start", "multifunction", "multi function", "safety package", 
            "ෆුල් ඔප්ෂන්", "පුෂ් ස්ටාට්"
        ],
        "points": 4,
        "label": "Full Option / High Spec",
    },
    "verified_service_history": {
        "keywords": [
            "service records", "service history", "full service", "company maintained",
            "agent maintained", "dealer maintained", "all records available", "genuine mileage",
            "toyota maintained", "honda maintained", "stafford maintained", "sterling maintained",
            "සියලුම වාර්තා ඇත", "සර්විස් රෙකෝඩ්", "නියෝජිතයා මගින් නඩත්තු කරන ලද"
        ],
        "points": 4,
        "label": "Service Records & Mileage Verified",
    },
    "engine_running_solid": {
        "keywords": [
            "engine in good condition", "smooth engine", "no smoke", "no leaks",
            "100% running", "engine 100%", "running 100%", "hybrid battery replaced", "abs replaced",
            "එන්ජින් 100%", "ධාවන තත්ත්වය 100%", "දෝෂ නොමැත"
        ],
        "points": 3,
        "label": "Engine & Mechanical 100%",
    },
    "careful_personal_use": {
        "keywords": [
            "home used", "carefully used", "personal used", "house used", "family used",
            "ගෙදර පාවිච්චි කළ", "පෞද්ගලික පාවිච්චිය", "පවුලේ පාවිච්චිය"
        ],
        "points": 3,
        "label": "Personal/Home Used",
    },
    "new_wear_and_tear": {
        "keywords": [
            "new battery", "new tyres", "new tires", "alloy wheels",
            "අලුත් බැටරිය", "අලුත් ටයර්", "ටයර් 4 අලුත්"
        ],
        "points": 2,
        "label": "New Tyres/Battery",
    },
    "new_unregistered": {
        "keywords": [
            "brand new", "unregistered", "zero km", "0 km",
            "showroom", "recondition", "reconditioned"
        ],
        "points": 3,
        "label": "New/Unregistered Stock",
    },
}

NLP_NEGATIVE = {
    "fatal_paperwork_issues": {
        "keywords": [
            "duplicate book", "cr duplicate", "open papers", "lost book", "second book",
            "ඩුප්ලිකේට් පොත", "පොත නැතිවී ඇත", "පොත ඩුප්ලිකේට්", "දෙවෙනි පොත"
        ],
        "points": -10,
        "label": "Duplicate Book / Invalid Papers (High Risk)",
    },
    "engine_and_mechanical_issues": {
        "keywords": [
            "engine issue", "engine problem", "engine repair", "gearbox issue",
            "needs repair", "not working", "smoke issue", "oil leak", "head gasket",
            "hybrid battery issue", "abs issue", "dual clutch issue",
            "සුළු අලුත්වැඩියාවන් ඇත", "ගියර් බොක්ස් ලෙඩක්", "එන්ජින් රෙපෙයාර්",
            "බැටරි ලෙඩක්", "ඒබීඑස් ලෙඩක්", "දුම දමයි"
        ],
        "points": -10,
        "label": "Engine/Hybrid/Mechanical Faults",
    },
    "accident_and_structural_damage": {
        "keywords": [
            "accident damage", "collision damage", "front damage", "rear damage",
            "accident repaired", "reconstructed", "salvage", "cut and join", "had an accident",
            "අනතුරකට ලක්වූ", "හැප්පුන", "කපලා ගහපු", "ඇක්සිඩන්ට් වී ඇත", "ඇක්සිඩන්ට් වූ"
        ],
        "points": -10,
        "label": "Major Structural/Accident History",
    },
    "corrosion_and_body_rot": {
        "keywords": [
            "body damage", "panel damage", "dents", "scratches", "tinkering needed",
            "paint faded", "need to paint", "floor rusted", "heavy rust", "chassis rust",
            "පේන්ට් කරගත යුතුයි", "තුඩු", "පොඩි වැඩ වගයක් තියෙනවා",
            "පොඩි පොඩි වැඩ තියෙනවා", "ටින්කරින් ඇත", "දිරුම් ඇත", "දිරා ඇත", "මලකඩ කා ඇත"
        ],
        "points": -6,
        "label": "Rust & Bodywork Required",
    },
    "high_commercial_abuse": {
        "keywords": [
            "taxi", "uber used", "pickme used", "hire", "used heavily",
            "fleet vehicle", "high mileage", "rental", "rent a car",
            "හයර් දුවපු", "ටැක්සි", "පික්මී"
        ],
        "points": -6,
        "label": "Heavy Commercial/Taxi/Hire Use",
    },
    "urgent_or_distressed_sale": {
        "keywords": [
            "urgent", "urgent sale", "quick sale", "must sell", "need to sell",
            "asap", "money urgent", "migrating", "going abroad", "owner migrating",
            "හදිසි විකිණීමක්", "සල්ලි හදිස්සියක්", "ඉක්මනින් විකිණීමට", "රට යන බැවින්"
        ],
        "points": -4,
        "label": "Urgent/Distress Sale (Risk)",
    },
    "finance_lease_burden": {
        "keywords": [
            "finance available", "leasing can be arranged", "lease", "finance settle",
            "ලීසිං මාරු කළ හැක", "ෆිනෑන්ස්", "ලීසිං ගෙවාගෙන යා හැක"
        ],
        "points": -3,
        "label": "Lease/Finance Involved",
    },
}


def extract_nlp_signals(text: str) -> dict:
    """Extract NLP signals from listing title/description."""
    if not text or pd.isna(text):
        return {"signals": [], "nlp_score": 0, "has_fatal_issue": False}

    text_lower = str(text).lower()
    detected   = []
    total_pts  = 0
    has_fatal_issue = False

    # Check Negative Signals First (to trigger fatal overrides)
    for signal_key, cfg in NLP_NEGATIVE.items():
        for kw in cfg["keywords"]:
            if kw.lower() in text_lower:
                detected.append({
                    "type"  : "negative",
                    "key"   : signal_key,
                    "label" : cfg["label"],
                    "points": cfg["points"],
                })
                total_pts += cfg["points"]
                if cfg["points"] <= -10:
                    has_fatal_issue = True
                break

    # Check Positive Signals (Ignore positive points if fatal issue exists)
    for signal_key, cfg in NLP_POSITIVE.items():
        for kw in cfg["keywords"]:
            if kw.lower() in text_lower:
                if has_fatal_issue:
                    detected.append({
                        "type"  : "positive",
                        "key"   : signal_key,
                        "label" : cfg["label"] + " (Ignored)",
                        "points": 0,
                    })
                else:
                    detected.append({
                        "type"  : "positive",
                        "key"   : signal_key,
                        "label" : cfg["label"],
                        "points": cfg["points"],
                    })
                    total_pts += cfg["points"]
                break

    if has_fatal_issue:
        total_pts = -50
    else:
        total_pts = max(-20, min(30, total_pts))

    return {"signals": detected, "nlp_score": total_pts, "has_fatal_issue": has_fatal_issue}


def compute_base_score(listing_price: float, predicted_price: float) -> int:
    """Base price score (0–70). Rewards fair pricing, penalises over/underpricing."""
    if predicted_price <= 0:
        return 35
    deviation_pct = ((listing_price - predicted_price) / predicted_price) * 100

    if   deviation_pct <= -30: return 25   # suspiciously cheap
    elif deviation_pct <= -15: return 55   # good deal
    elif deviation_pct <=  -5: return 65   # slightly below — buyer advantage
    elif deviation_pct <=   5: return 70   # fairly priced ← best
    elif deviation_pct <=  15: return 50   # slightly overpriced
    elif deviation_pct <=  30: return 30   # overpriced
    else:                      return 10   # severely overpriced


def compute_fair_score(listing_price: float, predicted_price: float, nlp_score: int, has_fatal_issue: bool = False) -> dict:
    """Combine base score + NLP modifiers -> final fair score 0-100."""
    base_score    = compute_base_score(listing_price, predicted_price)
    deviation_pct = ((listing_price - predicted_price) / predicted_price) * 100
    
    if has_fatal_issue:
        final_score = 20
        label = "High Risk 🔴"
    else:
        final_score = min(100, max(0, base_score + nlp_score))
        if final_score >= 65:
            label = "Fairly Priced ✅"
        elif final_score >= 45:
            label = "Review Carefully ⚠️"
        elif deviation_pct < -25:
            label = "Suspiciously Underpriced 🔵"
        else:
            label = "Overpriced ❌"

    return {
        "base_score"    : base_score,
        "nlp_score"     : nlp_score,
        "final_score"   : final_score,
        "label"         : label,
        "deviation_pct" : round(deviation_pct, 2),
    }


def get_confidence(record_count: int) -> str:
    """Confidence level based on training data available for this model."""
    if   record_count >= 30: return "High"
    elif record_count >= 10: return "Medium"
    else:                    return "Low — insufficient market data"


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


# ══════════════════════════════════════════════════════════════
#  YEAR-WISE MARKET SUMMARY
# ══════════════════════════════════════════════════════════════

def generate_yearwise_market_summary(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    summary_df = df.groupby(
        ["brand", "model", "variant", "model_year"]
    )["price"].agg(
        mean="mean", median="median",
        min="min", max="max",
        observed_count="count"
    ).reset_index()

    summary_df["mean"]   = summary_df["mean"].round(0).astype(int)
    summary_df["median"] = summary_df["median"].round(0).astype(int)
    summary_df["min"]    = summary_df["min"].astype(int)
    summary_df["max"]    = summary_df["max"].astype(int)
    summary_df = summary_df.sort_values(
        ["brand", "model", "variant", "model_year"]
    )
    summary_df.to_csv(output_path, index=False)
    print(f"  Saved year-wise market summary → {output_path.name}")
    return summary_df


# ══════════════════════════════════════════════════════════════
#  DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════════

def load_and_clean_data() -> pd.DataFrame:
    print(f"Loading datasets from: {DATA_DIR} ...")

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
        print(f"  Loaded  {brand:<20} {len(df):>5} records")

    if not dfs:
        print(f"ERROR: No JSON files found in {DATA_DIR}")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total raw records: {len(combined):,}")

    # ── Numeric cleaning ──
    combined["price"]      = pd.to_numeric(combined["price"],      errors="coerce")
    combined["model_year"] = pd.to_numeric(combined["model_year"], errors="coerce")
    combined["mileage_km"] = pd.to_numeric(combined["mileage_km"], errors="coerce")
    combined = combined.dropna(subset=["price", "model_year"]).copy()

    # ── Price filter ──
    before = len(combined)
    combined = combined[
        (combined["price"] >= PRICE_MIN) &
        (combined["price"] <= PRICE_MAX)
    ].copy()
    print(f"  Price filter removed     : {before - len(combined):,}")

    # ── Year filter ──
    before = len(combined)
    combined = combined[
        (combined["model_year"] >= YEAR_MIN) &
        (combined["model_year"] <= YEAR_MAX)
    ].copy()
    combined["model_year"] = combined["model_year"].astype(int)
    print(f"  Year filter removed      : {before - len(combined):,}")

    # ── Mileage filter ──
    before = len(combined)
    combined = combined[
        combined["mileage_km"].isna() |
        (combined["mileage_km"] <= MILEAGE_MAX)
    ].copy()
    print(f"  Mileage outliers removed : {before - len(combined):,}")

    # ── Car only ──
    if "vehicle_type" in combined.columns:
        before = len(combined)
        combined = combined[combined["vehicle_type"] == "Car"].copy()
        print(f"  Non-car removed          : {before - len(combined):,}")

    # ── Mileage imputation ──
    missing_mil = combined["mileage_km"].isna().sum()
    yr_mil_med  = combined.groupby("model_year")["mileage_km"].median()
    overall_med = combined["mileage_km"].median()
    combined["mileage_km"] = combined.apply(
        lambda r: yr_mil_med.get(r["model_year"], overall_med)
        if pd.isna(r["mileage_km"]) else r["mileage_km"], axis=1
    )
    print(f"  Mileage imputed          : {missing_mil:,} records")

    # ── Minimum combo filter ──
    # Remove brand+model+year combinations with < MIN_COMBO_RECORDS
    # These rare combos teach the model wrong patterns
    # (e.g. Daihatsu Passo 2017 = 1 record → prediction was 22% off)
    before = len(combined)
    combo_counts = combined.groupby(
        ["brand", "model", "model_year"]
    )["price"].transform("count")
    combined = combined[combo_counts >= MIN_COMBO_RECORDS].copy()
    print(f"  Min combo filter removed : {before - len(combined):,} "
          f"(combos with <{MIN_COMBO_RECORDS} records)")

    # ── Feature engineering ──
    combined["vehicle_age"] = (2026 - combined["model_year"]).clip(lower=1)

    # ── Categoricals ──
    combined["fuel_type"]    = combined["fuel_type"].fillna("Petrol").str.strip()
    combined["transmission"] = combined["transmission"].fillna("Automatic").str.strip()
    combined["model"]        = combined["model"].fillna("Unknown").str.strip()

    if "variant" not in combined.columns:
        combined["variant"] = "Standard"
    combined["variant"] = (
        combined["variant"]
        .fillna("Standard")
        .replace("", "Standard")
        .str.strip()
    )

    # ── title_raw for NLP ──
    combined["title_raw"] = combined.get(
        "title_raw", pd.Series("", index=combined.index)
    ).fillna("")

    # ── Record count lookup (for confidence) ──
    combined["_combo_count"] = combined.groupby(
        ["brand", "model"]
    )["price"].transform("count")

    print(f"\n  Final clean records: {len(combined):,}")
    return combined


# ══════════════════════════════════════════════════════════════
#  DATA SUMMARY
# ══════════════════════════════════════════════════════════════

def print_data_summary(df: pd.DataFrame):
    print(f"\n{'='*65}")
    print("DATASET SUMMARY")
    print(f"{'='*65}")
    print(f"  Total records : {len(df):,}")
    print(f"  Brands        : {df['brand'].nunique()}")
    print(f"  Models        : {df['model'].nunique()}")
    print(f"  Year range    : {int(df['model_year'].min())} – {int(df['model_year'].max())}")
    print(f"  Price range   : {df['price'].min()/1e6:.2f}M – {df['price'].max()/1e6:.2f}M LKR")
    print()
    print(f"  {'Brand':<20} {'Records':>7}  {'Mean (M)':>9}  {'Median (M)':>11}")
    print(f"  {'─'*52}")
    brand_stats = df.groupby("brand")["price"].agg(
        ["count", "mean", "median"]
    ).sort_values("count", ascending=False)
    for brand, r in brand_stats.iterrows():
        print(f"  {brand:<20} {int(r['count']):>7}  "
              f"{r['mean']/1e6:>8.2f}M  {r['median']/1e6:>10.2f}M")


# ══════════════════════════════════════════════════════════════
#  MODEL TRAINING
# ══════════════════════════════════════════════════════════════

def train_all_models(df: pd.DataFrame):
    print(f"\n{'='*65}")
    print("TRAINING — 4 MODEL PIPELINE")
    print(f"{'='*65}")

    X = df[ALL_FEATURES].copy()
    y = np.log1p(df[TARGET].copy())   # log transform for skewed price distribution

    # OrdinalEncoder for XGBoost / Gradient Boosting / Random Forest
    encoder   = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_encoded = X.copy()
    X_encoded[CATEGORICAL_FEATURES] = encoder.fit_transform(
        X[CATEGORICAL_FEATURES].astype(str)
    )

    # CatBoost uses original string categoricals
    cat_indices = [ALL_FEATURES.index(c) for c in CATEGORICAL_FEATURES]

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
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            cat_features=cat_indices,
            random_seed=RANDOM_STATE,
            verbose=False,
        ),
        # XGBoost selected for production — monotone constraints guarantee:
        #   newer year   → always higher price  (+1)
        #   more mileage → always lower price   (-1)
        #   older age    → always lower price   (-1)
        # Random Forest MAE: ~563K | XGBoost MAE: ~583K
        # Difference ~20K LKR is acceptable for logical consistency guarantee
        "XGBRegressor": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            verbosity=0,
            monotone_constraints=MONOTONE_CONSTRAINTS,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []
    trained = {}

    for name, model in models.items():
        print(f"  Training {name} ...")
        if name == "CatBoost":
            model.fit(X_cat_tr, y_tr)
            y_pred = model.predict(X_cat_te)
        else:
            model.fit(X_enc_tr, y_tr)
            y_pred = model.predict(X_enc_te)

        metrics = compute_metrics(np.expm1(y_te), np.expm1(y_pred))
        results.append({"Model": name, **metrics})
        trained[name] = model
        print(f"    MAE: {metrics['MAE']:>12,.0f} LKR  "
              f"RMSE: {metrics['RMSE']:>12,.0f} LKR  "
              f"R²: {metrics['R2_Score']:.4f}")

    results_df = pd.DataFrame(results).sort_values("MAE")

    # ── Report honest best by MAE ──
    honest_best = results_df.iloc[0]["Model"]
    print(f"\n  Best by MAE         : {honest_best} "
          f"(MAE={results_df.iloc[0]['MAE']:,.0f})")

    # ── Production model = XGBoost (monotone constraints) ──
    production_model_name = "XGBRegressor"
    production_model      = trained[production_model_name]
    xgb_row = results_df[results_df["Model"] == production_model_name].iloc[0]

    print(f"  Production model    : {production_model_name} "
          f"(MAE={xgb_row['MAE']:,.0f}, R²={xgb_row['R2_Score']:.4f})")
    print(f"  Reason              : Monotone constraints ensure logical pricing")

    # ── Build display predictions ──
    final_preds_log = production_model.predict(X_enc_te)
    display_df = X_cat_te.copy()
    display_df["Actual_Price_LKR"]    = np.expm1(y_te).round(0).values
    display_df["Predicted_Price_LKR"] = np.expm1(final_preds_log).round(0)
    display_df["Difference_LKR"]      = (
        display_df["Predicted_Price_LKR"] - display_df["Actual_Price_LKR"]
    )
    display_df["Error_Margin_%"] = (
        abs(display_df["Difference_LKR"]) /
        display_df["Actual_Price_LKR"] * 100
    ).round(2)

    # Add confidence level
    combo_count_map = df.groupby(["brand","model"])["price"].count().to_dict()
    display_df["Confidence"] = display_df.apply(
        lambda r: get_confidence(
            combo_count_map.get((r["brand"], r["model"]), 0)
        ), axis=1
    )

    return (results_df, trained, encoder,
            production_model, production_model_name,
            display_df, X_cat_te, X_enc_te, y_te,
            X_cat_tr, X_enc_tr, y_tr, cat_indices)


# ══════════════════════════════════════════════════════════════
#  PRINT TABLES
# ══════════════════════════════════════════════════════════════

def print_model_comparison(results_df, production_model_name):
    print(f"\n{'='*65}")
    print("OVERALL MODEL COMPARISON")
    print(f"{'='*65}")
    print(f"  {'Model':<22} {'MAE':>14}  {'RMSE':>14}  {'R²':>8}  Note")
    print(f"  {'─'*65}")
    for _, r in results_df.iterrows():
        note = ""
        if r["Model"] == results_df.iloc[0]["Model"]:
            note = "← Best MAE"
        if r["Model"] == production_model_name:
            note = (note + "  ← Production (monotone)").strip()
        print(f"  {r['Model']:<22} {r['MAE']:>14,.0f}  "
              f"{r['RMSE']:>14,.0f}  {r['R2_Score']:>8.4f}  {note}")


def print_per_brand_table(display_df, production_model_name):
    print(f"\n{'─'*70}")
    print(f"PER-BRAND PERFORMANCE  ({production_model_name})")
    print(f"  {'Brand':<20} {'Test N':>6}  {'MAE (LKR)':>12}  {'R²':>8}")
    print(f"  {'─'*50}")
    for brand in sorted(display_df["brand"].unique()):
        sub = display_df[display_df["brand"] == brand]
        if len(sub) < 2:
            continue
        mae = mean_absolute_error(sub["Actual_Price_LKR"], sub["Predicted_Price_LKR"])
        r2  = r2_score(sub["Actual_Price_LKR"], sub["Predicted_Price_LKR"])
        print(f"  {brand:<20} {len(sub):>6}  {mae:>12,.0f}  {r2:>8.4f}")


def print_per_variant_table(display_df, production_model_name):
    print(f"\n{'─'*85}")
    print(f"PER-VARIANT PERFORMANCE  ({production_model_name})")
    print(f"  {'Brand':<12} | {'Model':<15} | {'Variant':<15} | "
          f"{'N':>4} | {'MAE (LKR)':>12} | Confidence")
    print(f"  {'─'*80}")
    groups = display_df.groupby(["brand", "model", "variant"])
    for (brand, model_name, variant), grp in groups:
        if len(grp) < 2:
            continue
        mae  = mean_absolute_error(grp["Actual_Price_LKR"], grp["Predicted_Price_LKR"])
        conf = grp["Confidence"].iloc[0]
        m_str = (model_name[:12] + "..") if len(model_name) > 14 else model_name
        v_str = (variant[:12]    + "..") if len(variant)    > 14 else variant
        print(f"  {brand:<12} | {m_str:<15} | {v_str:<15} | "
              f"{len(grp):>4} | {mae:>12,.0f} | {conf}")
    print(f"  {'─'*80}")


def print_sample_predictions(display_df):
    print(f"\n{'─'*90}")
    print("SAMPLE PREDICTIONS  (10 random listings)")
    print(f"  {'Brand':<12} | {'Model':<12} | {'Year':>5} | "
          f"{'Actual':>12} | {'Predicted':>12} | {'Err%':>6} | Confidence")
    print(f"  {'─'*85}")
    sample = display_df.sample(min(10, len(display_df)), random_state=RANDOM_STATE)
    for _, r in sample.iterrows():
        flag = " ⚠️" if r["Error_Margin_%"] > 15 else ""
        print(f"  {str(r['brand']):<12} | {str(r['model'])[:12]:<12} | "
              f"{int(r['model_year']):>5} | "
              f"{r['Actual_Price_LKR']:>12,.0f} | "
              f"{r['Predicted_Price_LKR']:>12,.0f} | "
              f"{r['Error_Margin_%']:>5.1f}% | "
              f"{r['Confidence']}{flag}")
    print(f"  {'─'*85}")
    high_err = display_df[display_df["Error_Margin_%"] > 15]
    print(f"\n  Predictions >15% error : {len(high_err)} / {len(display_df)} "
          f"({len(high_err)/len(display_df)*100:.1f}%)")
    print(f"  Note: High errors typically occur in Low confidence groups")


# ══════════════════════════════════════════════════════════════
#  CROSS VALIDATION
# ══════════════════════════════════════════════════════════════

def run_cross_validation(df, cat_indices):
    print(f"\n{'─'*65}")
    print("5-FOLD CROSS VALIDATION  (XGBRegressor — Production Model)")

    X = df[ALL_FEATURES].copy()
    y = np.log1p(df[TARGET].copy())

    encoder   = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_encoded = X.copy()
    X_encoded[CATEGORICAL_FEATURES] = encoder.fit_transform(
        X[CATEGORICAL_FEATURES].astype(str)
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rows = []

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X_encoded), 1):
        X_tr, X_te = X_encoded.iloc[tr_idx], X_encoded.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        m = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_STATE, verbosity=0,
            monotone_constraints=MONOTONE_CONSTRAINTS,
        )
        m.fit(X_tr, y_tr)
        metrics = compute_metrics(np.expm1(y_te), np.expm1(m.predict(X_te)))
        cv_rows.append({"Fold": fold, **metrics})
        print(f"  Fold {fold}: MAE={metrics['MAE']:>10,.0f}  "
              f"RMSE={metrics['RMSE']:>10,.0f}  R²={metrics['R2_Score']:.4f}")

    cv_df = pd.DataFrame(cv_rows)
    print(f"\n  CV Mean MAE  : {cv_df['MAE'].mean():>12,.0f} LKR  "
          f"(±{cv_df['MAE'].std():,.0f})")
    print(f"  CV Mean RMSE : {cv_df['RMSE'].mean():>12,.0f} LKR  "
          f"(±{cv_df['RMSE'].std():,.0f})")
    print(f"  CV Mean R²   : {cv_df['R2_Score'].mean():>12.4f}       "
          f"(±{cv_df['R2_Score'].std():.4f})")
    return cv_df


# ══════════════════════════════════════════════════════════════
#  NLP LAYER DEMO
# ══════════════════════════════════════════════════════════════

def demonstrate_nlp(df, production_model, encoder):
    print(f"\n{'='*65}")
    print("NLP SCORING LAYER — Sample Demonstration")
    print(f"{'='*65}")

    sample = df[df["title_raw"].str.len() > 5].sample(
        min(5, len(df)), random_state=RANDOM_STATE
    )

    for _, row in sample.iterrows():
        inp = pd.DataFrame([{f: row[f] for f in ALL_FEATURES}])
        inp_enc = inp.copy()
        inp_enc[CATEGORICAL_FEATURES] = encoder.transform(
            inp[CATEGORICAL_FEATURES].astype(str)
        )
        predicted     = float(np.expm1(production_model.predict(inp_enc)[0]))
        listing_price = float(row["price"])
        nlp_result    = extract_nlp_signals(row["title_raw"])
        
        is_fatal = nlp_result.get("has_fatal_issue", False)
        fair_result   = compute_fair_score(
            listing_price, predicted, nlp_result["nlp_score"], is_fatal
        )

        print(f"\n  {row['brand']} {row['model']} {int(row['model_year'])}")
        print(f"  Title       : {str(row['title_raw'])[:70]}")
        print(f"  Listing     : Rs. {listing_price:>12,.0f}")
        print(f"  Predicted   : Rs. {predicted:>12,.0f}")
        print(f"  Deviation   : {fair_result['deviation_pct']:>+.1f}%")
        print(f"  Base Score  : {fair_result['base_score']:>3} / 70")
        for sig in nlp_result["signals"]:
            sign = "+" if sig["points"] > 0 else ""
            print(f"  NLP Signal  : {sig['label']:<30} {sign}{sig['points']} pts")
        print(f"  NLP Score   : {fair_result['nlp_score']:>+3}")
        print(f"  FINAL SCORE : {fair_result['final_score']:>3} / 100  "
              f"→  {fair_result['label']}")


# ══════════════════════════════════════════════════════════════
#  SAVE ARTIFACTS
# ══════════════════════════════════════════════════════════════

def save_artifacts(results_df, trained, encoder,
                   production_model, display_df,
                   cv_df, df_clean):
    ensure_dirs()

    # Save all 4 models
    for name, model in trained.items():
        fname = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, MODEL_DIR / fname)

    # Save production model
    joblib.dump(production_model, MODEL_DIR / "best_model.pkl")
    joblib.dump(encoder,          MODEL_DIR / "ordinal_encoder.pkl")

    # Save outputs
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    display_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    cv_df.to_csv(OUTPUT_DIR / "cross_validation_results.csv", index=False)

    # Save NLP config for API
    nlp_config = {
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
            "base_max"             : 70,
            "nlp_max"              : 30,
            "final_max"            : 100,
            "fairly_priced_min"    : 65,
            "review_min"           : 45,
        }
    }
    with open(OUTPUT_DIR / "nlp_config.json", "w") as f:
        json.dump(nlp_config, f, indent=2)

    # Brand-model lookup (for browser extension dropdowns + confidence)
    lookup = df_clean.groupby(["brand", "model"]).agg(
        record_count=("price","count"),
        mean_price=("price","mean"),
        min_year=("model_year","min"),
        max_year=("model_year","max"),
    ).reset_index()
    lookup["confidence"] = lookup["record_count"].apply(get_confidence)
    lookup["mean_price"] = lookup["mean_price"].round(0).astype(int)
    lookup = lookup.sort_values(["brand","record_count"], ascending=[True,False])
    lookup.to_csv(OUTPUT_DIR / "brand_model_lookup.csv", index=False)

    # Year-wise market summary
    generate_yearwise_market_summary(
        df_clean, OUTPUT_DIR / "yearwise_car_summary.csv"
    )

    print(f"\n  models/combined/best_model.pkl          ← XGBoost production")
    print(f"  models/combined/ordinal_encoder.pkl     ← for inference")
    print(f"  models/combined/[catboost/xgb/gb/rf].pkl")
    print(f"  outputs/combined/model_comparison.csv")
    print(f"  outputs/combined/test_predictions.csv")
    print(f"  outputs/combined/cross_validation_results.csv")
    print(f"  outputs/combined/nlp_config.json        ← NLP keywords for API")
    print(f"  outputs/combined/brand_model_lookup.csv ← for extension dropdowns")
    print(f"  outputs/combined/yearwise_car_summary.csv")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    ensure_dirs()

    df = load_and_clean_data()
    print_data_summary(df)

    (results_df, trained, encoder,
     production_model, production_model_name,
     display_df, X_cat_te, X_enc_te, y_te,
     X_cat_tr, X_enc_tr, y_tr,
     cat_indices) = train_all_models(df)

    print_model_comparison(results_df, production_model_name)
    print_per_brand_table(display_df, production_model_name)
    print_per_variant_table(display_df, production_model_name)
    print_sample_predictions(display_df)

    cv_df = run_cross_validation(df, cat_indices)

    demonstrate_nlp(df, production_model, encoder)

    print(f"\n{'='*65}")
    print("SAVING ALL ARTIFACTS")
    print(f"{'='*65}")
    save_artifacts(results_df, trained, encoder,
                   production_model, display_df,
                   cv_df, df)

    print(f"\n{'='*65}")
    print("TRAINING COMPLETE")
    print(f"{'='*65}")
    xgb_row = results_df[results_df["Model"] == "XGBRegressor"].iloc[0]
    print(f"  Models trained  : {len(results_df)} "
          f"(CatBoost, XGBoost, Gradient Boosting, Random Forest)")
    print(f"  Production model: XGBRegressor  "
          f"MAE={xgb_row['MAE']:,.0f}  R²={xgb_row['R2_Score']:.4f}")
    print(f"  CV Mean R²      : {cv_df['R2_Score'].mean():.4f}")
    print(f"  Total records   : {len(df):,} across {df['brand'].nunique()} brands")


if __name__ == "__main__":
    main()