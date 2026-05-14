"""
Suzuki Alto Price Prediction - Training Script
===============================================
Trains SEPARATE models for each Alto group because each group
represents a fundamentally different market segment.

Groups:
  G1 → 2000-2012  Manual              — Old gen 800cc
  G2 → 2013-2015  Manual              — New gen 800cc
  G3 → 2016-2019  Manual              — Recent manual
  G4 → 2000-2015  Automatic <700cc    — Old auto 660/650cc

Why separate models (not one combined)?
  - Manual 2000-2012 prices: 2.4M - 3.85M
  - Manual 2013-2015 prices: 3.0M - 4.7M
  - Automatic 2000-2015 prices: 2.7M - 4.7M
  These ranges overlap but have different depreciation curves.
  One combined model would confuse them — separate models are cleaner.

Feature used:
  - model_year (individual year, treated as categorical via OneHotEncoder)

NOT used:
  - fuel_type    : 97% Petrol — adds zero information
  - transmission : already separated into groups
  - engine_cc    : used only for group filtering, not as feature
  - mileage      : used in fair price scoring layer, not price prediction

Target: price_lkr

Three ML algorithms compared per group:
  - XGBRegressor        (primary — matches research proposal)
  - Gradient Boosting
  - Random Forest

Data cleaning applied:
  - Drop years outside 2000-2020 (2024/2025/2026 are data errors)
  - Drop price below 1,500,000 LKR (typos/damaged cars)
"""

import os
import sys
import json
import warnings
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data",    "clean_alto_dataset.json")
MODEL_DIR  = os.path.join(BASE_DIR, "models",  "alto")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "alto")

RANDOM_STATE = 42

# Price bounds — records outside these are data errors
PRICE_MIN = 1_500_000
PRICE_MAX = 8_000_000

# Group definitions — each group gets its own model
GROUPS = {
    "G1_Manual_2000-2012": {
        "transmission": "Manual",
        "year_min": 2000,
        "year_max": 2012,
        "engine_max": None,
    },
    "G2_Manual_2013-2015": {
        "transmission": "Manual",
        "year_min": 2013,
        "year_max": 2015,
        "engine_max": None,
    },
    "G3_Manual_2016-2019": {
        "transmission": "Manual",
        "year_min": 2016,
        "year_max": 2019,
        "engine_max": None,
    },
    "G4_Auto_lt700_2000-2015": {
        "transmission": "Automatic",
        "year_min": 2000,
        "year_max": 2015,
        "engine_max": 700,  # filter: engine_cc < 700
    },
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred):
    """Calculate MAE, RMSE, and R² for a set of predictions."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred) if len(y_true) >= 2 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2_Score": r2}


def build_pipeline(model):
    """
    Build sklearn Pipeline with OneHotEncoder for model_year_str.
    model_year is treated as categorical — not numeric —
    so the model learns a separate price level per year.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             ["model_year_str"])
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])


def get_ml_models():
    """Return the three ensemble models to compare."""
    return {
        "XGBRegressor": XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
        ),
    }


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_and_clean_data():
    """Load JSON dataset and apply cleaning rules."""
    print("Loading Alto dataset ...")
    if not os.path.exists(DATA_PATH):
        print(f"  ERROR: Dataset not found at {DATA_PATH}")
        sys.exit(1)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)

    # Verify required columns exist
    required = ["price_lkr", "model_year", "transmission", "engine_cc"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ERROR: Missing columns: {missing}")
        sys.exit(1)

    # Convert to numeric (coerce bad values to NaN)
    df["price_lkr"]  = pd.to_numeric(df["price_lkr"],  errors="coerce")
    df["model_year"] = pd.to_numeric(df["model_year"],  errors="coerce")
    df["engine_cc"]  = pd.to_numeric(df["engine_cc"],   errors="coerce")

    # Drop rows missing price or year
    df = df.dropna(subset=["price_lkr", "model_year"]).copy()
    df["model_year"] = df["model_year"].astype(int)

    # Drop bad years (2024/2025/2026 are data entry errors)
    before = len(df)
    df = df[df["model_year"].between(2000, 2020)].copy()
    print(f"  Dropped {before - len(df)} records with invalid years (outside 2000-2020)")

    # Drop price outliers
    before = len(df)
    df = df[(df["price_lkr"] >= PRICE_MIN) & (df["price_lkr"] <= PRICE_MAX)].copy()
    print(f"  Dropped {before - len(df)} records outside price range "
          f"({PRICE_MIN:,.0f} - {PRICE_MAX:,.0f} LKR)")

    # Add string version of year for OneHotEncoder
    df["model_year_str"] = df["model_year"].astype(str)

    return df


# ─── Group filtering ──────────────────────────────────────────────────────────

def filter_group(df, group_name):
    """Filter dataframe to records belonging to a specific group."""
    cfg = GROUPS[group_name]
    mask = (
        (df["transmission"] == cfg["transmission"]) &
        (df["model_year"] >= cfg["year_min"]) &
        (df["model_year"] <= cfg["year_max"])
    )
    # Apply engine_cc filter only for groups that need it (G4)
    if cfg["engine_max"] is not None:
        mask = mask & (df["engine_cc"] < cfg["engine_max"])
    return df[mask].copy()


# ─── Training ─────────────────────────────────────────────────────────────────

def train_group(group_df, group_name):
    """Train all 3 models on one group. Returns results + best pipeline."""
    print(f"\n{'-' * 60}")
    print(f"Training group: {group_name} ({len(group_df)} records)")
    print(f"{'-' * 60}")

    # Skip groups with too few records
    if len(group_df) < 10:
        print(f"  [SKIP] Too few records to train reliably.")
        return None, None, None, None, None, None

    X = group_df[["model_year_str"]]
    y = group_df["price_lkr"]

    # Try stratified split so each year is represented in test set
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE,
            stratify=group_df["model_year"]
        )
    except ValueError:
        # Falls back to normal split if a year has only 1 sample
        print("  [WARNING] Stratified split failed -> using normal split.")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )

    ml_models = get_ml_models()
    results   = []
    pipelines = {}

    # Train each algorithm and record metrics
    for ml_name, ml_model in ml_models.items():
        pipe = build_pipeline(ml_model)
        pipe.fit(X_tr, y_tr)
        metrics = compute_metrics(y_te, pipe.predict(X_te))
        results.append({"Model": ml_name, **metrics})
        pipelines[ml_name] = pipe

    results_df = pd.DataFrame(results).sort_values("MAE")

    # Best model = lowest MAE
    best_name = results_df.iloc[0]["Model"]
    best_pipe = pipelines[best_name]

    # Print comparison table
    print(f"\n  Model Comparison:")
    print(f"  {'Model':<27}  {'MAE':>12}  {'RMSE':>12}  {'R2':>8}")
    print(f"  {'-' * 65}")
    for _, r in results_df.iterrows():
        marker = " <- best" if r["Model"] == best_name else ""
        print(f"  {r['Model']:<27}  {r['MAE']:>12,.0f}  "
              f"{r['RMSE']:>12,.0f}  {r['R2_Score']:>8.4f}{marker}")

    print(f"\n  Best Model : {best_name}")
    print(f"  MAE        : {results_df.iloc[0]['MAE']:,.2f} LKR")
    print(f"  R2_Score   : {results_df.iloc[0]['R2_Score']:.4f}")

    return results_df, best_name, best_pipe, pipelines, X_te, y_te


# ─── Predicted prices ─────────────────────────────────────────────────────────

def get_predicted_prices(group_df, best_pipe, group_name):
    """Get predicted fair price per year for this group."""
    years = sorted(group_df["model_year"].unique())
    rows  = []
    for yr in years:
        subset = group_df[group_df["model_year"] == yr]
        pred_input = pd.DataFrame({"model_year_str": [str(yr)]})
        predicted  = best_pipe.predict(pred_input)[0]
        rows.append({
            "group":                    group_name,
            "model_year":               yr,
            "observed_count":           len(subset),
            "actual_mean_price_lkr":    round(subset["price_lkr"].mean(), 2),
            "actual_median_price_lkr":  round(subset["price_lkr"].median(), 2),
            "predicted_fair_price_lkr": round(predicted, 2),
        })
    return pd.DataFrame(rows)


# ─── Cross validation ─────────────────────────────────────────────────────────

def run_cv(group_df, best_pipe, group_name):
    """5-fold stratified CV. Skip if fewer than 25 records."""
    X = group_df[["model_year_str"]]
    y = group_df["price_lkr"]

    n_splits = 5

    # Need at least 25 records for meaningful 5-fold CV
    if len(group_df) < 25:
        print(f"\n  [CV] Only {len(group_df)} records - skipping CV (need 25+)")
        return None

    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        cv_results = cross_validate(
            best_pipe, X, y,
            cv=list(skf.split(X, group_df["model_year"])),  # list() to avoid generator bug
            scoring={
                "mae":  "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2":   "r2",
            },
            return_train_score=False,
        )
        cv_df = pd.DataFrame({
            "Fold": range(1, n_splits + 1),
            "MAE":  -cv_results["test_mae"],
            "RMSE": -cv_results["test_rmse"],
            "R2":   cv_results["test_r2"],
        })

        # Print fold-by-fold results
        print(f"\n  5-Fold Cross-Validation:")
        print(cv_df.to_string(index=False))

        # Print summary stats
        print(f"\n  CV Mean MAE  : {cv_df['MAE'].mean():>12,.2f} LKR  "
              f"(+/-{cv_df['MAE'].std():,.2f})")
        print(f"  CV Mean RMSE : {cv_df['RMSE'].mean():>12,.2f} LKR  "
              f"(+/-{cv_df['RMSE'].std():,.2f})")
        print(f"  CV Mean R2   : {cv_df['R2'].mean():>12.4f}       "
              f"(+/-{cv_df['R2'].std():.4f})")
        return cv_df

    except Exception as e:
        print(f"\n  (CV skipped: {e})")
        return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ensure_dirs()
    df = load_and_clean_data()

    print(f"\nTotal records after cleaning: {len(df)}")
    print("\nRecords per group:")

    all_results     = []
    all_predictions = []
    all_cv_results  = []

    # Print group summary first
    for group_name in GROUPS:
        group_df = filter_group(df, group_name)
        n = len(group_df)
        cfg = GROUPS[group_name]
        engine_note = f" engine<{cfg['engine_max']}cc" if cfg["engine_max"] else ""
        print(f"  {group_name:<30} {n:>4} records  "
              f"({cfg['transmission']}, {cfg['year_min']}-{cfg['year_max']}{engine_note})")
        if n < 10:
            print(f"    [WARNING] Too few records - this group will be skipped")

    print()

    # ── Train each group ──────────────────────────────────────────────────────
    best_models = {}

    for group_name in GROUPS:
        group_df = filter_group(df, group_name)

        # Skip groups that are too small
        if len(group_df) < 10:
            print(f"\n[SKIP] {group_name} - only {len(group_df)} records")
            continue

        result = train_group(group_df, group_name)
        results_df, best_name, best_pipe, pipelines, X_te, y_te = result

        # Skip if training returned None (too few records)
        if results_df is None:
            continue

        # Save all 3 models + best_model.pkl per group
        group_model_dir = os.path.join(MODEL_DIR, group_name)
        os.makedirs(group_model_dir, exist_ok=True)

        for ml_name, pipe in pipelines.items():
            fname = ml_name.lower().replace(" ", "_") + ".pkl"
            joblib.dump(pipe, os.path.join(group_model_dir, fname))

        joblib.dump(best_pipe, os.path.join(group_model_dir, "best_model.pkl"))
        best_models[group_name] = best_pipe

        # Generate predicted fair prices per year
        pred_df = get_predicted_prices(group_df, best_pipe, group_name)
        all_predictions.append(pred_df)

        # Print predicted prices table
        print(f"\n  Predicted Fair Prices ({group_name}):")
        print(f"  {'Year':>6}  {'N':>4}  "
              f"{'Actual Mean':>14}  {'Predicted':>14}  {'Diff':>12}")
        print(f"  {'-' * 60}")
        for _, r in pred_df.iterrows():
            diff = r["predicted_fair_price_lkr"] - r["actual_mean_price_lkr"]
            print(f"  {int(r['model_year']):>6}  {int(r['observed_count']):>4}  "
                  f"{r['actual_mean_price_lkr']:>14,.0f}  "
                  f"{r['predicted_fair_price_lkr']:>14,.0f}  "
                  f"{diff:>+12,.0f}")

        # Append results for combined CSV
        results_df["group"] = group_name
        all_results.append(results_df)

        # Run cross-validation
        cv_df = run_cv(group_df, best_pipe, group_name)
        if cv_df is not None:
            cv_df["group"] = group_name
            all_cv_results.append(cv_df)

    # ── Save combined outputs ─────────────────────────────────────────────────
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_csv(
            os.path.join(OUTPUT_DIR, "alto_overall_model_comparison.csv"), index=False
        )

    if all_predictions:
        combined_preds = pd.concat(all_predictions, ignore_index=True)
        combined_preds.to_csv(
            os.path.join(OUTPUT_DIR, "alto_year_predictions.csv"), index=False
        )

    if all_cv_results:
        combined_cv = pd.concat(all_cv_results, ignore_index=True)
        combined_cv.to_csv(
            os.path.join(OUTPUT_DIR, "alto_cross_validation_results.csv"), index=False
        )

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("ALTO TRAINING COMPLETE")
    print(f"{'=' * 60}")
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        summary = combined_results.loc[
            combined_results.groupby("group")["MAE"].idxmin()
        ][["group", "Model", "MAE", "R2_Score"]]
        print("\nBest model per group:")
        print(f"  {'Group':<30} {'Model':<27} {'MAE':>12}  {'R2':>8}")
        print(f"  {'-' * 85}")
        for _, r in summary.iterrows():
            print(f"  {r['group']:<30} {r['Model']:<27} "
                  f"{r['MAE']:>12,.0f}  {r['R2_Score']:>8.4f}")

    print(f"\nOutputs saved to:")
    print(f"  models/alto/<group_name>/best_model.pkl")
    print(f"  outputs/alto/alto_year_predictions.csv")
    print(f"  outputs/alto/alto_overall_model_comparison.csv")
    print(f"  outputs/alto/alto_cross_validation_results.csv")


if __name__ == "__main__":
    main()
