"""
Toyota Corolla Combined Price Prediction - Training Script
=========================================================
Trains ONE combined regression model for four Corolla variants:
  121, 141, AE110, DX/KE72

Features used:
  - year_range     (categorical, variant-specific)
  - variant        (categorical)
  - transmission   (categorical)
  - fuel_type      (categorical - optional)

Target: price_lkr

Compares Feature Sets A/B/C/D and three ML algorithms.
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

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data",   "clean_corolla_dataset_final.json")
MODEL_DIR   = os.path.join(BASE_DIR, "models", "corolla_combined")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs","corolla")

VALID_VARIANTS = ["121", "141", "AE110", "DX/KE72"]

# Feature sets to compare
FEATURE_SETS = {
    "A": ["year_range", "variant"],
    "B": ["year_range", "variant", "transmission"],
    "C": ["year_range", "variant", "fuel_type"],
    "D": ["year_range", "variant", "transmission", "fuel_type"],
}

NUMERIC_FEATURES      = [] # Using range as categorical
CATEGORICAL_FEATURES  = ["year_range", "variant", "transmission", "fuel_type"]

RANDOM_STATE = 42


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_year_range(variant, year):
    """Maps year to range based on variant as requested by USER."""
    if variant == "121":
        if 2000 <= year <= 2001: return "2000-2001"
        if 2002 <= year <= 2003: return "2002-2003"
        if 2004 <= year <= 2008: return "2004-2008"
    elif variant == "141":
        if 2007 <= year <= 2009: return "2007-2009"
        if 2010 <= year <= 2013: return "2010-2013"
    elif variant == "AE110":
        if 1994 <= year <= 1996: return "1994-1996"
        if 1997 <= year <= 2000: return "1997-2000"
    elif variant == "DX/KE72":
        if 1980 <= year <= 1984: return "1980-1984"
        if 1985 <= year <= 1990: return "1985-1990"
    
    # Fallback for years outside specific ranges
    if year < 1990: return "Pre-1990"
    if year < 2000: return "1990-1999"
    if year < 2010: return "2000-2009"
    return "2010-Later"

def ensure_dirs():
    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred):
    """Return dict with MAE, MSE, RMSE, R2_Score."""
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred) if len(y_true) >= 2 else float("nan")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2_Score": r2}


def build_pipeline(feature_set_keys, model):
    """Build a sklearn Pipeline with ColumnTransformer + model."""
    num_cols = [f for f in NUMERIC_FEATURES if f in feature_set_keys]
    cat_cols = [f for f in CATEGORICAL_FEATURES if f in feature_set_keys]

    transformers = []
    if cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        )
    if num_cols:
        from sklearn.preprocessing import FunctionTransformer
        transformers.append(("num", FunctionTransformer(), num_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="passthrough")
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])


# ─── Data loading & filtering ─────────────────────────────────────────────────

def load_and_filter_data():
    print("Loading Corolla dataset ...")
    if not os.path.exists(DATA_PATH):
        print(f"  ERROR: Dataset not found at:\n  {DATA_PATH}")
        print("  Please ensure 'clean_corolla_dataset_final.json' is in the data/ folder.")
        sys.exit(1)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)

    # Validate required columns
    required = ["price_lkr", "model_year", "variant", "transmission",
                "fuel_type", "record_status", "variant_confidence",
                "seller_description_clean"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ERROR: Missing columns: {missing}")
        sys.exit(1)

    # ── Suspect price rows (down-payment listings) ──
    desc_col = df["seller_description_clean"].fillna("").str.lower()
    suspect_mask = desc_col.str.contains("down payment|downpayment", regex=True)
    suspect_df = df[suspect_mask].copy()
    suspect_path = os.path.join(OUTPUT_DIR, "suspect_price_rows.csv")
    suspect_df.to_csv(suspect_path, index=False)
    if not suspect_df.empty:
        print(f"  [WARNING] {len(suspect_df)} suspect 'down payment' rows found -> saved to outputs/corolla/suspect_price_rows.csv")

    # ── Apply filters ──
    df = df[
        (df["record_status"] == "keep") &
        (df["variant_confidence"] == "High") &
        (df["variant"].isin(VALID_VARIANTS)) &
        (~suspect_mask)
    ].copy()

    # ── Clean numerics ──
    df["price_lkr"]  = pd.to_numeric(df["price_lkr"],  errors="coerce")
    df["model_year"] = df["model_year"].astype(int)

    # ── Add Year Range Column ──
    df["year_range"] = df.apply(lambda x: get_year_range(x["variant"], x["model_year"]), axis=1)

    return df


# ─── Print helpers ────────────────────────────────────────────────────────────

def print_data_summary(df):
    print(f"\nTotal Corolla records after cleaning and filtering: {len(df)}\n")

    counts = df["variant"].value_counts()
    print("Records per Corolla variant:")
    for v in VALID_VARIANTS:
        print(f"  {v:<10} {counts.get(v, 0)}")

    print("\nActual Price Summary by Variant:")
    summary = df.groupby("variant")["price_lkr"].agg(
        Count="count", Min="min", Max="max", Mean="mean", Median="median"
    )
    print(summary.to_string())

    # Warn small groups
    for v in VALID_VARIANTS:
        n = counts.get(v, 0)
        if n < 10:
            print(f"\n  [WARNING] Variant '{v}' has only {n} records - metrics may be unreliable.")


# ─── Save price summary outputs ───────────────────────────────────────────────

def save_price_summaries(df):
    # 1. Variant price summary
    vs = df.groupby("variant")["price_lkr"].agg(
        record_count="count",
        min_actual_price_lkr="min",
        max_actual_price_lkr="max",
        mean_actual_price_lkr="mean",
        median_actual_price_lkr="median"
    ).reset_index()
    vs.to_csv(os.path.join(OUTPUT_DIR, "corolla_variant_price_summary.csv"), index=False)

    # 2. Variant × year price summary
    vys = df.groupby(["variant", "year_range"])["price_lkr"].agg(
        record_count="count",
        min_actual_price_lkr="min",
        max_actual_price_lkr="max",
        mean_actual_price_lkr="mean",
        median_actual_price_lkr="median"
    ).reset_index()
    vys.to_csv(os.path.join(OUTPUT_DIR, "corolla_variant_range_price_summary.csv"), index=False)

    # 3. Variant counts
    vc = df["variant"].value_counts().reset_index()
    vc.columns = ["variant", "record_count"]
    vc.to_csv(os.path.join(OUTPUT_DIR, "corolla_variant_counts.csv"), index=False)


# ─── Training loop ────────────────────────────────────────────────────────────

def train_all(df):
    """Train all model × feature-set combos. Returns best pipeline + results."""
    X_all = df[["year_range", "variant", "transmission", "fuel_type"]]
    y     = df["price_lkr"]

    # Stratified split by variant
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y, test_size=0.2, random_state=RANDOM_STATE,
            stratify=df["variant"]
        )
    except ValueError:
        print("  [WARNING] Stratified split failed -> using normal split.")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y, test_size=0.2, random_state=RANDOM_STATE
        )

    ml_models = {
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "XGBRegressor":            XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=RANDOM_STATE),
        "Gradient Boosting":       GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)
    }

    overall_results   = []
    per_variant_results = []
    trained_pipelines = {}   # (ml_name, fs_label) → pipeline

    print("\nTraining and evaluating combined Corolla models ...")

    for fs_label, fs_cols in FEATURE_SETS.items():
        for ml_name, ml_model in ml_models.items():
            pipe = build_pipeline(fs_cols, ml_model.__class__(**ml_model.get_params()))
            pipe.fit(X_tr[fs_cols], y_tr)

            y_pred_te = pipe.predict(X_te[fs_cols])

            overall = compute_metrics(y_te, y_pred_te)
            overall_results.append({
                "Model"       : ml_name,
                "Feature_Set" : fs_label,
                "Features"    : ", ".join(fs_cols),
                **overall,
            })
            trained_pipelines[(ml_name, fs_label)] = (pipe, fs_cols)

            # Per-variant scores on test set
            for var in VALID_VARIANTS:
                mask_var = X_te["variant"] == var
                if mask_var.sum() == 0:
                    continue
                X_var = X_te[mask_var][fs_cols]
                y_var = y_te[mask_var]
                y_pred_var = pipe.predict(X_var)
                vm = compute_metrics(y_var, y_pred_var)
                per_variant_results.append({
                    "Variant"      : var,
                    "ML_Model"     : ml_name,
                    "Feature_Set"  : fs_label,
                    "Test_Records" : int(mask_var.sum()),
                    **vm,
                })

    return overall_results, per_variant_results, trained_pipelines, X_tr, X_te, y_tr, y_te


# ─── Best model selection & saving ───────────────────────────────────────────

def select_and_save_best(overall_results, trained_pipelines, df, X_te, y_te):
    ov_df = pd.DataFrame(overall_results)

    # Best = lowest overall MAE (Feature Set D preferred, tie-break by MAE)
    best_row  = ov_df.loc[ov_df["MAE"].idxmin()]
    best_name = best_row["Model"]
    best_fs   = best_row["Feature_Set"]
    best_pipe, best_cols = trained_pipelines[(best_name, best_fs)]

    print(f"\nBest Combined Corolla Model:")
    print(f"  ML Model    : {best_name}")
    print(f"  Feature Set : {best_fs}  ({', '.join(best_cols)})")
    print(f"  MAE         : {best_row['MAE']:,.2f} LKR")
    print(f"  RMSE        : {best_row['RMSE']:,.2f} LKR")
    print(f"  R2_Score    : {best_row['R2_Score']:.4f}")

    # Save individual ML models (Feature Set D by default for each algo)
    model_filenames = {
        "Gradient Boosting"       : "gradient_boosting_regressor.pkl",
        "Random Forest Regressor" : "random_forest_regressor.pkl",
        "XGBRegressor"            : "xgbregressor.pkl",
    }
    preferred_fs = "D"  # Save the fullest feature set per algorithm
    for ml_name, fname in model_filenames.items():
        key = (ml_name, preferred_fs)
        if key in trained_pipelines:
            p, _ = trained_pipelines[key]
            joblib.dump(p, os.path.join(MODEL_DIR, fname))

    # Save best overall model
    joblib.dump(best_pipe, os.path.join(MODEL_DIR, "best_corolla_combined_model.pkl"))

    # ── Test predictions file ──
    X_te_full = df.loc[X_te.index, ["variant", "year_range", "transmission", "fuel_type"]].copy()
    X_te_full["actual_price_lkr"]    = y_te.values
    X_te_full["predicted_price_lkr"] = best_pipe.predict(X_te[best_cols])
    X_te_full["prediction_error_lkr"] = X_te_full["predicted_price_lkr"] - X_te_full["actual_price_lkr"]
    X_te_full.to_csv(os.path.join(OUTPUT_DIR, "corolla_test_predictions.csv"), index=False)

    # ── Variant predictions file (detailed) ──
    group_cols = ["variant", "year_range", "transmission", "fuel_type"]
    grp = df.groupby(group_cols).agg(
        observed_count       =("price_lkr", "count"),
        actual_mean_price_lkr=("price_lkr", "mean")
    ).reset_index()
    grp_input = grp[best_cols].copy()
    grp["predicted_fair_price_lkr"] = best_pipe.predict(grp_input)
    grp = grp.sort_values(["variant", "year_range", "transmission", "fuel_type"])
    grp.to_csv(os.path.join(OUTPUT_DIR, "corolla_variant_predictions.csv"), index=False)

    # ── Variant average predicted price summary ──
    avg_pred = grp.groupby("variant").apply(
        lambda g: pd.Series({
            "record_count"                    : int(g["observed_count"].sum()),
            "actual_mean_price_lkr"           : round(g["actual_mean_price_lkr"].mean(), 2),
            "average_predicted_fair_price_lkr": round(g["predicted_fair_price_lkr"].mean(), 2),
        })
    ).reset_index()
    avg_pred.to_csv(
        os.path.join(OUTPUT_DIR, "corolla_variant_average_predicted_price_summary.csv"), index=False
    )

    return best_name, best_fs, best_pipe, best_cols, ov_df, grp, avg_pred


# ─── Print final tables ───────────────────────────────────────────────────────

def print_overall_table(ov_df):
    print("\nOverall Corolla Model Performance Comparison:")
    cols = ["Model", "Feature_Set", "MAE", "MSE", "RMSE", "R2_Score"]
    # Format manually to avoid pandas display width cutting off headers
    rows = ov_df[cols].sort_values("MAE")
    header = f"  {'Model':<27} {'FS':>3}  {'MAE':>14}  {'MSE':>22}  {'RMSE':>14}  {'R2_Score':>9}"
    print(header)
    print("  " + "-" * 100)
    for _, r in rows.iterrows():
        print(f"  {r['Model']:<27} {r['Feature_Set']:>3}  {r['MAE']:>14,.2f}  {r['MSE']:>22,.2f}  {r['RMSE']:>14,.2f}  {r['R2_Score']:>9.4f}")


def print_per_variant_table(pv_df, best_name, best_fs):
    """Print only the best-model per-variant scores with a clear header."""
    print(f"\nPer-Variant Scores of Best Combined Corolla Model ({best_name} | Feature Set {best_fs}):")
    mask = (pv_df["ML_Model"] == best_name) & (pv_df["Feature_Set"] == best_fs)
    rows = pv_df[mask].sort_values("Variant")
    header = f"  {'Variant':<10} {'ML_Model':<27} {'FS':>3}  {'N':>4}  {'MAE':>14}  {'RMSE':>14}  {'R2_Score':>9}"
    print(header)
    print("  " + "-" * 90)
    for _, r in rows.iterrows():
        print(f"  {r['Variant']:<10} {r['ML_Model']:<27} {r['Feature_Set']:>3}  {int(r['Test_Records']):>4}  "
              f"{r['MAE']:>14,.2f}  {r['RMSE']:>14,.2f}  {r['R2_Score']:>9.4f}")


def print_variant_average_summary(avg_pred):
    """Print one-row-per-variant average predicted price summary."""
    print("\nAverage Predicted Fair Price by Corolla Variant:")
    header = f"  {'Variant':<10}  {'Records':>8}  {'Actual Mean (LKR)':>20}  {'Predicted Mean (LKR)':>22}"
    print(header)
    print("  " + "-" * 68)
    for _, r in avg_pred.iterrows():
        print(f"  {r['variant']:<10}  {int(r['record_count']):>8}  "
              f"{r['actual_mean_price_lkr']:>20,.2f}  {r['average_predicted_fair_price_lkr']:>22,.2f}")


def print_predicted_prices_table(grp):
    """Print detailed predicted fair prices grouped by variant/range/transmission/fuel_type."""
    print("\nPredicted Fair Prices by Corolla Variant / Year Range / Transmission / Fuel Type:")
    header = (f"  {'Variant':<10} {'Range':<11}  {'Transmission':<14} {'Fuel':<8}"
              f"  {'N':>4}  {'Actual Mean (LKR)':>20}  {'Predicted Fair Price (LKR)':>26}")
    print(header)
    print("  " + "-" * 98)
    current_variant = None
    for _, r in grp.iterrows():
        if r["variant"] != current_variant:
            if current_variant is not None:
                print("  " + "-" * 98)
            current_variant = r["variant"]
        print(f"  {r['variant']:<10} {r['year_range']:<11}  {r['transmission']:<14} {r['fuel_type']:<8}"
              f"  {int(r['observed_count']):>4}  {r['actual_mean_price_lkr']:>20,.2f}  "
              f"{r['predicted_fair_price_lkr']:>26,.2f}")


# ─── Cross-validation ─────────────────────────────────────────────────────────

def run_cross_validation(df, best_pipe, best_cols):
    """5-fold CV with variant stratification where possible."""
    X = df[best_cols]
    y = df["price_lkr"]

    try:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_results = cross_validate(
            best_pipe, X, y,
            cv=list(skf.split(X, df["variant"])),  # ← fix: list() not generator
            scoring={"mae": "neg_mean_absolute_error",
                     "rmse": "neg_root_mean_squared_error",
                     "r2":   "r2"},
            return_train_score=False
        )
        cv_df = pd.DataFrame({
            "Fold" : range(1, 6),
            "MAE"  : -cv_results["test_mae"],
            "RMSE" : -cv_results["test_rmse"],
            "R2"   : cv_results["test_r2"],
        })
        cv_df.to_csv(os.path.join(OUTPUT_DIR, "corolla_cross_validation_results.csv"), index=False)

        print("\n5-Fold Cross-Validation Results (Best Model):")
        print(cv_df.to_string(index=False))

        # ── Summary line for viva ──
        print(f"\n  CV Mean MAE  : {cv_df['MAE'].mean():>12,.2f} LKR  (±{cv_df['MAE'].std():,.2f})")
        print(f"  CV Mean RMSE : {cv_df['RMSE'].mean():>12,.2f} LKR  (±{cv_df['RMSE'].std():,.2f})")
        print(f"  CV Mean R²   : {cv_df['R2'].mean():>12.4f}       (±{cv_df['R2'].std():.4f})")

    except Exception as e:
        print(f"\n  (Cross-validation skipped: {e})")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ensure_dirs()
    df = load_and_filter_data()
    print_data_summary(df)
    save_price_summaries(df)

    overall_results, per_variant_results, trained_pipelines, X_tr, X_te, y_tr, y_te = train_all(df)

    ov_df = pd.DataFrame(overall_results)
    pv_df = pd.DataFrame(per_variant_results)

    # Save CSVs
    ov_df.to_csv(os.path.join(OUTPUT_DIR, "corolla_overall_model_comparison.csv"),     index=False)
    pv_df.to_csv(os.path.join(OUTPUT_DIR, "corolla_per_variant_model_comparison.csv"), index=False)

    # Feature set comparison (best MAE per set)
    fs_best = ov_df.groupby("Feature_Set")["MAE"].min().reset_index()
    fs_best.columns = ["Feature_Set", "Best_MAE"]
    fs_best.to_csv(os.path.join(OUTPUT_DIR, "corolla_feature_set_comparison.csv"), index=False)

    print_overall_table(ov_df)

    best_name, best_fs, best_pipe, best_cols, _, grp, avg_pred = select_and_save_best(
        overall_results, trained_pipelines, df, X_te, y_te
    )

    print_per_variant_table(pv_df, best_name, best_fs)
    print_variant_average_summary(avg_pred)
    print_predicted_prices_table(grp)
    run_cross_validation(df, best_pipe, best_cols)

    print("\nSuccess! Corolla models, comparisons, summaries, and predictions saved to:")
    print(f"  models/corolla_combined/")
    print(f"  outputs/corolla/")
    print()
    print("  Key output files:")
    print("    outputs/corolla/corolla_variant_predictions.csv          <- Detailed predicted fair prices")
    print("    outputs/corolla/corolla_variant_average_predicted_price_summary.csv <- Summary per variant")
    print("    outputs/corolla/corolla_overall_model_comparison.csv     <- All model MAE/RMSE/R2 results")
    print("    outputs/corolla/corolla_per_variant_model_comparison.csv <- Per-variant accuracy scores")


if __name__ == "__main__":
    main()
