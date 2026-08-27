#!/usr/bin/env python
"""
GPU Price Predictor — Phase 3: Fit Condition & Warranty Residual Correction Model
================================================================================
Regresses price residuals (log(actual) - log(predicted_specs)) on extracted
condition & warranty tags to derive empirical, statistically defensible coefficients.

Saves output to: artifacts/condition_correction_coefficients.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gpu_price_predictor.pipeline import (
    build_inference_feature_frame,
    normalize_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TAGGED_JSON = PROJECT_ROOT / "data" / "final" / "condition_tags.json"
ENRICHED_CSV = PROJECT_ROOT / "data" / "final" / "gpu_enriched_dataset.csv"
MODEL_V2_PATH = ARTIFACTS_DIR / "gpu_price_model_v2.joblib"
COEFFICIENTS_OUT = ARTIFACTS_DIR / "condition_correction_coefficients.json"

TAG_FEATURE_NAMES = [
    "warranty_months",
    "needs_repair",
    "urgent_sale",
    "tested_working",
    "good_condition",
    "brand_new",
    "price_negotiable",
    "is_shop",
]


def load_model_and_data():
    if not MODEL_V2_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_V2_PATH}")
    if not TAGGED_JSON.exists():
        raise FileNotFoundError(f"Tagged listings not found: {TAGGED_JSON}")

    log.info("Loading model artifact...")
    artifact = joblib.load(MODEL_V2_PATH)
    best_model = artifact["best_model"]
    feature_cols = artifact["feature_columns"]

    log.info("Loading enriched dataset for spec lookups...")
    enriched_df = pd.read_csv(ENRICHED_CSV) if ENRICHED_CSV.exists() else None

    with open(TAGGED_JSON, "r", encoding="utf-8") as f:
        tagged_records = json.load(f)

    return best_model, feature_cols, enriched_df, tagged_records


def calculate_residuals(best_model, feature_cols, enriched_df, tagged_records):
    log.info("Computing predictions and residuals across %d records...", len(tagged_records))

    # Memoize inference feature frames by (model, vram, brand)
    feature_cache: dict[tuple, pd.DataFrame] = {}

    rows = []
    skipped = 0

    for item in tagged_records:
        price = item.get("Price_LKR")
        model = item.get("Extracted_Model")
        if not price or not model:
            skipped += 1
            continue

        try:
            p_val = float(price)
            if p_val < 3000:
                skipped += 1
                continue
        except (ValueError, TypeError):
            skipped += 1
            continue

        vram = item.get("VRAM_GB")
        try:
            vram_val = float(vram) if vram else 4.0
        except (ValueError, TypeError):
            vram_val = 4.0

        brand = item.get("Brand") or "Any"

        cache_key = (normalize_model(model), vram_val, str(brand).upper())
        if cache_key not in feature_cache:
            try:
                df_inf = build_inference_feature_frame(
                    model_name=model,
                    vram_gb=vram_val,
                    brand=brand,
                    enriched_df=enriched_df,
                    feature_columns=feature_cols,
                )
                feature_cache[cache_key] = df_inf
            except Exception:
                feature_cache[cache_key] = None

        df_inf = feature_cache[cache_key]
        if df_inf is None:
            skipped += 1
            continue

        tags = item.get("condition_tags", {})
        row_dict = {
            "price_lkr": p_val,
            "actual_log_price": float(np.log1p(p_val)),
            "warranty_months": float(tags.get("warranty_months", 0.0)),
            "needs_repair": 1.0 if tags.get("needs_repair") else 0.0,
            "urgent_sale": 1.0 if tags.get("urgent_sale") else 0.0,
            "tested_working": 1.0 if tags.get("tested_working") else 0.0,
            "good_condition": 1.0 if tags.get("good_condition") else 0.0,
            "brand_new": 1.0 if tags.get("brand_new") else 0.0,
            "price_negotiable": 1.0 if tags.get("price_negotiable") else 0.0,
            "is_shop": 1.0 if tags.get("is_shop") else 0.0,
            "_df_inf": df_inf,
        }
        rows.append(row_dict)

    if not rows:
        raise ValueError("No valid records found for residual calculation.")

    # Batch prediction
    X_all_inference = pd.concat([r["_df_inf"] for r in rows], ignore_index=True)
    preds_log = best_model.predict(X_all_inference)

    for i, r in enumerate(rows):
        pred_log = float(preds_log[i])
        r["pred_log_price"] = pred_log
        r["residual"] = r["actual_log_price"] - pred_log
        del r["_df_inf"]

    log.info("Successfully calculated residuals for %d valid listings (skipped %d)", len(rows), skipped)
    return pd.DataFrame(rows)


def fit_residual_model(df_res: pd.DataFrame):
    log.info("Fitting Ridge regression on price residuals...")

    X = df_res[TAG_FEATURE_NAMES].values
    y = df_res["residual"].values

    # Fit Ridge with mild regularization to ensure stability
    reg = Ridge(alpha=1.0, fit_intercept=True, random_state=42)
    reg.fit(X, y)

    y_pred_residual = reg.predict(X)
    r2 = float(r2_score(y, y_pred_residual))

    # Evaluate accuracy improvement on original LKR price
    unadjusted_pred_lkr = np.expm1(df_res["pred_log_price"].values)
    adjusted_log_pred = df_res["pred_log_price"].values + y_pred_residual
    adjusted_pred_lkr = np.expm1(adjusted_log_pred)
    actual_lkr = df_res["price_lkr"].values

    mae_unadjusted = float(mean_absolute_error(actual_lkr, unadjusted_pred_lkr))
    mae_adjusted = float(mean_absolute_error(actual_lkr, adjusted_pred_lkr))
    mape_unadjusted = float(np.mean(np.abs((actual_lkr - unadjusted_pred_lkr) / actual_lkr)) * 100)
    mape_adjusted = float(np.mean(np.abs((actual_lkr - adjusted_pred_lkr) / actual_lkr)) * 100)

    # Standard errors and t-stats estimation (OLS / Bayesian approximation)
    n = len(X)
    p = X.shape[1] + 1
    residuals = y - y_pred_residual
    s2 = np.sum(residuals ** 2) / max(1, (n - p))
    X_design = np.hstack([np.ones((n, 1)), X])
    cov_matrix = np.linalg.pinv(X_design.T @ X_design + np.eye(p) * 1.0) * s2
    std_errs = np.sqrt(np.diagonal(cov_matrix))

    coefficients_dict = {}
    coef_table = []

    # Intercept
    intercept_val = float(reg.intercept_)
    coefficients_dict["intercept"] = round(intercept_val, 4)

    for i, name in enumerate(TAG_FEATURE_NAMES):
        coef_val = float(reg.coef_[i])
        se_val = float(std_errs[i + 1])
        t_stat = coef_val / se_val if se_val > 0 else 0.0
        pct_effect = (np.exp(coef_val) - 1.0) * 100.0

        coefficients_dict[name] = round(coef_val, 4)
        coef_table.append({
            "feature": name,
            "coef_beta": round(coef_val, 4),
            "std_err": round(se_val, 4),
            "t_stat": round(t_stat, 2),
            "pct_multiplier": f"{pct_effect:+.1f}%",
        })

    # Save artifact
    output_payload = {
        "version": "v1.0",
        "description": "Hedonic condition correction coefficients estimated from GPU market residuals",
        "features": TAG_FEATURE_NAMES,
        "intercept": coefficients_dict["intercept"],
        "coefficients": {k: v for k, v in coefficients_dict.items() if k != "intercept"},
        "metrics": {
            "r2_on_residuals": round(r2, 4),
            "n_samples": n,
            "mae_before_correction_lkr": round(mae_unadjusted, 0),
            "mae_after_correction_lkr": round(mae_adjusted, 0),
            "mape_before_correction": round(mape_unadjusted, 2),
            "mape_after_correction": round(mape_adjusted, 2),
            "mape_reduction_pct": round(mape_unadjusted - mape_adjusted, 2),
        },
        "statistical_summary": coef_table,
    }

    with open(COEFFICIENTS_OUT, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    log.info("[SUCCESS] Saved condition correction coefficients to %s", COEFFICIENTS_OUT.relative_to(PROJECT_ROOT))

    # Print reporting table
    print("\n" + "=" * 78)
    print(f"{'Condition Tag':<22} | {'Coefficient (Beta)':<18} | {'Std Error':<10} | {'Effect (%)':<12}")
    print("=" * 78)
    print(f"{'Intercept (Baseline)':<22} | {intercept_val:>+16.4f}  | {std_errs[0]:>8.4f} | {'Baseline':<12}")
    for row in coef_table:
        print(f"{row['feature']:<22} | {row['coef_beta']:>+16.4f}  | {row['std_err']:>8.4f} | {row['pct_multiplier']:>10}")
    print("=" * 78)
    print(f"Residual R2: {r2:.4f} across N={n} real marketplace listings")
    print(f"Overall Model MAPE: {mape_unadjusted:.2f}% -> {mape_adjusted:.2f}% (Improved by {mape_unadjusted - mape_adjusted:+.2f}%)")
    print(f"Overall Model MAE:  LKR {mae_unadjusted:,.0f} -> LKR {mae_adjusted:,.0f}")
    print("=" * 78 + "\n")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 70)
    print("  GPU Price Predictor -- Fit Condition Residual Correction Model")
    print("=" * 70 + "\n")

    best_model, feature_cols, enriched_df, tagged_records = load_model_and_data()
    df_res = calculate_residuals(best_model, feature_cols, enriched_df, tagged_records)
    fit_residual_model(df_res)


if __name__ == "__main__":
    main()
