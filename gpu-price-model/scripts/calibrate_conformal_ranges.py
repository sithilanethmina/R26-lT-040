"""
Conformal Prediction Calibration Script
=======================================
Computes out-of-fold log residuals across 5 cross-validation folds on the GPU enriched dataset,
calculates tier-stratified conformal prediction quantiles for 90% coverage,
and saves the calibration metrics into the model artifact.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# Add src to path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gpu_price_predictor.pipeline import (
    FEATURE_COLUMNS_V2,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_CSV = ROOT / "data" / "final" / "gpu_enriched_dataset.csv"
MODEL_PATH = ROOT / "artifacts" / "gpu_price_model_v2.joblib"
SUMMARY_PATH = ROOT / "artifacts" / "training_summary_v2.json"

RANDOM_STATE = 42

def determine_price_tier(price_lkr: float) -> str:
    if price_lkr < 35000:
        return "entry"
    elif price_lkr <= 110000:
        return "mid"
    else:
        return "high"

def main():
    log.info("Starting Conformal Prediction Calibration...")
    
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_CSV}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")

    df = pd.read_csv(DATA_CSV)
    artifact = joblib.load(MODEL_PATH)
    
    best_model_name = artifact.get("best_model_name", "xgboost")
    best_model = artifact["all_models"][best_model_name]
    feature_cols = artifact.get("feature_columns", FEATURE_COLUMNS_V2)

    X = df[feature_cols].copy()
    y_log = df["log_price_lkr"].to_numpy(dtype=float)
    y_lkr = np.expm1(y_log)

    log.info("Computing 5-fold Out-Of-Fold predictions for conformal calibration...")
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_preds_log = np.zeros_like(y_log)

    from sklearn.base import clone
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log)):
        X_tr, y_tr = X.iloc[train_idx], y_log[train_idx]
        X_va = X.iloc[val_idx]
        
        fold_model = clone(best_model)
        fold_model.fit(X_tr, y_tr)
        oof_preds_log[val_idx] = fold_model.predict(X_va)

    # Compute raw log residuals (actual - predicted) and absolute nonconformity scores
    residuals_log = y_log - oof_preds_log
    abs_residuals_log = np.abs(residuals_log)

    # 1. Global Conformal Quantiles
    # For 90% coverage, quantile level = 0.90 * (1 + 1/N)
    n_samples = len(y_log)
    q_level_90 = min(1.0, 0.90 * (1.0 + 1.0 / n_samples))
    q_level_85 = min(1.0, 0.85 * (1.0 + 1.0 / n_samples))
    q_level_95 = min(1.0, 0.95 * (1.0 + 1.0 / n_samples))

    global_q90 = float(np.quantile(abs_residuals_log, q_level_90))
    global_q85 = float(np.quantile(abs_residuals_log, q_level_85))
    global_q95 = float(np.quantile(abs_residuals_log, q_level_95))

    signed_q_lower_90 = float(np.quantile(residuals_log, 0.05))
    signed_q_upper_90 = float(np.quantile(residuals_log, 0.95))

    log.info(f"Global 90% Conformal Log-Quantile (|residual|): {global_q90:.4f}")
    log.info(f"Global Signed Residual Quantiles (5%, 95%): ({signed_q_lower_90:.4f}, {signed_q_upper_90:.4f})")

    # 2. Tier-Stratified Calibration
    tiers = [determine_price_tier(p) for p in y_lkr]
    tier_series = pd.Series(tiers, index=df.index)

    tier_calibration = {}
    for tier_name in ["entry", "mid", "high"]:
        mask = tier_series == tier_name
        n_tier = int(mask.sum())
        if n_tier > 0:
            tier_abs_res = abs_residuals_log[mask]
            tier_signed_res = residuals_log[mask]
            
            q_tier_90 = float(np.quantile(tier_abs_res, min(1.0, 0.90 * (1.0 + 1.0 / n_tier))))
            q_tier_lower_90 = float(np.quantile(tier_signed_res, 0.05))
            q_tier_upper_90 = float(np.quantile(tier_signed_res, 0.95))
            
            cover_global = np.mean((y_lkr[mask] >= np.expm1(oof_preds_log[mask] - global_q90)) & 
                                   (y_lkr[mask] <= np.expm1(oof_preds_log[mask] + global_q90))) * 100
            cover_tier = np.mean((y_lkr[mask] >= np.expm1(oof_preds_log[mask] - q_tier_90)) & 
                                 (y_lkr[mask] <= np.expm1(oof_preds_log[mask] + q_tier_90))) * 100
            
            log.info(f"Tier [{tier_name.upper()}] (n={n_tier}): q90={q_tier_90:.4f} | Coverage Global={cover_global:.1f}%, Tier={cover_tier:.1f}%")
            
            tier_calibration[tier_name] = {
                "n_samples": n_tier,
                "q90": round(q_tier_90, 5),
                "q_lower_5pct": round(q_tier_lower_90, 5),
                "q_upper_95pct": round(q_tier_upper_90, 5),
                "mean_abs_residual": round(float(np.mean(tier_abs_res)), 5),
                "coverage_pct": round(float(cover_tier), 2)
            }

    # Evaluate Overall Coverage Ratio
    oof_lower = np.expm1(oof_preds_log - global_q90)
    oof_upper = np.expm1(oof_preds_log + global_q90)
    overall_coverage = float(np.mean((y_lkr >= oof_lower) & (y_lkr <= oof_upper)) * 100)
    mriw = float(np.mean((oof_upper - oof_lower) / y_lkr))

    log.info(f"Overall OOF Empirical Coverage: {overall_coverage:.2f}% (Target: 90.0%)")
    log.info(f"Overall Mean Relative Interval Width (MRIW): {mriw * 100:.2f}% of price")

    calibration_bundle = {
        "confidence_level": 0.90,
        "global": {
            "q90": round(global_q90, 5),
            "q85": round(global_q85, 5),
            "q95": round(global_q95, 5),
            "q_signed_lower_5pct": round(signed_q_lower_90, 5),
            "q_signed_upper_95pct": round(signed_q_upper_90, 5),
            "overall_coverage_pct": round(overall_coverage, 2),
            "mriw": round(mriw, 4),
        },
        "tiers": tier_calibration
    }

    # Update Joblib Artifact
    artifact["conformal_calibration"] = calibration_bundle
    joblib.dump(artifact, MODEL_PATH)
    log.info(f"Updated joblib artifact with conformal calibration parameters -> {MODEL_PATH}")

    # Update summary JSON
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
        summary_data["conformal_calibration"] = calibration_bundle
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        log.info(f"Updated summary JSON -> {SUMMARY_PATH}")

    print("SUCCESS: Conformal Prediction Calibration completed.")

if __name__ == "__main__":
    main()
