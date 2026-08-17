"""
Conformal Prediction Validation & Backtesting Script
====================================================
Evaluates Empirical Coverage Ratio (ECR), Mean Relative Interval Width (MRIW),
and verdict distributions on the 20% holdout test set to validate the fair market price range.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Add src to path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gpu_price_predictor.pipeline import (
    FEATURE_COLUMNS_V2,
    calculate_fair_market_range,
    get_fairness_verdict,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_CSV = ROOT / "data" / "final" / "gpu_enriched_dataset.csv"
MODEL_PATH = ROOT / "artifacts" / "gpu_price_model_v2.joblib"

RANDOM_STATE = 42

def main():
    log.info("Starting Backtesting & Conformal Validation on Holdout Set...")

    if not DATA_CSV.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError("Dataset or model artifact missing.")

    df = pd.read_csv(DATA_CSV)
    artifact = joblib.load(MODEL_PATH)
    
    best_model_name = artifact.get("best_model_name", "xgboost")
    best_model = artifact["all_models"][best_model_name]
    feature_cols = artifact.get("feature_columns", FEATURE_COLUMNS_V2)
    calibration_data = artifact.get("conformal_calibration", {})

    from sklearn.model_selection import train_test_split
    price_quintile = pd.qcut(df["log_price_lkr"], q=5, labels=False, duplicates="drop")
    _, test_df = train_test_split(df, test_size=0.20, random_state=RANDOM_STATE, stratify=price_quintile)

    log.info(f"Loaded {len(test_df)} test listings for backtesting.")

    X_test = test_df[feature_cols].copy()
    y_test_log = test_df["log_price_lkr"].to_numpy(dtype=float)
    y_test_lkr = test_df["price_lkr"].to_numpy(dtype=float)

    preds_log = best_model.predict(X_test)

    inside_count = 0
    below_count = 0
    above_count = 0
    relative_widths = []
    verdicts = []

    for i in range(len(test_df)):
        pred_log = float(preds_log[i])
        actual_price = float(y_test_lkr[i])
        
        range_info = calculate_fair_market_range(
            predicted_log_price=pred_log,
            sample_count=20,
            calibration_data=calibration_data,
            confidence_level="90%"
        )
        
        lower = range_info["lower_price_lkr"]
        upper = range_info["upper_price_lkr"]
        
        verdict = get_fairness_verdict(actual_price, lower, upper)
        verdicts.append(verdict["verdict_code"])

        width = upper - lower
        relative_widths.append(width / actual_price)

        if lower <= actual_price <= upper:
            inside_count += 1
        elif actual_price < lower:
            below_count += 1
        else:
            above_count += 1

    ecr = (inside_count / len(test_df)) * 100.0
    mriw = float(np.mean(relative_widths)) * 100.0
    median_riw = float(np.median(relative_widths)) * 100.0

    print("\n" + "="*60)
    print("      CONFORMAL RANGE BACKTESTING RESULTS (HOLD-OUT SET)")
    print("="*60)
    print(f"Total Holdout Listings Tested : {len(test_df)}")
    print(f"Empirical Coverage Ratio (ECR): {ecr:.2f}%  (Target Nominal: 90.0%)")
    print(f"Mean Relative Interval Width  : {mriw:.2f}% of price")
    print(f"Median Relative Interval Width: {median_riw:.2f}% of price")
    print("-" * 60)
    print("Listing Distribution vs. Fair Range:")
    print(f"  • Within Expected Range     : {inside_count} ({inside_count/len(test_df)*100:.1f}%)")
    print(f"  • Below Typical Market Range: {below_count} ({below_count/len(test_df)*100:.1f}%)")
    print(f"  • Above Typical Market Range: {above_count} ({above_count/len(test_df)*100:.1f}%)")
    print("="*60 + "\n")

    if 85.0 <= ecr <= 93.0:
        print("[OK] VALIDATION SUCCESS: Conformal Fair Market Range achieved target coverage!")
    else:
        print("[WARNING] Empirical coverage deviates from nominal target.")

if __name__ == "__main__":
    main()
