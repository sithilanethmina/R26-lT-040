import json
import logging
import re
import sys
from pathlib import Path
from datetime import datetime
import joblib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path for local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from gpu_price_predictor.pipeline import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    NUMERIC_SPEC_COLUMNS,
    CATEGORICAL_SPEC_COLUMNS,
    UNKNOWN,
    normalize_model,
    evaluate_predictions,
    baseline_median_by_model,
    build_segment_metrics,
    build_stratify_labels,
    derive_series_family,
    derive_model_number,
    derive_ti_flag,
    compute_iqr_summary,
    TrainingDatasetBundle,
)
from gpu_price_predictor.model_training import (
    RANDOM_STATE,
    build_candidates,
    fit_and_evaluate_model,
    build_linear_diagnostics,
    build_xgboost_feature_importance,
)

# Configuration
DATA_DIR = PROJECT_ROOT / "data" / "final"
TRAINING_DATA_V1_PATH = DATA_DIR / "training_data_v1.json"
TRAINING_DATA_V2_PATH = DATA_DIR / "training_data_v2.json"
TRAINING_DATA_V3_PATH = DATA_DIR / "training_data_v3.json"
TRUSTED_SPECS_PATH = DATA_DIR / "trusted_gpu_specs.json"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def parse_to_mb(value: any) -> float:
    """Safely parse a string like '4 GB' or '512 MB' into a float in MB."""
    if value is None or str(value).lower() == "unknown" or str(value).strip() == "":
        return np.nan
    
    text = str(value).upper()
    match = re.search(r"([\d\.]+)", text)
    if not match:
        return np.nan
    
    val = float(match.group(1))
    
    if "KB" in text:
        return val / 1024.0
    if "GB" in text:
        return val * 1024.0
    # Default is MB
    return val


def parse_raw_number(value: any) -> float:
    """Safely extract the first number from a string."""
    if value is None or str(value).lower() == "unknown" or str(value).strip() == "":
        return np.nan
    match = re.search(r"([\d\.]+)", str(value))
    return float(match.group(1)) if match else np.nan


def load_consolidated_dataset() -> tuple[TrainingDatasetBundle, dict]:
    """
    Loads training_data.json and training_data_old.json and enriches them with trusted_gpu_specs.json.
    Maps the new JSON structure to the internal FEATURE_COLUMNS.
    """
    raw_training = []
    
    # Load v1 training data
    if TRAINING_DATA_V1_PATH.exists():
        print(f"[*] Loading training data: {TRAINING_DATA_V1_PATH.name}")
        with open(TRAINING_DATA_V1_PATH, "r", encoding="utf-8") as f:
            v1_data = json.load(f)
            raw_training.extend(v1_data)
    
    # Load v2 training data
    if TRAINING_DATA_V2_PATH.exists():
        print(f"[*] Loading training data: {TRAINING_DATA_V2_PATH.name}")
        with open(TRAINING_DATA_V2_PATH, "r", encoding="utf-8") as f:
            v2_data = json.load(f)
            raw_training.extend(v2_data)

    # Load v3 training data
    if TRAINING_DATA_V3_PATH.exists():
        print(f"[*] Loading training data: {TRAINING_DATA_V3_PATH.name}")
        with open(TRAINING_DATA_V3_PATH, "r", encoding="utf-8") as f:
            v3_data = json.load(f)
            raw_training.extend(v3_data)
            
    if not raw_training:
        print("[!] No training data found. Using empty list.")
    
    print(f"[*] Loading trusted specs: {TRUSTED_SPECS_PATH.name}")
    with open(TRUSTED_SPECS_PATH, "r", encoding="utf-8") as f:
        trusted_specs = json.load(f)

    # Use the pipeline's normalize_model for robust matching
    spec_lookup = {normalize_model(k): v for k, v in trusted_specs.items()}
    
    enriched_records = []
    missing_specs_count = 0
    
    processed_spec_lookup = {} # To store processed specs for the app

    for item in raw_training:
        model_name = item.get("Extracted_Model", "")
        norm_name = normalize_model(model_name)
        specs = spec_lookup.get(norm_name, {})
        
        if not specs:
            missing_specs_count += 1
        
        # Build the feature row
        record = {
            "source": "Consolidated", 
            "model": model_name,
            "price_lkr": item.get("Price_LKR"),
            "brand": item.get("Brand", UNKNOWN),
            "vram_gb": item.get("VRAM_GB"),
        }
        
        # Map Kaggle/TechPowerUp specs to our feature names
        record["manufacturer"] = specs.get("Manufacturer", UNKNOWN)
        record["vendor"] = specs.get("Brand", UNKNOWN)
        record["release_year"] = extract_year(specs.get("Graphics Card__Release Date"))
        record["memory_size_mb"] = parse_to_mb(specs.get("Memory__Memory Size"))
        record["memory_type"] = specs.get("Memory__Memory Type", UNKNOWN)
        record["buswidth_bits"] = parse_raw_number(specs.get("Memory__Memory Bus"))
        record["gpu_clockspeed_mhz"] = parse_raw_number(specs.get("Clock Speeds__GPU Clock"))
        record["memory_clockspeed_mhz"] = parse_raw_number(specs.get("Clock Speeds__Memory Clock"))
        record["max_bandwidth_mb_s"] = parse_raw_number(specs.get("Memory__Bandwidth"))
        record["process_size_nm"] = parse_raw_number(specs.get("Graphics Processor__Process Size"))
        record["transistors_million"] = parse_raw_number(specs.get("Graphics Processor__Transistors"))
        record["external_power"] = specs.get("Board Design__Power Connectors", UNKNOWN)
        
        # Shaders/Stream Processors mapping
        shader_keys = ["Render Config__Shading Units", "Render Config__Pixel Shaders", "Top__PIXEL SHADERS", "Top__CORES"]
        shaders = np.nan
        for sk in shader_keys:
            val = parse_raw_number(specs.get(sk))
            if not np.isnan(val) and val > 0:
                shaders = val
                break
        record["shader_cores_or_stream_processors"] = shaders
        record["boost_clock_mhz"] = parse_raw_number(specs.get("Clock Speeds__Boost Clock"))
        
        # Derived features
        record["series_family"] = derive_series_family(model_name)
        record["model_number"] = derive_model_number(model_name)
        record["ti_variant"] = derive_ti_flag(model_name)
        record["vram_gb_missing"] = "No" if record["vram_gb"] is not None else "Yes"
        
        enriched_records.append(record)
        
        # Keep track of specs for the inference artifact
        if norm_name not in processed_spec_lookup:
            processed_spec_lookup[norm_name] = {k: v for k, v in record.items() if k in FEATURE_COLUMNS or k == "vendor"}

    if missing_specs_count > 0:
        print(f"   [!] Warning: {missing_specs_count} records could not be matched with trusted specs.")

    df = pd.DataFrame(enriched_records)
    df["vram_gb"] = df["vram_gb"].fillna(df["memory_size_mb"] / 1024.0)
    df = df.dropna(subset=["price_lkr"])
    
    for col in NUMERIC_SPEC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isna().all():
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(df[col].median())

    iqr_res = compute_iqr_summary(df, "price_lkr")
    df = df[(df["price_lkr"] >= iqr_res["lower_bound"]) & (df["price_lkr"] <= iqr_res["upper_bound"])]
    
    bundle = TrainingDatasetBundle(
        dataset=df.reset_index(drop=True),
        unmatched_models=pd.DataFrame(),
        alias_table=pd.DataFrame(),
        spec_table=pd.DataFrame(),
        iqr_summary=iqr_res,
    )
    return bundle, processed_spec_lookup


def extract_year(value: any) -> float:
    if value is None or str(value).lower() == "unknown":
        return np.nan
    match = re.search(r"(\d{4})", str(value))
    return float(match.group(1)) if match else np.nan


def train_and_select_model():
    """Execute the training pipeline on consolidated JSON data."""
    print("Starting GPU Price Prediction Model Training (JSON Pipeline)")

    bundle, spec_lookup = load_consolidated_dataset()
    dataset = bundle.dataset
    print(f"Final dataset: {len(dataset)} enriched records")

    stratify_labels = build_stratify_labels(dataset)
    
    train_df, test_df = train_test_split(
        dataset,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify_labels,
    )
    print(f"Training set: {len(train_df)} records | Test set: {len(test_df)} records")

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=float)
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    evaluation = {}
    candidate_pipelines = {}
    best_name = None
    best_pipeline = None
    best_mae = float("inf")

    baseline_preds = baseline_median_by_model(train_df, test_df)
    evaluation["baseline_median_by_model"] = {
        "status": "baseline",
        "metrics": evaluate_predictions(y_test, baseline_preds),
        "segment_metrics": build_segment_metrics(test_df, y_test, baseline_preds),
        "cv_summary": {"status": "baseline_no_cv"},
        "target_strategy": "raw",
    }

    for name, candidate in build_candidates().items():
        print(f"Training candidate: {name}...")
        result = fit_and_evaluate_model(
            name=name,
            candidate=candidate,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            test_df=test_df,
        )
        evaluation[name] = {
            "status": "trained",
            "metrics": result["selected_metrics"],
            "segment_metrics": result["selected_segment_metrics"],
            "cv_summary": result["selected_cv_summary"],
            "target_strategy": result["selected_strategy"],
            "target_strategy_comparison": result["target_strategy_comparison"],
        }
        candidate_pipelines[name] = result["selected_pipeline"]

        selection_mae = result["selected_metrics"]["mae_lkr"]
        print(f"  - Overall MAE: LKR {selection_mae:,.0f}")

        if selection_mae < best_mae:
            best_mae = selection_mae
            best_name = name
            best_pipeline = result["selected_pipeline"]

    print(f"\nBest model: {best_name} (MAE: LKR {best_mae:,.0f})")

    linear_coefficients_df, linear_diagnostics = build_linear_diagnostics(candidate_pipelines["linear_regression"], x_train)
    linear_coefficients_df.to_csv(ARTIFACTS_DIR / "linear_coefficients.csv", index=False)
    xgboost_importance_df = build_xgboost_feature_importance(candidate_pipelines["xgboost"])
    xgboost_importance_df.to_csv(ARTIFACTS_DIR / "xgboost_importance.csv", index=False)

    model_artifact = {
        "best_model": best_name,
        "selected_model": best_name,
        "all_pipelines": candidate_pipelines,
        "evaluation": evaluation,
        "spec_lookup": spec_lookup,
        "supported_values": {
            "model": sorted(dataset["model"].unique().tolist()),
            "manufacturer": sorted(dataset["manufacturer"].unique().tolist()),
            "source": sorted(dataset["source"].unique().tolist()),
        },
        "defaults": {
            "model": dataset["model"].mode()[0] if not dataset.empty else "",
            "vram_gb": dataset["vram_gb"].median() if not dataset.empty else 4.0,
        },
        "benchmark_results": {
            "models_compared": list(candidate_pipelines.keys()),
            "best_mae": best_mae,
        },
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "trained_at_utc": datetime.now().isoformat(),
        "selection_metric": "MAE (LKR)",
        "candidate_models": list(candidate_pipelines.keys()),
    }
    
    joblib.dump(model_artifact, ARTIFACTS_DIR / "gpu_price_model.joblib")
    dataset.to_csv(ARTIFACTS_DIR / "gpu_training_dataset_enriched.csv", index=False)

    summary = {
        "trained_at": datetime.now().isoformat(),
        "best_model": best_name,
        "best_mae": best_mae,
        "dataset_size": len(dataset),
        "evaluation": evaluation,
        "iqr_summary": bundle.iqr_summary,
        "linear_diagnostics": linear_diagnostics,
        "features": FEATURE_COLUMNS,
    }
    with open(ARTIFACTS_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, default=str)
    
    print(f"Training complete. Artifacts saved to: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    try:
        train_and_select_model()
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
