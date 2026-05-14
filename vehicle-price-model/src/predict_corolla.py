"""
Toyota Corolla Combined Model - Prediction Script
=================================================
Loads the best saved combined Corolla model and predicts price.

Usage examples:
  python src/predict_corolla.py --model_year 2005 --variant 121 --transmission Automatic --fuel_type Petrol
  python src/predict_corolla.py --model_year 2008 --variant 141 --transmission Automatic --fuel_type Petrol
  python src/predict_corolla.py --model_year 1998 --variant AE110 --transmission Manual --fuel_type Petrol
  python src/predict_corolla.py --model_year 1986 --variant "DX/KE72" --transmission Manual --fuel_type Petrol
"""

import os
import sys
import argparse
import pandas as pd
import joblib

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "corolla_combined", "best_corolla_combined_model.pkl")

VALID_VARIANTS      = ["121", "141", "AE110", "DX/KE72"]
VALID_YEAR_RANGE    = (1970, 2030)
VALID_TRANSMISSIONS = ["Automatic", "Manual"]
VALID_FUEL_TYPES    = ["Petrol", "Diesel", "Hybrid"]


# ─── Validation helpers ───────────────────────────────────────────────────────

def validate_variant(variant_str):
    """Check that variant is one of the four supported Corolla types."""
    if variant_str not in VALID_VARIANTS:
        print(f"\n  ERROR: Unsupported variant '{variant_str}'.")
        print(f"  Supported variants: {', '.join(VALID_VARIANTS)}")
        sys.exit(1)
    return variant_str


def validate_year(year_str):
    try:
        year = int(year_str)
    except ValueError:
        print(f"\n  ERROR: model_year must be a number. Got: '{year_str}'")
        sys.exit(1)
    if not (VALID_YEAR_RANGE[0] <= year <= VALID_YEAR_RANGE[1]):
        print(f"\n  WARNING: model_year {year} looks unusual. Expected between "
              f"{VALID_YEAR_RANGE[0]} and {VALID_YEAR_RANGE[1]}.")
    return year


def validate_transmission(trans_str):
    """Case-insensitive match for transmission."""
    for valid in VALID_TRANSMISSIONS:
        if trans_str.strip().lower() == valid.lower():
            return valid
    print(f"\n  ERROR: Unsupported transmission '{trans_str}'.")
    print(f"  Supported: {', '.join(VALID_TRANSMISSIONS)}")
    sys.exit(1)


def validate_fuel_type(fuel_str):
    """Case-insensitive match for fuel type."""
    for valid in VALID_FUEL_TYPES:
        if fuel_str.strip().lower() == valid.lower():
            return valid
    print(f"\n  ERROR: Unsupported fuel_type '{fuel_str}'.")
    print(f"  Supported: {', '.join(VALID_FUEL_TYPES)}")
    sys.exit(1)


# ─── Prediction ───────────────────────────────────────────────────────────────

def predict(model_year, variant, transmission, fuel_type):
    # Load the best combined Corolla model
    if not os.path.exists(MODEL_PATH):
        print("\n  ERROR: Trained model not found.")
        print(f"  Expected at: {MODEL_PATH}")
        print("  Please run 'python src/train_corolla_combined.py' first.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)

    # Build input DataFrame (include all possible feature columns;
    # the pipeline will use only the ones it was trained with)
    input_df = pd.DataFrame([{
        "model_year"   : model_year,
        "variant"      : variant,
        "transmission" : transmission,
        "fuel_type"    : fuel_type,
    }])

    # The pipeline stores which columns it expects via the ColumnTransformer
    try:
        predicted_price = model.predict(input_df)[0]
    except Exception as e:
        # If the model was trained with a subset of features, try selecting them
        try:
            feature_names = (
                model.named_steps["preprocessor"]
                .feature_names_in_
            )
            predicted_price = model.predict(input_df[list(feature_names)])[0]
        except Exception:
            print(f"\n  ERROR during prediction: {e}")
            sys.exit(1)

    # ─── Print result ───
    print("\n" + "=" * 50)
    print("  Toyota Corolla Price Prediction")
    print("=" * 50)
    print(f"  Variant      : {variant}")
    print(f"  Model Year   : {model_year}")
    print(f"  Transmission : {transmission}")
    print(f"  Fuel Type    : {fuel_type}")
    print("-" * 50)
    print(f"  => Predicted Fair Price : LKR {predicted_price:,.2f}")
    print("=" * 50 + "\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Predict Toyota Corolla price using the combined model."
    )
    parser.add_argument("--model_year",   required=True,
                        help="Model year (e.g., 2005)")
    parser.add_argument("--variant",      required=True,
                        help="Corolla variant: 121 | 141 | AE110 | DX/KE72")
    parser.add_argument("--transmission", required=True,
                        help="Transmission: Automatic | Manual")
    parser.add_argument("--fuel_type",    required=True,
                        help="Fuel type: Petrol | Diesel | Hybrid")
    args = parser.parse_args()

    model_year   = validate_year(args.model_year)
    variant      = validate_variant(args.variant)
    transmission = validate_transmission(args.transmission)
    fuel_type    = validate_fuel_type(args.fuel_type)

    predict(model_year, variant, transmission, fuel_type)


if __name__ == "__main__":
    main()
