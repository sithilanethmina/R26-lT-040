import os
import argparse
import pandas as pd
import joblib
from utils import normalize_variant, get_year_range

def predict_price(year_range, variant_raw):
    """Predicts the price of a Toyota Aqua based on year range and variant."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'best_model.pkl')
    
    if not os.path.exists(model_path):
        print("Error: Best model not found. Please run 'python src/train.py' first.")
        return
        
    # Load models/best_model.pkl
    model = joblib.load(model_path)
    
    # Normalize variant
    variant = normalize_variant(variant_raw)
    
    # Validate year_range
    valid_years = ["2012-2014", "2015-2017"]
    if year_range not in valid_years:
        print(f"Warning: '{year_range}' is not a standard year range used in training.")
        print(f"Standard ranges are: {valid_years}")
        
    print(f"\nPredicting for:")
    print(f"Year Range: {year_range}")
    print(f"Variant   : {variant} (normalized from '{variant_raw}')")
    
    # Predict price
    input_data = pd.DataFrame({
        'year_range': [year_range],
        'variant': [variant]
    })
    
    try:
        predicted_price = model.predict(input_data)[0]
        # Print the predicted Toyota Aqua price in Sri Lankan Rupees
        print(f"\n=> Predicted Fair Price: LKR {predicted_price:,.2f}\n")
    except Exception as e:
        print(f"Prediction failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Toyota Aqua Price")
    parser.add_argument("--year_range", required=True, help="Year range (e.g., '2012-2014')")
    parser.add_argument("--variant", required=True, help="Vehicle variant (e.g., 'S Grade')")
    
    args = parser.parse_args()
    predict_price(args.year_range, args.variant)
