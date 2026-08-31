import pandas as pd
import numpy as np
import joblib
import os

def predict_laptop_price(
    brand="Dell", 
    model="Latitude", 
    cpu="Core i5", 
    generation=11, 
    ram_gb=16, 
    storage_gb=256, 
    storage_type="SSD", 
    gpu="Integrated", 
    is_touchscreen=0, 
    condition="Used", 
    location="Colombo"
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(script_dir, "..", "..", "models", "best_laptop_model.pkl"))
    
    if not os.path.exists(model_path):
        print(f"[!] Error: Model not found at {model_path}. Please run train_model.py first.")
        return
        
    bundle = joblib.load(model_path)
    model_obj = bundle['model']
    model_name = bundle.get('model_name', 'CatBoost')
    model_type = bundle.get('model_type', 'native_cat')
    features = bundle['features']
    encoder = bundle.get('encoder')
    
    # Prepare input dictionary
    input_dict = {
        'Brand_Cleaned': [str(brand).capitalize()],
        'Model_Cleaned': [str(model).title()],
        'CPU_Cleaned': [str(cpu)],
        'Storage_Type': [str(storage_type).upper()],
        'GPU_Tier': [str(gpu)],
        'Condition_Cleaned': ['Brand New' if 'NEW' in str(condition).upper() else 'Used'],
        'Location_Cleaned': [str(location).capitalize()],
        'Generation_Cleaned': [int(generation)],
        'RAM_GB': [float(ram_gb)],
        'Storage_Capacity_GB': [float(storage_gb)],
        'Is_Touchscreen': [int(is_touchscreen)]
    }
    
    df_input = pd.DataFrame(input_dict)[features]
    
    # Pre-process if encoded
    if model_type == 'encoded' and encoder is not None:
        cat_cols = bundle['cat_cols']
        df_input[cat_cols] = encoder.transform(df_input[cat_cols])
    elif model_type == 'lgb_cat':
        cat_cols = bundle['cat_cols']
        for c in cat_cols:
            df_input[c] = df_input[c].astype('category')
            
    # Predict in log scale and convert back to LKR
    pred_log = model_obj.predict(df_input)[0]
    predicted_lkr = float(np.expm1(pred_log))
    
    print("\n" + "=" * 50)
    print("LAPTOP SPECIFICATIONS ESTIMATOR")
    print("=" * 50)
    print(f"Brand         : {brand}")
    print(f"Model / Series: {model}")
    print(f"Processor     : {cpu} ({generation}th Gen)")
    print(f"Memory (RAM)  : {ram_gb} GB")
    print(f"Storage       : {storage_gb} GB {storage_type.upper()}")
    print(f"Graphics (GPU): {gpu}")
    print(f"Touchscreen   : {'Yes' if is_touchscreen else 'No'}")
    print(f"Condition     : {condition}")
    print(f"Market Region : {location}")
    print("-" * 50)
    print(f"Estimated Price: Rs {predicted_lkr:,.2f} LKR")
    print(f"Model Engine   : {model_name} (Trained on 8,600+ listings)")
    print("=" * 50 + "\n")
    return predicted_lkr

if __name__ == "__main__":
    # Test 1: Mid-range Workstation
    predict_laptop_price(
        brand="Dell",
        model="Latitude",
        cpu="Core i5",
        generation=11,
        ram_gb=16,
        storage_gb=256,
        storage_type="SSD",
        condition="Used",
        location="Colombo"
    )
    
    # Test 2: High-end Gaming Laptop
    predict_laptop_price(
        brand="Asus",
        model="ROG",
        cpu="Core i7",
        generation=12,
        ram_gb=16,
        storage_gb=512,
        storage_type="SSD",
        gpu="RTX 30-Series",
        condition="Used",
        location="Colombo"
    )
