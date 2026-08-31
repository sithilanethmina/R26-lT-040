import pandas as pd
import numpy as np
import joblib
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "models"))

def predict_laptop_price(
    brand=None, 
    model=None, 
    cpu=None, 
    generation=None, 
    ram_gb=None, 
    storage_gb=None, 
    storage_type=None, 
    gpu=None, 
    is_touchscreen=0, 
    location=None
):
    model_path = os.path.join(MODELS_DIR, "best_laptop_model.pkl")
    if not os.path.exists(model_path):
        print(f"[!] Error: Model not found at {model_path}. Please run train_model.py first.")
        return None
        
    bundle = joblib.load(model_path)
    model_obj = bundle['model']
    model_name = bundle.get('model_name', 'XGBoost')
    model_type = bundle.get('model_type', 'encoded')
    features = bundle['features']
    encoder = bundle.get('encoder')
    
    # Safe domain normalization for missing / null values
    brand_val = str(brand).capitalize() if brand else "Dell"
    model_val = str(model).title() if model else "Latitude"
    cpu_val = str(cpu) if cpu else "Core i5"
    gen_val = int(generation) if generation is not None else 10
    ram_val = float(ram_gb) if ram_gb is not None else 8.0
    storage_val = float(storage_gb) if storage_gb is not None else 256.0
    storage_type_val = str(storage_type).upper() if storage_type else "SSD"
    gpu_val = str(gpu) if gpu else "Integrated"  # Standard laptops without dedicated GPU are Integrated
    touch_val = int(bool(is_touchscreen))
    loc_val = str(location).capitalize() if location else "Colombo"
    
    input_dict = {
        'Brand_Cleaned': [brand_val],
        'Model_Cleaned': [model_val],
        'CPU_Cleaned': [cpu_val],
        'Storage_Type': [storage_type_val],
        'GPU_Tier': [gpu_val],
        'Location_Cleaned': [loc_val],
        'Generation_Cleaned': [gen_val],
        'RAM_GB': [ram_val],
        'Storage_Capacity_GB': [storage_val],
        'Is_Touchscreen': [touch_val]
    }
    
    df_input = pd.DataFrame(input_dict)[features]
    
    if model_type == 'encoded' and encoder is not None:
        cat_cols = bundle['cat_cols']
        df_input[cat_cols] = encoder.transform(df_input[cat_cols])
    elif model_type == 'lgb_cat':
        cat_cols = bundle['cat_cols']
        for c in cat_cols:
            df_input[c] = df_input[c].astype('category')
            
    pred_log = model_obj.predict(df_input)[0]
    predicted_lkr = float(np.expm1(pred_log))
    
    print("\n" + "=" * 50)
    print("LAPTOP FAIR MARKET ESTIMATOR (USED)")
    print("=" * 50)
    print(f"Brand         : {brand_val}")
    print(f"Model / Series: {model_val}")
    print(f"Processor     : {cpu_val} ({gen_val}th Gen)")
    print(f"Memory (RAM)  : {ram_val} GB")
    print(f"Storage       : {storage_val} GB {storage_type_val}")
    print(f"Graphics (GPU): {gpu_val}")
    print(f"Touchscreen   : {'Yes' if touch_val else 'No'}")
    print(f"Market Region : {loc_val}")
    print("-" * 50)
    print(f"Estimated Price: Rs {predicted_lkr:,.2f} LKR")
    print(f"Model Engine   : {model_name} (Trained on 4,050+ used listings)")
    print("=" * 50 + "\n")
    return predicted_lkr

def predict_tablet_price(
    brand=None,
    model=None,
    storage_gb=None,
    ram_gb=None,
    connectivity=None,
    screen_size=None,
    location=None
):
    model_path = os.path.join(MODELS_DIR, "best_tablet_model.pkl")
    if not os.path.exists(model_path):
        print(f"[!] Error: Model not found at {model_path}. Please run train_tablets_model.py first.")
        return None
        
    bundle = joblib.load(model_path)
    model_obj = bundle['model']
    model_name = bundle.get('model_name', 'CatBoost')
    model_type = bundle.get('model_type', 'native_cat')
    features = bundle['features']
    encoder = bundle.get('encoder')
    
    brand_val = str(brand).capitalize() if brand else "Apple"
    model_val = str(model).title() if model else "iPad"
    storage_val = float(storage_gb) if storage_gb is not None else 64.0
    ram_val = float(ram_gb) if ram_gb is not None else 4.0
    conn_val = str(connectivity) if connectivity else "WiFi / Standard"
    
    size_str = str(screen_size) if screen_size is not None else "10.0 Inch"
    if not "Inch" in size_str:
        size_str = f"{size_str} Inch"
        
    loc_val = str(location).capitalize() if location else "Colombo"
    
    input_dict = {
        'Brand_Cleaned': [brand_val],
        'Model_Cleaned': [model_val],
        'Storage_GB': [storage_val],
        'RAM_GB': [ram_val],
        'Connectivity_Cleaned': [conn_val],
        'Screen_Size': [size_str],
        'Location_Cleaned': [loc_val]
    }
    
    df_input = pd.DataFrame(input_dict)[features]
    
    if model_type == 'encoded' and encoder is not None:
        cat_cols = bundle['cat_cols']
        df_input[cat_cols] = encoder.transform(df_input[cat_cols])
    elif model_type == 'lgb_cat':
        cat_cols = bundle['cat_cols']
        for c in cat_cols:
            df_input[c] = df_input[c].astype('category')
            
    pred_log = model_obj.predict(df_input)[0]
    predicted_lkr = float(np.expm1(pred_log))
    
    print("\n" + "=" * 50)
    print("TABLET FAIR MARKET ESTIMATOR (USED)")
    print("=" * 50)
    print(f"Brand         : {brand_val}")
    print(f"Model / Series: {model_val}")
    print(f"Storage       : {storage_val} GB")
    print(f"RAM           : {ram_val} GB")
    print(f"Connectivity  : {conn_val}")
    print(f"Screen Size   : {size_str}")
    print(f"Market Region : {loc_val}")
    print("-" * 50)
    print(f"Estimated Price: Rs {predicted_lkr:,.2f} LKR")
    print(f"Model Engine   : {model_name} (Trained on 600+ used listings)")
    print("=" * 50 + "\n")
    return predicted_lkr

def predict_monitor_price(
    brand=None,
    size_inches=None,
    refresh_rate_hz=None,
    resolution=None,
    panel_type=None,
    is_curved=0,
    is_gaming=0,
    is_frameless=0,
    location=None
):
    model_path = os.path.join(MODELS_DIR, "best_monitor_model.pkl")
    if not os.path.exists(model_path):
        print(f"[!] Error: Model not found at {model_path}. Please run train_monitors_model.py first.")
        return None
        
    bundle = joblib.load(model_path)
    model_obj = bundle['model']
    model_name = bundle.get('model_name', 'LightGBM')
    model_type = bundle.get('model_type', 'lgb_cat')
    features = bundle['features']
    encoder = bundle.get('encoder')
    
    raw_brand = str(brand).upper() if brand else "DELL"
    brand_fmt = raw_brand if raw_brand in ["MSI", "AOC", "HP", "LG"] else raw_brand.capitalize()
    
    size_val = float(size_inches) if size_inches is not None else 24.0
    hz_val = float(refresh_rate_hz) if refresh_rate_hz is not None else 60.0
    res_val = str(resolution) if resolution else "1080p FHD"
    panel_val = str(panel_type) if panel_type else "Standard"
    curved_val = int(bool(is_curved))
    gaming_val = int(bool(is_gaming or hz_val >= 100))
    frameless_val = int(bool(is_frameless))
    loc_val = str(location).capitalize() if location else "Colombo"
    
    input_dict = {
        'Brand_Cleaned': [brand_fmt],
        'Resolution_Cleaned': [res_val],
        'Panel_Type': [panel_val],
        'Location_Cleaned': [loc_val],
        'Size_Inches': [size_val],
        'Refresh_Rate_Hz': [hz_val],
        'Is_Curved': [curved_val],
        'Is_Gaming': [gaming_val],
        'Is_Frameless': [frameless_val]
    }
    
    df_input = pd.DataFrame(input_dict)[features]
    
    if model_type == 'encoded' and encoder is not None:
        cat_cols = bundle['cat_cols']
        df_input[cat_cols] = encoder.transform(df_input[cat_cols])
    elif model_type == 'lgb_cat':
        cat_cols = bundle['cat_cols']
        for c in cat_cols:
            df_input[c] = df_input[c].astype('category')
            
    pred_log = model_obj.predict(df_input)[0]
    predicted_lkr = float(np.expm1(pred_log))
    
    print("\n" + "=" * 50)
    print("MONITOR FAIR MARKET ESTIMATOR (USED)")
    print("=" * 50)
    print(f"Brand         : {brand_fmt}")
    print(f"Size          : {size_val} Inches")
    print(f"Refresh Rate  : {hz_val} Hz")
    print(f"Resolution    : {res_val}")
    print(f"Panel Type    : {panel_val}")
    print(f"Curved        : {'Yes' if curved_val else 'No'}")
    print(f"Gaming Display: {'Yes' if gaming_val else 'No'}")
    print(f"Frameless     : {'Yes' if frameless_val else 'No'}")
    print(f"Market Region : {loc_val}")
    print("-" * 50)
    print(f"Estimated Price: Rs {predicted_lkr:,.2f} LKR")
    print(f"Model Engine   : {model_name} (Trained on 1,200+ used listings)")
    print("=" * 50 + "\n")
    return predicted_lkr

def predict_electronics_price(category="laptop", **kwargs):
    cat_lower = str(category).lower()
    if cat_lower == "laptop":
        return predict_laptop_price(**kwargs)
    elif cat_lower == "tablet":
        return predict_tablet_price(**kwargs)
    elif cat_lower == "monitor":
        return predict_monitor_price(**kwargs)
    else:
        raise ValueError(f"Unknown electronics category: {category}. Supported: 'laptop', 'tablet', 'monitor'")

if __name__ == "__main__":
    print("Testing with None/Null parameters to verify graceful domain handling:\n")
    
    # Test 1: Laptop with gpu=None
    predict_laptop_price(
        brand="Lenovo",
        model="ThinkPad T480",
        cpu="Core i5",
        generation=8,
        ram_gb=16,
        storage_gb=256,
        gpu=None,  # Handled safely as Integrated
        location="Galle"
    )
    
    # Test 2: Tablet with screen_size=None
    predict_tablet_price(
        brand="Samsung",
        model="Galaxy Tab S6",
        storage_gb=128,
        ram_gb=6,
        screen_size=None,
        location="Colombo"
    )
    
    # Test 3: Monitor with panel_type=None
    predict_monitor_price(
        brand="MSI",
        size_inches=24,
        refresh_rate_hz=144,
        panel_type=None,
        is_gaming=1,
        location="Kandy"
    )
