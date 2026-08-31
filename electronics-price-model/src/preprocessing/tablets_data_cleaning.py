import pandas as pd
import numpy as np
import os
import re

def clean_price(price_val):
    if pd.isna(price_val):
        return np.nan
    clean_str = str(price_val).replace('Rs', '').replace(',', '').strip()
    try:
        val = float(clean_str)
        if 4000 <= val <= 650000:
            return val
        return np.nan
    except ValueError:
        return np.nan

def clean_storage(storage_val, title_str=""):
    storage_str = str(storage_val).upper() if not pd.isna(storage_val) else ""
    title_str = str(title_str).upper()
    full_text = f"{storage_str} {title_str}"
    
    # 1. Look for TB (e.g. 1TB)
    tb_match = re.search(r'\b(\d+)\s*TB\b', full_text)
    if tb_match:
        val = int(tb_match.group(1))
        if val in [1, 2]:
            return float(val * 1024)
            
    # 2. Look for explicit GB storage patterns
    gb_match = re.search(r'(\d+)\s*(?:GB|ROM|STORAGE|INTERNAL)\b', full_text)
    if gb_match:
        val = int(gb_match.group(1))
        if val in [16, 32, 64, 128, 256, 512, 1024]:
            return float(val)
            
    # 3. Dual storage pattern e.g. 8/256GB or 8+128GB
    dual_match = re.search(r'\d+\s*[\/\+]\s*(\d+)\s*(?:GB)?', full_text)
    if dual_match:
        val = int(dual_match.group(1))
        if val in [16, 32, 64, 128, 256, 512, 1024]:
            return float(val)
            
    # Fallback to standard 64GB
    return 64.0

def clean_ram(ram_val, title_str="", storage_gb=64.0):
    ram_str = str(ram_val).upper() if not pd.isna(ram_val) else ""
    title_str = str(title_str).upper()
    full_text = f"{ram_str} {title_str}"
    
    # 1. Dual RAM pattern e.g. 8/256GB or 4+64GB
    dual_match = re.search(r'(\d+)\s*[\/\+]\s*\d+\s*(?:GB)?', full_text)
    if dual_match:
        val = int(dual_match.group(1))
        if val in [2, 3, 4, 6, 8, 12, 16]:
            return float(val)
            
    # 2. Explicit RAM match
    ram_match = re.search(r'(\d+)\s*(?:GB|MB)?\s*RAM', full_text)
    if not ram_match:
        ram_match = re.search(r'RAM\s*(\d+)', full_text)
    if ram_match:
        val = int(ram_match.group(1))
        if val in [2, 3, 4, 6, 8, 12, 16]:
            return float(val)
            
    # Smart default based on storage
    if storage_gb >= 256: return 8.0
    if storage_gb >= 128: return 6.0
    if storage_gb >= 64: return 4.0
    return 3.0

def clean_brand(brand_val, title_str=""):
    title_upper = str(title_str).upper()
    brand_upper = str(brand_val).strip().upper() if not pd.isna(brand_val) else ""
    
    brands = [
        "APPLE", "SAMSUNG", "XIAOMI", "REDMI", "HONOR", "HUAWEI", 
        "LENOVO", "AMAZON", "MICROSOFT", "BLACKVIEW", "TECLAST", "REALME", "NOKIA", "CHUWI"
    ]
    for b in brands:
        if b in brand_upper or b in title_upper:
            return b.capitalize()
    if "IPAD" in title_upper: return "Apple"
    if "GALAXY TAB" in title_upper: return "Samsung"
    if "SURFACE" in title_upper: return "Microsoft"
    if "KINDLE" in title_upper or "FIRE HD" in title_upper: return "Amazon"
    
    return "Other"

def clean_model(brand, title_str):
    title = str(title_str).upper()
    brand = str(brand).upper()
    
    models = {
        'APPLE': ['IPAD PRO', 'IPAD AIR', 'IPAD MINI', 'IPAD 10', 'IPAD 9', 'IPAD 8', 'IPAD 7', 'IPAD 6', 'IPAD 5', 'IPAD'],
        'SAMSUNG': ['GALAXY TAB S10', 'GALAXY TAB S9', 'GALAXY TAB S8', 'GALAXY TAB S7', 'GALAXY TAB S6', 'GALAXY TAB A9', 'GALAXY TAB A8', 'GALAXY TAB A7', 'GALAXY TAB A11', 'GALAXY TAB'],
        'XIAOMI': ['PAD 6', 'PAD 5', 'REDMI PAD PRO', 'REDMI PAD SE', 'REDMI PAD', 'MI PAD'],
        'HONOR': ['PAD X9', 'PAD X8', 'PAD 9', 'PAD 8', 'HONOR PAD'],
        'HUAWEI': ['MATEPAD PRO', 'MATEPAD 11', 'MATEPAD SE', 'MATEPAD', 'MEDIAPAD'],
        'LENOVO': ['TAB P12', 'TAB P11', 'TAB M10', 'TAB M9', 'TAB M8', 'YOGA TAB'],
        'MICROSOFT': ['SURFACE PRO', 'SURFACE GO', 'SURFACE'],
        'AMAZON': ['KINDLE PAPERWHITE', 'KINDLE OASIS', 'KINDLE', 'FIRE HD 10', 'FIRE HD 8', 'FIRE 7'],
        'BLACKVIEW': ['TAB 60', 'TAB 50', 'TAB 30', 'TAB 16', 'TAB 15', 'ZENO 10', 'ZENO 5']
    }
    
    if brand in models:
        for m in models[brand]:
            if m in title:
                return m.title()
                
    return "Standard Tablet"

def clean_connectivity(conn_str, title_str=""):
    title_lower = str(title_str).lower()
    conn = str(conn_str).strip()
    
    if "5g" in title_lower or "5g" in conn.lower():
        return "5G + WiFi"
    if any(c in title_lower for c in ["4g", "lte", "cellular", "sim"]):
        return "4G LTE / SIM"
    if "wifi only" in title_lower or "wifi only" in conn.lower():
        return "WiFi Only"
    return "WiFi / Standard"

def clean_size(size_val, title_str=""):
    title = str(title_str)
    size_str = str(size_val) if not pd.isna(size_val) else ""
    full_text = f"{size_str} {title}"
    
    match = re.search(r'(\d{1,2}(?:\.\d)?)\s*(?:["\']|inch|\-inch)', full_text, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        if 6.0 <= val <= 16.0:
            return f"{val:g} Inch"
    return "10.0 Inch"  # Standard median tablet size

def clean_location(loc_str):
    if pd.isna(loc_str): return "Colombo"
    loc = str(loc_str).strip()
    top_districts = ["Colombo", "Gampaha", "Kandy", "Kalutara", "Kurunegala", "Galle", "Matara", "Batticaloa", "Jaffna", "Anuradhapura", "Ratnapura", "Badulla"]
    for d in top_districts:
        if d.lower() in loc.lower():
            return d
    return "Other"

def process_tablets_dataset(input_csv, output_csv):
    print("=" * 65)
    print("TABLET DATA PREPROCESSING & FEATURE ENGINEERING PIPELINE")
    print("=" * 65)
    print(f"Loading raw dataset from: {input_csv}")
    
    if not os.path.exists(input_csv):
        print(f"[!] Error: Raw file not found at {input_csv}")
        return
        
    df = pd.read_csv(input_csv)
    print(f"[*] Raw Records Loaded: {len(df):,}")
    
    # 1. Price Outlier Cleaning
    price_col = 'price' if 'price' in df.columns else 'Price'
    df['Price_Cleaned'] = df[price_col].apply(clean_price)
    df = df.dropna(subset=['Price_Cleaned'])
    print(f"[*] Filtered Price Outliers (Rs 4,000 - 650,000): {len(df):,} valid rows")
    
    # 2. Extract Specifications
    df['Brand_Cleaned'] = df.apply(lambda r: clean_brand(r.get('brand', r.get('Brand', '')), r['title']), axis=1)
    df['Model_Cleaned'] = df.apply(lambda r: clean_model(r['Brand_Cleaned'], r['title']), axis=1)
    df['Storage_GB'] = df.apply(lambda r: clean_storage(r.get('storage', r.get('Storage', '')), r['title']), axis=1)
    df['RAM_GB'] = df.apply(lambda r: clean_ram(r.get('ram', r.get('RAM', '')), r['title'], r['Storage_GB']), axis=1)
    df['Connectivity_Cleaned'] = df.apply(lambda r: clean_connectivity(r.get('connectivity', ''), r['title']), axis=1)
    df['Screen_Size'] = df.apply(lambda r: clean_size(r.get('size', ''), r['title']), axis=1)
    df['Condition_Cleaned'] = df.get('condition', df.get('Condition', 'Used')).apply(lambda x: 'Brand New' if 'Brand New' in str(x) or 'New' in str(x) else 'Used')
    df['Location_Cleaned'] = df.get('location', df.get('Location', 'Colombo')).apply(clean_location)
    
    feature_cols = [
        'title',
        'Brand_Cleaned',
        'Model_Cleaned',
        'Storage_GB',
        'RAM_GB',
        'Connectivity_Cleaned',
        'Screen_Size',
        'Condition_Cleaned',
        'Location_Cleaned',
        'Price_Cleaned'
    ]
    
    cleaned_df = df[feature_cols].copy()
    
    # Save cleaned file
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    cleaned_df.to_csv(output_csv, index=False)
    
    print("\n" + "=" * 65)
    print(f"TABLET PREPROCESSING SUCCESSFUL!")
    print(f"Cleaned Dataset Shape: {cleaned_df.shape[0]:,} rows, {cleaned_df.shape[1]} columns")
    print(f"Saved to: {output_csv}")
    print("=" * 65)
    
    print("\nBrand Distribution:")
    print(cleaned_df['Brand_Cleaned'].value_counts())
    print("\nCondition Distribution:")
    print(cleaned_df['Condition_Cleaned'].value_counts())
    print("\nPrice Summary (LKR):")
    print(cleaned_df['Price_Cleaned'].describe().apply(lambda x: f"{x:,.2f}"))
    return cleaned_df

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "raw", "ikman_tablets_all.csv"))
    output_path = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "processed", "tablets_cleaned.csv"))
    
    process_tablets_dataset(input_path, output_path)
