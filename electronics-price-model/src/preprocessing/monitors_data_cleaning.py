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
        if 3000 <= val <= 500000:
            return val
        return np.nan
    except ValueError:
        return np.nan

def clean_size(size_val, title_str=""):
    title = str(title_str)
    size_str = str(size_val) if not pd.isna(size_val) else ""
    full_text = f"{size_str} {title}"
    
    match = re.search(r'(\d{2}(?:\.\d)?)\s*(?:["\']|inch|\-inch|\s*in\b)', full_text, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        if 15.0 <= val <= 55.0:
            return val
            
    num_match = re.search(r'\b(17|18|19|20|21|22|23|24|25|27|28|29|32|34|43|49)\b', full_text)
    if num_match:
        return float(num_match.group(1))
        
    return 24.0  # Standard default 24-inch

def clean_hz(hz_val, title_str=""):
    title = str(title_str)
    hz_str = str(hz_val) if not pd.isna(hz_val) else ""
    full_text = f"{hz_str} {title}"
    
    match = re.search(r'(\d{2,3})\s*hz\b', full_text, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        if 50 <= val <= 500:
            return val
    return 60.0  # Standard default 60Hz

def clean_resolution(res_val, title_str="", size=24.0):
    full_text = f"{str(res_val)} {str(title_str)}".lower()
    
    if any(k in full_text for k in ["4k", "uhd", "2160p", "3840x2160"]):
        return "4K UHD"
    elif any(k in full_text for k in ["2k", "qhd", "1440p", "wqhd", "2560x1440"]):
        return "2K QHD"
    elif any(k in full_text for k in ["fhd", "1080p", "full hd", "1920x1080", "frameless"]):
        return "1080p FHD"
    elif any(k in full_text for k in ["hd", "720p", "1366x768", "1600x900", "1440x900"]):
        return "HD"
    return "1080p FHD" if size >= 21.5 else "HD"

def clean_brand(brand_val, title_str=""):
    title_upper = str(title_str).upper()
    brand_upper = str(brand_val).strip().upper() if not pd.isna(brand_val) else ""
    
    brands = [
        "SAMSUNG", "DELL", "HP", "LG", "ASUS", "ACER", "BENQ", "VIEWSONIC", 
        "MSI", "AOC", "GIGABYTE", "PHILIPS", "LENOVO", "XIAOMI", "REDMI", 
        "PROLINK", "ARMAGGEDDON", "HUAWEI", "APPLE", "FUJITSU", "EPSON", "NEC"
    ]
    for b in brands:
        if b in brand_upper or b in title_upper:
            if b in ["MSI", "AOC", "HP", "LG", "NEC"]:
                return b
            return b.capitalize()
    return "Other"

def clean_panel(panel_val, title_str=""):
    full_text = f"{str(panel_val)} {str(title_str)}".lower()
    if "ips" in full_text: return "IPS"
    if "oled" in full_text: return "OLED"
    if " va " in full_text or "va panel" in full_text: return "VA"
    return "Standard"

def clean_location(loc_str):
    if pd.isna(loc_str): return "Colombo"
    loc = str(loc_str).strip()
    top_districts = ["Colombo", "Gampaha", "Kandy", "Kalutara", "Kurunegala", "Galle", "Matara", "Batticaloa", "Jaffna", "Anuradhapura", "Ratnapura", "Badulla"]
    for d in top_districts:
        if d.lower() in loc.lower():
            return d
    return "Other"

def process_monitors_dataset(input_csv, output_csv):
    print("=" * 65)
    print("MONITOR DATA PREPROCESSING & FEATURE ENGINEERING PIPELINE")
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
    print(f"[*] Filtered Price Outliers (Rs 3,000 - 500,000): {len(df):,} valid rows")
    
    # 2. Extract Specifications
    df['Brand_Cleaned'] = df.apply(lambda r: clean_brand(r.get('brand', r.get('Brand', '')), r['title']), axis=1)
    df['Size_Inches'] = df.apply(lambda r: clean_size(r.get('size', ''), r['title']), axis=1)
    df['Refresh_Rate_Hz'] = df.apply(lambda r: clean_hz(r.get('refresh_rate', ''), r['title']), axis=1)
    df['Resolution_Cleaned'] = df.apply(lambda r: clean_resolution(r.get('resolution', ''), r['title'], r['Size_Inches']), axis=1)
    df['Panel_Type'] = df.apply(lambda r: clean_panel(r.get('panel_type', ''), r['title']), axis=1)
    
    df['Is_Curved'] = df.apply(lambda r: 1 if "curved" in str(r['title']).lower() or r.get('is_curved') == True else 0, axis=1)
    df['Is_Gaming'] = df.apply(lambda r: 1 if "gaming" in str(r['title']).lower() or r['Refresh_Rate_Hz'] >= 100 or r.get('is_gaming') == True else 0, axis=1)
    df['Is_Frameless'] = df.apply(lambda r: 1 if any(f in str(r['title']).lower() for f in ["frameless", "borderless", "bezel"]) or r.get('is_frameless') == True else 0, axis=1)
    
    df['Condition_Cleaned'] = df.get('condition', df.get('Condition', 'Used')).apply(lambda x: 'Brand New' if 'Brand New' in str(x) or 'New' in str(x) else 'Used')
    df['Location_Cleaned'] = df.get('location', df.get('Location', 'Colombo')).apply(clean_location)
    
    feature_cols = [
        'title',
        'Brand_Cleaned',
        'Size_Inches',
        'Refresh_Rate_Hz',
        'Resolution_Cleaned',
        'Panel_Type',
        'Is_Curved',
        'Is_Gaming',
        'Is_Frameless',
        'Condition_Cleaned',
        'Location_Cleaned',
        'Price_Cleaned'
    ]
    
    cleaned_df = df[feature_cols].copy()
    
    # Save cleaned file
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    cleaned_df.to_csv(output_csv, index=False)
    
    print("\n" + "=" * 65)
    print(f"MONITOR PREPROCESSING SUCCESSFUL!")
    print(f"Cleaned Dataset Shape: {cleaned_df.shape[0]:,} rows, {cleaned_df.shape[1]} columns")
    print(f"Saved to: {output_csv}")
    print("=" * 65)
    
    print("\nBrand Distribution:")
    print(cleaned_df['Brand_Cleaned'].value_counts().head(10))
    print("\nResolution Distribution:")
    print(cleaned_df['Resolution_Cleaned'].value_counts())
    print("\nRefresh Rate Distribution:")
    print(cleaned_df['Refresh_Rate_Hz'].value_counts().head(8))
    print("\nPrice Summary (LKR):")
    print(cleaned_df['Price_Cleaned'].describe().apply(lambda x: f"{x:,.2f}"))
    return cleaned_df

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "raw", "ikman_monitors_all.csv"))
    output_path = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "processed", "monitors_cleaned.csv"))
    
    process_monitors_dataset(input_path, output_path)
