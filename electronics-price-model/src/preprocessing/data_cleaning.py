import pandas as pd
import numpy as np
import re
import os

def clean_price(price_val):
    if pd.isna(price_val):
        return np.nan
    clean_str = str(price_val).replace('Rs', '').replace(',', '').strip()
    try:
        val = float(clean_str)
        # Valid realistic price bounds for laptops (15k LKR to 1.5M LKR)
        if 15000 <= val <= 1500000:
            return val
        return np.nan
    except ValueError:
        return np.nan

def clean_ram(ram_val, title_str=""):
    ram_str = str(ram_val).upper() if not pd.isna(ram_val) else ""
    title_str = str(title_str).upper()
    
    # 1. Try from RAM column
    match = re.search(r'(\d+)\s*(GB|MB)?', ram_str)
    if match:
        val = float(match.group(1))
        if val in [2, 4, 6, 8, 12, 16, 24, 32, 64, 128]:
            return val
            
    # 2. Try from Title
    match_title = re.search(r'(\d+)\s*(?:GB|MB)\s*RAM', title_str)
    if not match_title:
        match_title = re.search(r'RAM\s*(\d+)\s*(?:GB|MB)?', title_str)
    if not match_title:
        match_title = re.search(r'(\d+)\s*GB\s*(?:DDR\d|RAM|MEMORY)', title_str)
    if match_title:
        val = float(match_title.group(1))
        if val in [2, 4, 6, 8, 12, 16, 24, 32, 64, 128]:
            return val
            
    return np.nan

def clean_storage(storage_val, title_str=""):
    storage_str = str(storage_val).upper() if not pd.isna(storage_val) else ""
    title_str = str(title_str).upper()
    full_text = f"{storage_str} {title_str}"
    
    # Capacity extraction
    capacity = np.nan
    match = re.search(r'(\d+)\s*(GB|TB)?\s*(?:HDD|SSD|STORAGE|NVME|M\.2|EMMC)', full_text)
    if match:
        val = float(match.group(1))
        unit = match.group(2)
        if val <= 4 or unit == 'TB':
            capacity = val * 1024
        elif val in [64, 128, 256, 512, 1000, 1024, 2000, 2048]:
            capacity = val
            
    # Fallback to standalone numbers
    if pd.isna(capacity):
        num_match = re.search(r'\b(128|256|512|1024|1TB|2TB)\b', full_text)
        if num_match:
            v_str = num_match.group(1)
            capacity = 1024.0 if '1TB' in v_str else (2048.0 if '2TB' in v_str else float(v_str))

    # Storage type determination
    storage_type = "SSD"  # modern default
    if any(k in full_text for k in ["NVME", "M.2", "SSD", "PCIE"]):
        storage_type = "SSD"
    elif "HDD" in full_text or "HARD DISK" in full_text:
        storage_type = "HDD"
        
    return capacity, storage_type

def clean_brand(brand_val, title_str=""):
    title_upper = str(title_str).upper()
    brand_upper = str(brand_val).strip().upper() if not pd.isna(brand_val) else ""
    
    brands_priority = [
        "APPLE", "DELL", "HP", "LENOVO", "ASUS", "ACER", "MSI", 
        "TOSHIBA", "SAMSUNG", "MICROSOFT", "RAZER", "FUJITSU", "HUAWEI"
    ]
    
    for b in brands_priority:
        if b in brand_upper or b in title_upper:
            return b.capitalize()
    if "MACBOOK" in title_upper or "IPAD" in title_upper:
        return "Apple"
    if "SURFACE" in title_upper:
        return "Microsoft"
        
    return "Other"

def clean_model(brand, title_str):
    title = str(title_str).upper()
    brand = str(brand).upper()
    
    models = {
        'DELL': ['LATITUDE', 'INSPIRON', 'VOSTRO', 'PRECISION', 'XPS', 'ALIENWARE', 'G15', 'G3', 'G5', 'G7', 'CHROMEBOOK'],
        'HP': ['ELITEBOOK', 'PROBOOK', 'PAVILION', 'SPECTRE', 'ENVY', 'VICTUS', 'OMEN', 'ZBOOK', '14S', '15S', 'NOTEBOOK', 'DRAGONFLY', 'ELITE'],
        'LENOVO': ['THINKPAD', 'IDEAPAD', 'LEGION', 'YOGA', 'LOQ', 'THINKBOOK', 'V15', 'V14', 'FLEX'],
        'ASUS': ['VIVOBOOK', 'ZENBOOK', 'ROG', 'TUF', 'EXPERTBOOK', 'ZEPHYRUS', 'STRIX'],
        'ACER': ['ASPIRE', 'NITRO', 'PREDATOR', 'SWIFT', 'TRAVELMATE', 'SPIN', 'EXTENSA'],
        'APPLE': ['MACBOOK PRO', 'MACBOOK AIR', 'MACBOOK'],
        'MSI': ['MODERN', 'STEALTH', 'KATANA', 'SWORD', 'CYBORG', 'BRAVO', 'GF63', 'PULSE', 'THIN', 'CREATOR'],
        'MICROSOFT': ['SURFACE PRO', 'SURFACE LAPTOP', 'SURFACE BOOK', 'SURFACE GO', 'SURFACE']
    }
    
    if brand in models:
        for m in models[brand]:
            if m in title:
                return m.title()
                
    return "Standard / Other"

def clean_cpu(title_str):
    title = str(title_str).upper()
    
    cpu_rules = [
        ('Core i9', r'\b(?:CORE\s*I9|I9\b)'),
        ('Core i7', r'\b(?:CORE\s*I7|I7\b)'),
        ('Core i5', r'\b(?:CORE\s*I5|I5\b)'),
        ('Core i3', r'\b(?:CORE\s*I3|I3\b)'),
        ('Core Ultra', r'\bCORE\s*ULTRA\b'),
        ('Ryzen 9', r'\bRYZEN\s*9\b'),
        ('Ryzen 7', r'\bRYZEN\s*7\b'),
        ('Ryzen 5', r'\bRYZEN\s*5\b'),
        ('Ryzen 3', r'\bRYZEN\s*3\b'),
        ('Apple M4', r'\bM4\b'),
        ('Apple M3', r'\bM3\b'),
        ('Apple M2', r'\bM2\b'),
        ('Apple M1', r'\bM1\b'),
        ('Celeron', r'\bCELERON\b'),
        ('Pentium', r'\bPENTIUM\b'),
        ('Quad Core', r'\bQUAD[\s\-]CORE\b'),
        ('Dual Core', r'\bDUAL[\s\-]CORE\b')
    ]
    
    for cpu_name, pattern in cpu_rules:
        if re.search(pattern, title):
            return cpu_name
            
    return "Other / Dual Core"

def clean_generation(title_str, cpu_name=""):
    title = str(title_str).upper()
    
    # Check for "Xth Gen"
    match = re.search(r'(\d+)(?:ST|ND|RD|TH)?\s*(?:GEN|GENERATION)', title)
    if match:
        gen = int(match.group(1))
        if 2 <= gen <= 15:
            return gen
            
    # Check model number pattern e.g., i5-1135G7 (11th Gen), i7-8550U (8th Gen), i5-7200U (7th Gen)
    model_match = re.search(r'I[3579][\s\-](1[0-4]|\d)\d{2,3}', title)
    if model_match:
        gen = int(model_match.group(1))
        if 2 <= gen <= 15:
            return gen
            
    # Apple M-series mapping to generation tiers
    if "Apple M4" in cpu_name: return 15
    if "Apple M3" in cpu_name: return 14
    if "Apple M2" in cpu_name: return 13
    if "Apple M1" in cpu_name: return 12
    
    # Year-based estimation for MacBooks
    year_match = re.search(r'\b(201[5-9]|202[0-6])\b', title)
    if year_match:
        year = int(year_match.group(1))
        if year >= 2023: return 14
        if year >= 2020: return 12
        if year >= 2018: return 8
        if year >= 2015: return 6
        
    return 8  # Median default generation for used market

def clean_gpu(title_str):
    title = str(title_str).lower()
    if re.search(r'rtx\s*40\d0', title): return 'RTX 40-Series'
    if re.search(r'rtx\s*30\d0', title): return 'RTX 30-Series'
    if re.search(r'rtx\s*20\d0', title): return 'RTX 20-Series'
    if re.search(r'gtx\s*(?:16\d0|10\d0|9\d0)', title): return 'GTX Series'
    if any(k in title for k in ['nvidia', 'geforce', 'radeon', 'dedicated', '4gb vga', '6gb vga', '8gb vga', 'graphics', 'quadro', 'rx ']):
        return 'Other Dedicated'
    return 'Integrated'

def clean_touchscreen(title_str):
    title = str(title_str).lower()
    return 1 if any(k in title for k in ['touch', 'x360', 'convertible', '2-in-1', '2in1', 'flip', 'yoga']) else 0

def clean_location(loc_str):
    if pd.isna(loc_str): return "Colombo"
    loc = str(loc_str).strip()
    top_districts = ["Colombo", "Gampaha", "Kandy", "Kalutara", "Kurunegala", "Galle", "Matara", "Batticaloa", "Jaffna", "Anuradhapura", "Ratnapura", "Badulla"]
    for d in top_districts:
        if d.lower() in loc.lower():
            return d
    return "Other"

def process_laptops_dataset(input_csv, output_csv):
    print("=" * 65)
    print("LAPTOP DATA PREPROCESSING & FEATURE ENGINEERING PIPELINE")
    print("=" * 65)
    print(f"Loading raw dataset from: {input_csv}")
    
    if not os.path.exists(input_csv):
        print(f"[!] Error: Raw file not found at {input_csv}")
        return
        
    df = pd.read_csv(input_csv)
    print(f"[*] Raw Records Loaded: {len(df):,}")
    
    # 1. Price Cleaning & Outlier Removal
    price_col = 'price' if 'price' in df.columns else 'Price'
    df['Price_Cleaned'] = df[price_col].apply(clean_price)
    initial_len = len(df)
    df = df.dropna(subset=['Price_Cleaned'])
    print(f"[*] Filtered Price Outliers (Rs 15,000 - 1.5M): {len(df):,} valid rows ({initial_len - len(df)} dropped)")
    
    # 2. Extract Specifications
    print("[*] Extracting and standardizing laptop features...")
    df['Brand_Cleaned'] = df.apply(lambda r: clean_brand(r.get('brand', r.get('Brand', '')), r['title']), axis=1)
    df['Model_Cleaned'] = df.apply(lambda r: clean_model(r['Brand_Cleaned'], r['title']), axis=1)
    df['CPU_Cleaned'] = df['title'].apply(clean_cpu)
    df['Generation_Cleaned'] = df.apply(lambda r: clean_generation(r['title'], r['CPU_Cleaned']), axis=1)
    
    # RAM Cleaning with smart imputation
    df['RAM_GB'] = df.apply(lambda r: clean_ram(r.get('ram', r.get('RAM', '')), r['title']), axis=1)
    # Smart RAM imputation based on CPU tier
    def impute_ram(row):
        if not pd.isna(row['RAM_GB']):
            return row['RAM_GB']
        cpu = str(row['CPU_Cleaned'])
        if any(c in cpu for c in ['Core i9', 'Core i7', 'Ryzen 7', 'Ryzen 9', 'Apple M']):
            return 16.0
        elif any(c in cpu for c in ['Celeron', 'Pentium', 'Dual Core']):
            return 4.0
        return 8.0  # standard median for i3/i5
    df['RAM_GB'] = df.apply(impute_ram, axis=1)
    
    # Storage Cleaning with smart imputation
    storage_tuples = df.apply(lambda r: clean_storage(r.get('storage', r.get('Storage', '')), r['title']), axis=1)
    df['Storage_Capacity_GB'] = storage_tuples.apply(lambda x: x[0])
    df['Storage_Type'] = storage_tuples.apply(lambda x: x[1])
    
    def impute_storage(row):
        if not pd.isna(row['Storage_Capacity_GB']):
            return row['Storage_Capacity_GB']
        cpu = str(row['CPU_Cleaned'])
        if any(c in cpu for c in ['Core i9', 'Core i7', 'Apple M']):
            return 512.0
        return 256.0
    df['Storage_Capacity_GB'] = df.apply(impute_storage, axis=1)
    
    # GPU, Touchscreen, Condition, and Location
    df['GPU_Tier'] = df['title'].apply(clean_gpu)
    df['Is_Touchscreen'] = df['title'].apply(clean_touchscreen)
    df['Condition_Cleaned'] = df.get('condition', df.get('Condition', 'Used')).apply(lambda x: 'Brand New' if 'Brand New' in str(x) or 'New' in str(x) else 'Used')
    df['Location_Cleaned'] = df.get('location', df.get('Location', 'Colombo')).apply(clean_location)
    
    # 3. Final Feature Selection
    feature_cols = [
        'title',
        'Brand_Cleaned',
        'Model_Cleaned',
        'CPU_Cleaned',
        'Generation_Cleaned',
        'RAM_GB',
        'Storage_Capacity_GB',
        'Storage_Type',
        'GPU_Tier',
        'Is_Touchscreen',
        'Condition_Cleaned',
        'Location_Cleaned',
        'Price_Cleaned'
    ]
    
    cleaned_df = df[feature_cols].copy()
    
    # Deduplication
    cleaned_df.drop_duplicates(subset=['Brand_Cleaned', 'Model_Cleaned', 'CPU_Cleaned', 'RAM_GB', 'Storage_Capacity_GB', 'Price_Cleaned', 'Location_Cleaned'], inplace=True)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    cleaned_df.to_csv(output_csv, index=False)
    
    print("\n" + "=" * 65)
    print(f"PREPROCESSING SUCCESSFUL!")
    print(f"Final Cleaned Dataset Shape: {cleaned_df.shape[0]:,} rows, {cleaned_df.shape[1]} columns")
    print(f"Saved to: {output_csv}")
    print("=" * 65)
    
    print("\nBrand Distribution in Cleaned Dataset:")
    print(cleaned_df['Brand_Cleaned'].value_counts())
    print("\nCPU Distribution in Cleaned Dataset:")
    print(cleaned_df['CPU_Cleaned'].value_counts().head(8))
    print("\nPrice Statistics (LKR):")
    print(cleaned_df['Price_Cleaned'].describe().apply(lambda x: f"{x:,.2f}"))
    return cleaned_df

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "raw", "ikman_used_laptops_all.csv"))
    output_path = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "processed", "laptops_cleaned.csv"))
    
    process_laptops_dataset(input_path, output_path)
