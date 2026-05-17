import pandas as pd
import numpy as np
import re
import os

def clean_price(price_str):
    if pd.isna(price_str) or price_str == "Unknown" or price_str.strip() == "":
        return np.nan
    # Remove "Rs", commas, and whitespace
    clean_str = str(price_str).replace('Rs', '').replace(',', '').strip()
    try:
        return float(clean_str)
    except ValueError:
        return np.nan

def clean_ram(ram_str):
    if pd.isna(ram_str) or ram_str == "Unknown":
        return np.nan
    
    ram_str = str(ram_str).upper()
    # Extract digits before GB or MB
    match = re.search(r'(\d+)\s*(GB|MB)?', ram_str)
    if match:
        value = float(match.group(1))
        # If the value is somehow absurdly high, it might be storage misclassified, or MB
        if value > 128: 
            return np.nan # Unlikely to be RAM
        return value
    return np.nan

def clean_storage(storage_str):
    if pd.isna(storage_str) or storage_str == "Unknown":
        return np.nan, "Unknown"
    
    storage_str = str(storage_str).upper()
    
    # Extract digits and type
    match = re.search(r'(\d+)\s*(GB|TB)?', storage_str)
    capacity = np.nan
    if match:
        val = float(match.group(1))
        unit = match.group(2)
        
        # Assume if it's <= 2, it's probably TB. E.g., "1 HDD" -> 1 TB
        if val <= 4 or unit == 'TB':
            capacity = val * 1024 # Convert to GB
        else:
            capacity = val
            
    # Determine type
    storage_type = "HDD"
    if "SSD" in storage_str or "NVME" in storage_str or "M.2" in storage_str:
        storage_type = "SSD"
        
    return capacity, storage_type

def clean_brand(brand_str, title_str):
    # Try to extract brand from Brand column, if Unknown, try to guess from Title
    if pd.isna(brand_str) or brand_str == "Unknown":
        title_upper = str(title_str).upper()
        brands = ["DELL", "HP", "ASUS", "ACER", "APPLE", "MICROSOFT"]
        for b in brands:
            if b in title_upper:
                return b
        return "Other"
    
    return str(brand_str).strip().upper()

def clean_model(brand, title):
    title = str(title).upper()
    brand = str(brand).upper()
    
    models = {
        'DELL': ['LATITUDE', 'INSPIRON', 'VOSTRO', 'PRECISION', 'XPS', 'ALIENWARE', 'G15', 'G3', 'G5', 'G7'],
        'HP': ['ELITEBOOK', 'PROBOOK', 'PAVILION', 'SPECTRE', 'ENVY', 'OMEN', 'VICTUS', 'ZBOOK', 'NOTEBOOK', 'ELITE'],
        'LENOVO': ['THINKPAD', 'IDEAPAD', 'LEGION', 'YOGA', 'V15', 'THINKBOOK', 'T470', 'T480', 'T490', 'X1'],
        'ASUS': ['VIVOBOOK', 'ZENBOOK', 'ROG', 'TUF', 'EXPERTBOOK'],
        'ACER': ['ASPIRE', 'SWIFT', 'NITRO', 'PREDATOR', 'TRAVELMATE', 'SPIN'],
        'APPLE': ['MACBOOK PRO', 'MACBOOK AIR', 'MACBOOK'],
        'MSI': ['MODERN', 'STEALTH', 'KATANA', 'SWORD', 'CYBORG', 'GAMING']
    }
    
    if brand in models:
        for m in models[brand]:
            if m in title:
                return m
                
    return "Other"

def clean_cpu(title):
    title = str(title).upper()
    # Patterns for CPU types
    cpu_patterns = {
        'I9': r'I9|CORE I9',
        'I7': r'I7|CORE I7',
        'I5': r'I5|CORE I5',
        'I3': r'I3|CORE I3',
        'RYZEN 9': r'RYZEN 9',
        'RYZEN 7': r'RYZEN 7',
        'RYZEN 5': r'RYZEN 5',
        'RYZEN 3': r'RYZEN 3',
        'M1': r'M1',
        'M2': r'M2',
        'M3': r'M3',
        'CELERON': r'CELERON',
        'PENTIUM': r'PENTIUM',
        'QUAD CORE': r'QUAD CORE|QUAD-CORE'
    }
    
    for cpu, pattern in cpu_patterns.items():
        if re.search(pattern, title):
            return cpu
    return "Other"

def clean_generation(title):
    title = str(title).upper()
    # Pattern to find "Xth Gen" or "X Gen"
    match = re.search(r'(\d+)(?:ST|ND|RD|TH)?\s*(?:GEN|GENERATION)', title)
    if match:
        return int(match.group(1))
    
    # Try to find year for MacBooks
    year_match = re.search(r'(20\d{2})', title)
    if year_match:
        year = int(year_match.group(1))
        if year >= 2020: return 12 # Approximation for M1/M2 era
        if year >= 2018: return 8
        if year >= 2015: return 5
        
    return 0 # Unknown or older

def process_data(input_csv, output_csv):
    print(f"Loading data from {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print("Data file not found. Wait for the scraper to finish!")
        return
        
    print(f"Raw data shape: {df.shape}")
    
    # Filter out items that are strictly just chargers or non-laptops based on title/price
    # E.g. Price < 5000 is probably a charger or part
    
    # 1. Clean Price
    df['Price_Cleaned'] = df['Price'].apply(clean_price)
    
    # Drop rows without a valid price or price too low (like chargers at Rs 2000)
    df = df.dropna(subset=['Price_Cleaned'])
    # Filter out outliers (e.g. Chargers or super-expensive misclassified items)
    df = df[(df['Price_Cleaned'] >= 15000) & (df['Price_Cleaned'] <= 1500000)]
    print(f"Data shape after price filtering: {df.shape}")
    
    # 2. Clean RAM
    df['RAM_GB'] = df['RAM'].apply(clean_ram)
    
    # 3. Clean Storage
    storage_data = df['Storage'].apply(clean_storage)
    df['Storage_Capacity_GB'] = storage_data.apply(lambda x: x[0])
    df['Storage_Type'] = storage_data.apply(lambda x: x[1])
    
    # 4. Clean Brand and Model
    df['Brand_Cleaned'] = df.apply(lambda row: clean_brand(row['Brand'], row['Title']), axis=1)
    df['Model_Cleaned'] = df.apply(lambda row: clean_model(row['Brand_Cleaned'], row['Title']), axis=1)
    
    # 5. Clean CPU and Generation
    df['CPU_Cleaned'] = df['Title'].apply(clean_cpu)
    df['Generation_Cleaned'] = df['Title'].apply(clean_generation)
    
    # 5. Clean Condition
    df['Condition_Cleaned'] = df['Condition'].apply(lambda x: "New" if "New" in str(x) else "Used")
    
    # --- FILTERING STEP ---
    # Remove specific brands requested by user
    excluded_brands = ['SAMSUNG', 'TOSHIBA', 'MICROSOFT', 'LENOVO', 'MSI']
    df = df[~df['Brand_Cleaned'].isin(excluded_brands)]
    
    # Remove specific CPUs requested by user
    excluded_cpus = ['CELERON', 'PENTIUM']
    df = df[~df['CPU_Cleaned'].isin(excluded_cpus)]
    # ----------------------
    
    # Fill missing numeric values with median (Simple imputation for now)
    df['RAM_GB'] = df['RAM_GB'].fillna(df['RAM_GB'].median())
    df['Storage_Capacity_GB'] = df['Storage_Capacity_GB'].fillna(df['Storage_Capacity_GB'].median())
    
    # Drop columns that are no longer needed for modeling
    model_df = df[['Title', 'Brand_Cleaned', 'Model_Cleaned', 'CPU_Cleaned', 'Generation_Cleaned', 'Condition_Cleaned', 'RAM_GB', 'Storage_Capacity_GB', 'Storage_Type', 'Price_Cleaned']].copy()
    
    # Save the cleaned dataset
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    model_df.to_csv(output_csv, index=False)
    print(f"Cleaned data saved to {output_csv}. Shape: {model_df.shape}")
    
    # Display preview
    print("\nSample of cleaned data:")
    print(model_df.head())

if __name__ == "__main__":
    # Point to the large dataset (or fall back to the detailed one if large isn't ready)
    if os.path.exists('data/raw/laptops_large_dataset.csv'):
        input_file = 'data/raw/laptops_large_dataset.csv'
    else:
        print("Large dataset not found yet, using the initial 47 row dataset for testing...")
        input_file = 'data/raw/laptops_detailed.csv'
        
    output_file = 'data/processed/laptops_cleaned.csv'
    process_data(input_file, output_file)
