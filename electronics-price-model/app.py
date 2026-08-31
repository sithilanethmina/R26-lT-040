from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os
import re
import sys

# Ensure src/ is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

try:
    from llm_extractor import extract_electronics_specs_llm
except ImportError:
    try:
        from src.llm_extractor import extract_electronics_specs_llm
    except ImportError:
        extract_electronics_specs_llm = None

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "service": "electronics_price_predictor"})

# Dictionary to store all loaded models
models_db = {
    'laptop': {},
    'monitor': {},
    'tablet': {}
}

def load_models():
    # 1. Primary Laptop Model
    laptop_path = 'models/best_laptop_model.pkl'
    if os.path.exists(laptop_path):
        try:
            bundle = joblib.load(laptop_path)
            models_db['laptop']['default'] = bundle
            models_db['laptop']['catboost'] = bundle
            models_db['laptop']['xgboost'] = bundle
            models_db['laptop']['random_forest'] = bundle
            print(f"[+] Loaded Laptop Production Model: {bundle.get('model_name', 'CatBoost')}")
        except Exception as e:
            print(f"[!] Failed to load Laptop model: {e}")

    # 2. Primary Monitor Model
    monitor_path = 'models/best_monitor_model.pkl'
    if os.path.exists(monitor_path):
        try:
            bundle = joblib.load(monitor_path)
            models_db['monitor']['default'] = bundle
            models_db['monitor']['xgboost'] = bundle
            models_db['monitor']['random_forest'] = bundle
            models_db['monitor']['lightgbm'] = bundle
            print(f"[+] Loaded Monitor Production Model: {bundle.get('model_name', 'XGBoost')}")
        except Exception as e:
            print(f"[!] Failed to load Monitor model: {e}")

    # 3. Primary Tablet Model
    tablet_path = 'models/best_tablet_model.pkl'
    if os.path.exists(tablet_path):
        try:
            bundle = joblib.load(tablet_path)
            models_db['tablet']['default'] = bundle
            models_db['tablet']['lightgbm'] = bundle
            models_db['tablet']['xgboost'] = bundle
            models_db['tablet']['random_forest'] = bundle
            print(f"[+] Loaded Tablet Production Model: {bundle.get('model_name', 'LightGBM')}")
        except Exception as e:
            print(f"[!] Failed to load Tablet model: {e}")

load_models()

# --- Laptop Reference Database and Cleaning Helpers ---
laptops_db = None

def clean_price(price_str):
    if pd.isna(price_str) or price_str == "Unknown" or str(price_str).strip() == "":
        return np.nan
    clean_str = str(price_str).replace('Rs', '').replace(',', '').strip()
    try:
        return float(clean_str)
    except ValueError:
        return np.nan

def clean_ram(ram_str):
    if pd.isna(ram_str) or ram_str == "Unknown":
        return np.nan
    ram_str = str(ram_str).upper()
    match = re.search(r'(\d+)\s*(GB|MB)?', ram_str)
    if match:
        value = float(match.group(1))
        if value > 128: 
            return np.nan
        return value
    return np.nan

def clean_storage(storage_str):
    if pd.isna(storage_str) or storage_str == "Unknown":
        return np.nan, "Unknown"
    storage_str = str(storage_str).upper()
    match = re.search(r'(\d+)\s*(GB|TB)?', storage_str)
    capacity = np.nan
    if match:
        val = float(match.group(1))
        unit = match.group(2)
        if val <= 4 or unit == 'TB':
            capacity = val * 1024
        else:
            capacity = val
    storage_type = "HDD"
    if "SSD" in storage_str or "NVME" in storage_str or "M.2" in storage_str:
        storage_type = "SSD"
    return capacity, storage_type

def clean_brand(brand_str, title_str):
    title_upper = str(title_str).upper()
    
    # 1. Guess from title series/keyword first (highest accuracy)
    if "THINKPAD" in title_upper or "IDEAPAD" in title_upper or "THINKBOOK" in title_upper or "LEGION" in title_upper:
        return "LENOVO"
    if "MACBOOK" in title_upper:
        return "APPLE"
    if "VIVOBOOK" in title_upper or "ZENBOOK" in title_upper:
        return "ASUS"
    if "LATITUDE" in title_upper or "INSPIRON" in title_upper or "VOSTRO" in title_upper or "PRECISION" in title_upper or "XPS" in title_upper or "ALIENWARE" in title_upper:
        return "DELL"
    if "ELITEBOOK" in title_upper or "PROBOOK" in title_upper or "PAVILION" in title_upper or "VICTUS" in title_upper or "SPECTRE" in title_upper or "ENVY" in title_upper or "ZBOOK" in title_upper:
        return "HP"
    if "ASPIRE" in title_upper or "SWIFT" in title_upper or "NITRO" in title_upper or "PREDATOR" in title_upper:
        return "ACER"
        
    brands = ["DELL", "HP", "LENOVO", "ASUS", "ACER", "APPLE", "SAMSUNG", "MSI", "TOSHIBA", "MICROSOFT"]
    for b in brands:
        if b in title_upper:
            return b

    # 2. Fallback to passed brand_str
    if not pd.isna(brand_str) and str(brand_str).strip() != "":
        brand_upper = str(brand_str).strip().upper()
        if brand_upper not in ["UNKNOWN", "GENERIC", "OTHER"]:
            for b in brands:
                if b in brand_upper:
                    return b
            return brand_upper
            
    return "Other"

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
    
    # Check specific brand first
    if brand in models:
        for m in models[brand]:
            if m in title:
                return m
                
    # Search across ALL brands (in case brand was misidentified or Generic)
    for b, m_list in models.items():
        for m in m_list:
            if m in title:
                return m
                
    return "Other"

def clean_cpu(title):
    title = str(title).upper()
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
    match = re.search(r'(\d+)(?:ST|ND|RD|TH)?\s*(?:GEN|GENERATION)', title)
    if match:
        return int(match.group(1))
    year_match = re.search(r'(20\d{2})', title)
    if year_match:
        year = int(year_match.group(1))
        if year >= 2020: return 12
        if year >= 2018: return 8
        if year >= 2015: return 5
    return 0

def load_laptops_database():
    global laptops_db
    csv_path = 'data/raw/laptops_large_dataset.csv'
    if not os.path.exists(csv_path):
        csv_path = 'data/raw/laptops_detailed.csv'
        
    if os.path.exists(csv_path):
        try:
            print(f"Loading reference laptops database from {csv_path}...")
            df = pd.read_csv(csv_path)
            
            # Clean prices
            df['Price_Cleaned'] = df['Price'].apply(clean_price)
            df = df.dropna(subset=['Price_Cleaned'])
            
            # Extract clean columns
            df['Brand_Cleaned'] = df.apply(lambda row: clean_brand(row['Brand'], row['Title']), axis=1)
            df['Model_Cleaned'] = df.apply(lambda row: clean_model(row['Brand_Cleaned'], row['Title']), axis=1)
            df['CPU_Cleaned'] = df['Title'].apply(clean_cpu)
            df['Generation_Cleaned'] = df['Title'].apply(clean_generation)
            df['Is_Used'] = df['Title'].apply(lambda x: 0 if "NEW" in str(x).upper() else 1)
            
            # Clean RAM and Storage
            df['RAM_GB'] = df['RAM'].apply(clean_ram)
            df['RAM_GB'] = df['RAM_GB'].fillna(8.0)
            
            storage_data = df['Storage'].apply(clean_storage)
            df['Storage_Capacity_GB'] = storage_data.apply(lambda x: x[0])
            df['Storage_Capacity_GB'] = df['Storage_Capacity_GB'].fillna(256.0)
            df['Storage_Type'] = storage_data.apply(lambda x: x[1])
            
            laptops_db = df
            print(f"Laptops reference database loaded. Total rows: {len(laptops_db)}")
        except Exception as e:
            print(f"Error loading reference laptops database: {e}")

def find_matching_laptops(brand, model_title, ram, storage, storage_type, cpu=None, generation=None):
    if laptops_db is None or laptops_db.empty:
        return None, 0.0, None
    
    q_brand = clean_brand(brand, model_title)
    q_model = clean_model(q_brand, model_title)
    q_cpu = str(cpu).strip().upper() if cpu else clean_cpu(model_title)
    q_gen = safe_float(generation) if (generation is not None and str(generation).strip() != "") else clean_generation(model_title)
    q_ram = safe_float(ram, 8.0)
    q_storage = safe_float(storage, 256.0)
    q_storage_type = "SSD" if "SSD" in str(storage_type).upper() or "SSD" in str(model_title).upper() else "HDD"
    
    # 1. Brand match (20 / 5)
    brand_match = (laptops_db['Brand_Cleaned'] == q_brand)
    brand_score = np.where(brand_match, 20, np.where((q_brand == "OTHER") | (laptops_db['Brand_Cleaned'] == "OTHER"), 5, 0))
    
    # 2. Model match (25 / 10)
    model_match = (laptops_db['Model_Cleaned'] == q_model)
    model_score = np.where(model_match & (q_model != "Other"), 25, np.where((q_model == "Other") | (laptops_db['Model_Cleaned'] == "Other"), 10, 0))
    
    # 3. CPU match (20 / 5 / -50 penalty for mismatch of known CPUs)
    cpu_match = (laptops_db['CPU_Cleaned'] == q_cpu)
    both_known_cpu = (q_cpu != "Other") & (laptops_db['CPU_Cleaned'] != "Other")
    cpu_score = np.where(cpu_match & (q_cpu != "Other"), 20,
                         np.where(both_known_cpu & ~cpu_match, -50,
                                  np.where((q_cpu == "Other") | (laptops_db['CPU_Cleaned'] == "Other"), 5, 0)))
    
    # 4. Generation match (15 / 5 / -40 penalty for mismatch of known Gen)
    gen_match = (q_gen > 0) & (laptops_db['Generation_Cleaned'] == q_gen)
    both_known_gen = (q_gen > 0) & (laptops_db['Generation_Cleaned'] > 0)
    gen_score = np.where(gen_match, 15,
                         np.where(both_known_gen & ~gen_match, -40,
                                  np.where((q_gen == 0) | (laptops_db['Generation_Cleaned'] == 0), 5, 0)))
    
    # 5. RAM match (10 / 5 / 2)
    ram_diff = (laptops_db['RAM_GB'] - q_ram).abs()
    ram_score = np.where(ram_diff == 0, 10, np.where(ram_diff <= 4, 5, np.where(ram_diff <= 8, 2, 0)))
    
    # 6. Storage match (10 / 5 / 2)
    storage_diff = (laptops_db['Storage_Capacity_GB'] - q_storage).abs()
    storage_score = np.where(storage_diff == 0, 10, np.where(storage_diff <= 128, 5, np.where(storage_diff <= 256, 2, 0)))
    
    # 7. Storage Type match (5)
    st_match = (laptops_db['Storage_Type'] == q_storage_type)
    st_score = np.where(st_match, 5, 0)
    
    # 8. Condition (Used vs New) match (10 / -30 penalty for New when query is Used)
    cond_score = np.where(laptops_db['Is_Used'] == 1, 10, -30)
    
    total_scores = brand_score + model_score + cpu_score + gen_score + ram_score + storage_score + st_score + cond_score
    
    matched_df = laptops_db.copy()
    matched_df['match_score'] = total_scores
    
    good_matches = matched_df[matched_df['match_score'] >= 70]
    
    if good_matches.empty:
        good_matches = matched_df[matched_df['match_score'] >= 50]
        
    if good_matches.empty:
        good_matches = matched_df[matched_df['match_score'] >= 30]
        
    if good_matches.empty:
        return None, 0.0, None
    
    good_matches = good_matches.sort_values(by='match_score', ascending=False)
    top_matches = good_matches.head(15)
    
    # Feature-based Price Correction
    adjusted_prices = top_matches['Price_Cleaned'].copy()
    
    # RAM correction: LKR 1000 per GB
    ram_diff = top_matches['RAM_GB'] - q_ram
    adjusted_prices -= ram_diff * 1000.0
    
    # Storage correction: LKR 15 per GB
    storage_diff = top_matches['Storage_Capacity_GB'] - q_storage
    adjusted_prices -= storage_diff * 15.0
    
    # Bound the adjustments so they don't produce ridiculous prices (e.g. min 50% of original price)
    min_bound = top_matches['Price_Cleaned'] * 0.5
    adjusted_prices = np.maximum(adjusted_prices, min_bound)
    
    est_price = float(adjusted_prices.median())
    best_score = float(top_matches['match_score'].max())
    
    return est_price, best_score, top_matches

load_laptops_database()

# --- Tablet Reference Database and Cleaning Helpers ---
tablets_db = None

def clean_tablet_brand(brand_str, title_str):
    title_upper = str(title_str).upper()
    if "IPAD" in title_upper:
        return "APPLE"
    if "GALAXY TAB" in title_upper or "SAMSUNG TAB" in title_upper or "SAMSUNG" in title_upper:
        return "SAMSUNG"
    if "MATEPAD" in title_upper or "MEDIAPAD" in title_upper:
        return "HUAWEI"
    if "MI PAD" in title_upper or "XIAOMI PAD" in title_upper or "REDMI PAD" in title_upper:
        return "XIAOMI"
    if "SURFACE" in title_upper:
        return "MICROSOFT"
    if "KINDLE" in title_upper or "FIRE HD" in title_upper:
        return "AMAZON"
    if "REALME PAD" in title_upper:
        return "REALME"
        
    brands = ["APPLE", "SAMSUNG", "LENOVO", "HUAWEI", "XIAOMI", "MICROSOFT", "AMAZON", "REALME", "ASUS", "ACER"]
    for b in brands:
        if b in title_upper:
            return b
            
    if not pd.isna(brand_str) and str(brand_str).strip() != "":
        brand_upper = str(brand_str).strip().upper()
        if brand_upper not in ["UNKNOWN", "GENERIC", "OTHER"]:
            for b in brands:
                if b in brand_upper:
                    return b
            return brand_upper
    return "Other"

def clean_tablet_model(brand, title):
    title = str(title).upper()
    brand = str(brand).upper()
    
    if "IPAD PRO" in title: return "IPAD PRO"
    if "IPAD AIR" in title: return "IPAD AIR"
    if "IPAD MINI" in title: return "IPAD MINI"
    if "IPAD" in title: return "IPAD"
    
    if "SURFACE PRO" in title: return "SURFACE PRO"
    if "SURFACE GO" in title: return "SURFACE GO"
    if "SURFACE" in title: return "SURFACE"
    
    if brand == "SAMSUNG" or "SAMSUNG" in title:
        if any(x in title for x in ["TAB S", "S9", "S8", "S7", "S6", "S5", "S4", "S3", "S2", "S10"]):
            return "GALAXY TAB S"
        if any(x in title for x in ["TAB A", "A9", "A8", "A7", "A6", "A10", "A11", "A2"]):
            return "GALAXY TAB A"
        return "GALAXY TAB"
        
    if "GALAXY TAB S" in title: return "GALAXY TAB S"
    if "GALAXY TAB A" in title: return "GALAXY TAB A"
    if "GALAXY TAB" in title: return "GALAXY TAB"
    
    if "MATEPAD" in title: return "MATEPAD"
    if "MEDIAPAD" in title: return "MEDIAPAD"
    if "REDMI PAD" in title: return "REDMI PAD"
    if "XIAOMI PAD" in title: return "XIAOMI PAD"
    if "MI PAD" in title: return "MI PAD"
    if "KINDLE" in title: return "KINDLE"
    if "FIRE HD" in title: return "FIRE"
    if "REALME PAD" in title: return "REALME PAD"
    
    series = ["IPAD", "GALAXY TAB", "SURFACE", "MATEPAD", "MEDIAPAD", "REDMI PAD", "XIAOMI PAD", "MI PAD", "KINDLE", "FIRE", "REALME PAD"]
    for s in series:
        if s in title:
            return s
            
    return "Other"

def load_tablets_database():
    global tablets_db
    csv_path = 'data/raw/tablets_large_dataset.csv'
    if not os.path.exists(csv_path):
        csv_path = 'data/raw/tablets_dataset.csv'
        
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df['Price_Cleaned'] = df['Price'].apply(clean_price)
            df = df.dropna(subset=['Price_Cleaned'])
            
            df['RAM_GB'] = df['Title'].apply(clean_ram)
            df['RAM_GB'] = df.apply(lambda r: r['RAM_GB'] if not pd.isna(r['RAM_GB']) else clean_ram(r.get('Description', '')), axis=1)
            df['RAM_GB'] = df['RAM_GB'].fillna(4.0)
            
            df['Storage_Capacity_GB'] = df['Title'].apply(lambda x: clean_storage(x)[0])
            df['Storage_Capacity_GB'] = df.apply(lambda r: r['Storage_Capacity_GB'] if not pd.isna(r['Storage_Capacity_GB']) else clean_storage(r.get('Description', ''))[0], axis=1)
            df['Storage_Capacity_GB'] = df['Storage_Capacity_GB'].fillna(64.0)
            
            df['Brand_Cleaned'] = df.apply(lambda row: clean_tablet_brand(row.get('Brand'), row.get('Title')), axis=1)
            df['Model_Cleaned'] = df.apply(lambda row: clean_tablet_model(row['Brand_Cleaned'], row.get('Title')), axis=1)
            
            df['Is_Used'] = df['Condition'].apply(lambda x: 0 if str(x).upper() == "NEW" or str(x).upper() == "BRAND NEW" else 1)
            
            tablets_db = df
            print(f"Tablets reference database loaded. Total rows: {len(tablets_db)}")
        except Exception as e:
            print(f"Error loading reference tablets database: {e}")

def find_matching_tablets(brand, model_title, ram, storage, condition=None):
    if tablets_db is None or tablets_db.empty:
        return None, 0.0, None
        
    q_brand = clean_tablet_brand(brand, model_title)
    q_model = clean_tablet_model(q_brand, model_title)
    q_ram = safe_float(ram, 4.0)
    q_storage = safe_float(storage, 64.0)
    
    brand_match = (tablets_db['Brand_Cleaned'] == q_brand)
    brand_score = np.where(brand_match, 25, np.where((q_brand == "OTHER") | (tablets_db['Brand_Cleaned'] == "OTHER"), 5, 0))
    
    model_match = (tablets_db['Model_Cleaned'] == q_model)
    model_score = np.where(model_match & (q_model != "Other"), 35, np.where((q_model == "Other") | (tablets_db['Model_Cleaned'] == "Other"), 10, 0))
    
    ram_diff = (tablets_db['RAM_GB'] - q_ram).abs()
    ram_score = np.where(ram_diff == 0, 15, np.where(ram_diff <= 2, 8, np.where(ram_diff <= 4, 3, 0)))
    
    storage_diff = (tablets_db['Storage_Capacity_GB'] - q_storage).abs()
    storage_score = np.where(storage_diff == 0, 15, np.where(storage_diff <= 32, 10, np.where(storage_diff <= 64, 5, 0)))
    
    cond_score = np.where(tablets_db['Is_Used'] == 1, 10, -30)
    
    total_scores = brand_score + model_score + ram_score + storage_score + cond_score
    
    matched_df = tablets_db.copy()
    matched_df['match_score'] = total_scores
    
    good_matches = matched_df[matched_df['match_score'] >= 70]
    if good_matches.empty:
        good_matches = matched_df[matched_df['match_score'] >= 50]
    if good_matches.empty:
        good_matches = matched_df[matched_df['match_score'] >= 30]
    if good_matches.empty:
        return None, 0.0, None
        
    good_matches = good_matches.sort_values(by='match_score', ascending=False)
    top_matches = good_matches.head(15)
    
    adjusted_prices = top_matches['Price_Cleaned'].copy()
    
    ram_diffs = top_matches['RAM_GB'] - q_ram
    adjusted_prices -= ram_diffs * 1500.0
    
    storage_diffs = top_matches['Storage_Capacity_GB'] - q_storage
    adjusted_prices -= storage_diffs * 75.0
    
    adjusted_prices = adjusted_prices.clip(lower=5000.0)
    
    est_price = float(adjusted_prices.median())
    best_score = float(top_matches['match_score'].max())
    
    return est_price, best_score, top_matches

# --- Monitors Reference Database and Cleaning Helpers ---
monitors_db = None

def clean_monitor_brand(brand_str, title_str):
    title_upper = str(title_str).upper()
    brands = ["DELL", "HP", "SAMSUNG", "LG", "ASUS", "ACER", "BENQ", "VIEWSONIC", "PHILIPS", "MSI", "AOC", "LENOVO", "EIZO", "HKC"]
    for b in brands:
        if b in title_upper:
            return b
            
    if not pd.isna(brand_str) and str(brand_str).strip() != "":
        brand_upper = str(brand_str).strip().upper()
        if brand_upper not in ["UNKNOWN", "GENERIC", "OTHER"]:
            for b in brands:
                if b in brand_upper:
                    return b
            return brand_upper
    return "Other"

def clean_monitor_size(title_str, desc_str=""):
    text = (str(title_str) + " " + str(desc_str)).upper()
    match = re.search(r'(\d{2})\s*(?:INCH|")', text)
    if match:
        return float(match.group(1))
    return 24.0

def clean_monitor_hz(title_str, desc_str=""):
    text = (str(title_str) + " " + str(desc_str)).upper()
    match = re.search(r'(\d{2,3})\s*HZ', text)
    if match:
        return float(match.group(1))
    return 60.0

def clean_monitor_resolution(title_str, desc_str=""):
    text = (str(title_str) + " " + str(desc_str)).upper()
    if "4K" in text or "UHD" in text or "2160P" in text or "3840" in text:
        return "4K"
    if "2K" in text or "QHD" in text or "1440P" in text or "2560" in text:
        return "2K"
    return "FHD"

def load_monitors_database():
    global monitors_db
    csv_path = 'data/raw/monitors_large_dataset.csv'
    if not os.path.exists(csv_path):
        csv_path = 'data/raw/monitors_dataset.csv'
        
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df['Price_Cleaned'] = df['Price'].apply(clean_price)
            df = df.dropna(subset=['Price_Cleaned'])
            
            df['Brand_Cleaned'] = df.apply(lambda r: clean_monitor_brand(r.get('Brand'), r.get('Title')), axis=1)
            df['Size_Cleaned'] = df.apply(lambda r: clean_monitor_size(r.get('Title'), r.get('Description', '')), axis=1)
            df['Hz_Cleaned'] = df.apply(lambda r: clean_monitor_hz(r.get('Title'), r.get('Description', '')), axis=1)
            df['Resolution_Cleaned'] = df.apply(lambda r: clean_monitor_resolution(r.get('Title'), r.get('Description', '')), axis=1)
            
            df['Is_Used'] = df['Condition'].apply(lambda x: 0 if str(x).upper() in ["NEW", "BRAND NEW"] else 1)
            
            monitors_db = df
            print(f"Monitors reference database loaded. Total rows: {len(monitors_db)}")
        except Exception as e:
            print(f"Error loading reference monitors database: {e}")

def find_matching_monitors(brand, title, size, refresh_rate, resolution, condition=None):
    if monitors_db is None or monitors_db.empty:
        return None, 0.0, None
        
    q_brand = clean_monitor_brand(brand, title)
    q_size = safe_float(size, 24.0)
    q_hz = safe_float(refresh_rate, 60.0)
    q_res = clean_monitor_resolution(title, resolution)
    
    brand_match = (monitors_db['Brand_Cleaned'] == q_brand)
    brand_score = np.where(brand_match, 30, np.where((q_brand == "OTHER") | (monitors_db['Brand_Cleaned'] == "OTHER"), 10, 0))
    
    size_diff = (monitors_db['Size_Cleaned'] - q_size).abs()
    size_score = np.where(size_diff == 0, 25, np.where(size_diff <= 2, 15, np.where(size_diff <= 5, 5, 0)))
    
    hz_diff = (monitors_db['Hz_Cleaned'] - q_hz).abs()
    hz_score = np.where(hz_diff == 0, 20, np.where(hz_diff <= 30, 10, np.where(hz_diff <= 75, 5, 0)))
    
    res_match = (monitors_db['Resolution_Cleaned'] == q_res)
    res_score = np.where(res_match, 15, 0)
    
    cond_score = np.where(monitors_db['Is_Used'] == 1, 10, -30)
    
    total_scores = brand_score + size_score + hz_score + res_score + cond_score
    
    matched_df = monitors_db.copy()
    matched_df['match_score'] = total_scores
    
    good_matches = matched_df[matched_df['match_score'] >= 70]
    if good_matches.empty:
        good_matches = matched_df[matched_df['match_score'] >= 50]
    if good_matches.empty:
        good_matches = matched_df[matched_df['match_score'] >= 30]
    if good_matches.empty:
        return None, 0.0, None
        
    good_matches = good_matches.sort_values(by='match_score', ascending=False)
    top_matches = good_matches.head(15)
    
    adjusted_prices = top_matches['Price_Cleaned'].copy()
    
    size_diffs = top_matches['Size_Cleaned'] - q_size
    adjusted_prices -= size_diffs * 3000.0
    
    hz_diffs = top_matches['Hz_Cleaned'] - q_hz
    adjusted_prices -= hz_diffs * 100.0
    
    adjusted_prices = adjusted_prices.clip(lower=4000.0)
    
    est_price = float(adjusted_prices.median())
    best_score = float(top_matches['match_score'].max())
    
    return est_price, best_score, top_matches

load_tablets_database()
load_monitors_database()

def safe_float(value, default=0.0):
    try:
        return float(value) if value else default
    except (ValueError, TypeError):
        return default

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}
        category = data.get('category', 'laptop')
        algorithm = data.get('algorithm', 'default')
        
        # 0. Check for unstructured text or screenshot and run Gemini LLM Extractor
        image_input = data.get('image') or data.get('image_base64') or data.get('screenshot')
        has_unstructured_input = bool(data.get('title') or data.get('description') or data.get('raw_text') or image_input)
        extraction_source = "DOM / Regex Extractor"
        
        if extract_electronics_specs_llm and has_unstructured_input:
            llm_result = extract_electronics_specs_llm(
                title=data.get('title') or data.get('model') or "",
                description=data.get('description', ""),
                raw_text=data.get('raw_text', ""),
                image_base64=image_input
            )
            if llm_result:
                extraction_source = "Gemini Flash Lite (Vision/LLM)" if image_input else "Gemini Flash Lite (LLM)"
                category = llm_result.get('category') or category
                if llm_result.get('brand'): data['brand'] = llm_result['brand']
                if llm_result.get('model'): data['model'] = llm_result['model']
                if llm_result.get('cpu'): data['cpu'] = llm_result['cpu']
                if llm_result.get('generation') is not None: data['generation'] = llm_result['generation']
                if llm_result.get('ram_gb') is not None: data['ram_gb'] = llm_result['ram_gb']
                if llm_result.get('storage_gb') is not None: data['storage_gb'] = llm_result['storage_gb']
                if llm_result.get('storage_type'): data['storage_type'] = llm_result['storage_type']
                if llm_result.get('gpu'): data['gpu'] = llm_result['gpu']
                if llm_result.get('screen_size_inch') is not None: data['size'] = llm_result['screen_size_inch']
                if llm_result.get('refresh_rate_hz') is not None: data['refresh_rate'] = llm_result['refresh_rate_hz']
                if llm_result.get('resolution'): data['resolution'] = llm_result['resolution']
                if llm_result.get('condition'): data['condition'] = llm_result['condition']
        if category in ['desktop_pc', 'other', 'unsupported']:
            return jsonify({
                'success': False,
                'category': 'unsupported',
                'error': 'Desktop Computers, PC Towers, and Computer Accessories are currently not supported for automated valuation. FairPriceLK supports Laptops, Tablets, and Monitors.'
            })

        if category not in models_db or not models_db[category]:
            return jsonify({'success': False, 'error': f'Models for {category} not loaded.'})
        
        # Select bundle (fallback to default)
        model_data = models_db[category].get(algorithm) or models_db[category].get('default') or list(models_db[category].values())[0]
        
        # Check if bundle or raw model
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
            features = model_data['features']
            model_type = model_data.get('model_type', 'native_cat')
            encoder = model_data.get('encoder')
            model_r2 = model_data.get('r2_score', 0.85)
            model_mae = model_data.get('mae', 0.0)
            model_display_name = model_data.get('model_name', 'Production Model')
        else:
            model = model_data
            features = getattr(model, 'feature_names_', None)
            model_type = 'raw'
            encoder = None
            model_r2 = 0.80
            model_mae = 0.0
            model_display_name = algorithm
            
        # Prepare input data
        if category == 'laptop':
            brand_val = data.get('brand', 'Dell')
            model_val = data.get('model', 'Latitude')
            ram_val = safe_float(data.get('ram') or data.get('ram_gb'), 16.0)
            storage_val = safe_float(data.get('storage') or data.get('storage_gb'), 256.0)
            storage_type_val = data.get('storageType') or data.get('storage_type', 'SSD')
            cpu_val = data.get('cpu', 'Core i5')
            generation_val = safe_float(data.get('generation'), 11.0)
            gpu_val = data.get('gpu', 'Integrated')
            is_touch = 1 if data.get('is_touchscreen') or data.get('touchscreen') else 0
            location_val = str(data.get('location', 'Colombo')).capitalize()
            
            # Predict using production pipeline
            row_dict = {
                'Brand_Cleaned': [str(brand_val).capitalize()],
                'Model_Cleaned': [str(model_val).title()],
                'CPU_Cleaned': [str(cpu_val)],
                'Storage_Type': [str(storage_type_val).upper()],
                'GPU_Tier': [str(gpu_val)],
                'Location_Cleaned': [location_val],
                'Generation_Cleaned': [int(generation_val)],
                'RAM_GB': [float(ram_val)],
                'Storage_Capacity_GB': [float(storage_val)],
                'Is_Touchscreen': [int(is_touch)]
            }
            df_in = pd.DataFrame(row_dict)[features]
            
            if model_type == 'encoded' and encoder is not None:
                cat_cols = model_data['cat_cols']
                df_in[cat_cols] = encoder.transform(df_in[cat_cols])
            elif model_type == 'lgb_cat':
                cat_cols = model_data['cat_cols']
                for c in cat_cols:
                    df_in[c] = df_in[c].astype('category')
                    
            pred_log = model.predict(df_in)[0]
            predicted_price = float(np.expm1(pred_log))
            
            lower_price = max(15000.0, predicted_price * 0.88)
            upper_price = predicted_price * 1.12
            
            return jsonify({
                'success': True,
                'category': 'laptop',
                'price': f"Rs {predicted_price:,.2f}",
                'predicted_price': predicted_price,
                'model_name': f"{model_display_name} ({extraction_source})",
                'accuracy': float(model_r2),
                'fair_market_range': {
                    'lower_price_lkr': lower_price,
                    'upper_price_lkr': upper_price
                },
                'extraction_source': extraction_source,
                'extracted_specs': {
                    'brand': brand_val,
                    'model': model_val,
                    'cpu': cpu_val,
                    'generation': generation_val,
                    'ram_gb': ram_val,
                    'storage_gb': storage_val,
                    'storage_type': storage_type_val,
                    'gpu': gpu_val,
                    'condition': 'Used',
                    'location': location_val
                }
            })
            
        elif category == 'tablet':
            brand_val = data.get('brand', 'Apple')
            model_val = data.get('model', 'iPad')
            storage_val = safe_float(data.get('storage') or data.get('storage_gb'), 64.0)
            ram_val = safe_float(data.get('ram') or data.get('ram_gb'), 4.0)
            conn_val = data.get('connectivity', 'WiFi / Standard')
            size_val = str(data.get('size') or data.get('screen_size_inch') or '10.0 Inch')
            if not "Inch" in size_val: size_val = f"{size_val} Inch"
            location_val = str(data.get('location', 'Colombo')).capitalize()
            
            row_dict = {
                'Brand_Cleaned': [str(brand_val).capitalize()],
                'Model_Cleaned': [str(model_val).title()],
                'Storage_GB': [float(storage_val)],
                'RAM_GB': [float(ram_val)],
                'Connectivity_Cleaned': [conn_val],
                'Screen_Size': [size_val],
                'Location_Cleaned': [location_val]
            }
            df_in = pd.DataFrame(row_dict)[features]
            
            if model_type == 'encoded' and encoder is not None:
                cat_cols = model_data['cat_cols']
                df_in[cat_cols] = encoder.transform(df_in[cat_cols])
            elif model_type == 'lgb_cat':
                cat_cols = model_data['cat_cols']
                for c in cat_cols:
                    df_in[c] = df_in[c].astype('category')
                    
            pred_log = model.predict(df_in)[0]
            predicted_price = float(np.expm1(pred_log))
            
            lower_price = max(4000.0, predicted_price * 0.88)
            upper_price = predicted_price * 1.12
            
            return jsonify({
                'success': True,
                'category': 'tablet',
                'price': f"Rs {predicted_price:,.2f}",
                'predicted_price': predicted_price,
                'model_name': f"{model_display_name} ({extraction_source})",
                'accuracy': float(model_r2),
                'fair_market_range': {
                    'lower_price_lkr': lower_price,
                    'upper_price_lkr': upper_price
                },
                'extraction_source': extraction_source,
                'extracted_specs': {
                    'brand': brand_val,
                    'model': model_val,
                    'storage_gb': storage_val,
                    'ram_gb': ram_val,
                    'connectivity': conn_val,
                    'screen_size': size_val,
                    'condition': 'Used',
                    'location': location_val
                }
            })
            
        elif category == 'monitor':
            brand_val = data.get('brand', 'Dell')
            size_val = safe_float(data.get('size') or data.get('size_inch') or data.get('screen_size_inch'), 24.0)
            hz_val = safe_float(data.get('refreshRate') or data.get('refresh_rate') or data.get('refresh_rate_hz'), 60.0)
            res_val = data.get('resolution', '1080p FHD')
            panel_val = data.get('panel_type', 'Standard')
            is_curved = 1 if data.get('is_curved') else 0
            is_gaming = 1 if data.get('is_gaming') or hz_val >= 100 else 0
            is_frameless = 1 if data.get('is_frameless') else 0
            location_val = str(data.get('location', 'Colombo')).capitalize()
            
            row_dict = {
                'Brand_Cleaned': [str(brand_val).capitalize() if brand_val.upper() not in ["MSI", "AOC", "HP", "LG"] else brand_val.upper()],
                'Resolution_Cleaned': [res_val],
                'Panel_Type': [panel_val],
                'Location_Cleaned': [location_val],
                'Size_Inches': [float(size_val)],
                'Refresh_Rate_Hz': [float(hz_val)],
                'Is_Curved': [int(is_curved)],
                'Is_Gaming': [int(is_gaming)],
                'Is_Frameless': [int(is_frameless)]
            }
            df_in = pd.DataFrame(row_dict)[features]
            
            if model_type == 'encoded' and encoder is not None:
                cat_cols = model_data['cat_cols']
                df_in[cat_cols] = encoder.transform(df_in[cat_cols])
            elif model_type == 'lgb_cat':
                cat_cols = model_data['cat_cols']
                for c in cat_cols:
                    df_in[c] = df_in[c].astype('category')
                    
            pred_log = model.predict(df_in)[0]
            predicted_price = float(np.expm1(pred_log))
            
            lower_price = max(3000.0, predicted_price * 0.90)
            upper_price = predicted_price * 1.10
            
            return jsonify({
                'success': True,
                'category': 'monitor',
                'price': f"Rs {predicted_price:,.2f}",
                'predicted_price': predicted_price,
                'model_name': f"{model_display_name} ({extraction_source})",
                'accuracy': float(model_r2),
                'fair_market_range': {
                    'lower_price_lkr': lower_price,
                    'upper_price_lkr': upper_price
                },
                'extraction_source': extraction_source,
                'extracted_specs': {
                    'brand': brand_val,
                    'model': data.get('model') or f"{brand_val} {size_val}\" Monitor",
                    'size_inches': size_val,
                    'refresh_rate_hz': hz_val,
                    'resolution': res_val,
                    'panel_type': panel_val,
                    'is_curved': bool(is_curved),
                    'is_gaming': bool(is_gaming),
                    'condition': 'Used',
                    'location': location_val
                }
            })
            
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})

@app.route('/model_info')
def model_info():
    info = {}
    for cat in models_db:
        info[cat] = {}
        for algo, m in models_db[cat].items():
            if isinstance(m, dict):
                info[cat][algo] = {
                    'r2_score': float(m.get('r2_score', 0)),
                    'mae': float(m.get('mae', 0)),
                    'name': m.get('model_name', algo)
                }
    return jsonify(info)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8004, use_reloader=False)
