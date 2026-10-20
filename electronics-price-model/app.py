from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os
import re
import sys

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
except Exception:
    pass

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Ensure src/ is on sys.path
sys.path.insert(0, str(BASE_DIR / 'src'))

try:
    from llm_extractor import extract_electronics_specs_llm
except ImportError:
    try:
        from src.llm_extractor import extract_electronics_specs_llm
    except ImportError:
        extract_electronics_specs_llm = None

app = Flask(__name__)

try:
    from flask_cors import CORS
    CORS(app)
except Exception:
    pass

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
    # 1. Laptop Models (with multi-file fallback)
    laptop_candidates = [
        'best_laptop_model.pkl',
        'laptop_production_model.pkl',
        'laptop_xgboost.pkl',
        'laptop_lightgbm.pkl',
        'laptop_catboost.pkl',
        'laptop_random_forest.pkl'
    ]
    for fname in laptop_candidates:
        l_path = os.path.join(BASE_DIR, 'models', fname)
        if os.path.exists(l_path):
            try:
                bundle = joblib.load(l_path)
                algo = str(bundle.get('model_name', 'XGBoost')).lower().replace(' ', '_')
                models_db['laptop'][algo] = bundle
                if 'default' not in models_db['laptop']:
                    models_db['laptop']['default'] = bundle
                    print(f"[+] Loaded Laptop Production Model: {bundle.get('model_name', 'XGBoost')} (from {fname})")
            except Exception as e:
                print(f"[!] Warning: Could not load laptop candidate {fname}: {e}")

    # 2. Monitor Models (with multi-file fallback)
    monitor_candidates = [
        'best_monitor_model.pkl',
        'monitor_production_model.pkl',
        'monitor_lightgbm.pkl',
        'monitor_xgboost.pkl',
        'monitor_catboost.pkl',
        'monitor_random_forest.pkl'
    ]
    for fname in monitor_candidates:
        m_path = os.path.join(BASE_DIR, 'models', fname)
        if os.path.exists(m_path):
            try:
                bundle = joblib.load(m_path)
                algo = str(bundle.get('model_name', 'LightGBM')).lower().replace(' ', '_')
                models_db['monitor'][algo] = bundle
                if 'default' not in models_db['monitor']:
                    models_db['monitor']['default'] = bundle
                    print(f"[+] Loaded Monitor Production Model: {bundle.get('model_name', 'LightGBM')} (from {fname})")
            except Exception as e:
                print(f"[!] Warning: Could not load monitor candidate {fname}: {e}")

    # 3. Tablet Models (with multi-file fallback)
    tablet_candidates = [
        'best_tablet_model.pkl',
        'tablet_production_model.pkl',
        'tablet_lightgbm.pkl',
        'tablet_xgboost.pkl',
        'tablet_catboost.pkl',
        'tablet_random_forest.pkl'
    ]
    for fname in tablet_candidates:
        t_path = os.path.join(BASE_DIR, 'models', fname)
        if os.path.exists(t_path):
            try:
                bundle = joblib.load(t_path)
                algo = str(bundle.get('model_name', 'LightGBM')).lower().replace(' ', '_')
                models_db['tablet'][algo] = bundle
                if 'default' not in models_db['tablet']:
                    models_db['tablet']['default'] = bundle
                    print(f"[+] Loaded Tablet Production Model: {bundle.get('model_name', 'LightGBM')} (from {fname})")
            except Exception as e:
                print(f"[!] Warning: Could not load tablet candidate {fname}: {e}")

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


#home page
@app.route('/')
def home():
    return render_template('index.html')

def log_required_fields_table(category, fields_list, predicted_price=None, lower_price=None, upper_price=None, asking_price=None, model_display_name="Production ML Model", missing_fields=None):
    col1_w = 22
    col2_w = 26
    col3_w = 26
    total_w = col1_w + col2_w + col3_w + 10

    header_title = f"📋 REQUIRED PREDICTION FIELDS: {category.upper()}"
    print("\n" + "=" * total_w)
    print(f" {header_title}")
    print("=" * total_w)
    print(f"┌─{'─'*col1_w}─┬─{'─'*col2_w}─┬─{'─'*col3_w}─┐")
    print(f"│ {'Required Feature':<{col1_w}} │ {'Applied Value':<{col2_w}} │ {'Source / Status':<{col3_w}} │")
    print(f"├─{'─'*col1_w}─┼─{'─'*col2_w}─┼─{'─'*col3_w}─┤")
    for f in fields_list:
        name = str(f.get("name", ""))[:col1_w]
        val = str(f.get("value", ""))[:col2_w]
        src = str(f.get("status", ""))[:col3_w]
        print(f"│ {name:<{col1_w}} │ {val:<{col2_w}} │ {src:<{col3_w}} │")
    print(f"└─{'─'*col1_w}─┴─{'─'*col2_w}─┴─{'─'*col3_w}─┘")

    if predicted_price is not None:
        print("-" * total_w)
        print(f" 🔮 Model Engine:       {model_display_name}")
        print(f" 💰 Fair Market Price:  Rs {predicted_price:,.2f} LKR")
        if lower_price is not None and upper_price is not None:
            print(f" 📊 Fair Price Range:   Rs {lower_price:,.2f} - Rs {upper_price:,.2f} LKR")
        if asking_price:
            try:
                clean_p = float(re.sub(r'[^0-9.]', '', str(asking_price)))
                if clean_p < lower_price:
                    diff = lower_price - clean_p
                    print(f" 🟢 Deal Verdict:       GREAT DEAL / UNDERPRICED (Save ~Rs {diff:,.0f} below lower fair bound)")
                elif clean_p > upper_price:
                    diff = clean_p - upper_price
                    print(f" 🔴 Deal Verdict:       OVERPRICED (Asking Rs {diff:,.0f} above upper fair bound)")
                else:
                    print(f" 🟡 Deal Verdict:       FAIR MARKET PRICE (Within estimated market range)")
            except Exception:
                pass
    else:
        print("-" * total_w)
        missing_str = ", ".join(missing_fields) if missing_fields else "Required specifications missing"
        print(f" ❌ PREDICTION HALTED: Missing required field(s): {missing_str}")
        print("    No synthetic baseline fallback used. All required fields must be present.")
    print("=" * total_w + "\n", flush=True)

#prediction page
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}
        raw_cat = str(data.get('category') or 'laptop').strip().lower()
        if raw_cat in ['tablets', 'ipad', 'tabs', 'tab', 'pad']:
            category = 'tablet'
        elif raw_cat in ['monitors', 'screen', 'display']:
            category = 'monitor'
        elif raw_cat in ['laptops', 'notebook']:
            category = 'laptop'
        else:
            category = raw_cat
        algorithm = data.get('algorithm', 'default')
        
        # 0. Check for unstructured text, dom_tree, or screenshot and run Gemini LLM Extractor
        image_input = data.get('image') or data.get('image_base64') or data.get('screenshot')
        dom_tree_input = data.get('dom_tree')
        has_unstructured_input = bool(data.get('title') or data.get('description') or data.get('raw_text') or image_input or dom_tree_input)
        extraction_source = "DOM / Regex Extractor"
        llm_result = None
        
        if extract_electronics_specs_llm and has_unstructured_input:
            llm_result = extract_electronics_specs_llm(
                title=data.get('title') or data.get('model') or "",
                description=data.get('description', ""),
                raw_text=data.get('raw_text', ""),
                image_base64=image_input,
                dom_tree=dom_tree_input
            )
            if llm_result:
                has_img_flag = bool(image_input and len(str(image_input).strip()) > 50)
                has_dom_flag = bool(dom_tree_input and isinstance(dom_tree_input, dict) and any(dom_tree_input.values()))
                if has_img_flag and has_dom_flag:
                    extraction_source = "Gemini Flash Lite (Dual Fusion: Vision + DOM)"
                elif has_img_flag:
                    extraction_source = "Gemini Flash Lite (Vision AI)"
                else:
                    extraction_source = "Gemini Flash Lite (DOM Tree)"
                llm_cat = str(llm_result.get('category') or "").strip().lower()
                if llm_cat in ['tablets', 'ipad', 'tabs', 'tab', 'pad']:
                    category = 'tablet'
                elif llm_cat in ['monitors', 'screen', 'display']:
                    category = 'monitor'
                elif llm_cat in ['laptops', 'notebook']:
                    category = 'laptop'
                elif llm_cat in ['laptop', 'tablet', 'monitor', 'desktop_pc', 'other']:
                    category = llm_cat
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
                if llm_result.get('panel_type'): data['panel_type'] = llm_result['panel_type']
                if llm_result.get('condition'): data['condition'] = llm_result['condition']
                if llm_result.get('location'): data['location'] = llm_result['location']
                if llm_result.get('is_touchscreen') is not None: data['is_touchscreen'] = llm_result['is_touchscreen']
                if llm_result.get('is_curved') is not None: data['is_curved'] = llm_result['is_curved']
                if llm_result.get('is_gaming') is not None: data['is_gaming'] = llm_result['is_gaming']

        if category in ['desktop_pc', 'other', 'unsupported']:
            print(f"⛔ [UNSUPPORTED CATEGORY] '{category}' is not supported for automated valuation.")
            return jsonify({
                'success': False,
                'category': 'unsupported',
                'error': 'Desktop Computers, PC Towers, and Computer Accessories are currently not supported for automated valuation. FairPriceLK supports Laptops, Tablets, and Monitors.'
            })

        if category not in models_db or not models_db[category]:
            print(f"❌ [MODEL ERROR] No machine learning models loaded for category '{category}'. Available categories: {[k for k, v in models_db.items() if v]}")
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
            missing_fields = []

            # Detect Chromebook
            is_chromebook = (
                "CHROMEBOOK" in str(data.get('model', '')).upper() or
                "CROMEBOOK" in str(data.get('model', '')).upper() or
                "CHROMEBOOK" in str(data.get('title', '')).upper() or
                "CROMEBOOK" in str(data.get('title', '')).upper()
            )

            # 1. Brand (Strict - No Baseline Default)
            raw_brand = data.get('brand')
            if raw_brand and str(raw_brand).strip().lower() not in ['', 'generic', 'unknown', 'none']:
                brand_val = str(raw_brand).strip()
                brand_status = "✅ Extracted by LLM" if (llm_result and llm_result.get('brand')) else "🌐 Scraped from Page"
            else:
                brand_val = None
                brand_status = "❌ Missing (Required)"
                missing_fields.append("Brand")

            # 2. Model (Strict - No Baseline Default)
            raw_model = data.get('model')
            if is_chromebook:
                model_val = 'Chromebook'
                model_status = "ℹ️ Chromebook Model"
            elif raw_model and str(raw_model).strip().lower() not in ['', 'standard model', 'unknown', 'laptop', 'none']:
                model_val = str(raw_model).strip()
                model_status = "✅ Extracted by LLM" if (llm_result and llm_result.get('model')) else "🌐 Scraped from Page"
            else:
                model_val = None
                model_status = "❌ Missing (Required)"
                missing_fields.append("Model / Series")

            # 3. CPU Processor (Strict - No Baseline Default)
            raw_cpu = data.get('cpu')
            if raw_cpu and str(raw_cpu).strip().lower() not in ['', 'unknown', 'none']:
                cpu_val = str(raw_cpu).strip()
                if cpu_val.lower() == 'other':
                    cpu_val = 'Other / Dual Core'
                cpu_status = "✅ Extracted by LLM" if (llm_result and llm_result.get('cpu')) else "🌐 Scraped from Page"
            elif is_chromebook:
                cpu_val = 'Celeron'
                cpu_status = "ℹ️ Chromebook Celeron"
            else:
                cpu_val = None
                cpu_status = "❌ Missing (Required)"
                missing_fields.append("CPU Processor")

            # 4. CPU Generation (Strict - Required for Intel Core / Ryzen, N/A for Apple M & Budget/Chromebook)
            raw_gen = data.get('generation')
            is_non_core = False
            if cpu_val:
                cpu_upper = cpu_val.upper()
                if any(m in cpu_upper for m in ["APPLE", "M1", "M2", "M3", "M4"]):
                    is_non_core = True
                    if "M4" in cpu_upper: generation_val = 15.0
                    elif "M3" in cpu_upper: generation_val = 14.0
                    elif "M2" in cpu_upper: generation_val = 13.0
                    else: generation_val = 12.0
                    gen_status = "✅ Apple Silicon Tier"
                elif any(b in cpu_upper for b in ["CELERON", "PENTIUM", "DUAL CORE", "QUAD CORE"]) or is_chromebook:
                    is_non_core = True
                    generation_val = 0.0
                    gen_status = "✅ N/A (Budget / Non-Core)"
            
            if not is_non_core:
                if raw_gen is not None and str(raw_gen).strip() not in ['', '0', 'None', 'null']:
                    generation_val = safe_float(raw_gen, 0.0)
                    if generation_val > 0:
                        gen_status = "✅ Extracted by LLM" if (llm_result and llm_result.get('generation') is not None) else "🌐 Scraped from Page"
                    else:
                        generation_val = None
                        gen_status = "❌ Missing (Required)"
                        missing_fields.append("CPU Generation")
                else:
                    generation_val = None
                    gen_status = "❌ Missing (Required)"
                    missing_fields.append("CPU Generation")

            # 5. RAM (Strict - No Baseline Default)
            raw_ram = data.get('ram') or data.get('ram_gb')
            if raw_ram is not None and str(raw_ram).strip() not in ['', '0', 'None', 'null']:
                ram_val = safe_float(raw_ram, 0.0)
                if ram_val > 0:
                    ram_status = "✅ Extracted by LLM" if (llm_result and llm_result.get('ram_gb') is not None) else "🌐 Scraped from Page"
                else:
                    ram_val = None
                    ram_status = "❌ Missing (Required)"
                    missing_fields.append("RAM Memory")
            else:
                ram_val = None
                ram_status = "❌ Missing (Required)"
                missing_fields.append("RAM Memory")

            # 6. Storage (Strict - No Baseline Default)
            raw_storage = data.get('storage') or data.get('storage_gb')
            if raw_storage is not None and str(raw_storage).strip() not in ['', '0', 'None', 'null']:
                storage_val = safe_float(raw_storage, 0.0)
                if storage_val > 0:
                    storage_status = "✅ Extracted by LLM" if (llm_result and llm_result.get('storage_gb') is not None) else "🌐 Scraped from Page"
                else:
                    storage_val = None
                    storage_status = "❌ Missing (Required)"
                    missing_fields.append("Storage Capacity")
            else:
                storage_val = None
                storage_status = "❌ Missing (Required)"
                missing_fields.append("Storage Capacity")

            # 7. Storage Type (Strict - No Baseline Default)
            raw_stype = data.get('storageType') or data.get('storage_type')
            if raw_stype and str(raw_stype).strip().lower() not in ['', 'unknown', 'none', 'null']:
                storage_type_val = str(raw_stype).strip().upper()
                stype_status = "✅ Extracted by LLM" if (llm_result and llm_result.get('storage_type')) else "🌐 Scraped from Page"
            elif is_chromebook:
                storage_type_val = 'SSD'
                stype_status = "ℹ️ Chromebook eMMC/SSD"
            else:
                storage_type_val = None
                stype_status = "❌ Missing (Required)"
                missing_fields.append("Storage Type")

            # 8. GPU Tier (Strict - No Baseline Default)
            raw_gpu = data.get('gpu')
            if raw_gpu and str(raw_gpu).strip().lower() not in ['', 'unknown', 'none', 'null']:
                gpu_val = str(raw_gpu).strip()
                gpu_status = "✅ Extracted by LLM" if (llm_result and llm_result.get('gpu')) else "🌐 Scraped from Page"
            elif is_chromebook:
                gpu_val = 'Integrated'
                gpu_status = "ℹ️ Integrated"
            else:
                gpu_val = None
                gpu_status = "❌ Missing (Required)"
                missing_fields.append("GPU Tier")

            # 9. Touchscreen (ALLOWED FALLBACK: Defaults to Non-Touch)
            raw_touch = data.get('is_touchscreen') if data.get('is_touchscreen') is not None else data.get('touchscreen')
            if raw_touch is not None and str(raw_touch).strip().lower() not in ['', 'none', 'null']:
                is_touch = 1 if raw_touch in [True, 1, '1', 'true', 'True'] else 0
                touch_status = "✅ Extracted by LLM" if (llm_result and llm_result.get('is_touchscreen') is not None) else "🌐 Scraped from Page"
            else:
                is_touch = 0
                touch_status = "ℹ️ Default Non-touch"

            # 10. Location (ALLOWED FALLBACK: Defaults to Colombo)
            raw_loc = data.get('location')
            if raw_loc and str(raw_loc).strip().lower() not in ['', 'other', 'unknown', 'none', 'null']:
                location_val = str(raw_loc).strip().capitalize()
                loc_status = "✅ Extracted by LLM" if (llm_result and llm_result.get('location')) else "🌐 Scraped from Page"
            else:
                location_val = 'Colombo'
                loc_status = "ℹ️ Default Hub (Colombo)"

            laptop_fields = [
                {"name": "Brand", "value": brand_val if brand_val else "[MISSING]", "status": brand_status},
                {"name": "Model / Series", "value": model_val if model_val else "[MISSING]", "status": model_status},
                {"name": "CPU Processor", "value": cpu_val if cpu_val else "[MISSING]", "status": cpu_status},
                {"name": "CPU Generation", "value": (f"{int(generation_val)}th Gen" if generation_val > 0 else "0 (N/A)") if generation_val is not None else "[MISSING]", "status": gen_status},
                {"name": "RAM Memory", "value": f"{ram_val:.0f} GB" if ram_val is not None else "[MISSING]", "status": ram_status},
                {"name": "Storage Capacity", "value": f"{storage_val:.0f} GB" if storage_val is not None else "[MISSING]", "status": storage_status},
                {"name": "Storage Type", "value": storage_type_val if storage_type_val else "[MISSING]", "status": stype_status},
                {"name": "GPU Tier", "value": gpu_val if gpu_val else "[MISSING]", "status": gpu_status},
                {"name": "Touchscreen", "value": "Yes" if is_touch else "No", "status": touch_status},
                {"name": "Location", "value": location_val, "status": loc_status},
            ]

            if missing_fields:
                log_required_fields_table('laptop', laptop_fields, predicted_price=None, missing_fields=missing_fields)
                return jsonify({
                    'success': False,
                    'category': 'laptop',
                    'error': f"Unable to predict laptop price. Missing required field(s): {', '.join(missing_fields)}. Please provide these specifications.",
                    'missing_fields': missing_fields,
                    'status': 'missing_required_fields'
                })

            # Model title normalization for encoder
            clean_brand_title = str(brand_val).capitalize()
            clean_model_title = str(model_val).title()
            if is_chromebook or 'chromebook' in clean_model_title.lower() or 'cromebook' in clean_model_title.lower():
                clean_model_title = 'Chromebook'
            
            clean_cpu_title = str(cpu_val)
            if clean_cpu_title.lower() in ['other', 'dual core', 'other / dual core']:
                clean_cpu_title = 'Other / Dual Core'

            # Predict using production pipeline
            row_dict = {
                'Brand_Cleaned': [clean_brand_title],
                'Model_Cleaned': [clean_model_title],
                'CPU_Cleaned': [clean_cpu_title],
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
            
            log_required_fields_table('laptop', laptop_fields, predicted_price, lower_price, upper_price, data.get('price') or data.get('listed_price'), model_display_name)

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
            missing_fields = []

            # 1. Brand (Strict - No Baseline Default)
            raw_brand = data.get('brand')
            if raw_brand and str(raw_brand).strip().lower() not in ['', 'generic', 'unknown', 'none']:
                brand_val = str(raw_brand).strip()
                brand_status = "✅ Extracted by LLM" if (llm_result and llm_result.get('brand')) else "🌐 Scraped from Page"
            else:
                brand_val = None
                brand_status = "❌ Missing (Required)"
                missing_fields.append("Brand")

            # 2. Model / Series (Strict - No Baseline Default)
            raw_model = data.get('model')
            if raw_model and str(raw_model).strip().lower() not in ['', 'unknown', 'tablet', 'none']:
                model_val = str(raw_model).strip()
                model_status = "✅ Extracted by LLM" if (llm_result and llm_result.get('model')) else "🌐 Scraped from Page"
            else:
                model_val = None
                model_status = "❌ Missing (Required)"
                missing_fields.append("Model / Series")

            # 3. Storage Capacity (Strict - No Baseline Default)
            raw_storage = data.get('storage') or data.get('storage_gb')
            if raw_storage is not None and str(raw_storage).strip() not in ['', '0', 'None', 'null']:
                storage_val = safe_float(raw_storage, 0.0)
                if storage_val > 0:
                    storage_status = "✅ Extracted by LLM" if (llm_result and llm_result.get('storage_gb') is not None) else "🌐 Scraped from Page"
                else:
                    storage_val = None
                    storage_status = "❌ Missing (Required)"
                    missing_fields.append("Storage Capacity")
            else:
                storage_val = None
                storage_status = "❌ Missing (Required)"
                missing_fields.append("Storage Capacity")

            # 4. RAM Memory (Strict - No Baseline Default)
            raw_ram = data.get('ram') or data.get('ram_gb')
            if raw_ram is not None and str(raw_ram).strip() not in ['', '0', 'None', 'null']:
                ram_val = safe_float(raw_ram, 0.0)
                if ram_val > 0:
                    ram_status = "✅ Extracted by LLM" if (llm_result and llm_result.get('ram_gb') is not None) else "🌐 Scraped from Page"
                else:
                    ram_val = None
                    ram_status = "❌ Missing (Required)"
                    missing_fields.append("RAM Memory")
            else:
                ram_val = None
                ram_status = "❌ Missing (Required)"
                missing_fields.append("RAM Memory")

            # 5. Screen Size (Strict - No Baseline Default)
            raw_size = data.get('size') or data.get('size_inch') or data.get('size_inches') or data.get('screen_size_inch')
            if raw_size and str(raw_size).strip() not in ['', '0', 'None', 'null']:
                size_val = str(raw_size).strip()
                if "Inch" not in size_val: size_val = f"{size_val} Inch"
                size_status = "✅ Extracted by LLM" if (llm_result and llm_result.get('screen_size_inch') is not None) else "🌐 Scraped from Page"
            else:
                size_val = None
                size_status = "❌ Missing (Required)"
                missing_fields.append("Screen Size")

            # 6. Connectivity (ALLOWED FALLBACK: Defaults to WiFi / Standard)
            raw_conn = data.get('connectivity')
            if raw_conn and str(raw_conn).strip().lower() not in ['', 'unknown', 'none', 'null']:
                conn_val = str(raw_conn).strip()
                conn_status = "✅ Extracted by LLM" if (llm_result and llm_result.get('connectivity')) else "🌐 Scraped from Page"
            else:
                conn_val = 'WiFi / Standard'
                conn_status = "ℹ️ Default WiFi / Standard"

            # 7. Location (ALLOWED FALLBACK: Defaults to Colombo)
            raw_loc = data.get('location')
            if raw_loc and str(raw_loc).strip().lower() not in ['', 'other', 'unknown', 'none', 'null']:
                location_val = str(raw_loc).strip().capitalize()
                loc_status = "✅ Extracted by LLM" if (llm_result and llm_result.get('location')) else "🌐 Scraped from Page"
            else:
                location_val = 'Colombo'
                loc_status = "ℹ️ Default Hub (Colombo)"

            tablet_fields = [
                {"name": "Brand", "value": brand_val if brand_val else "[MISSING]", "status": brand_status},
                {"name": "Model / Series", "value": model_val if model_val else "[MISSING]", "status": model_status},
                {"name": "Storage Capacity", "value": f"{storage_val:.0f} GB" if storage_val is not None else "[MISSING]", "status": storage_status},
                {"name": "RAM Memory", "value": f"{ram_val:.0f} GB" if ram_val is not None else "[MISSING]", "status": ram_status},
                {"name": "Screen Size", "value": size_val if size_val else "[MISSING]", "status": size_status},
                {"name": "Connectivity", "value": conn_val, "status": conn_status},
                {"name": "Location", "value": location_val, "status": loc_status},
            ]

            if missing_fields:
                log_required_fields_table('tablet', tablet_fields, predicted_price=None, missing_fields=missing_fields)
                return jsonify({
                    'success': False,
                    'category': 'tablet',
                    'error': f"Unable to predict tablet price. Missing required field(s): {', '.join(missing_fields)}. Please provide these specifications.",
                    'missing_fields': missing_fields,
                    'status': 'missing_required_fields'
                })

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
            
            log_required_fields_table('tablet', tablet_fields, predicted_price, lower_price, upper_price, data.get('price') or data.get('listed_price'), model_display_name)

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
            missing_fields = []

            # 1. Brand (Strict - No Baseline Default)
            raw_brand = data.get('brand')
            if raw_brand and str(raw_brand).strip().lower() not in ['', 'generic', 'unknown', 'none']:
                brand_val = str(raw_brand).strip()
                brand_status = "✅ Extracted by LLM" if (llm_result and llm_result.get('brand')) else "🌐 Scraped from Page"
            else:
                brand_val = None
                brand_status = "❌ Missing (Required)"
                missing_fields.append("Brand")

            # 2. Screen Size (Strict - No Baseline Default)
            raw_size = data.get('size') or data.get('size_inch') or data.get('size_inches') or data.get('screen_size_inch')
            if raw_size is not None and str(raw_size).strip() not in ['', '0', 'None', 'null']:
                size_val = safe_float(raw_size, 0.0)
                if size_val > 0:
                    size_status = "✅ Extracted by LLM" if (llm_result and llm_result.get('screen_size_inch') is not None) else "🌐 Scraped from Page"
                else:
                    size_val = None
                    size_status = "❌ Missing (Required)"
                    missing_fields.append("Screen Size")
            else:
                size_val = None
                size_status = "❌ Missing (Required)"
                missing_fields.append("Screen Size")

            # 3. Refresh Rate (Strict - No Baseline Default)
            raw_hz = data.get('refreshRate') or data.get('refresh_rate') or data.get('refresh_rate_hz')
            if raw_hz is not None and str(raw_hz).strip() not in ['', '0', 'None', 'null']:
                hz_val = safe_float(raw_hz, 0.0)
                if hz_val > 0:
                    hz_status = "✅ Extracted by LLM" if (llm_result and llm_result.get('refresh_rate_hz') is not None) else "🌐 Scraped from Page"
                else:
                    hz_val = None
                    hz_status = "❌ Missing (Required)"
                    missing_fields.append("Refresh Rate")
            else:
                hz_val = None
                hz_status = "❌ Missing (Required)"
                missing_fields.append("Refresh Rate")

            # 4. Resolution (Strict - No Baseline Default)
            raw_res = data.get('resolution')
            if raw_res and str(raw_res).strip().lower() not in ['', 'unknown', 'none', 'null']:
                res_val = str(raw_res).strip()
                res_status = "✅ Extracted by LLM" if (llm_result and llm_result.get('resolution')) else "🌐 Scraped from Page"
            else:
                res_val = None
                res_status = "❌ Missing (Required)"
                missing_fields.append("Resolution")

            # 5. Panel Type (Strict - No Baseline Default)
            raw_panel = data.get('panel_type')
            if raw_panel and str(raw_panel).strip().lower() not in ['', 'unknown', 'none', 'null']:
                panel_val = str(raw_panel).strip()
                panel_status = "✅ Extracted by LLM" if (llm_result and llm_result.get('panel_type')) else "🌐 Scraped from Page"
            else:
                panel_val = None
                panel_status = "❌ Missing (Required)"
                missing_fields.append("Panel Type")

            # 6. Curved Display (Strict - No Baseline Default)
            raw_curved = data.get('is_curved')
            if raw_curved is not None and str(raw_curved).strip().lower() not in ['', 'none', 'null']:
                is_curved = 1 if raw_curved in [True, 1, '1', 'true', 'True'] else 0
                curved_status = "✅ Extracted by LLM" if (llm_result and llm_result.get('is_curved') is not None) else "🌐 Scraped from Page"
            else:
                is_curved = None
                curved_status = "❌ Missing (Required)"
                missing_fields.append("Curved Display")

            # 7. Gaming Tier (Strict - No Baseline Default)
            raw_gaming = data.get('is_gaming')
            if raw_gaming is not None and str(raw_gaming).strip().lower() not in ['', 'none', 'null']:
                is_gaming = 1 if raw_gaming in [True, 1, '1', 'true', 'True'] else 0
                gaming_status = "✅ Extracted by LLM" if (llm_result and llm_result.get('is_gaming') is not None) else "🌐 Scraped from Page"
            elif hz_val is not None and hz_val > 0:
                is_gaming = 1 if hz_val >= 100 else 0
                gaming_status = f"ℹ️ Inferred from {hz_val:.0f}Hz"
            else:
                is_gaming = None
                gaming_status = "❌ Missing (Required)"
                missing_fields.append("Gaming Tier")

            # 8. Frameless Bezel (ALLOWED FALLBACK: Defaults to Standard Bezel)
            raw_frameless = data.get('is_frameless')
            if raw_frameless is not None and str(raw_frameless).strip().lower() not in ['', 'none', 'null']:
                is_frameless = 1 if raw_frameless in [True, 1, '1', 'true', 'True'] else 0
                frameless_status = "🌐 Scraped from Page"
            else:
                is_frameless = 0
                frameless_status = "ℹ️ Default Bezel"

            # 9. Location (ALLOWED FALLBACK: Defaults to Colombo)
            raw_loc = data.get('location')
            if raw_loc and str(raw_loc).strip().lower() not in ['', 'other', 'unknown', 'none', 'null']:
                location_val = str(raw_loc).strip().capitalize()
                loc_status = "✅ Extracted by LLM" if (llm_result and llm_result.get('location')) else "🌐 Scraped from Page"
            else:
                location_val = 'Colombo'
                loc_status = "ℹ️ Default Hub (Colombo)"

            monitor_fields = [
                {"name": "Brand", "value": brand_val if brand_val else "[MISSING]", "status": brand_status},
                {"name": "Screen Size", "value": f"{size_val:.1f} Inches" if size_val is not None else "[MISSING]", "status": size_status},
                {"name": "Refresh Rate", "value": f"{hz_val:.0f} Hz" if hz_val is not None else "[MISSING]", "status": hz_status},
                {"name": "Resolution", "value": res_val if res_val else "[MISSING]", "status": res_status},
                {"name": "Panel Type", "value": panel_val if panel_val else "[MISSING]", "status": panel_status},
                {"name": "Curved Display", "value": ("Yes" if is_curved else "No") if is_curved is not None else "[MISSING]", "status": curved_status},
                {"name": "Gaming Tier", "value": ("Yes" if is_gaming else "No") if is_gaming is not None else "[MISSING]", "status": gaming_status},
                {"name": "Frameless Bezel", "value": "Yes" if is_frameless else "No", "status": frameless_status},
                {"name": "Location", "value": location_val, "status": loc_status},
            ]

            if missing_fields:
                log_required_fields_table('monitor', monitor_fields, predicted_price=None, missing_fields=missing_fields)
                return jsonify({
                    'success': False,
                    'category': 'monitor',
                    'error': f"Unable to predict monitor price. Missing required field(s): {', '.join(missing_fields)}. Please provide these specifications.",
                    'missing_fields': missing_fields,
                    'status': 'missing_required_fields'
                })

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
            
            log_required_fields_table('monitor', monitor_fields, predicted_price, lower_price, upper_price, data.get('price') or data.get('listed_price'), model_display_name)

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

@app.route('/metadata', methods=['GET'])
@app.route('/model_info', methods=['GET'])
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
    port = int(os.environ.get("PORT", 8004))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1")
    app.run(host='0.0.0.0', debug=debug, port=port, use_reloader=False)
