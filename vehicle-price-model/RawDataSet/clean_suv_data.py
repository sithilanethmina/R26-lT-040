import os
import json
import re
import pandas as pd
import numpy as np

# Directory Setup
INPUT_DIR = "Separated_Suv_Brands"
OUTPUT_DIR = "Cleaned_Suv_Brands"

KNOWN_COMPLEX_MODELS = [
    "Land Cruiser Prado", "Land Cruiser", "Range Rover Sport", "Range Rover Autobiography", 
    "Range Rover", "Montero Sport", "Pajero Mini", "Pajero IO", "Pajero", "Glory I-Auto", 
    "Glory 580", "Glory 330", "Grand Vitara", "Urban Cruiser Taisor", "Taisor", 
    "Yaris Cross", "Discovery 4S", "Discovery 4", "Discovery 5", "Discovery 2", 
    "Discovery", "Freelander 2", "Freelander", "Defender", "Eclipse Cross", "KUV 100", 
    "Santa Fe", "Beijing X55", "Nomad Extreme", "Nomad", "Rexton W", "Rexton"
]
# Sort by length descending to match longest possible multi-word model first
KNOWN_COMPLEX_MODELS.sort(key=len, reverse=True)

GENERIC_TERMS = [
    "SUV", "(Used)", "Used", "Brand New", "Unregistered", "(Recondition)", "Recondition"
]

def clean_and_extract_model_variant(title, brand, year, engine_cc):
    if not isinstance(title, str):
        return pd.Series([None, None])
    
    cleaned = title
    
    # 1. Clean Title String
    # Remove year
    if pd.notna(year):
        cleaned = re.sub(rf'\b{int(year)}\b', '', cleaned)
        
    # Remove brand (case insensitive)
    if isinstance(brand, str) and brand:
        cleaned = re.sub(rf'\b{re.escape(brand)}\b', '', cleaned, flags=re.IGNORECASE)
        
    # Remove generic terms (case insensitive)
    for term in GENERIC_TERMS:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        cleaned = pattern.sub('', cleaned)
        
    # Cleanup extra spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    model = None
    variant = ""
    
    # 2. Multi-Word Complex Model Matching
    cleaned_lower = cleaned.lower()
    for cm in KNOWN_COMPLEX_MODELS:
        if cleaned_lower.startswith(cm.lower()):
            model = cm
            variant = cleaned[len(cm):].strip()
            break
            
    # 3. Single-Word Fallback
    if model is None:
        parts = cleaned.split(' ', 1)
        if len(parts) > 0 and parts[0]:
            model = parts[0]
            if len(parts) > 1:
                variant = parts[1].strip()
        else:
            model = "Unknown"
            variant = ""
            
    # ── ADVANCED SUV MODEL NORMALIZATION & BUSINESS RULES ──────────
    combined_check = f"{model} {title}".lower()
    
    # Audi Q-Series
    if re.search(r'\bq2\b', combined_check):
        model = "Q2"
    elif re.search(r'\bq3\b', combined_check):
        model = "Q3"
    elif re.search(r'\bq5\b', combined_check):
        model = "Q5"
    elif re.search(r'\bq7\b', combined_check):
        model = "Q7"
        
    # BMW
    elif re.search(r'\bx1\b', combined_check):
        model = "X1"
    elif re.search(r'\bx2\b', combined_check):
        model = "X2"
    elif re.search(r'\bx3\b', combined_check):
        model = "X3"
    elif re.search(r'\bx5\b', combined_check):
        model = "X5"

    # MG
    elif "zs" in combined_check:
        model = "ZS"

    # Mitsubishi Rules
    elif "outlander" in combined_check:
        model = "Outlander"
    elif "pajero io" in combined_check or "gdi io" in combined_check or "pajero dgi" in combined_check:
        model = "Pajero Io"
    elif "eclipse cross" in combined_check:
        model = "Eclipse Cross"
    elif "asx" in combined_check:
        model = "ASX"
    elif "montero sport" in combined_check:
        model = "Montero Sport"
    elif "montero" in combined_check:
        if pd.notna(year) and int(year) <= 2006:
            model = "Montero 3rd gen"
        elif pd.notna(year) and int(year) >= 2007:
            model = "Montero 4th gen"
        else:
            model = "Montero"

    # A. Box Prado
    elif "box prado" in combined_check or "bj75" in combined_check:
        model = "Box Prado"
    # B. X-Trail Variations
    elif "xtrail" in combined_check or "x-trail" in combined_check:
        model = "X-Trail"
    # C. CHR Variations
    elif "chr" in combined_check or "c-hr" in combined_check:
        model = "CHR"
    # D. RAV4 Variations
    elif "rav4" in combined_check:
        model = "RAV4"
    # E. Raize Variations
    elif "raize" in combined_check:
        model = "Raize"
    # F. Vitara / Escudo Family -> Vitara
    elif any(k in combined_check for k in ["vitara", "escudo"]):
        model = "Vitara"
    # G. Xbee
    elif "xbee" in combined_check:
        model = "Xbee"
    # H. Fronx
    elif "fronx" in combined_check:
        model = "Fronx"
    # I. S Cross
    elif "s cross" in combined_check or "scross" in combined_check:
        model = "S Cross"
    # J. Jimny
    elif "jimny" in combined_check:
        model = "Jimny"
    # K. Vezel
    elif "vezel" in combined_check:
        model = "Vezel"
    # L. CRV / CR-V
    elif "crv" in combined_check or "cr-v" in combined_check:
        model = "CRV"
    # M. Tucson
    elif "tucson" in combined_check:
        model = "Tucson"
    # N. Sorento
    elif "sorento" in combined_check:
        model = "Sorento"
    # O. Rexton (including Ssangyong Rexton)
    elif "rexton" in combined_check:
        model = "Rexton"
    # P. Kyron
    elif "kyron" in combined_check:
        model = "Kyron"
    # Q. Actyon
    elif "actyon" in combined_check:
        model = "Actyon"
    # R. Korando
    elif "korando" in combined_check:
        model = "Korando"
    # S. KUV 100
    elif "kuv" in combined_check and "100" in combined_check:
        model = "KUV 100"
    # T. Peugeot 3008, 5008, 2008
    elif "3008" in combined_check:
        model = "3008"
    elif "5008" in combined_check:
        model = "5008"
    elif "2008" in combined_check:
        model = "2008"
    # U. Land Cruiser / Prado / V8 Logic
    elif "land cruiser" in combined_check or "prado" in combined_check or "v8" in combined_check:
        if "sahara" in combined_check or "v8" in combined_check or "zx" in combined_check:
            model = "V8"
        elif engine_cc and not pd.isna(engine_cc) and engine_cc > 4000:
            model = "V8"
        elif "land cruiser" in combined_check:
            model = "Prado"
        else:
            model = "Prado"
    # ───────────────────────────────────────────────────────────────

    # 4. Variant Fallback
    variant = re.sub(r'^[-,\s]+|[-,\s]+$', '', variant)
    
    if not variant or re.match(r'^[^a-zA-Z0-9]+$', variant):
        variant = "Standard"
        
    return pd.Series([model, variant])

def extract_year(title):
    if not isinstance(title, str):
        return np.nan
    match = re.search(r'\b(19[8-9]\d|20[0-1]\d|202[0-6])\b', title)
    if match:
        return int(match.group(1))
    return np.nan

def clean_price(price_str):
    if not isinstance(price_str, str):
        return np.nan
    if "negotiable" in price_str.lower():
        return -1
    
    num_str = re.sub(r'[^\d]', '', price_str)
    if num_str:
        return int(num_str)
    return np.nan

def process_files():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory {INPUT_DIR} does not exist.")
        return
        
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    
    for filename in files:
        filepath = os.path.join(INPUT_DIR, filename)
        brand_name = filename.replace('.json', '')
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, list) or len(data) == 0:
                print(f"[{brand_name}] Skipping: Empty or invalid JSON format.")
                continue
                
            df = pd.DataFrame(data)
            initial_count = len(df)
            
            required_keys = ["Price", "Mileage", "Engine (cc)", "Condition", "Title", "Fuel Type", "Location", "Gear", "Brand"]
            for key in required_keys:
                if key not in df.columns:
                    df[key] = np.nan
                    
            if df['Brand'].isnull().all():
                df['Brand'] = brand_name
                
            # 1. Price Validation
            df['price_temp'] = df['Price'].astype(str).apply(clean_price)
            df = df[df['price_temp'] != -1]
            df = df[(df['price_temp'] >= 1000000) & (df['price_temp'] <= 120000000)]
            
            # Model year extraction
            df['model_year'] = df['Title'].apply(extract_year)
            df = df.dropna(subset=['model_year'])
            df['model_year'] = df['model_year'].astype(int)
            
            # 2. Fake Mileage Detection
            def extract_mileage(row):
                mil_str = str(row['Mileage'])
                cond_str = str(row['Condition']).lower()
                
                num_str = re.sub(r'[^\d]', '', mil_str)
                if not num_str:
                    return np.nan
                    
                val = int(num_str)
                if val < 100 and ('registered' in cond_str or 'used' in cond_str):
                    return np.nan
                return val
            df['mileage_km'] = df.apply(extract_mileage, axis=1)
            
            # 3. Engine CC Filtering
            def extract_cc(cc_str):
                if pd.isna(cc_str):
                    return np.nan
                num_str = re.sub(r'[^\d]', '', str(cc_str))
                if num_str:
                    val = int(num_str)
                    if 600 <= val <= 6000:
                        return val
                return np.nan
            df['engine_cc'] = df['Engine (cc)'].apply(extract_cc)
            
            # 4. Intelligent Model & Variant Extraction Logic
            df[['model', 'variant']] = df.apply(lambda row: clean_and_extract_model_variant(row['Title'], row['Brand'], row['model_year'], row['engine_cc']), axis=1)
            
            # 5. Output Schema Definition & Key Mapping
            final_df = pd.DataFrame({
                'brand': df['Brand'],
                'vehicle_type': 'SUV',
                'model': df['model'],
                'title_raw': df['Title'],
                'model_year': df['model_year'],
                'mileage_km': df['mileage_km'],
                'fuel_type': df['Fuel Type'],
                'location': df['Location'],
                'price': df['price_temp'].astype(int),
                'engine_cc': df['engine_cc'],
                'transmission': df['Gear'],
                'condition': df['Condition'],
                'variant': df['variant']
            })
            
            final_df = final_df.replace({np.nan: None})
            
            final_count = len(final_df)
            dropped_count = initial_count - final_count
            
            out_filepath = os.path.join(OUTPUT_DIR, filename)
            
            with open(out_filepath, 'w', encoding='utf-8') as f:
                json.dump(final_df.to_dict(orient='records'), f, indent=4, ensure_ascii=False)
                
            print(f"[{brand_name}] Scanned: {initial_count} | Dropped: {dropped_count} | Cleaned: {final_count} exported.")
            
        except Exception as e:
            print(f"[{brand_name}] Error processing file: {e}")

if __name__ == "__main__":
    print("Starting SUV Dataset Cleaning Process...")
    process_files()
    print("Process Complete.")