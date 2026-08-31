import json
import os
import re
import glob

def clean_price(price_str):
    if not price_str:
        return None
    if 'negotiable' in str(price_str).lower():
        return None
    cleaned = re.sub(r'[^\d]', '', str(price_str))
    if cleaned:
        val = int(cleaned)
        # Drop prices below 500,000 LKR
        if val < 500000:
            return None
        return val
    return None

def clean_mileage(mileage_str):
    if not mileage_str:
        return None
    cleaned = re.sub(r'[^\d]', '', str(mileage_str))
    if cleaned:
        return int(cleaned)
    return None

def clean_engine_cc(engine_str):
    if not engine_str:
        return None
    cleaned = re.sub(r'[^\d]', '', str(engine_str))
    if cleaned:
        return int(cleaned)
    return None

def normalize_fuel(fuel_raw):
    if not fuel_raw:
        return "Unknown"
    fuel_lower = fuel_raw.lower()
    if 'petrol' in fuel_lower:
        return 'Petrol'
    if 'diesel' in fuel_lower:
        return 'Diesel'
    if 'hybrid' in fuel_lower:
        return 'Hybrid'
    if 'electric' in fuel_lower or 'ev' in fuel_lower:
        return 'Electric'
    return fuel_raw.strip()

def normalize_condition(condition_raw):
    if not condition_raw:
        return "Unknown"
    cond_lower = condition_raw.lower()
    if 'unregistered' in cond_lower:
        if 'recondition' in cond_lower:
            return "Unregistered Recondition"
        return "Unregistered Brand New"
    if 'brand new' in cond_lower:
        return "Unregistered Brand New"
    if 'registered' in cond_lower or 'used' in cond_lower:
        return "Registered Used"
    return condition_raw.strip()

def normalize_transmission(gear_raw):
    if not gear_raw:
        return "Unknown"
    gear_lower = gear_raw.lower()
    if 'auto' in gear_lower or 'tiptronic' in gear_lower or 'cvt' in gear_lower:
        return 'Automatic'
    if 'manual' in gear_lower:
        return 'Manual'
    return gear_raw.strip()

def extract_year(item):
    year_str = item.get("Year")
    if year_str:
        cleaned = re.sub(r'[^\d]', '', str(year_str))
        if cleaned and len(cleaned) >= 4:
            return int(cleaned[:4])
    title = item.get("Title", "")
    match = re.search(r'\b(19\d{2}|20\d{2})\b', title)
    if match:
        return int(match.group(1))
    return None

def extract_model_variant(title, brand, vehicle_type, year=None, price=None):
    clean_title = title
    if brand:
        clean_title = re.sub(r'(?i)\b' + re.escape(brand) + r'\b', '', clean_title)
    
    clean_title = re.sub(r'(?i)\b(Car|SUV|Van|Jeep|Cab|Truck|Bus|Used|Unregistered)\b', '', clean_title)
    clean_title = re.sub(r'\b(19\d{2}|20\d{2})\b', '', clean_title)
    clean_title = re.sub(r'[\(\)]', '', clean_title)
    clean_title = re.sub(r'\s+', ' ', clean_title).strip()
    
    if not clean_title:
        return "Unknown", ""

    model = ""
    variant = ""
    title_lower = title.lower()

    # --- SPECIFIC NORMALIZATION RULES FOR CARS ---
    
    # BMW E46 Drop Rule
    if "e46" in title_lower:
        return "DROP", ""

    # Tata
    if "indica" in title_lower: return "Indica", ""
    if "nano" in title_lower: return "Nano", ""
    if "indigo" in title_lower: return "Indigo", ""

    # Audi
    if "a1" in title_lower: return "A1", ""
    if "a3" in title_lower: return "A3", ""
    if "a4" in title_lower: return "A4", ""
    if "a5" in title_lower: return "A5", ""
    if "a6" in title_lower: return "A6", ""

    # BMW
    if "318i" in title_lower: return "318i", ""
    if "520d" in title_lower: return "520D", ""
    if "320d" in title_lower: return "320D", ""
    if "730ld" in title_lower: return "730Ld", ""
    if "530e" in title_lower: return "530e", ""
    if "523i" in title_lower: return "523i", ""
    if "mini cooper" in title_lower: return "Mini Cooper", ""
    if "i8" in title_lower: return "I8", ""

    # Mercedes-Benz
    if "c180" in title_lower: return "C180", ""
    if "e200" in title_lower: return "E200", ""
    if "a200" in title_lower: return "A200", ""
    if "e300" in title_lower: return "E300", ""
    if "slk200" in title_lower or "slk 200" in title_lower: return "SLK200", ""
    if "e220" in title_lower: return "E220", ""
    if "s350" in title_lower: return "S350", ""
    if "e180" in title_lower: return "E180", ""
    if "e240" in title_lower: return "E240", ""
    if "c200" in title_lower: return "C200", ""
    if "cla 200" in title_lower or "cla200" in title_lower: return "CLA 200", ""
    if "c250" in title_lower: return "C250", ""
    if "w210" in title_lower: return "W210", ""
    if "e350" in title_lower: return "E350", ""

    # Perodua
    if "kelisa" in title_lower: return "Kelisa", ""
    if "bezza" in title_lower: return "Bezza", ""
    if "kenari" in title_lower: return "Kenari", ""
    if "viva elite" in title_lower: return "Viva Elite", ""

    # Toyota
    if "aqua" in title_lower: return "Aqua", ""
    if "prius" in title_lower: return "Prius", ""
    if "vitz" in title_lower: return "Vitz", ""
    if "premio" in title_lower: return "Premio", ""
    if "axio" in title_lower: return "Axio", ""
    if "carina" in title_lower: return "Carina", ""
    if "allion" in title_lower: return "Allion", ""
    if "vios" in title_lower: return "Vios", ""
    if "passo" in title_lower: return "Passo", ""
    if "ae110" in title_lower: return "AE110", ""
    if "ce110" in title_lower: return "CE110", ""
    if "121" in title_lower and ("corolla" in title_lower or brand.lower() == "toyota"): return "Corolla 121", ""
    if "141" in title_lower and ("corolla" in title_lower or brand.lower() == "toyota"): return "Corolla 141", ""
    if "110" in title_lower and ("corolla" in title_lower or brand.lower() == "toyota"): return "110", ""

    # Honda
    if "fit gp1" in title_lower: return "Fit GP1", ""
    if "fit gp5" in title_lower: return "FIT GP5", ""
    if "civic fd3" in title_lower: return "Civic FD3", ""
    if "civic fd4" in title_lower: return "Civic FD4", ""
    if "civic fd1" in title_lower: return "Civic FD1", ""
    if "civic es8" in title_lower: return "Civic ES8", ""
    if "civic es5" in title_lower: return "Civic ES5", ""
    if "grace" in title_lower: return "Honda Grace", ""
    if "insight" in title_lower: return "Insight", ""
    if "civic" in title_lower and year is not None and year > 2012 and price is not None and price > 10000000:
        return "Civic", ""

    # Suzuki
    if "wagon r" in title_lower or "wagonr" in title_lower:
        if "stingray" in title_lower: return "Wagon R Stingray", ""
        if "fz" in title_lower: return "Wagon R FZ", ""
        if "fx" in title_lower: return "Wagon R FX", ""
    if "alto k10" in title_lower: return "Alto K10", ""
    if "alto" in title_lower: return "Alto", ""
    if "celerio" in title_lower: return "Celerio", ""
    if "swift" in title_lower: return "Swift", ""
    if "hustler" in title_lower: return "Hustler", ""
    if "maruti" in title_lower: return "Maruti", ""
    if "spacia" in title_lower: return "Spacia", ""

    # Micro
    if "panda cross" in title_lower: return "Panda Cross", ""
    if "panda" in title_lower: return "Panda", ""
    if "mx7" in title_lower: return "MX7", ""
    if "emgrand" in title_lower: return "Emgrand", ""

    # Hyundai
    if "accent" in title_lower: return "Accent", ""
    if "sonata" in title_lower: return "Sonata", ""
    if "eon" in title_lower: return "Eon", ""
    if "elantra" in title_lower: return "Elantra", ""

    # Mazda
    if any(x in title_lower for x in ["axela", "mazda 3", "mazda 6", "mazda3", "mazda6"]): 
        return "Axela", ""
    if "demio" in title_lower: return "Demio", ""
    if "familia" in title_lower: return "Familia", ""

    # Mitsubishi
    if "cs3" in title_lower: return "CS3", ""
    if "cs1" in title_lower: return "CS1", ""
    if "cs2" in title_lower: return "CS2", ""

    # Nissan
    if "fb13" in title_lower: return "FB13", ""
    if "fb14" in title_lower: return "FB14", ""
    if "fb15" in title_lower: return "FB15", ""
    if "n16" in title_lower: return "N16", ""
    if "n17" in title_lower: return "N17", ""
    if "leaf" in title_lower: return "Leaf", ""
    if "cefiro" in title_lower: return "Cefiro", ""
    if "march" in title_lower:
        if "k10" in title_lower: return "March K10", ""
        if "k11" in title_lower: return "March K11", ""
        if "k12" in title_lower or "ak12" in title_lower: return "March K12", ""
    if "tiida" in title_lower: return "Tiida", ""

    # Daihatsu
    if "mira" in title_lower: return "Mira", ""

    # --- END SPECIFIC RULES ---

    # Fallback to general logic for other vehicles
    multi_word_models = [
        "Wagon R", "Land Cruiser", "Range Rover", "CR V", "CR-V", 
        "HR V", "HR-V", "Santa Fe", "Alfa Romeo", "Aston Martin"
    ]
    
    clean_title_lower = clean_title.lower()
    
    for mwm in multi_word_models:
        if clean_title_lower.startswith(mwm.lower()):
            model = clean_title[:len(mwm)]
            variant = clean_title[len(mwm):].strip()
            break
            
    if not model:
        parts = clean_title.split(' ', 1)
        model = parts[0]
        variant = parts[1] if len(parts) > 1 else ""
        
    return model.strip(), variant.strip()

def process_file(filepath):
    cleaned_records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading JSON from {filepath}")
            return []
    
    for item in data:
        price_raw = item.get("Price", "")
        price = clean_price(price_raw)
        
        # Skip record if price is invalid, below 500000, empty, or negotiable
        if price is None:
            continue
            
        brand = item.get("Brand", "Unknown")
        vehicle_type = item.get("Vehicle_Type", "Car")
        title_raw = item.get("Title", "")
        
        year = extract_year(item)
        if year is None:
            year = 2010  # Fallback
            
        # Extract model with access to year and price logic
        model, variant = extract_model_variant(title_raw, brand, vehicle_type, year, price)
        
        # Skip record if it was flagged to be dropped (e.g., BMW E46)
        if model == "DROP":
            continue
            
        registration_year = year
        model_year = year
        manufacture_year = registration_year - 1 if registration_year >= 1 else registration_year
        
        vehicle_age = max(1, 2026 - registration_year)
        
        mileage_km = clean_mileage(item.get("Mileage"))
        mileage_per_year = None
        if mileage_km is not None:
            mileage_per_year = round(mileage_km / vehicle_age, 2)
            
        engine_cc = clean_engine_cc(item.get("Engine (cc)"))
        fuel_type_raw = item.get("Fuel Type", "")
        fuel_type = normalize_fuel(fuel_type_raw)
        transmission = normalize_transmission(item.get("Gear"))
        condition = normalize_condition(item.get("Condition"))
        location = item.get("Location", "Unknown")
        
        cleaned_record = {
            "brand": brand,
            "vehicle_type": vehicle_type,
            "model": model,
            "variant": variant,
            "title_raw": title_raw,
            "model_year": model_year,
            "manufacture_year": manufacture_year,
            "registration_year": registration_year,
            "vehicle_age": vehicle_age,
            "mileage_km": mileage_km,
            "mileage_per_year": mileage_per_year,
            "engine_cc": engine_cc,
            "fuel_type_raw": fuel_type_raw,
            "fuel_type": fuel_type,
            "transmission": transmission,
            "condition": condition,
            "location": location,
            "price": price
        }
        cleaned_records.append(cleaned_record)
        
    return cleaned_records

def main():
    input_dir = "Separated_Brands"
    output_dir = "Cleaned_Brands"
    master_file = "Master_Cleaned_Dataset.json"
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    all_files = glob.glob(os.path.join(input_dir, "*.json"))
    master_records = []
    
    print(f"Found {len(all_files)} files in '{input_dir}'. Processing...")
    
    for filepath in all_files:
        filename = os.path.basename(filepath)
        records = process_file(filepath)
        
        if records:
            out_filepath = os.path.join(output_dir, filename)
            try:
                with open(out_filepath, 'w', encoding='utf-8') as out_f:
                    json.dump(records, out_f, indent=4, ensure_ascii=False)
                master_records.extend(records)
            except Exception as e:
                print(f"Error saving {out_filepath}: {e}")
                
    # Save master dataset
    try:
        if master_records:
            with open(master_file, 'w', encoding='utf-8') as mf:
                json.dump(master_records, mf, indent=4, ensure_ascii=False)
            print(f"\nProcessing complete!")
            print(f"Saved {len(all_files)} cleaned files to '{output_dir}/'")
            print(f"Master compiled file saved as '{master_file}' ({len(master_records)} valid records with prices)")
        else:
            print("\nNo valid records with prices found across all files.")
    except Exception as e:
        print(f"Error saving master file: {e}")

if __name__ == "__main__":
    main()