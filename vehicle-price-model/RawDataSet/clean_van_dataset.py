import json
import re
import os

def clean_price(price_str):
    if not price_str or 'Negotiable' in price_str:
        return None
    digits = re.sub(r'[^\d]', '', price_str)
    return int(digits) if digits else None

def clean_number(val_str):
    if not val_str:
        return None
    digits = re.sub(r'[^\d.]', '', str(val_str))
    return float(digits) if digits else None

def extract_year(title):
    match = re.search(r'\b(19[89]\d|20[012]\d)\b', title)
    return int(match.group(1)) if match else None

def extract_location(url):
    if not url:
        return "Unknown"
    match = re.search(r'-sale-(.*?)-\d+$', url)
    if match:
        parts = match.group(1).split('-')
        return " ".join([p.capitalize() for p in parts])
    return "Unknown"

def extract_model_and_variant(title, brand, year):
    title_lower = title.lower()

    # --- SPECIFIC NORMALIZATION RULES FOR VANS ---
    
    # Toyota
    if "townace" in title_lower or "town ace" in title_lower: return "Townace", ""
    if "liteace" in title_lower or "lite ace" in title_lower: return "Liteace", ""
    if "dolphin" in title_lower: return "Dolphin", ""
    if "kdh" in title_lower: return "KDH", ""
    if "voxy" in title_lower: return "Voxy", ""

    # Suzuki / Toyota (Every)
    if "every" in title_lower: return "Every", ""

    # Nissan
    if "caravan e25" in title_lower or "e25" in title_lower: return "E25", ""
    if "caravan" in title_lower: return "Caravan", ""
    if "serena" in title_lower: return "Serena", ""
    if "vanette" in title_lower: return "Vanette", ""

    # Mazda
    if "bongo" in title_lower: return "Bongo", ""
    if "brawny" in title_lower: return "Brawny", ""

    # Isuzu
    if "fargo" in title_lower: return "Fargo", ""

    # Daihatsu
    if "hijet" in title_lower: return "Hijet", ""
    
    # --- END SPECIFIC RULES ---

    # Fallback to general logic
    t = title
    t = re.sub(rf'(?i)\b{brand}\b', '', t)
    if year:
        t = re.sub(rf'\b{year}\b', '', t)
    t = re.sub(r'(?i)\bVan\b', '', t)
    t = re.sub(r'(?i)Unregistered \(Recondition\)', '', t)
    t = re.sub(r'(?i)Registered \(Used\)', '', t)
    t = re.sub(r'(?i)\(Used\)', '', t)
    t = re.sub(r'(?i)Brand New', '', t)
    t = re.sub(r'(?i)Unregistered', '', t)
    t = re.sub(r'(?i)Recondition', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    
    parts = t.split(' ', 1)
    model = parts[0] if parts else "Unknown"
    variant = parts[1].strip() if len(parts) > 1 and parts[1].strip() else "Standard"
        
    return model.title() if model else "Unknown", variant

def main():
    input_file = "VAN_Data.json"
    output_dir = "Cleaned_Van_Brands"
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    brands_whitelist = {"Toyota", "Mazda", "Isuzu", "Nissan", "Daihatsu", "Suzuki", "Mitsubishi"}
    
    total_raw = len(data)
    filtered_out = 0
    brand_data = {brand: [] for brand in brands_whitelist}
    
    for item in data:
        raw_brand = item.get("Brand", "").strip().title()
        condition = item.get("Condition", "")
        title = item.get("Title", "")
        
        # Filter brand
        if raw_brand not in brands_whitelist:
            filtered_out += 1
            continue
            
        # Filter condition (remove unregistered/brand new/recondition)
        bad_conditions = ["unregistered", "brand new", "recondition"]
        is_bad = False
        for bc in bad_conditions:
            if bc in condition.lower() or bc in title.lower():
                is_bad = True
                break
                
        if is_bad:
            filtered_out += 1
            continue
            
        year = extract_year(title)
        model, variant = extract_model_and_variant(title, raw_brand, year)
        
        price = clean_price(item.get("Price"))
        if price is None:
            filtered_out += 1
            continue

        fuel_type = item.get("Fuel Type", "").title()
        engine_cc = clean_number(item.get("Engine (cc)"))

        # Force KDH Engine CC standardizations[cite: 3]
        if model == "KDH":
            if "Diesel" in fuel_type:
                engine_cc = 3000.0
            elif "Petrol" in fuel_type:
                engine_cc = 2000.0
            
        cleaned = {
            "brand": raw_brand,
            "vehicle_type": "Van",
            "model": model,
            "title_raw": title,
            "model_year": year,
            "mileage_km": clean_number(item.get("Mileage")),
            "fuel_type": fuel_type,
            "location": extract_location(item.get("Listing_URL")),
            "price": price,
            "engine_cc": engine_cc,
            "transmission": item.get("Gear", "").title(),
            "condition": "Registered (Used)",
            "variant": variant
        }
        
        brand_data[raw_brand].append(cleaned)
        
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Total raw records loaded: {total_raw}")
    print(f"Records filtered out: {filtered_out}")
    print("Final record count saved per brand:")
    
    for brand, records in brand_data.items():
        if records:
            out_file = os.path.join(output_dir, f"{brand}.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            print(f"  {brand}: {len(records)}")

if __name__ == "__main__":
    main()