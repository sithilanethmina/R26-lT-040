#!/usr/bin/env python
"""
Preprocess restructured data for model training.

This script:
- Filters records with valid price and model data
- Validates models against trusted GPU specs
- Enriches VRAM information from multiple fallback sources
- Extracts and standardizes brand information
- Outputs training-ready dataset
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd


# Common GPU brand names for extraction
COMMON_BRANDS = [
    "asus",
    "msi",
    "gigabyte",
    "zotac",
    "evga",
    "palit",
    "sapphire",
    "galax",
    "powercolor",
    "inno3d",
    "pny",
    "colorful",
    "xfx",
    "asrock",
    "emtek",
]


def main():
    """Preprocess data for training."""
    project_root = Path(__file__).resolve().parents[1]
    
    if str(project_root / "src") not in sys.path:
        sys.path.insert(0, str(project_root / "src"))
    from gpu_price_predictor.pipeline import normalize_model
    
    input_file = project_root / "data" / "final" / "restructured_scraped_data.json"
    trusted_file = project_root / "data" / "final" / "trusted_gpu_specs.json"
    output_file = project_root / "data" / "final" / "training_data_v2.json"
    dumps_dir = project_root / "data" / "dumps"
    
    # Ensure directories exist
    dumps_dir.mkdir(parents=True, exist_ok=True)

    # Load input data
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load trusted specs
    trusted_specs = {}
    if trusted_file.exists():
        with open(trusted_file, "r", encoding="utf-8") as f:
            raw_trusted_specs = json.load(f)
            # Normalize keys by removing spaces
            trusted_specs = {
                k.replace(" ", "").upper(): v for k, v in raw_trusted_specs.items()
            }

    final_data = []
    discarded_items = []

    for item in data:
        price = item.get("Price_LKR")
        model = item.get("Extracted_Model")
        
        # Standardize the model formatting (e.g. GTX 1050TI -> GTX 1050 TI)
        if model:
            model = normalize_model(model)
            
        vram = item.get("VRAM_GB")
        raw_title = item.get("Raw_Title", "")
        brand = item.get("Brand")

        # Filter: Drop records without price or model
        if price is None or model is None:
            item["Discard_Reason"] = "Missing Price or Model"
            discarded_items.append(item)
            continue

        # Validate: Check if model is in trusted dataset
        normalized_model = model.replace(" ", "").upper()
        trusted_info = trusted_specs.get(normalized_model)

        if not trusted_info:
            # Data Validation: Invalid/unknown models are likely typos or fakes
            # (e.g., GTX 6501). Discard to prevent training data poisoning.
            safe_title = raw_title[:50].encode('ascii', 'ignore').decode('ascii')
            print(f"WARNING: Discarding invalid model: '{model}' (Raw: {safe_title}...)")
            item["Discard_Reason"] = "Invalid or unknown GPU model"
            discarded_items.append(item)
            continue

        # Enrich VRAM from multiple fallback sources
        vram_val = None

        # Try 1: Convert existing VRAM_GB value
        if vram is not None:
            try:
                vram_val = float(vram)
            except ValueError:
                pass

        # Try 2: Extract VRAM from Raw_Title if still missing
        if vram_val is None and raw_title:
            match = re.search(r"(\d+)\s*(?:gb|g)\b", raw_title, re.IGNORECASE)
            if match:
                try:
                    vram_val = float(match.group(1))
                except ValueError:
                    pass

        # Try 3: Get VRAM from trusted specs
        if vram_val is None and trusted_info:
            vram_val = trusted_info.get("VRAM_GB")

        # Extract and standardize brand
        extracted_brand = None

        # Try 1: Use existing brand if valid
        if brand and brand.lower() != "random brand":
            extracted_brand = brand

        # Try 2: Extract brand from title
        if not extracted_brand and raw_title:
            title_lower = raw_title.lower()
            for b in COMMON_BRANDS:
                if b in title_lower:
                    extracted_brand = b.upper()
                    break

        # Try 3: Get manufacturer from trusted specs
        if not extracted_brand and trusted_info:
            extracted_brand = trusted_info.get("Manufacturer")

        # Ensure all brands are fully capitalized (e.g. ZOTAC)
        if extracted_brand:
            extracted_brand = extracted_brand.upper()

        # Add to final dataset
        final_data.append(
            {
                "Listing_ID": item.get("Product_ID"),
                "Listing_URL": item.get("Product_URL"),
                "Raw_Title": raw_title,
                "Price_LKR": price,
                "Extracted_Model": model,
                "VRAM_GB": vram_val,
                "Brand": extracted_brand,
                "Scraped_At_UTC": item.get("Scraped_At_UTC"),
            }
        )

    # --- OUTLIER REMOVAL ---
    if final_data:
        df = pd.DataFrame(final_data)
        original_count = len(df)
        
        def get_outlier_mask(s):
            if len(s) < 4:
                return pd.Series(True, index=s.index)
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                return pd.Series(True, index=s.index)
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return (s >= lower_bound) & (s <= upper_bound)
            
        # Group by BOTH Model and Brand so premium brands (like ASUS) aren't compared to cheaper brands
        mask = df.groupby(["Extracted_Model", "Brand"], dropna=False)["Price_LKR"].transform(get_outlier_mask)
        mask = mask.astype(bool)
        
        filtered_df = df[mask]
        outliers_df = df[~mask]
        outliers_data = outliers_df.to_dict(orient="records")
        outliers_removed = len(outliers_data)
        
        # Save outliers dump
        for item in outliers_data:
            item["Discard_Reason"] = "Price Outlier (IQR Filtering)"
            if pd.isna(item.get("VRAM_GB")):
                item["VRAM_GB"] = None
            if pd.isna(item.get("Brand")):
                item["Brand"] = None
        
        outliers_file = dumps_dir / "outliers_dump.json"
        with open(outliers_file, "w", encoding="utf-8") as f:
            json.dump(outliers_data, f, indent=4)
        
        # In case the DataFrame index gets messed up or it drops columns, we recreate the list of dicts
        # pandas sometimes returns an empty DataFrame if no groups, but we handled that
        final_data = filtered_df.to_dict(orient="records")
        # Ensure we don't have NaN for None values (pandas converts None to NaN sometimes)
        for item in final_data:
            if pd.isna(item.get("VRAM_GB")):
                item["VRAM_GB"] = None
            if pd.isna(item.get("Brand")):
                item["Brand"] = None
    else:
        outliers_removed = 0
    # -----------------------

    # Save training data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=4)

    # Save discarded items dump
    discarded_file = dumps_dir / "discarded_dump.json"
    with open(discarded_file, "w", encoding="utf-8") as f:
        json.dump(discarded_items, f, indent=4)

    print(f"[OK] Processed {len(final_data) + outliers_removed} valid records")
    print(f"[OK] Discarded {len(discarded_items)} invalid records (saved to {discarded_file.relative_to(project_root)})")
    print(f"[OK] Removed {outliers_removed} price outliers via IQR (saved to {outliers_file.relative_to(project_root)})")
    print(f"[OK] Saved {len(final_data)} clean records to {output_file.relative_to(project_root)}")


if __name__ == "__main__":
    main()
