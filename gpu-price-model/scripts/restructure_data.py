#!/usr/bin/env python
"""
Restructure cleaned data into a standardized format.

This script transforms cleaned GPU data from all sources (ikman, mdcomputers, msk)
into a unified data structure for downstream processing.
"""

import json
import os
from pathlib import Path


# --- DATA RESTRUCTURING UTILITY ---
# Stored in: scripts/restructure_data.py
# Purpose: This script standardizes the messy, multi-source scraped data into a unified schema.
# It ensures that fields like 'Price' and 'URL' use the same keys regardless of the source website.

def main():
    """Main execution block for data restructuring."""
    # Get paths relative to script location
    project_root = Path(__file__).resolve().parents[1]
    input_file = project_root / "data" / "cleaned" / "all_scraped_data.json"
    output_dir = project_root / "data" / "final"
    output_file = output_dir / "restructured_scraped_data.json"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load input data
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    restructured_data = []

    for item in data:
        # Every source has different keys (e.g. 'Listing_ID' vs 'Product_ID').
        # We normalize them into a single consistent dictionary here.
        source = item.get("Source", "unknown")

        # --- Base target structure (Unified Schema) ---
        new_item = {
            "Product_ID": None,
            "Raw_Title": item.get("Raw_Title"),
            "Raw_Price": item.get("Raw_Price"),
            "Price_LKR": None,
            "Category": None,
            "Stock": None,
            "Stock_Status": None,
            "Product_URL": None,
            "Image_URL": None,
            "Brand": None,
            "Scraped_At_UTC": item.get("Scraped_At_UTC"),
            "Extracted_Model": item.get("Extracted_Model"),
            "VRAM_GB": item.get("VRAM_GB"),
            "Manufacturer": None,
            "Source": source,
        }

        # --- SOURCE-SPECIFIC MAPPING ---
        if source == "ikman":
            # Ikman specific fields
            new_item["Product_ID"] = item.get("Listing_ID")

            # Ikman uses Clean_Price_LKR
            new_item["Price_LKR"] = item.get("Clean_Price_LKR")
            new_item["Product_URL"] = item.get("Listing_URL")

            # Extract Category from Details (usually "Location, Category")
            details = item.get("Details")
            if details:
                parts = details.split(",")
                if len(parts) > 1:
                    new_item["Category"] = parts[1].strip()
                else:
                    new_item["Category"] = details.strip()
        else:
            # MDComputers and MSK specific fields
            new_item["Product_ID"] = item.get("Product_ID")
            new_item["Price_LKR"] = item.get("Price_LKR")

            # Fallback if Price_LKR is missing but Clean_Price_LKR exists
            if new_item["Price_LKR"] is None and "Clean_Price_LKR" in item:
                new_item["Price_LKR"] = item.get("Clean_Price_LKR")

            new_item["Category"] = item.get("Category")
            new_item["Stock"] = item.get("Stock")
            new_item["Stock_Status"] = item.get("Stock_Status")
            new_item["Product_URL"] = item.get("Product_URL")
            new_item["Image_URL"] = item.get("Image_URL")
            new_item["Brand"] = item.get("Brand")
            new_item["Manufacturer"] = item.get("Manufacturer")

        restructured_data.append(new_item)

    # Save restructured data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(restructured_data, f, indent=4)

    # The 'restructured_scraped_data.json' will be used by 'build_benchmark_features.py'
    # to create the final training dataset.
    print(f"✓ Restructured {len(restructured_data)} records")
    print(f"✓ Saved to {output_file.relative_to(project_root)}")


if __name__ == "__main__":
    main()
