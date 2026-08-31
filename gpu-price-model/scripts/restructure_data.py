#!/usr/bin/env python
"""
Restructure cleaned data into a standardized format.

This script transforms cleaned GPU data from all sources (ikman, mdcomputers, msk)
into a unified data structure for downstream processing.
"""

import json
import os
from pathlib import Path


def main():
    """Normalize multi-source scraped data into a unified JSON schema."""
    project_root = Path(__file__).resolve().parents[1]
    input_file = project_root / "data" / "cleaned" / "all_scraped_data.json"
    output_dir = project_root / "data" / "final"
    output_file = output_dir / "restructured_scraped_data.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    restructured_data = []

    for item in data:
        source = item.get("Source", "unknown")

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
            "Full_Description": item.get("Full_Description"),
        }

        if source == "ikman":
            new_item["Product_ID"] = item.get("Listing_ID")
            new_item["Price_LKR"] = item.get("Clean_Price_LKR")
            new_item["Product_URL"] = item.get("Listing_URL")

            details = item.get("Details")
            if details:
                parts = details.split(",")
                new_item["Category"] = parts[1].strip() if len(parts) > 1 else details.strip()
        else:
            # MDComputers and MSK store structured retailer metadata
            new_item["Product_ID"] = item.get("Product_ID")
            new_item["Price_LKR"] = item.get("Price_LKR")

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

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(restructured_data, f, indent=4)

    print(f"[OK] Restructured {len(restructured_data)} records")
    print(f"[OK] Saved to {output_file.relative_to(project_root)}")


if __name__ == "__main__":
    main()
