#!/usr/bin/env python
"""
Extract structured condition tags from all scraped listings and save to data/final/condition_tags.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gpu_price_predictor.condition_extractor import extract_condition_tags


def main() -> None:
    input_file = PROJECT_ROOT / "data" / "final" / "restructured_scraped_data.json"
    output_file = PROJECT_ROOT / "data" / "final" / "condition_tags.json"

    if not input_file.exists():
        print(f"Error: {input_file} not found. Run restructure_data.py first.")
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        listings = json.load(f)

    print(f"Processing {len(listings)} listings for condition tagging...")

    tagged_listings = []
    tag_counts = {
        "has_warranty": 0,
        "warranty_months_gt_0": 0,
        "needs_repair": 0,
        "urgent_sale": 0,
        "tested_working": 0,
        "good_condition": 0,
        "brand_new": 0,
        "price_negotiable": 0,
        "is_shop": 0,
        "delivery_available": 0,
    }

    for item in listings:
        desc = item.get("Full_Description") or item.get("Raw_Title") or ""
        source = item.get("Source", "ikman")
        tags = extract_condition_tags(desc, source=source)

        for k in tag_counts:
            if k == "warranty_months_gt_0":
                if tags["warranty_months"] > 0:
                    tag_counts[k] += 1
            elif tags.get(k):
                tag_counts[k] += 1

        record = {
            "Product_ID": item.get("Product_ID"),
            "Product_URL": item.get("Product_URL"),
            "Raw_Title": item.get("Raw_Title"),
            "Price_LKR": item.get("Price_LKR"),
            "Extracted_Model": item.get("Extracted_Model"),
            "VRAM_GB": item.get("VRAM_GB"),
            "Brand": item.get("Brand"),
            "Source": source,
            "condition_tags": tags,
        }
        tagged_listings.append(record)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tagged_listings, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] Tagged {len(tagged_listings)} records -> {output_file.relative_to(PROJECT_ROOT)}")
    print("\n--- Condition Tag Summary ---")
    for k, v in tag_counts.items():
        print(f"  {k:<22}: {v:>4} ({v / len(tagged_listings) * 100:>5.1f}%)")


if __name__ == "__main__":
    main()
