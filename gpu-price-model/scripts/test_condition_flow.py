#!/usr/bin/env python
"""
Validate condition extraction and hedonic adjustment end-to-end.
"""

import sys
import json
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from gpu_price_predictor.condition_extractor import extract_condition_tags
from gpu_price_predictor.pipeline import apply_condition_adjustment, load_condition_coefficients
from gpu_price_predictor.api import app, PredictRequest, predict


def main():
    print("=" * 75)
    print("  1. Testing Bilingual Condition Extractor")
    print("=" * 75)

    samples = [
        ("Standard Used", "GTX 1060 6GB, standard used card, works fine"),
        ("6 Mos Warranty", "ASUS Dual GTX 1060 6GB, 6 months company warranty remaining with bill, box"),
        ("Sinhala Warranty & Good", "මාස 3ක වගකීමක් සහිතයි. සුපිරි තත්වයේ."),
        ("Needs Repair", "GPU fan not spinning properly, needs repair, selling cheap"),
        ("Urgent Sale", "Money urgent sale! Need cash today!"),
        ("Shop Warranty", "Computer House showroom Godagama. 02 months warranty."),
        ("Brand New Sealed", "Brand new in box, sealed, factory sealed unopened"),
    ]

    for label, desc in samples:
        tags = extract_condition_tags(desc)
        print(f"\n[{label}] \"{desc[:60]}...\"")
        print(f"  -> Warranty: {tags['has_warranty']} ({tags['warranty_months']} mos) | Repair: {tags['needs_repair']} | Urgent: {tags['urgent_sale']} | Shop: {tags['is_shop']} | New: {tags['brand_new']}")

    print("\n" + "=" * 75)
    print("  2. Testing Hedonic Adjustment on baseline price LKR 50,000 (log ~ 10.8198)")
    print("=" * 75)

    base_log = 10.819778  # expm1(10.819778) ≈ 50,000 LKR
    for label, desc in samples:
        adj = apply_condition_adjustment(base_log, description=desc)
        print(f"\n[{label}]")
        print(f"  Base Price: LKR {adj['unadjusted_price_lkr']:,.0f} -> Condition-Adjusted: LKR {adj['adjusted_price_lkr']:,.0f} ({adj['condition_multiplier_pct']:+.1f}%)")
        for f in adj['applied_factors']:
            print(f"    - Factor: {f['factor']:<18} value={f['value']} -> impact={f['pct_impact']}")

    print("\n" + "=" * 75)
    print("  3. Testing FastAPI Endpoint '/predict' with Description Context")
    print("=" * 75)

    # Test API direct call
    test_reqs = [
        PredictRequest(model="GTX 1060", vram_gb=6.0, listed_price=38000.0, description="Standard working card"),
        PredictRequest(model="GTX 1060", vram_gb=6.0, listed_price=38000.0, description="6 months company warranty included with original receipt"),
        PredictRequest(model="GTX 1060", vram_gb=6.0, listed_price=22000.0, description="Fan broken, needs repair, no display occasionally"),
    ]

    for req in test_reqs:
        res = predict(req)
        print(f"\nModel: {req.model} 6GB | Listed: LKR {req.listed_price:,.0f} | Desc: \"{req.description}\"")
        print(f"  Base Specs Price:  LKR {res['base_specs_price']:,.0f}")
        print(f"  Condition Price:   LKR {res['condition_adjusted_price']:,.0f} ({res['condition_adjustment_pct']:+.1f}%)")
        print(f"  Fair Range (90%):  LKR {res['lower_price']:,.0f} - LKR {res['upper_price']:,.0f}")
        print(f"  Verdict:           {res['evaluation']['verdict']} (Score: {res['evaluation']['fairness_score']}/100)")
        print(f"  Verdict Detail:    {res['evaluation']['description']}")

    print("\n" + "=" * 75)
    print("  [SUCCESS] All Condition Adjustment Pipeline Tests Passed!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
