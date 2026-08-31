import sys
import os
from pathlib import Path

# Add mobile-price-model to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from api import predict, PredictRequest, startup_event

startup_event()

print("=" * 70)
print("TESTING HYBRID LLM & SINGLE SOURCE OF TRUTH PIPELINE")
print("=" * 70)

# Test cases simulating unstructured marketplace listings
test_listings = [
    # 1. Messy iPhone listing with Singlish / slang
    PredictRequest(
        title="iPhone 13 Pro Graphite 128GB (exchange ok)",
        description="Phone is in 99% mint condition. Full set with box and original cable. Battery health 88%. TrueTone 100% working. No errors.",
        raw_text="Condition: Used, Brand: Apple, Memory: 128 GB"
    ),
    # 2. Android listing with missing RAM and 5G
    PredictRequest(
        title="Samsung Galaxy A34 128GB Awesome Silver",
        description="Very good condition phone. 128GB storage. Used carefully with tempered glass.",
        raw_text="Condition: Used, Brand: Samsung, Model: Galaxy A34"
    ),
    # 3. Direct structured request (Backward compatibility test)
    PredictRequest(
        brand="Apple",
        model="iPhone 13",
        storage_gb=128.0
    )
]

for idx, req in enumerate(test_listings, 1):
    print(f"\n>>> RUNNING TEST CASE #{idx} <<<")
    res = predict(req)
    print(f"RESULT #{idx}:")
    print(f"  Matched Model:   {res['matched_model']} ({res['phone_type']})")
    print(f"  Predicted Price: Rs. {res['predicted_price']:,.2f}")
    print(f"  Enriched Specs:  RAM={res['enriched_specs']['ram_gb']}GB, Storage={res['enriched_specs']['storage_gb']}GB, 5G={res['enriched_specs']['has_5g']}, Tier={res['enriched_specs']['model_tier']}")
