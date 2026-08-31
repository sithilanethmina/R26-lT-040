import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from api import predict, PredictRequest, startup_event

startup_event()

test_cases = [
    PredictRequest(brand="Apple", model="iPhone 13"),
    PredictRequest(brand="Apple", model="iPhone 13 Green 128GB LL/A"),
    PredictRequest(brand="Apple", model="iPhone 13 Pro", storage_gb=256.0, battery_health_percent=88.0),
    PredictRequest(brand="Samsung", model="Galaxy S22 Ultra"),
    PredictRequest(brand="Xiaomi", model="Redmi Note 13 Pro"),
]

for req in test_cases:
    res = predict(req)
    print("=" * 60)
    print(f"INPUT:  Brand='{req.brand}', Model='{req.model}'")
    print(f"OUTPUT: Matched='{res['matched_model']}', Type='{res['phone_type']}'")
    print(f"PRICE:  Rs. {res['predicted_price']:,.2f}")
    print(f"RANGE:  Rs. {res['fair_market_range']['lower_price_lkr']:,.2f} - Rs. {res['fair_market_range']['upper_price_lkr']:,.2f}")
    print("ENRICHED SPECS:")
    for k, v in res['enriched_specs'].items():
        print(f"  - {k}: {v}")
