import requests
import json

url = "http://127.0.0.1:8004/predict"

test_cases = [
    {
        "title": "Samsung s6 Lite Tab",
        "brand": "Samsung",
        "model": "s6 Lite Tab",
        "ram": 4,
        "storage": 64,
        "price": 60000.0
    },
    {
        "title": "samsung tab A9",
        "brand": "Samsung",
        "model": "tab A9",
        "ram": 4,
        "storage": 64,
        "price": 26000.0
    },
    {
        "title": "Samsung Galaxy Tab A8",
        "brand": "Samsung",
        "model": "Galaxy Tab A8",
        "ram": 6,
        "storage": 128,
        "price": 15000.0
    }
]

headers = {
    'Content-Type': 'application/json'
}

for case in test_cases:
    payload = {
        "category": "tablet",
        "brand": case["brand"],
        "model": case["model"],
        "ram": case["ram"],
        "storage": case["storage"],
        "algorithm": "xgboost",
        "price": case["price"],
        "listed_price": case["price"]
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        res = response.json()
        print(f"Title: {case['title']} | Listed: Rs {case['price']:,.0f}")
        print(f"  Model: {res.get('model_name')}")
        print(f"  Predicted Price: {res.get('price')}")
        print(f"  Range: Rs {res.get('fair_market_range', {}).get('lower_price_lkr', 0):,.0f} - Rs {res.get('fair_market_range', {}).get('upper_price_lkr', 0):,.0f}")
        print(f"  Accuracy/Score: {res.get('accuracy') * 100:.0f}%")
        print("-" * 50)
    except Exception as e:
        print(f"Request failed for {case['title']}: {e}")
