import requests
import json

url = "http://127.0.0.1:8004/predict"
payload = {
    "category": "tablet",
    "brand": "Apple",
    "model": "Apple iPad Pro 11-inch",
    "ram": 8,
    "storage": 128,
    "algorithm": "xgboost",
    "price": 120000.0,
    "listed_price": 120000.0
}

headers = {
    'Content-Type': 'application/json'
}

try:
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    print("Response Status Code:", response.status_code)
    print("Response JSON:")
    print(json.dumps(response.json(), indent=4))
except Exception as e:
    print("Request failed:", e)
