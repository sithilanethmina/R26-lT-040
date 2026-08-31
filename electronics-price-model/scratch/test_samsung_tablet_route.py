import requests
import json

url = "http://127.0.0.1:8004/predict"
payload = {
    "category": "tablet",
    "brand": "Samsung",
    "model": "Samsung Galaxy Tab A9",
    "ram": 4,
    "storage": 64,
    "algorithm": "xgboost",
    "price": 45000.0,
    "listed_price": 45000.0
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
