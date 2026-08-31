import requests
import json

url = "http://127.0.0.1:8004/predict"
payload = {
    "category": "monitor",
    "brand": "HP",
    "model": "HP Z24n 24-inch Professional Frameless Monitor",
    "size": 24,
    "refreshRate": 60,
    "resolution": "FHD",
    "algorithm": "xgboost",
    "price": 25000.0,
    "listed_price": 25000.0
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
