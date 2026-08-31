import requests
import json

url = "http://127.0.0.1:8004/predict"
payload = {
    "category": "monitor",
    "brand": "Dell",
    "model": "Dell E2318H 23-inch LED Monitor",
    "size": 23,
    "refreshRate": 60,
    "resolution": "FHD",
    "algorithm": "xgboost",
    "price": 19500.0,
    "listed_price": 19500.0
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
