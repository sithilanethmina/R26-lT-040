import requests
import json

url = "http://127.0.0.1:8004/predict"
payload = {
    "category": "laptop",
    "brand": "ASUS",
    "model": "Asus Vivobook i3 13th Gen",
    "ram": 8,
    "storage": 256,
    "generation": 13,
    "cpu": "I3",
    "price": 87000.0,
    "listed_price": 87000.0,
    "algorithm": "xgboost"
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
