import requests
import json

url = "http://127.0.0.1:8004/predict"
payload = {
    "category": "laptop",
    "brand": "ASUS",
    "model": "Asus Vivobook 15 X1502Z",
    "ram": 8,
    "storage": 512,
    "generation": 12,
    "cpu": "i5",
    "algorithm": "xgboost",
    "price": 135000.0,
    "listed_price": 135000.0
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
