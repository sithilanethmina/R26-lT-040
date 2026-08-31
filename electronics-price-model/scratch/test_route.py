import requests
import json

url = "http://127.0.0.1:8004/predict"
payload = {
    "category": "laptop",
    "brand": "ASUS",
    "model": "Asus Vivobook",
    "ram": 4,
    "storage": 256,
    "storageType": "SSD",
    "cpu": "i3",
    "generation": 13,
    "algorithm": "xgboost",
    "price": 91500.0,
    "listed_price": 91500.0
}

headers = {
    'Content-Type': 'application/json'
}

response = requests.post(url, headers=headers, data=json.dumps(payload))
print("Response Status Code:", response.status_code)
print("Response JSON:")
print(json.dumps(response.json(), indent=4))
