import requests
import json

url = "http://127.0.0.1:8004/predict"
payload = {
    "category": "laptop",
    "brand": "HP",
    "model": "HP 640 G10",
    "ram": 16,
    "storage": 512,
    "storageType": "SSD",
    "cpu": "i5",
    "generation": 13,
    "algorithm": "xgboost"
}

headers = {
    'Content-Type': 'application/json'
}

response = requests.post(url, headers=headers, data=json.dumps(payload))
print("Response Status Code:", response.status_code)
print("Response JSON:")
print(json.dumps(response.json(), indent=4))
