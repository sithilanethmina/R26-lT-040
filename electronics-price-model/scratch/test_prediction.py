import requests
import json

url = 'http://127.0.0.1:5000/predict'

def test_predict(data):
    response = requests.post(url, json=data)
    return response.json()

base_data = {
    'category': 'laptop',
    'brand': 'DELL',
    'model': 'LATITUDE',
    'ram': '8',
    'storage': '256',
    'storageType': 'SSD',
    'cpu': 'I5',
    'generation': '8',
    'condition': 'Used'
}

print("Base: Dell Latitude i5 8th Gen 8GB/256GB SSD")
print(test_predict(base_data)['price'])

print("\nChanging RAM: 8GB -> 16GB")
data_ram = base_data.copy()
data_ram['ram'] = '16'
print(test_predict(data_ram)['price'])

print("\nChanging Storage: 256GB -> 512GB")
data_storage = base_data.copy()
data_storage['storage'] = '512'
print(test_predict(data_storage)['price'])

print("\nChanging Brand: Dell -> Apple (MacBook Pro)")
data_apple = base_data.copy()
data_apple['brand'] = 'APPLE'
data_apple['model'] = 'MACBOOK PRO'
data_apple['cpu'] = 'M1'
data_apple['generation'] = '12'
print(test_predict(data_apple)['price'])

print("\nChanging Generation: 8th -> 12th")
data_gen = base_data.copy()
data_gen['generation'] = '12'
print(test_predict(data_gen)['price'])
