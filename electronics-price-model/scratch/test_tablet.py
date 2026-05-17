import requests

url = 'http://127.0.0.1:5000/predict'

def test_tablet(data):
    response = requests.post(url, json=data)
    return response.json()

base_tab = {
    'category': 'tablet',
    'brand': 'APPLE',
    'model': 'IPAD',
    'ram': '4',
    'storage': '64',
    'condition': 'Used'
}

print("Base: Apple iPad 4GB/64GB")
print(test_tablet(base_tab)['price'])

print("\nChanging Brand: Apple -> Samsung")
tab_samsung = base_tab.copy()
tab_samsung['brand'] = 'SAMSUNG'
tab_samsung['model'] = 'GALAXY TAB S'
print(test_tablet(tab_samsung)['price'])

print("\nChanging Storage: 64GB -> 256GB")
tab_storage = base_tab.copy()
tab_storage['storage'] = '256'
print(test_tablet(tab_storage)['price'])
