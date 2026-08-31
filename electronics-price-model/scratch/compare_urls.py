import requests
from bs4 import BeautifulSoup

url1 = "https://ikman.lk/en/ads/sri-lanka/computers-tablets?sort=relevance&buy_now=0&urgent=0&query=samsung&page=1&enum.item_type=tablet&enum.condition=used"
url2 = "https://ikman.lk/en/ads/sri-lanka/computers-tablets?sort=relevance&buy_now=0&urgent=0&query=samsung&enum.item_type=tablet&enum.condition=used&page=1"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

r1 = requests.get(url1, headers=headers, timeout=10)
s1 = BeautifulSoup(r1.text, 'html.parser')
cards1 = s1.find_all('li', class_=lambda c: c and 'gtm-normal-ad' in c)

r2 = requests.get(url2, headers=headers, timeout=10)
s2 = BeautifulSoup(r2.text, 'html.parser')
cards2 = s2.find_all('li', class_=lambda c: c and 'gtm-normal-ad' in c)

print("URL1 count:", len(cards1))
print("URL2 count:", len(cards2))
