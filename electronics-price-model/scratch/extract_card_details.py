import requests
from bs4 import BeautifulSoup

url = "https://ikman.lk/en/ads/sri-lanka/computers-tablets?sort=relevance&buy_now=0&urgent=0&query=samsung&page=1&enum.item_type=tablet&enum.condition=used"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

response = requests.get(url, headers=headers, timeout=10)
soup = BeautifulSoup(response.text, 'html.parser')

items = soup.find_all('li', class_=lambda c: c and 'gtm-normal-ad' in c)

with open("scratch/ad_card_structure.txt", "w", encoding="utf-8") as f:
    if items:
        f.write("Ad Card HTML structure:\n")
        f.write(items[0].prettify())
        f.write("\n" + "="*80 + "\n")
        f.write(items[1].prettify())
    else:
        f.write("No items found.")
print("Saved ad card structures to scratch/ad_card_structure.txt")
