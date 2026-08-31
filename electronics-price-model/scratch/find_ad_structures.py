import requests
from bs4 import BeautifulSoup
import re

url = "https://ikman.lk/en/ads/sri-lanka/computers-tablets?sort=relevance&buy_now=0&urgent=0&query=samsung&page=1&enum.item_type=tablet&enum.condition=used"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

response = requests.get(url, headers=headers, timeout=10)
soup = BeautifulSoup(response.text, 'html.parser')

print("Title:", soup.title.text if soup.title else "None")

# Find all links to ads
ad_links = []
for a in soup.find_all('a', href=True):
    href = a['href']
    if '/en/ad/' in href:
        ad_links.append(href)

print("Found", len(ad_links), "links containing /en/ad/")
if ad_links:
    print("Sample ad links:")
    for l in ad_links[:5]:
        print(" -", l)

# Let's inspect class names of div/li containing these links
for a in soup.find_all('a', href=True)[:150]:
    href = a['href']
    if '/en/ad/' in href:
        parent = a.parent
        print(f"Parent tag: {parent.name}, classes: {parent.get('class')}")
        # print some text
        print(f"Text inside parent: {parent.get_text(separator=' | ').strip()[:200]}")
        print("-" * 50)
