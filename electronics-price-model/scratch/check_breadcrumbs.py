import requests
from bs4 import BeautifulSoup
import re

url = "https://ikman.lk/en/ad/asus-vivobook-i3-13th-gen-for-sale-colombo-15"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

try:
    res = requests.get(url, headers=headers, timeout=15)
    print("Status:", res.status_code)
    soup = BeautifulSoup(res.text, 'html.parser')
    
    # 1. Print H1
    h1 = soup.find('h1')
    print("H1 Title:", h1.text.strip() if h1 else "None")
    
    # 2. Check breadcrumbs
    print("\n--- Breadcrumb link candidates ---")
    for a in soup.find_all('a'):
        href = a.get('href', '')
        text = a.text.strip()
        if '/ads/sri-lanka/' in href or 'breadcrumb' in str(a.get('class')) or 'breadcrumb' in str(a.parent.get('class')):
            print(f"Text: '{text}' | Href: '{href}' | Class: {a.get('class')}")
            
    # 3. Check for specific selectors
    print("\n--- Matching ol/ul/nav elements ---")
    for el in soup.find_all(['nav', 'ol', 'ul', 'div']):
        classes = str(el.get('class', ''))
        aria = str(el.get('aria-label', ''))
        if 'breadcrumb' in classes or 'breadcrumb' in aria:
            print(f"Tag: {el.name} | Class: {classes} | Aria: {aria}")
            print(f"Content: {el.text.strip()}")
            
except Exception as e:
    print("Error:", e)
