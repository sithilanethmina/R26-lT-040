import requests
from bs4 import BeautifulSoup
import re

url = "https://ikman.lk/en/ad/samsung-galaxy-tab-a-for-sale-2017-model-for-sale-colombo"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

response = requests.get(url, headers=headers, timeout=10)
soup = BeautifulSoup(response.text, 'html.parser')

print("Detail page title:", soup.title.text if soup.title else "None")

# Print all text from paragraphs or divs that look like description
for div in soup.find_all(['div', 'p', 'span']):
    classes = div.get('class')
    if classes:
        class_str = " ".join(classes)
        if 'description' in class_str or 'item-description' in class_str:
            print(f"Tag: {div.name}, class: {class_str}")
            print("Content:")
            print(div.get_text(separator=' | ').strip()[:500])
            print("="*50)
