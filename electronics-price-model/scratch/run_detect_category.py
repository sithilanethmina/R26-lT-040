import requests
from bs4 import BeautifulSoup
import re

url = "https://ikman.lk/en/ad/asus-vivobook-i3-13th-gen-for-sale-colombo-15"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

res = requests.get(url, headers=headers)
soup = BeautifulSoup(res.text, 'html.parser')

title = soup.find('h1').text.strip() if soup.find('h1') else ""
url_lower = url.lower()

# Scrape DOM key values
keyValues = {}
for row in soup.find_all(['tr', 'dl', 'div', 'li']):
    text = row.text.strip()
    if ":" in text and len(text) < 120:
        parts = text.split(":")
        k = parts[0].lower().strip()
        v = parts[1].strip() if len(parts) > 1 else ""
        if k and v:
            keyValues[k] = v

# Full text
fullCollectedText = ""
for el in soup.find_all(['div', 'span', 'p', 'li']):
    t = el.text.strip()
    if t and len(fullCollectedText) < 4000:
        fullCollectedText += t + " "

# Breadcrumbs
breadcrumbs = []
for a in soup.find_all('a'):
    href = a.get('href', '')
    text = a.text.strip()
    if '/ads/sri-lanka/' in href or 'breadcrumb' in str(a.get('class')):
        if text and text not in breadcrumbs:
            breadcrumbs.append(text)

bText = " ".join(breadcrumbs).lower()
text = f"{title} {fullCollectedText}".lower()

print("Title:", title)
print("Breadcrumbs extracted:", breadcrumbs)
print("bText:", bText)

# JS logic translation
NON_PHONE_PATTERNS = [
    r'\b(smart\s*watch|smartwatch|watch|wrist\s*watch|fitness\s*band|wristband|watch\s*strap)\b',
    r'\b(airpods|earbuds|earphones|headphones|headset|bluetooth\s*speaker|ear\s*buds|tws)\b',
    r'\b(charger|charging\s*cable|power\s*adapter|data\s*cable|fast\s*charger|wireless\s*charger)\b',
    r'\b(phone\s*case|back\s*cover|flip\s*cover|pouch|silicone\s*case|leather\s*case)\b',
    r'\b(tempered\s*glass|screen\s*protector|lens\s*protector|gorilla\s*glass)\b',
    r'\b(power\s*bank|powerbank|battery\s*pack)\b',
    r'\b(sim\s*tray|housing|display\s*panel|touch\s*display|lcd\s*panel|spare\s*parts|battery\s*replacement)\b'
]

isAccessoryBreadcrumb = any(kw in bText for kw in ["accessories", "wearables", "smart watch", "audio"])
isNonPhoneTitle = any(re.search(pat, title.lower()) for pat in NON_PHONE_PATTERNS)

print("isAccessoryBreadcrumb:", isAccessoryBreadcrumb)
print("isNonPhoneTitle:", isNonPhoneTitle)

category = None

# Step 0
if isNonPhoneTitle or isAccessoryBreadcrumb:
    if not ("riyasewana.com/buy/" in url_lower or "cars" in url_lower or "graphic-card" in url_lower or "laptop" in url_lower):
        if isNonPhoneTitle or not ("mobile phones" in bText):
            category = "unsupported"

if not category:
    # Step 1
    if "laptop" in bText or "computers & tablets" in bText or "computer parts" in bText:
        if "graphic card" in bText or "vga" in bText or "rtx" in text or "gtx" in text or "rx " in text or "graphics card" in text:
            category = "gpu"
        else:
            category = "electronics"

if not category:
    if "monitor" in bText or "display" in bText:
        category = "electronics"
    elif "tablet" in bText:
        category = "electronics"
    elif "mobile phone" in bText or "mobile_phone" in bText:
        category = "mobile" if not isNonPhoneTitle else "unsupported"

if not category:
    # Step 2
    if "computer-accessories" in url_lower or "graphic-card" in url_lower or "vga" in url_lower or "gpu" in url_lower:
        category = "gpu"
    elif "mobile-phones" in url_lower or "mobile_phones" in url_lower:
        category = "mobile" if not isNonPhoneTitle else "unsupported"
    elif "laptop" in url_lower or "computer" in url_lower or "monitor" in url_lower or "tablet" in url_lower or "electronics" in url_lower:
        category = "electronics"

if not category:
    # Step 3
    if re.search(r'\b(rtx|gtx|rx\s*\d{3,4}|graphics card|vga card|geforce|radeon)\b', title.lower()):
        category = "gpu"
    elif re.search(r'\b(iphone|samsung galaxy|redmi|poco|oneplus|pixel|android phone|mobile phone|huawei|vivo|oppo|realme|nokia|infinix|tecno)\b', title.lower()):
        category = "unsupported" if isNonPhoneTitle else "mobile"
    elif re.search(r'\b(toyota|suzuki|corolla|aqua|alto|honda|nissan|wagon r|prius|axio|premio|vezel|vitz|land cruiser|prado|dolphin|hiace)\b', title.lower()):
        category = "vehicle"
    elif re.search(r'\b(laptop|macbook|thinkpad|notebook|dell monitor|curved monitor|ipad)\b', title.lower()):
        category = "electronics"

if not category:
    # Step 4
    if isNonPhoneTitle:
        category = "unsupported"
    elif re.search(r'\b(iphone|samsung galaxy|redmi note|oneplus|google pixel)\b', text):
        category = "mobile"
    elif re.search(r'\b(rtx|gtx|geforce|radeon)\b', text):
        category = "gpu"
    elif re.search(r'\b(toyota|suzuki|corolla|aqua|alto)\b', text):
        category = "vehicle"
    else:
        category = "unsupported"

print("FINAL DETECTED CATEGORY:", category)
