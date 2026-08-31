import requests
import re

url = "https://ikman.lk/en/ads/sri-lanka/computers-tablets?sort=relevance&buy_now=0&urgent=0&query=samsung&page=1&enum.item_type=tablet&enum.condition=used"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

response = requests.get(url, headers=headers, timeout=10)
html = response.text

# Check for window.initialState
match = re.search(r'window\.__initialState__\s*=\s*(\{.*?\});', html)
if match:
    print("Found window.__initialState__!")
    print("Length of json state:", len(match.group(1)))
    # save snippet
    with open("scratch/initial_state.json", "w", encoding="utf-8") as f:
        f.write(match.group(1)[:5000])
else:
    print("Could not find window.__initialState__.")
    # Check for script tags that contain JSON
    scripts = re.findall(r'<script.*?>([\s\S]*?)</script>', html)
    print("Found", len(scripts), "script tags.")
    for i, s in enumerate(scripts):
        if "initialState" in s or "apollo" in s or "props" in s:
            print(f"Script {i} contains key terms. Length: {len(s)}")
