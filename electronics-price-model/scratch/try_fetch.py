import requests

url = "https://ikman.lk/en/ads/sri-lanka/computers-tablets?sort=relevance&buy_now=0&urgent=0&query=samsung&page=1&enum.item_type=tablet&enum.condition=used"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

try:
    response = requests.get(url, headers=headers, timeout=10)
    print("Status:", response.status_code)
    print("Length of content:", len(response.text))
    if response.status_code == 200:
        print("Success! Snippet of content:")
        print(response.text[:500])
    else:
        print("Failed to fetch directly.")
except Exception as e:
    print("Error:", e)
