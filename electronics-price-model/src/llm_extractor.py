import os
import json
import re
import base64
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import httpx

def _load_env_fallback():
    search_paths = [
        Path(__file__).resolve().parent.parent / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env",
        Path(__file__).resolve().parent.parent.parent / "mobile-price-model" / ".env"
    ]
    for p in search_paths:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k, v = k.strip(), v.strip().strip('"').strip("'")
                            if k and not os.environ.get(k):
                                os.environ[k] = v
            except Exception:
                pass

_load_env_fallback()

EXTRACTION_SYSTEM_PROMPT = """You are an expert electronics specification extraction assistant for second-hand marketplace listings (e.g., Ikman.lk, Facebook Marketplace, Riyasewana).
Your job is to analyze listing Title, Price, Description, and page text for Laptops, Tablets, and Monitors, extracting ONLY explicitly stated or directly inferable facts into structured JSON.

EXTRACTION GUIDELINES:
1. "category": Must be one of ["laptop", "tablet", "monitor"]. Infer from keywords (e.g. iPad, Tab -> tablet; Monitor, Hz, Display -> monitor; MacBook, ThinkPad, Core i5, RAM -> laptop).
2. "brand": Clean brand name (e.g., Apple, Dell, HP, Lenovo, Asus, Acer, Samsung, MSI, ViewSonic, LG, Xiaomi, Redmi, Honor, Huawei, Microsoft, Blackview, Other).
3. "model": Clean series / model name WITHOUT cosmetic colors or warranty noise (e.g. "Latitude 5420", "Victus 15", "ThinkPad T480", "iPad Pro M1", "Galaxy Tab S9", "Nitro VG270", "ThinkVision P24q").
4. "cpu": (For laptops/tablets) Processor family (e.g. "Core i3", "Core i5", "Core i7", "Core i9", "Ryzen 3", "Ryzen 5", "Ryzen 7", "Ryzen 9", "Apple M1", "Apple M2", "Apple M3", "Apple M4", "Celeron", "Pentium", "Other"). Return null if monitor.
5. "generation": (For laptops) Intel/AMD generation as an integer (e.g., "11th Gen" -> 11, "i5 12500H" -> 12, "Ryzen 5000" -> 5). Return null if not applicable.
6. "ram_gb": RAM in GB as a number (e.g. 4, 8, 16, 32, 64). Return null if not stated or if monitor.
7. "storage_gb": Total storage in GB as a number (e.g. 64, 128, 256, 512, 1024, 2048). Convert 1TB -> 1024. Return null if monitor.
8. "storage_type": "SSD", "HDD", or "NVMe". Return null if not applicable.
9. "gpu": Dedicated or integrated graphics tier (e.g. "RTX 40-Series", "RTX 30-Series", "RTX 20-Series", "GTX", "Integrated", "Other Dedicated").
10. "screen_size_inch": Display diagonal size in inches as a float (e.g. 14.0, 15.6, 24.0, 27.0, 10.2, 11.0, 32.0).
11. "refresh_rate_hz": Refresh rate in Hz as an integer (e.g. 60, 75, 100, 120, 144, 165, 180, 240).
12. "resolution": Resolution standard: "1080p FHD", "2K QHD", "4K UHD", "HD".
13. "panel_type": "IPS", "OLED", "VA", or "Standard".
14. "condition": "Used" (default) or "Brand New" (sealed / unopened).
15. "is_touchscreen": true if touchscreen, x360, flip, 2-in-1; otherwise false.
16. "is_curved": true if curved monitor/display; otherwise false.
17. "is_gaming": true if gaming laptop/monitor (e.g. TUF, ROG, Nitro, Legion, Victus, Alienware, >=100Hz); otherwise false.
18. "location": District/City (e.g. "Colombo", "Gampaha", "Kandy", "Kalutara", "Kurunegala", "Galle", "Other").

Respond ONLY with valid JSON matching the schema.
"""

EXTRACTION_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "category": {"type": "STRING", "enum": ["laptop", "tablet", "monitor"]},
        "brand": {"type": "STRING", "nullable": True},
        "model": {"type": "STRING", "nullable": True},
        "cpu": {"type": "STRING", "nullable": True},
        "generation": {"type": "NUMBER", "nullable": True},
        "ram_gb": {"type": "NUMBER", "nullable": True},
        "storage_gb": {"type": "NUMBER", "nullable": True},
        "storage_type": {"type": "STRING", "enum": ["SSD", "HDD", "NVMe"], "nullable": True},
        "gpu": {"type": "STRING", "enum": ["RTX 40-Series", "RTX 30-Series", "RTX 20-Series", "GTX", "Integrated", "Other Dedicated"], "nullable": True},
        "screen_size_inch": {"type": "NUMBER", "nullable": True},
        "refresh_rate_hz": {"type": "NUMBER", "nullable": True},
        "resolution": {"type": "STRING", "enum": ["1080p FHD", "2K QHD", "4K UHD", "HD"], "nullable": True},
        "panel_type": {"type": "STRING", "enum": ["IPS", "OLED", "VA", "Standard"], "nullable": True},
        "condition": {"type": "STRING", "enum": ["Used", "Brand New"]},
        "is_touchscreen": {"type": "BOOLEAN"},
        "is_curved": {"type": "BOOLEAN"},
        "is_gaming": {"type": "BOOLEAN"},
        "location": {"type": "STRING", "nullable": True},
        "listed_price": {"type": "NUMBER", "nullable": True}
    },
    "required": ["category", "brand", "model", "condition", "is_touchscreen", "is_curved", "is_gaming"]
}

def extract_electronics_specs_llm(
    title: str = "",
    description: str = "",
    raw_text: str = "",
    image_base64: str = None,
    model_name: str = None
) -> Optional[Dict[str, Any]]:
    """
    Calls Google Gemini Flash Lite (multimodal) to extract structured electronics specifications 
    from unstructured text and/or a listing screenshot using gemini-3.5-flash-lite.
    """
    _load_env_fallback()
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None

    # Default to gemini-3.5-flash-lite
    model = model_name or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    
    combined_content = f"Title: {title.strip()}\nDescription: {description.strip()}\nPage Specs Text: {raw_text.strip()}"
    has_image = bool(image_base64 and len(image_base64.strip()) > 50)
    
    if len(combined_content.strip()) < 8 and not has_image:
        return None

    # Construct request parts (multimodal)
    parts = []
    
    if has_image:
        clean_base64 = image_base64.strip()
        mime_type = "image/jpeg"
        if "data:" in clean_base64 and ";base64," in clean_base64:
            header, clean_base64 = clean_base64.split(";base64,", 1)
            if "png" in header:
                mime_type = "image/png"
            elif "webp" in header:
                mime_type = "image/webp"
                
        parts.append({
            "inlineData": {
                "mimeType": mime_type,
                "data": clean_base64
            }
        })
        
    prompt_text = f"{EXTRACTION_SYSTEM_PROMPT}\n\n"
    if has_image:
        prompt_text += "Carefully inspect this marketplace listing screenshot. Extract all visible hardware specifications, brand, model, condition, and listed price."
    if len(combined_content.strip()) >= 8:
        prompt_text += f"\n\nListing Text Context:\n{combined_content[:3500]}"
        
    parts.append({"text": prompt_text})

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    payload = {
        "contents": [
            {
                "parts": parts
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": EXTRACTION_RESPONSE_SCHEMA,
            "temperature": 0.1
        }
    }

    print("\n" + "-" * 70)
    print(f"[ELECTRONICS MULTIMODAL LLM EXTRACTION: {model}]")
    print("-" * 70)
    print(f"   * Screenshot attached: {'YES (Vision Mode)' if has_image else 'NO'}")
    if title.strip(): print(f"   * Title:               {title.strip()}")
    if description.strip(): print(f"   * Description:         {description.strip()[:150]}")

    # Save incoming screenshot with page title and timestamp
    if has_image:
        try:
            screenshots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "screenshots"))
            os.makedirs(screenshots_dir, exist_ok=True)
            clean_title = re.sub(r'[^\w\s-]', '', title or "page_screenshot").strip()
            clean_title = re.sub(r'[-\s]+', '_', clean_title)[:60]
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{clean_title}_{timestamp_str}.jpg" if clean_title else f"screenshot_{timestamp_str}.jpg"
            screenshot_path = os.path.join(screenshots_dir, filename)
            
            image_bytes = base64.b64decode(clean_base64)
            with open(screenshot_path, "wb") as f:
                f.write(image_bytes)
            print(f"   * Screenshot saved:    screenshots/{filename}")
        except Exception as save_err:
            print(f"   * [WARNING] Failed to save screenshot locally: {save_err}")

    try:
        with httpx.Client(timeout=6.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code != 200:
                # Fallback to gemini-1.5-flash if model endpoint differs
                if model != "gemini-1.5-flash":
                    fallback_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
                    resp = client.post(fallback_url, json=payload)
                
                if resp.status_code != 200:
                    print(f"   * [ERROR] Gemini API status {resp.status_code}: {resp.text[:120]}")
                    print("-" * 70)
                    return None

            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                print("   * [ERROR] No candidate returned from Gemini")
                print("-" * 70)
                return None
                
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                return None
                
            raw_json_str = parts[0].get("text", "").strip()
            parsed = json.loads(raw_json_str)

            print("\n>> STRUCTURED JSON OUTPUT FROM GEMINI:")
            for k, v in parsed.items():
                print(f"   * {k:20}: {v}")
            print("-" * 70 + "\n")

            return parsed
            
    except Exception as e:
        print(f"   * [ERROR] LLM extraction error: {e}")
        print("-" * 70 + "\n")
        return None

if __name__ == "__main__":
    # Self-test with tricky laptop listing
    sample_text = "HP Victus 15 gaming laptop, Core i5 12450H 12th gen, 16GB DDR4, 512GB NVMe SSD, RTX 3050 4GB GPU, 144Hz FHD screen, Used with box, Colombo"
    print("Testing Electronics LLM Extractor with sample:")
    res = extract_electronics_specs_llm(title=sample_text)
    print("Result:", res)
