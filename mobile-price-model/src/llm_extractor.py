import os
import json
import re
from typing import Optional, Dict, Any
from pathlib import Path
import httpx

def _load_env_fallback():
    for p in [Path(__file__).resolve().parent.parent / ".env", Path(__file__).resolve().parent.parent.parent / ".env"]:
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

EXTRACTION_SYSTEM_PROMPT = """You are an expert mobile phone specification extraction assistant for second-hand marketplace listings (e.g., Ikman.lk, Facebook Marketplace, Riyasewana).
Your job is to read the listing Title, Price, and Description, and extract ONLY the facts explicitly mentioned by the seller into structured JSON.

RULES:
1. "brand": The phone brand/manufacturer (e.g. Apple, Samsung, Xiaomi, Google, OnePlus, Huawei, Vivo, Oppo, Realme, Nokia, Sony, Motorola). Return null if not determinable.
2. "model": Clean phone model name WITHOUT colors (Graphite, Green, etc.), carrier/region codes (LL/A, ZP/A), or condition words (e.g., "iPhone 13 Pro", "Galaxy S22 Ultra", "Redmi Note 13 Pro", "Pixel 7a"). Return null if unclear.
3. "storage_gb": Internal storage in GB as a number (e.g. 64, 128, 256, 512, 1024). Return null if omitted.
4. "ram_gb": Explicitly stated RAM in GB as a number (e.g. 4, 6, 8, 12). DO NOT GUESS; return null if the seller did not explicitly state the RAM.
5. "battery_health_percent": Battery health percentage as a number between 50 and 100 (e.g. 88). Return null if not stated or if it is an Android device.
6. "warranty_days": Remaining warranty converted to total days (e.g. "6 months" -> 180, "1 year" -> 365, "no warranty" -> 0). Return null if unmentioned.
7. "condition": "Used" (default), "Brand New" (sealed pack / company sealed), "For Parts" (faulty/broken), or "Unknown".
8. "is_installment_trap": true IF the asking price is explicitly described as a down-payment, lease advance, or installment rather than the full cash price; otherwise false.
9. "repairs_or_defects": Any explicit mention of replaced parts, defects, or TrueTone status (e.g. "Display changed, TrueTone working", "Face ID error"); otherwise null.

Respond ONLY with valid JSON.
"""

EXTRACTION_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "brand": {"type": "STRING", "nullable": True},
        "model": {"type": "STRING", "nullable": True},
        "storage_gb": {"type": "NUMBER", "nullable": True},
        "ram_gb": {"type": "NUMBER", "nullable": True},
        "battery_health_percent": {"type": "NUMBER", "nullable": True},
        "warranty_days": {"type": "NUMBER", "nullable": True},
        "condition": {"type": "STRING", "enum": ["Used", "Brand New", "For Parts", "Unknown"]},
        "is_installment_trap": {"type": "BOOLEAN"},
        "repairs_or_defects": {"type": "STRING", "nullable": True}
    },
    "required": ["brand", "model", "storage_gb", "ram_gb", "battery_health_percent", "warranty_days", "condition", "is_installment_trap"]
}

def extract_mobile_specs_llm(
    title: str = "",
    description: str = "",
    raw_text: str = "",
    model_name: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Extracts mobile phone specs using Gemini Flash Lite.
    Falls back gracefully to None if the API key is not set or request times out.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None

    # Default to Gemini 2.5 Flash Lite or user configured model
    model = model_name or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
    
    # Clean input text
    combined_content = f"Title: {title.strip()}\nDescription: {description.strip()}\nPage Specs Text: {raw_text.strip()}"
    if len(combined_content.strip()) < 10:
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"{EXTRACTION_SYSTEM_PROMPT}\n\nListing to extract:\n{combined_content[:3500]}"}
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": EXTRACTION_RESPONSE_SCHEMA,
            "temperature": 0.1
        }
    }

    print("\n" + "-" * 70)
    print(f"[LLM EXTRACTION INVOKED: {model}]")
    print("-" * 70)
    print(">> TEXT SENT TO GEMINI:")
    print(f"   * Title:       {title.strip() or 'N/A'}")
    print(f"   * Description: {description.strip()[:200] or 'N/A'}")
    if raw_text.strip():
        print(f"   * Page Specs:  {raw_text.strip()[:150]}")

    try:
        with httpx.Client(timeout=4.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code != 200:
                # If model is not available on standard v1beta, try fallback to gemini-1.5-flash
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
                print(f"   * {k:24}: {v}")
            print("-" * 70 + "\n")

            return parsed
            
    except Exception as e:
        print(f"   * [ERROR] LLM extraction error: {e}")
        print("-" * 70 + "\n")
        return None
