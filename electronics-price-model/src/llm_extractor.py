import os
import sys
import json
import re
import base64
import time
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import httpx

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
except Exception:
    pass


# Load the env variables
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
Your job is to analyze listing Title, Asking Price, Ikman Defined Fields, and Description for Laptops, Tablets, Monitors, and Desktop PCs, extracting ONLY explicitly stated or directly inferable facts into structured JSON.

CRITICAL FIELD PRIORITY & DISCREPANCY RESOLUTION RULES (MANDATORY & ABSOLUTE):
Marketplace listings contain multiple information sources with differing levels of reliability. You MUST strictly obey this priority hierarchy:

1. TIER 1 (HIGHEST PRIORITY - ABSOLUTE GROUND TRUTH): IKMAN DEFINED FIELDS / SPECIFICATION TABLE
   - The attributes explicitly selected by the seller in the marketplace form (e.g., "Condition: Used", "Brand: Acer", "Item type: Monitor").
   - DISCREPANCY RULE: If there is ANY contradiction between Ikman Defined Fields and the Description, IKMAN DEFINED FIELDS ALWAYS WIN.
     * Example: If Ikman Defined Field says "Condition: Used" and the Description says "Brand new", "Brand new condition", or "100% brand new", you MUST classify "condition": "Used". NEVER allow the description to override the Ikman defined field!
     * Example: If Ikman Defined Field says "Brand: Apple" and description claims "better than Dell", brand MUST be "Apple".

2. TIER 2 (HIGH PRIORITY): LISTING TITLE & LOCATION SUBTITLE
   - Listing Title is authoritative for exact model series (e.g. "Acer Chromebook C733 Series", "AOC 27 Inch Flameless IPS Monitor").
   - BRAND RESOLUTION: If the Ikman Defined Field states "Brand: Other brand" or "Brand: Other", the Listing Title is authoritative for identifying the actual brand (e.g. "AOC", "MSI", "ViewSonic", "Xiaomi", "Chuwi").
   - Location Subtitle (directly under title, e.g. "Matale City, Matale") is authoritative for the "location" field (e.g. "Matale").

3. TIER 3 (LOW PRIORITY - UNRELIABLE / SUPPLEMENTARY ONLY): SELLER DESCRIPTION
   - Descriptions are noisy, subjective, and prone to seller exaggeration, hype, and keyword stuffing.
   - USE THE DESCRIPTION EXCLUSIVELY AS A SECONDARY SOURCE to extract technical hardware specifications that Ikman has no dedicated form fields for:
     * Display Refresh Rate (e.g. 60Hz, 75Hz, 100Hz, 120Hz, 144Hz, 165Hz)
     * Display Panel Type (IPS, OLED, VA, Standard)
     * Display Resolution (1080p FHD, 2K QHD, 4K UHD, HD)
     * Screen diagonal size in inches (e.g. 24", 27", 15.6", 11.6")
     * Dedicated GPU tier (RTX, GTX, etc.)
     * Exact sub-model number (e.g. "27B1H", "C733-C8SZ")
     * Minor ports/features (e.g. HDMI, VGA, frameless, curved)
   - NEVER use the Description to override Condition, Item Category, or Core Brand established by Tier 1 or Tier 2.

CRITICAL INSTRUCTIONS FOR NOISY MARKETPLACE DESCRIPTIONS & CHATTER:
1. FILTER OUT INFORMAL SINHALA / SINGLISH & SELLER HYPE:
   - Second-hand ads frequently mix English with Sinhala/Singlish words and hashtags (e.g. "# conditiom supiriyatama thiyenava 100%", "# displaya and battery condition 100%", "# adupaduvakata thiyenne OS eka gahanna", "# ikmanata denawa").
   - "100%", "95%", or any other percentage in the description refers to COSMETIC/BATTERY CONDITION. It is NEVER RAM and NEVER Storage!
   - Filter out hashtags (`#`), seller guarantees, urgency phrases ("urgent sale", "supiri", "superb condition"), and minor defect notes ("only need to install OS").
   - DO NOT mistake seller phone numbers (e.g. 0788XXXXXX) or asking prices (e.g. "# price - 17000") for RAM, Storage, Model, or Generation numbers!

2. CHROMEBOOKS & BUDGET LAPTOPS:
   - Devices titled "Chromebook" or "Cromebook" (e.g., "Acer Cromebook C733 Series", "HP Chromebook 11", "Lenovo Chromebook") are "category": "laptop".
   - Normalize the spelling of "Cromebook" to "Chromebook" (e.g., model: "Chromebook C733").
   - Chromebooks typically run on budget processors (Intel Celeron, Intel Pentium, AMD A-series, or MediaTek ARM) with 16GB/32GB/64GB eMMC storage.
   - NEVER assume or hallucinate high-end specs like "Core i5" or "12th Gen" for a Chromebook! If the CPU is not explicitly stated as an Intel Core processor, return "cpu": "Celeron" (if Celeron) or "Other", and return "generation": null.

3. STRICT CPU GENERATION RULES:
   - "generation" MUST ONLY be an integer when an Intel Core (i3/i5/i7/i9) or AMD Ryzen processor generation is explicitly stated (e.g., "11th Gen" -> 11, "i5-1135G7" -> 11, "i7 8th Gen" -> 8, "i5-12400" -> 12, "Ryzen 5 5600" -> 5).
   - NEVER use listing years (e.g. 2024, 2023, 2020) or ad posting dates as the CPU generation!
   - If the processor is Celeron, Pentium, Apple M-series (M1/M2/M3), or generation is not explicitly stated, "generation" MUST be null.

4. IGNORE TRADE/EXCHANGE OFFERS, ACCESSORIES & COMPARISONS:
   - If the seller mentions "Exchange with iPhone or MacBook", the device being sold is NOT an iPhone or MacBook. Extract ONLY the device for sale.
   - If the seller mentions free gifts or accessories (e.g., "Free 64GB pen drive", "free external HDD", "free mouse"), extract the laptop's INTERNAL storage, NOT the accessory.
   - If the seller makes comparative claims (e.g., "better than i7", "equivalent to Ryzen 5"), DO NOT extract the compared spec.

SPECIFICATION GUIDELINES:
1. "category": Must be one of ["laptop", "tablet", "monitor", "desktop_pc", "other"].
   - "laptop": Portable clamshell computer with built-in screen & keyboard (e.g. MacBook, ThinkPad, Latitude, ZenBook, Inspiron, Chromebook).
   - "tablet": Touchscreen slate tablet (e.g. iPad, Galaxy Tab, MediaPad, MatePad, Surface Pro).
   - "monitor": Standalone display monitor / screen (e.g. 24", 27", IPS, 144Hz, FHD, Gaming Monitor).
   - "desktop_pc": Desktop computer, CPU tower, custom gaming rig, all-in-one desktop, workstation.
   - "other": Computer parts, accessories, cables, software.
2. "brand": Clean brand name (e.g., Apple, Dell, HP, Lenovo, Asus, Acer, Samsung, MSI, ViewSonic, AOC, LG, Xiaomi, Redmi, Huawei, Microsoft, Other).
3. "model": Clean series / model name WITHOUT cosmetic colors or warranty noise (e.g. "Chromebook C733", "27B1H", "Latitude 5420", "Victus 15", "ThinkPad T480", "iPad Pro M1", "Galaxy Tab S9", "Nitro VG270").
4. "cpu": (For laptops/tablets/desktops) Processor family ("Core i3", "Core i5", "Core i7", "Core i9", "Ryzen 3", "Ryzen 5", "Ryzen 7", "Ryzen 9", "Apple M1", "Apple M2", "Apple M3", "Apple M4", "Celeron", "Pentium", "Other"). Return null if monitor.
5. "generation": (For laptops/desktops) Intel/AMD generation as an integer (e.g., "11th Gen" -> 11). Return null if not applicable or Celeron/Pentium/Apple.
6. "ram_gb": RAM in GB as a number (e.g. 2, 4, 8, 16, 32, 64). Return null if not stated or if monitor.
7. "storage_gb": Total storage in GB as a number (e.g. 16, 32, 64, 128, 256, 512, 1024, 2048). Convert 1TB -> 1024. Return null if monitor.
8. "storage_type": "SSD", "HDD", or "NVMe". (Treat eMMC storage as "SSD").
9. "gpu": Dedicated or integrated graphics tier ("RTX 40-Series", "RTX 30-Series", "RTX 20-Series", "GTX", "Integrated", "Other Dedicated").
10. "screen_size_inch": Display diagonal size in inches as a float (e.g. 11.6, 14.0, 15.6, 24.0, 27.0).
11. "refresh_rate_hz": Refresh rate in Hz as an integer (e.g. 60, 75, 100, 120, 144, 165, 180, 240).
12. "resolution": Resolution standard: "1080p FHD", "2K QHD", "4K UHD", "HD".
13. "panel_type": "IPS", "OLED", "VA", or "Standard".
14. "condition": "Used" (default) or "Brand New" (sealed / unopened). Obey Tier 1 over Tier 3!
15. "is_touchscreen": true if touchscreen, x360, flip, 2-in-1; otherwise false.
16. "is_curved": true if curved monitor/display; otherwise false.
17. "is_gaming": true if gaming laptop/monitor/pc (e.g. TUF, ROG, Nitro, Legion, Victus, Alienware, >=100Hz); otherwise false.
18. "location": District/City (e.g. "Colombo", "Gampaha", "Kandy", "Kalutara", "Kurunegala", "Galle", "Matara", "Matale", "Other").

Respond ONLY with valid JSON matching the schema.
"""

EXTRACTION_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "category": {"type": "STRING", "enum": ["laptop", "tablet", "monitor", "desktop_pc", "other"]},
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
    dom_tree: Optional[Dict[str, Any]] = None,
    model_name: str = None
) -> Optional[Dict[str, Any]]:
    """
    Calls Google Gemini Flash Lite (multimodal) to extract structured electronics specifications 
    using Dual-Modal Fusion (DOM Tree + Screenshot) or single-source fallback (DOM-only / Vision-only).
    """
    _load_env_fallback()
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("\n⚠️ [GEMINI AI EXTRACTOR] GEMINI_API_KEY is missing from environment. Skipping AI image/spec extraction.")
        return None

    # Default to gemini-3.5-flash-lite
    model = model_name or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    
    combined_content = f"Title: {title.strip()}\nDescription: {description.strip()}\nPage Specs Text: {raw_text.strip()}"
    has_image = bool(image_base64 and len(image_base64.strip()) > 50)
    has_dom = bool(dom_tree and isinstance(dom_tree, dict) and any(dom_tree.values()))
    
    if len(combined_content.strip()) < 8 and not has_image and not has_dom:
        return None

    # Format structured DOM tree text into clear priority tiers
    formatted_dom_text = ""
    if has_dom:
        d_title = dom_tree.get("title") or title or ""
        d_loc = dom_tree.get("location") or ""
        d_loc_sub = dom_tree.get("location_subtitle") or ""
        d_price = dom_tree.get("price") or ""
        d_bc = " > ".join(dom_tree.get("breadcrumbs") or [])
        d_specs = dom_tree.get("specification_table") or {}
        d_desc = dom_tree.get("description") or description or ""

        formatted_dom_text = "\n" + "=" * 78 + "\n"
        formatted_dom_text += "=== TIER 1: IKMAN DEFINED ATTRIBUTES (HIGHEST PRIORITY - ABSOLUTE GROUND TRUTH) ===\n"
        if d_specs:
            for k, v in d_specs.items():
                if k and v:
                    formatted_dom_text += f"  • {k.capitalize()}: {v}\n"
        else:
            formatted_dom_text += "  (No structured attribute table detected)\n"

        formatted_dom_text += "\n=== TIER 2: LISTING TITLE & LOCATION (HIGH PRIORITY) ===\n"
        if d_title:
            formatted_dom_text += f"Listing Title: {d_title}\n"
        if d_loc_sub:
            formatted_dom_text += f"Location Subtitle: {d_loc_sub}\n"
        if d_loc:
            formatted_dom_text += f"Detected Location: {d_loc}\n"
        if d_price:
            formatted_dom_text += f"Marketplace Listed Price: Rs. {d_price}\n"
        if d_bc:
            formatted_dom_text += f"Category Breadcrumb Path: {d_bc}\n"

        formatted_dom_text += "\n=== TIER 3: SELLER DESCRIPTION (LOW PRIORITY - SUPPLEMENTARY SPECS ONLY) ===\n"
        formatted_dom_text += "(CRITICAL: Descriptions can be noisy and unreliable. Use ONLY to extract missing specs like Refresh Rate, Panel Type, Resolution, and exact Model. If any discrepancy with Tier 1 Condition or Brand, TIER 1 ALWAYS PREVAILS!)\n"
        if d_desc:
            formatted_dom_text += f"{d_desc.strip()[:4000]}\n"
        else:
            formatted_dom_text += "(No seller description provided)\n"
        formatted_dom_text += "=" * 78 + "\n"
    elif len(combined_content.strip()) >= 8:
        formatted_dom_text = (
            "\n" + "=" * 78 + "\n"
            "=== LISTING TEXT CONTEXT ===\n"
            f"{combined_content[:3500]}\n"
            "=" * 78 + "\n"
        )

    # Construct request parts (multimodal)
    parts = []
    
    clean_base64 = ""
    mime_type = "image/jpeg"
    raw_bytes_len = 0
    if has_image:
        clean_base64 = image_base64.strip()
        if "data:" in clean_base64 and ";base64," in clean_base64:
            header, clean_base64 = clean_base64.split(";base64,", 1)
            if "png" in header:
                mime_type = "image/png"
            elif "webp" in header:
                mime_type = "image/webp"
                
        raw_bytes_len = int(len(clean_base64) * 0.75)

        parts.append({
            "inlineData": {
                "mimeType": mime_type,
                "data": clean_base64
            }
        })
        
    prompt_text = f"{EXTRACTION_SYSTEM_PROMPT}\n\n"
    if has_image and has_dom:
        prompt_text += (
            "DUAL-MODAL CROSS-VERIFICATION PROTOCOL:\n"
            "You are provided with BOTH: (1) An un-truncated visual screenshot of the listing page, and (2) The structured DOM tree & specification table scraped directly from the HTML.\n"
            "1. Primary Spec Verification: Use the Tier 1 Ikman Defined Table and Tier 2 Listing Title for brand, model, condition, and location.\n"
            "2. Visual Cross-Examination: Inspect the visual screenshot to confirm hardware details: look for physical badges/stickers (e.g. Intel Core i5/i7, Nvidia RTX/GTX, AMD Radeon), photographed 'About PC' or BIOS screens, physical ports, and overall device category/form factor.\n"
            "3. Conflict Resolution: If informal text conflicts with explicit structured table rows or verified hardware badges, prioritize verified hardware badges and structured tables.\n\n"
        )
    elif has_image:
        prompt_text += (
            "VISION-ONLY EXTRACTION PROTOCOL:\n"
            "Carefully inspect this marketplace listing screenshot. Extract all visible hardware specifications, brand, model, condition, and listed price.\n\n"
        )
    else:
        prompt_text += (
            "DOM-TEXT EXTRACTION PROTOCOL:\n"
            "Extract all explicit hardware specifications from the structured DOM tree and listing text provided below.\n\n"
        )

    if formatted_dom_text:
        prompt_text += formatted_dom_text
        
    parts.append({"text": prompt_text})

    # ONLY SHOW WHAT WE SENT TO THE LLM (User Instruction)
    print("\n" + "=" * 80)
    print("📤 [DATA SENT TO GEMINI LLM]")
    print("=" * 80)
    print(f"🤖 Target Model:   {model}")
    if has_image:
        print(f"🖼️ Attached Image: {mime_type.upper()} (~{raw_bytes_len / 1024:.1f} KB)")
    else:
        print(f"🖼️ Attached Image: None (Text / DOM Tree Mode)")
    print("-" * 80)
    print("📝 PROMPT CONTENT DISPATCHED TO LLM:")
    print("-" * 80)
    print(prompt_text.strip())
    print("=" * 80 + "\n", flush=True)

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

    # Save incoming screenshot with page title and timestamp if explicitly enabled
    save_screenshots = os.environ.get("SAVE_SCREENSHOTS", "false").lower() in ("true", "1")
    if has_image and save_screenshots:
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
        except Exception:
            pass

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(url, json=payload)

            if resp.status_code != 200:
                # Fallback to gemini-1.5-flash if model endpoint differs
                if model != "gemini-1.5-flash":
                    fallback_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
                    resp = client.post(fallback_url, json=payload)

                if resp.status_code != 200:
                    print(f"   ❌ [ERROR] Gemini API call failed with HTTP {resp.status_code}: {resp.text[:150]}")
                    print("=" * 78 + "\n")
                    return None

            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                print("   ❌ [ERROR] No response candidate returned from Gemini.")
                print("=" * 78 + "\n")
                return None
                
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                print("   ❌ [ERROR] Response content parts list was empty.")
                print("=" * 78 + "\n")
                return None
                
            raw_json_str = parts[0].get("text", "").strip()
            parsed = json.loads(raw_json_str)
            return parsed
            
    except Exception as e:
        print(f"   ❌ [EXCEPTION] Gemini Multimodal LLM Extraction Error: {e}")
        print("=" * 78 + "\n")
        return None

if __name__ == "__main__":
    # Self-test with tricky laptop listing
    sample_text = "HP Victus 15 gaming laptop, Core i5 12450H 12th gen, 16GB DDR4, 512GB NVMe SSD, RTX 3050 4GB GPU, 144Hz FHD screen, Used with box, Colombo"
    print("Testing Electronics LLM Extractor with sample:")
    res = extract_electronics_specs_llm(title=sample_text)
    print("Result:", res)
