"""
Bilingual (English + Sinhala) Condition Tag Extractor for GPU Listings.
======================================================================
Extracts structured condition tags from free-text descriptions:
- has_warranty (bool)
- warranty_months (float: 0.0, 1.0, 2.0, 3.0, 6.0, 12.0, 24.0, etc.)
- needs_repair (bool)
- urgent_sale (bool)
- tested_working (bool)
- good_condition (bool)
- brand_new (bool)
- price_negotiable (bool)
- is_shop (bool)
- delivery_available (bool)
"""

from __future__ import annotations

import re
from typing import Any, Dict


def extract_condition_tags(text: str | None, source: str = "ikman") -> Dict[str, Any]:
    """
    Extracts structured condition attributes from listing description text.
    Handles English, Sinhala, and Singlish expressions.
    """
    tags: Dict[str, Any] = {
        "has_warranty": False,
        "warranty_months": 0.0,
        "needs_repair": False,
        "urgent_sale": False,
        "tested_working": False,
        "good_condition": False,
        "brand_new": False,
        "price_negotiable": False,
        "is_shop": False,
        "delivery_available": False,
        "has_description": bool(text and str(text).strip()),
    }

    # Verified retail computer stores in Sri Lanka
    if source in ("md", "msk"):
        tags["is_shop"] = True
        tags["has_warranty"] = True
        tags["warranty_months"] = 2.0  # Standard retail testing warranty
        tags["good_condition"] = True
        tags["tested_working"] = True
        return tags

    if not text or not isinstance(text, str) or not text.strip():
        return tags

    s = text.lower()

    # --------------------------------------------------------------------------
    # 1. Warranty Detection (Duration & Existence)
    # --------------------------------------------------------------------------
    # Negation check: e.g. "no warranty", "without warranty", "warranty expired", "out of warranty", "වගකීමක් නැත"
    has_neg_warranty = bool(
        re.search(
            r"(?:no\s+warranty|without\s+warranty|warranty\s+(?:over|expired|ended|done|out|n[ae]th?[ai]|iwarai|nehe)|out\s+of\s+warranty|0\s*warranty|no\s+checking\s+warranty|වගකී[මම්]\s*(?:නැත|නොමැත|නෑ|නැහැ|ඉවරයි|අවසන්))",
            s,
        )
    )

    if not has_neg_warranty:
        # Years warranty - Number MUST be explicitly tied to warranty keyword
        yr_match = re.search(
            r"(?:(\d+)\s*(?:years?|yrs?|වසර|අවුරුදු)\s*(?:of\s+)?(?:company\s+|store\s+|shop\s+|seller\s+|remaining\s+|left\s+)?(?:warranty|waranty|warrenty|වගකී|වොරන්ටි|ගැරන්ටි)|(?:warranty|waranty|warrenty|වගකී|වොරන්ටි|ගැරන්ටි)\s*(?:remaining\s+|left\s+|for\s+|of\s+)?(\d+)\s*(?:years?|yrs?|වසර|අවුරුදු))",
            s,
        )
        if yr_match:
            y_val = float(yr_match.group(1) or yr_match.group(2))
            if 0 < y_val <= 5:
                tags["has_warranty"] = True
                tags["warranty_months"] = y_val * 12.0

        # Months warranty - Number MUST be explicitly tied to warranty keyword
        if tags["warranty_months"] == 0.0:
            mo_match = re.search(
                r"(?:(\d{1,2})\s*(?:months?|mnths?|mos?|මාස)\s*(?:of\s+)?(?:company\s+|store\s+|shop\s+|seller\s+|checking\s+|testing\s+|remaining\s+|left\s+)?(?:warranty|waranty|warrenty|වගකී|වොරන්ටි|ගැරන්ටි)|(?:warranty|waranty|warrenty|වගකී|වොරන්ටි|ගැරන්ටි)\s*(?:remaining\s+|left\s+|for\s+|of\s+)?(\d{1,2})\s*(?:months?|mnths?|mos?|මාස))",
                s,
            )
            if mo_match:
                m_val = float(mo_match.group(1) or mo_match.group(2))
                if 0 < m_val <= 36:
                    tags["has_warranty"] = True
                    tags["warranty_months"] = m_val

        # Specific calendar year expiry (e.g. Warranty 2026 June, till 2027)
        if tags["warranty_months"] == 0.0:
            date_match = re.search(
                r"(?:warranty|waranty|වගකී).*?\b(202[5-9])\b", s
            )
            if date_match:
                end_yr = int(date_match.group(1))
                curr_yr = 2026
                rem_months = max(1.0, float((end_yr - curr_yr) * 12 + 6))
                tags["has_warranty"] = True
                tags["warranty_months"] = min(36.0, rem_months)

        # General warranty mentioned without specific duration
        if tags["warranty_months"] == 0.0 and re.search(
            r"\b(?:company\s+warranty|shop\s+warranty|checking\s+warranty|seller\s+warranty|warranty\s+available|with\s+warranty|වගකීමක්\s*සහිත|වගකීම\s*ඇත)\b",
            s,
        ):
            tags["has_warranty"] = True
            tags["warranty_months"] = 1.0  # Default 1 month for unspecified warranty

    # --------------------------------------------------------------------------
    # 2. Defective / Needs Repair
    # --------------------------------------------------------------------------
    if re.search(
        r"(?:repair|broken|defective|not\s+working|faulty|damaged|crack|dead|fans?\s+not\s+spin|display\s+no|no\s+display|artifacts?|අඩුපාඩු|ලෙඩ|හදන්න|කැඩි)",
        s,
    ):
        # Negation filter: e.g. "no repair", "no defects", "කිසිම ලෙඩක් නෑ"
        if not re.search(
            r"(?:no\s+repair|no\s+defects?|no\s+issues?|no\s+error|no\s+fault|ලෙඩ\s*(?:නැ|නෑ|නොමැත)|කිසිම\s*ලෙඩක්\s*(?:නැ|නෑ|නොමැත)|කිසිම\s*අඩුපාඩුවක්\s*(?:නැ|නෑ|නොමැත)|කිසිම\s*ප්‍රශ්නයක්\s*(?:නැ|නෑ|නොමැත))",
            s,
        ):
            tags["needs_repair"] = True

    # --------------------------------------------------------------------------
    # 3. Urgent / Quick Sale
    # --------------------------------------------------------------------------
    if re.search(
        r"(?:urgent\s+(?:sale|sell)|quick\s+sale|need\s+cash|money\s+urgent|emergency\s+sale|හදිසි|ඉක්මනින්)",
        s,
    ):
        tags["urgent_sale"] = True

    # --------------------------------------------------------------------------
    # 4. Tested & Stress Tested Working
    # --------------------------------------------------------------------------
    if re.search(
        r"(?:tested|working|furmark|stress\s*test|100%\s*work|perfectly\s*work|no\s+errors?|temps?\s*(?:good|ok|normal)|පරීක්ෂා\s*කර|චෙක්\s*කර|හොඳටම\s*වැඩ)",
        s,
    ):
        tags["tested_working"] = True

    # --------------------------------------------------------------------------
    # 5. Good Physical & Operational Condition
    # --------------------------------------------------------------------------
    if re.search(
        r"(?:good\s+condition|best\s+condition|superb\s+condition|well\s+maintained|excellent\s+condition|mint\s+condition|සුපිරි|හොඳම?\s*තත්ව|හොඳට\s*තියෙනවා)",
        s,
    ):
        tags["good_condition"] = True

    # --------------------------------------------------------------------------
    # 6. Brand New / Sealed (Exclude comparative or past-purchase statements)
    # --------------------------------------------------------------------------
    if re.search(
        r"(?:brand\s+new(?!\s*(?:condition|look|wage|vage|thathwaye|thathwe))|factory\s+sealed|company\s+sealed|unopened|සීල්\s*කරන\s*ලද)",
        s,
    ):
        # Exclude "bought brand new", "like brand new", "brand new condition"
        if not re.search(
            r"(?:like\s+brand\s+new|as\s+brand\s+new|bought\s+(?:as\s+)?brand\s+new|purchased\s+brand\s+new|bought\s+new|brand\s+new\s+condition|brand\s+new\s+look|brand\s+new\s+wage|අලුත්\s*වගේ)",
            s,
        ):
            tags["brand_new"] = True

    # --------------------------------------------------------------------------
    # 7. Price Negotiable
    # --------------------------------------------------------------------------
    if re.search(
        r"(?:negotiable|can\s+negotiate|price\s+(?:is\s+)?negotiable|මිල\s*වෙනස්|අඩු\s*කරල\s*දෙන්න|සාකච්ඡා\s*කර)",
        s,
    ):
        tags["price_negotiable"] = True

    # --------------------------------------------------------------------------
    # 8. Commercial / Shop Listing
    # --------------------------------------------------------------------------
    if re.search(
        r"(?:showroom|computer\s+house|u\s*tec\s*zone|nanotek|barclays|redline|chama\s*computers|ශාඛාව|පැමිණ|ලිපිනය)",
        s,
    ):
        tags["is_shop"] = True

    # --------------------------------------------------------------------------
    # 9. Delivery Available
    # --------------------------------------------------------------------------
    if re.search(
        r"(?:delivery|island\s*wide|courier|cod|ඩිලිවරි|ලංකාව\s*පුරා)",
        s,
    ):
        tags["delivery_available"] = True

    return tags
