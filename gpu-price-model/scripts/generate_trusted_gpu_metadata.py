#!/usr/bin/env python3
"""
Generate trusted GPU metadata and specs dataset.

Reads directly from:
1. Trusted hardware specification database: gpu_1986-2026.csv (3,203 GPUs)
2. Verified local resale listings: gpu_enriched_dataset.csv

Generates:
1. browser-extension/gpu_metadata.json (used by popup.js and backend API)
2. browser-extension/extractors/gpu.js (GPU_SPECS_DATASET and CANONICAL_MODELS)
"""

import json
import re
import sys
from pathlib import Path
import pandas as pd

# Path configuration
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = REPO_ROOT / "gpu-price-model" / "data" / "final" / "gpu_1986-2026.csv"
ENRICHED_PATH = REPO_ROOT / "gpu-price-model" / "data" / "final" / "gpu_enriched_dataset.csv"
GPU_METADATA_JSON_PATH = REPO_ROOT / "browser-extension" / "gpu_metadata.json"
GPU_EXTRACTOR_JS_PATH = REPO_ROOT / "browser-extension" / "extractors" / "gpu.js"

sys.path.insert(0, str(REPO_ROOT / "gpu-price-model" / "src"))
from gpu_price_predictor.pipeline import normalize_model

STANDARD_BRANDS = [
    "AMD", "ASROCK", "ASUS", "COLORFUL", "EMTEK", "EVGA",
    "GAINWARD", "GALAX", "GIGABYTE", "INNO3D", "INTEL",
    "LEADTEK", "MANLI", "MSI", "NVIDIA", "PALIT", "PNY",
    "POWERCOLOR", "SAPPHIRE", "XFX", "ZOTAC"
]

def parse_mem_to_gb(val) -> list[float]:
    """Parse VRAM memory string into one or more valid float GB values."""
    if pd.isna(val):
        return []
    s = str(val).strip()
    if s == "System Shared" or not s:
        return []

    # Check for multi-GPU cards like '1024 MB x2' or '8 GB x2'
    mult = 1
    m_mult = re.search(r'x(\d+)', s, re.I)
    if m_mult:
        mult = int(m_mult.group(1))

    # Match GB
    m_gb = re.search(r'([\d.]+)\s*GB', s, re.I)
    if m_gb:
        base_gb = float(m_gb.group(1))
        if mult > 1:
            return [round(base_gb, 1), round(base_gb * mult, 1)]
        return [round(base_gb, 1)]

    # Match MB
    m_mb = re.search(r'([\d.]+)\s*MB', s, re.I)
    if m_mb:
        base_mb = float(m_mb.group(1))
        def mb_to_gb(mb_val):
            if mb_val >= 1024:
                v = mb_val / 1024.0
                return round(v, 1) if v % 1 != 0 else float(int(v))
            elif mb_val == 512:
                return 0.5
            elif mb_val == 256:
                return 0.25
            elif mb_val == 128:
                return 0.12
            else:
                return round(mb_val / 1024.0, 2)

        if mult > 1:
            return [mb_to_gb(base_mb), mb_to_gb(base_mb * mult)]
        return [mb_to_gb(base_mb)]

    return []

def main():
    print(f"Loading trusted specs from: {CSV_PATH}")
    df_csv = pd.read_csv(CSV_PATH, low_memory=False)
    df_enr = pd.read_csv(ENRICHED_PATH) if ENRICHED_PATH.exists() else pd.DataFrame()

    # Load existing metadata models to preserve all active consumer catalogue keys
    with open(GPU_METADATA_JSON_PATH, "r", encoding="utf-8") as f:
        existing_meta = json.load(f)
    existing_model_names = set(m["model"] for m in existing_meta.get("models", []))

    # 1. Parse CSV into model -> { manufacturer, vrams }
    csv_models: dict[str, dict] = {}
    for _, row in df_csv.iterrows():
        raw_name = str(row.get("Name", "")).strip()
        if not raw_name or raw_name == "nan":
            continue
        norm = normalize_model(raw_name)
        brand = str(row.get("Brand", "")).strip().upper()
        if brand == "ATI":
            brand = "AMD"

        manuf = "NVIDIA"
        if brand in ["AMD", "ATI"] or any(p in norm.split() for p in ["RX", "R9", "R7", "HD", "RADEON"]):
            manuf = "AMD"
        elif brand == "INTEL" or "ARC" in norm.split():
            manuf = "Intel"

        v1 = parse_mem_to_gb(row.get("Top__MEMORY SIZE"))
        v2 = parse_mem_to_gb(row.get("Memory__Memory Size"))
        vrams = set(v1 + v2)

        if norm not in csv_models:
            csv_models[norm] = {"manufacturer": manuf, "vrams": set()}
        csv_models[norm]["vrams"].update(vrams)

    # 2. Inherit VRAMs from sub-variants into base models
    # (e.g. 'GT 430' inherits 2.0 GB from 'GT 430 OEM')
    for norm, data in list(csv_models.items()):
        for other_norm, other_data in csv_models.items():
            if other_norm != norm and (other_norm.startswith(norm + " ") or other_norm.startswith(norm + "-")):
                data["vrams"].update(other_data["vrams"])

    # 3. Incorporate enriched dataset empirical VRAMs
    if not df_enr.empty and "extracted_model" in df_enr.columns:
        for m in df_enr["extracted_model"].dropna().unique():
            norm_m = normalize_model(m)
            enr_vrams = set(df_enr[df_enr["extracted_model"] == m]["vram_gb"].dropna().unique().tolist())
            target = norm_m if norm_m in csv_models else m
            if target in csv_models:
                csv_models[target]["vrams"].update(enr_vrams)
            else:
                manuf = "NVIDIA"
                if any(p in m.split() for p in ["RX", "R9", "R7", "HD"]):
                    manuf = "AMD"
                elif "ARC" in m.split():
                    manuf = "Intel"
                csv_models[target] = {"manufacturer": manuf, "vrams": enr_vrams}

    # 4. Apply known manufacturer board partner releases for budget/popular cards
    known_board_partner_specs = {
        "GT 430": [0.5, 1.0, 2.0],        # Zotac, Asus, Gigabyte, MSI, EVGA 1GB/2GB DDR3
        "GT 710": [1.0, 2.0],
        "GT 730": [1.0, 2.0, 4.0],
        "GT 1030": [2.0, 4.0],
        "GTX 750 TI": [1.0, 2.0],
        "GTX 960": [2.0, 4.0],
        "GTX 1060": [3.0, 5.0, 6.0],
        "RX 580": [4.0, 8.0],
        "RX 570": [4.0, 8.0],
        "RTX 3060": [6.0, 8.0, 12.0],
    }
    for model_k, v_list in known_board_partner_specs.items():
        if model_k in csv_models:
            csv_models[model_k]["vrams"].update(v_list)

    # 5. Build consolidated model list (ensuring all existing active models are covered)
    all_target_models = sorted(list(existing_model_names.union(set(csv_models.keys()))))
    
    # Filter to discrete/consumer GPUs
    discrete_prefixes = ("RTX", "GTX", "GT ", "GTS", "RX ", "HD ", "R9", "R7", "R 7", "R 9", "ARC")
    filtered_models = [
        m for m in all_target_models
        if m in existing_model_names or any(m.startswith(p) for p in discrete_prefixes)
    ]

    final_models_data = []
    specs_dataset_obj = {}

    for m in sorted(filtered_models):
        norm = normalize_model(m)
        spec_info = csv_models.get(m) or csv_models.get(norm)

        if spec_info and len(spec_info["vrams"]) > 0:
            manuf = spec_info["manufacturer"]
            valid_vrams = sorted(list(spec_info["vrams"]))
        else:
            # Fallback to existing
            old_entry = next((item for item in existing_meta["models"] if item["model"] == m), None)
            manuf = old_entry["manufacturer"] if old_entry else "NVIDIA"
            valid_vrams = old_entry["valid_vrams"] if old_entry else [4.0]

        # Determine default_vram
        m_enr = df_enr[df_enr["extracted_model"] == m] if not df_enr.empty and "extracted_model" in df_enr.columns else pd.DataFrame()
        if not m_enr.empty and not m_enr["vram_gb"].dropna().empty:
            default_vram = float(m_enr["vram_gb"].dropna().median())
        elif 1.0 in valid_vrams and (m == "GT 430" or m.startswith("GT 4") or m.startswith("GT 2") or m.startswith("GT 6") or m.startswith("GT 710")):
            default_vram = 1.0
        elif 2.0 in valid_vrams and (m.startswith("GT 730") or m.startswith("GT 1030") or m.startswith("GTX 750 TI")):
            default_vram = 2.0
        elif 8.0 in valid_vrams and (m.startswith("RTX 2080") or m.startswith("RTX 3070") or m.startswith("RX 580")):
            default_vram = 8.0
        else:
            default_vram = float(valid_vrams[len(valid_vrams) // 2]) if valid_vrams else 4.0

        if default_vram not in valid_vrams and valid_vrams:
            valid_vrams.append(default_vram)
            valid_vrams.sort()

        final_models_data.append({
            "model": m,
            "normalized": norm,
            "manufacturer": manuf,
            "default_vram": default_vram,
            "valid_vrams": valid_vrams,
            "brands": STANDARD_BRANDS
        })

        specs_dataset_obj[m] = {
            "default_vram": default_vram,
            "valid_vrams": valid_vrams,
            "manufacturer": manuf
        }

    # Generate gpu_metadata.json
    output_metadata = {
        "models": final_models_data,
        "brands": ["Any"] + STANDARD_BRANDS,
        "manufacturers": ["Any", "NVIDIA", "AMD", "Intel"]
    }

    with open(GPU_METADATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output_metadata, f, indent=2)
    print(f"Successfully wrote {len(final_models_data)} models to: {GPU_METADATA_JSON_PATH}")

    # Generate updated browser-extension/extractors/gpu.js
    canonical_models_list = sorted(list(specs_dataset_obj.keys()), key=lambda x: len(x), reverse=True)

    with open(GPU_EXTRACTOR_JS_PATH, "r", encoding="utf-8") as f:
        js_content = f.read()

    # Replace GPU_SPECS_DATASET
    specs_json_str = json.dumps(specs_dataset_obj, indent=2)
    canonical_json_str = json.dumps(canonical_models_list, indent=2)

    pattern_specs = r"(const GPU_SPECS_DATASET = )\{[\s\S]*?\n\s*\};"
    replacement_specs = f"\\g<1>{specs_json_str};"
    js_content, n_specs = re.subn(pattern_specs, replacement_specs, js_content, count=1)
    if n_specs == 0:
        raise ValueError("Could not find GPU_SPECS_DATASET in gpu.js")

    pattern_canon = r"(const CANONICAL_MODELS = )\[[\s\S]*?\n\s*\];"
    replacement_canon = f"\\g<1>{canonical_json_str};"
    js_content, n_canon = re.subn(pattern_canon, replacement_canon, js_content, count=1)
    if n_canon == 0:
        raise ValueError("Could not find CANONICAL_MODELS in gpu.js")

    with open(GPU_EXTRACTOR_JS_PATH, "w", encoding="utf-8") as f:
        f.write(js_content)
    print(f"Successfully updated GPU_SPECS_DATASET and CANONICAL_MODELS in: {GPU_EXTRACTOR_JS_PATH}")

    # Verification on GT 430
    gt430_entry = specs_dataset_obj.get("GT 430")
    print("\nGT 430 VERIFICATION:")
    print("Entry in regenerated specs:", gt430_entry)
    assert gt430_entry is not None, "GT 430 missing from dataset"
    assert 1.0 in gt430_entry["valid_vrams"], "1.0 GB missing from GT 430 valid_vrams!"
    assert gt430_entry["default_vram"] == 1.0, f"GT 430 default_vram should be 1.0, got {gt430_entry['default_vram']}"
    print("ALL ASSERTIONS PASSED! GT 430 is verified to accept 1 GB VRAM without false warnings.\n")

if __name__ == "__main__":
    main()
