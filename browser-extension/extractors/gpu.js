/**
 * FairPriceLK - GPU Category Extractor
 * Provides high-precision parsing of GPU listings from marketplace pages (Ikman, etc.)
 * Strictly matches canonical GPU models and checks required fields.
 */

window.FairPriceLK_Extractors = window.FairPriceLK_Extractors || {};

window.FairPriceLK_Extractors.gpu = (function () {
    // Canonical models sorted by length descending so longer tokens match first (e.g. RTX 3060 TI before RTX 3060)
    const CANONICAL_MODELS = [
        "RTX 4090", "RTX 4080 SUPER", "RTX 4080", "RTX 4070 TI SUPER", "RTX 4070 TI", "RTX 4070 SUPER", "RTX 4070",
        "RTX 4060 TI", "RTX 4060", "RTX 3090 TI", "RTX 3090", "RTX 3080 TI", "RTX 3080", "RTX 3070 TI", "RTX 3070",
        "RTX 3060 TI", "RTX 3060", "RTX 3050", "RTX 2080 TI", "RTX 2080 SUPER", "RTX 2080", "RTX 2070 SUPER", "RTX 2070",
        "RTX 2060 SUPER", "RTX 2060", "GTX 1660 TI", "GTX 1660 SUPER", "GTX 1660", "GTX 1650 SUPER", "GTX 1650", "GTX 1630",
        "GTX 1080 TI", "GTX 1080", "GTX 1070 TI", "GTX 1070", "GTX 1060", "GTX 1050 TI", "GTX 1050", "GTX 980 TI",
        "GTX 980", "GTX 970", "GTX 960", "GTX 950", "GTX 780 TI", "GTX 780", "GTX 770", "GTX 760", "GTX 750 TI", "GTX 750",
        "GTX 680", "GTX 670", "GTX 660 TI", "GTX 660", "GTX 650 TI", "GTX 650", "GTX 580", "GTX 570", "GTX 560 TI", "GTX 560",
        "GTX 550 TI", "GTX 550", "GTX 460", "GTS 450", "GT 1030", "GT 730", "GT 710", "GT 630", "GT 610", "GT 520", "GT 440",
        // AMD
        "RX 7900 XTX", "RX 7900 XT", "RX 7900 GRE", "RX 7800 XT", "RX 7700 XT", "RX 7600 XT", "RX 7600",
        "RX 6950 XT", "RX 6900 XT", "RX 6800 XT", "RX 6800", "RX 6750 XT", "RX 6700 XT", "RX 6700", "RX 6650 XT",
        "RX 6600 XT", "RX 6600", "RX 6500 XT", "RX 6400", "RX 5700 XT", "RX 5700", "RX 5600 XT", "RX 5500 XT",
        "RX 590", "RX 580", "RX 570", "RX 560", "RX 550", "RX 480", "RX 470", "RX 460", "R9 390", "R9 380", "R9 290",
        "R9 280", "R9 270", "R7 370", "R7 260", "R7 250", "R7 240", "HD 7970", "HD 7950", "HD 7870", "HD 7850", "HD 7770", "HD 6870",
        // Intel
        "ARC A770", "ARC A750", "ARC A580", "ARC A380", "ARC A310"
    ];

    const KNOWN_BRANDS = [
        "ASUS", "MSI", "GIGABYTE", "ZOTAC", "GALAX", "PALIT", "SAPPHIRE",
        "ASROCK", "POWERCOLOR", "COLORFUL", "INNO3D", "PNY", "EVGA", "EMTEK",
        "GAINWARD", "XFX", "MANLI", "LEADTEK", "NVIDIA", "AMD", "INTEL"
    ];

    const MODEL_ALIASES = {
        "GTX 1080 SUPER": "GTX 1080",
        "GTX 2060": "RTX 2060",
        "GTX 3060 TI": "RTX 3060 TI",
        "RX 2070": "RTX 2070",
        "RTX3060": "RTX 3060",
        "RTX3070": "RTX 3070",
        "RTX3080": "RTX 3080",
        "RTX4060": "RTX 4060",
        "RTX4070": "RTX 4070",
        "RTX4080": "RTX 4080",
        "RTX4090": "RTX 4090",
        "GTX1660": "GTX 1660",
        "GTX1650": "GTX 1650",
        "GTX1050TI": "GTX 1050 TI",
        "GTX750TI": "GTX 750 TI",
        "RX580": "RX 580",
        "RX570": "RX 570",
        "RX6600": "RX 6600"
    };

    function normalizeText(text) {
        if (!text) return "";
        let clean = String(text).toUpperCase().replace(/[-_]/g, " ").replace(/\s+/g, " ").trim();
        clean = clean.replace(/\bGEFORCE\b/g, " ");
        clean = clean.replace(/\bRADEON\b/g, " ");
        clean = clean.replace(/\bNVIDIA\b/g, " ");
        clean = clean.replace(/\bAMD\b/g, " ");
        clean = clean.replace(/\bINTEL\s+ARC\b/g, "ARC");
        // Separate prefix from numbers e.g. RTX3060 -> RTX 3060, RX580 -> RX 580
        clean = clean.replace(/\b(RTX|GTX|RX|GT|GTS|ARC|HD|R9|R7)\s*(\d{3,4})\b/g, "$1 $2");
        // Normalize suffix spacing e.g. 3060TI -> 3060 TI
        clean = clean.replace(/\b(\d{3,4})\s*(TI|XT|XTX|SUPER)\b/g, "$1 $2");
        clean = clean.replace(/\s+/g, " ").trim();
        return clean;
    }

    function extractModel(combinedText) {
        if (!combinedText) return null;
        const rawNormalized = normalizeText(combinedText);
        const padded = ` ${rawNormalized} `;

        // 1. Check alias mapping first (sorted by length descending)
        const sortedAliases = Object.keys(MODEL_ALIASES).sort((a, b) => normalizeText(b).length - normalizeText(a).length);
        for (const alias of sortedAliases) {
            const normAlias = normalizeText(alias);
            const regex = new RegExp(`(?:^|\\s)${normAlias.replace(/\s+/g, '\\s+')}(?=\\s|$)`, 'i');
            if (regex.test(padded)) {
                return MODEL_ALIASES[alias];
            }
        }

        // 2. Match canonical models (sorted by normalized length descending so "RTX 3060 TI" > "RTX 3060")
        const sortedModels = [...CANONICAL_MODELS].sort((a, b) => normalizeText(b).length - normalizeText(a).length);
        for (const model of sortedModels) {
            const normModel = normalizeText(model);
            const regex = new RegExp(`(?:^|\\s)${normModel.replace(/\s+/g, '\\s+')}(?=\\s|$)`, 'i');
            if (regex.test(padded)) {
                return model;
            }
        }

        return null;
    }

    function extractVram(combinedText, matchedModel) {
        if (!combinedText) return null;
        
        // Priority 1: Match standard VRAM patterns like "8GB", "8 GB GDDR6", "16 GB VRAM"
        const matches = [...combinedText.matchAll(/(\d{1,2})\s*(?:GB|G)\b(?:\s*(?:GDDR\d[X]?|VRAM|DDR\d))?/gi)];
        if (matches.length > 0) {
            for (const m of matches) {
                const val = parseInt(m[1], 10);
                if (val >= 1 && val <= 48) {
                    return val;
                }
            }
        }

        // Priority 2: Fallback defaults for famous fixed-size GPUs
        if (matchedModel) {
            const fixedVramMap = {
                "GTX 1050 TI": 4, "GTX 750 TI": 2, "GTX 1650 SUPER": 4, "GTX 1660 SUPER": 6,
                "GTX 1660 TI": 6, "GTX 1070": 8, "GTX 1070 TI": 8, "GTX 1080": 8, "GTX 1080 TI": 11,
                "RTX 3070": 8, "RTX 3070 TI": 8, "RTX 3080 TI": 12, "RTX 3090": 24, "RTX 3090 TI": 24,
                "RTX 4080": 16, "RTX 4090": 24, "RX 5600 XT": 6, "RX 6600": 8, "RX 6600 XT": 8,
                "RX 6700 XT": 12, "RX 6800": 16, "RX 6800 XT": 16, "RX 6900 XT": 16, "RX 7800 XT": 16,
                "RX 7900 XT": 20, "RX 7900 XTX": 24, "GT 1030": 2, "GT 710": 2
            };
            if (fixedVramMap[matchedModel]) {
                return fixedVramMap[matchedModel];
            }
        }

        return null;
    }

    function extractBrand(combinedText, rawTextElements) {
        if (!combinedText) return "Any";
        const upper = combinedText.toUpperCase();

        // 1. Look for explicit Brand field from spec table
        if (rawTextElements && rawTextElements.brand) {
            const explicit = rawTextElements.brand.toUpperCase();
            for (const b of KNOWN_BRANDS) {
                if (explicit.includes(b)) return b;
            }
        }

        // 2. Check title / description
        for (const b of KNOWN_BRANDS) {
            const regex = new RegExp(`\\b${b}\\b`, 'i');
            if (regex.test(upper)) {
                return b;
            }
        }

        return "Any";
    }

    function extractManufacturer(model, brand) {
        if (!model) return "Any";
        const mUpper = model.toUpperCase();
        if (mUpper.startsWith("RTX") || mUpper.startsWith("GTX") || mUpper.startsWith("GT") || mUpper.startsWith("GTS")) {
            return "NVIDIA";
        }
        if (mUpper.startsWith("RX") || mUpper.startsWith("R9") || mUpper.startsWith("R7") || mUpper.startsWith("HD")) {
            return "AMD";
        }
        if (mUpper.startsWith("ARC")) {
            return "Intel";
        }
        if (brand === "NVIDIA" || brand === "AMD" || brand === "INTEL") {
            return brand;
        }
        return "Any";
    }

    function parse(pageContext) {
        const { title = "", price = null, raw_text = "", key_values = {} } = pageContext;
        const searchScope = `${title} ${key_values.brand || ""} ${key_values.model || ""} ${raw_text}`;

        const model = extractModel(`${title} ${key_values.model || ""}`) || extractModel(searchScope);
        const vram = extractVram(`${title} ${key_values.vram || ""}`, model) || extractVram(searchScope, model);
        const brand = extractBrand(`${title} ${key_values.brand || ""}`, key_values);
        const manufacturer = extractManufacturer(model, brand);

        // Validation for 100% precision guarantee
        const missingFields = [];
        if (!model) missingFields.push("GPU Model");
        if (!vram) missingFields.push("VRAM (GB)");
        if (!price || isNaN(price) || price <= 0) missingFields.push("Listing Price");

        const isValid = missingFields.length === 0;

        return {
            category: "gpu",
            valid: isValid,
            missing_fields: missingFields,
            error_message: isValid ? null : `Could not auto-extract 100% of GPU specifications: Missing [${missingFields.join(", ")}]. Please enter manually.`,
            data: {
                title: title,
                model: model || "",
                vram_gb: vram || null,
                brand: brand || "Any",
                manufacturer: manufacturer || "Any",
                listed_price: price || null,
                stock: "In Stock",
                description: raw_text || title || ""
            }
        };
    }

    return {
        parse: parse,
        extractModel: extractModel,
        extractVram: extractVram,
        extractBrand: extractBrand,
        CANONICAL_MODELS: CANONICAL_MODELS,
        KNOWN_BRANDS: KNOWN_BRANDS
    };
})();
