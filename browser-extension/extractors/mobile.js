/**
 * FairPriceLK - Mobile Phone Category Extractor
 * Parses Android & iPhone listings from marketplace pages.
 */

window.FairPriceLK_Extractors = window.FairPriceLK_Extractors || {};

window.FairPriceLK_Extractors.mobile = (function () {
    const PHONE_BRANDS = [
        "APPLE", "SAMSUNG", "XIAOMI", "REDMI", "POCO", "ONEPLUS", "GOOGLE",
        "HUAWEI", "VIVO", "OPPO", "REALME", "SONY", "NOKIA", "MOTOROLA", "ASUS", "HONOR"
    ];

    function extractPhoneType(combinedText) {
        const lower = (combinedText || "").toLowerCase();
        if (lower.includes("iphone") || lower.includes("apple") || lower.includes("ios")) {
            return "iphone";
        }
        return "android";
    }

    function extractBrand(combinedText, keyValues) {
        if (keyValues && keyValues.brand) {
            const bUpper = keyValues.brand.toUpperCase().trim();
            for (const b of PHONE_BRANDS) {
                if (bUpper.includes(b)) return b;
            }
        }
        const upper = (combinedText || "").toUpperCase();
        for (const b of PHONE_BRANDS) {
            if (new RegExp(`\\b${b}\\b`, 'i').test(upper)) {
                return b;
            }
        }
        return "Unknown";
    }

    function extractStorage(combinedText) {
        const match = (combinedText || "").match(/(\d{1,4})\s*(?:GB|TB)\s*(?:rom|storage|internal)?/i);
        if (match) {
            let num = parseInt(match[1], 10);
            if (match[0].toUpperCase().includes("TB")) num *= 1024;
            if ([16, 32, 64, 128, 256, 512, 1024].includes(num)) {
                return num;
            }
        }
        return 128;
    }

    function extractRam(combinedText) {
        const match = (combinedText || "").match(/(\d{1,2})\s*(?:GB)?\s*(?:ram|\/)/i);
        if (match) {
            const val = parseInt(match[1], 10);
            if ([2, 3, 4, 6, 8, 12, 16].includes(val)) {
                return val;
            }
        }
        return 6;
    }

    function extractWarranty(combinedText) {
        const lower = (combinedText || "").toLowerCase();
        if (lower.includes("year warranty") || lower.includes("1 year")) return 365;
        if (lower.includes("6 month") || lower.includes("6 months")) return 180;
        if (lower.includes("3 month") || lower.includes("3 months")) return 90;
        if (lower.includes("1 month") || lower.includes("month warranty")) return 30;
        if (lower.includes("checking warranty") || lower.includes("check warranty")) return 7;
        return 0;
    }

    function parse(pageContext) {
        const { title = "", price = null, raw_text = "", key_values = {} } = pageContext;
        const scope = `${title} ${key_values.brand || ""} ${key_values.model || ""} ${raw_text}`;

        const phoneType = extractPhoneType(scope);
        const brand = extractBrand(scope, key_values);
        let model = key_values.model || "";
        if (!model) {
            // Remove brand from title to approximate model
            model = title.replace(new RegExp(`\\b${brand}\\b`, 'gi'), '').trim();
            model = model.replace(/for sale.*$/i, '').replace(/brand new.*$/i, '').trim();
        }

        const storage = extractStorage(scope);
        const ram = extractRam(scope);
        const warranty = extractWarranty(scope);

        const missingFields = [];
        if (!brand || brand === "Unknown") missingFields.push("Brand");
        if (!model) missingFields.push("Model");
        if (!price || isNaN(price) || price <= 0) missingFields.push("Listing Price");

        return {
            category: "mobile",
            valid: missingFields.length === 0,
            missing_fields: missingFields,
            error_message: missingFields.length > 0 ? `Missing required mobile details: [${missingFields.join(", ")}].` : null,
            data: {
                phone_type: phoneType,
                brand: brand,
                model: model,
                storage_gb: storage,
                ram_gb: ram,
                warranty_days: warranty,
                listed_price: price || null
            }
        };
    }

    return { parse: parse };
})();
