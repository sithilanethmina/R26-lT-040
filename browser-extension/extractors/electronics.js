/**
 * FairPriceLK - Electronics Category Extractor
 * Parses Laptops, Monitors, and Tablets from marketplace pages.
 */

window.FairPriceLK_Extractors = window.FairPriceLK_Extractors || {};

window.FairPriceLK_Extractors.electronics = (function () {
    const ELEC_BRANDS = [
        "DELL", "HP", "LENOVO", "ASUS", "ACER", "APPLE", "SAMSUNG", "MSI",
        "LG", "VIEWSONIC", "BENQ", "AOC", "TOSHIBA", "SONY", "HUAWEI", "XIAOMI"
    ];

    function extractSubCategory(combinedText) {
        const lower = (combinedText || "").toLowerCase();
        if (lower.includes("monitor") || lower.includes("display") || lower.includes("screen")) {
            return "monitor";
        }
        if (lower.includes("tablet") || lower.includes("ipad") || lower.includes("tab")) {
            return "tablet";
        }
        return "laptop";
    }

    function extractBrand(combinedText, keyValues) {
        if (keyValues && keyValues.brand) {
            const bUpper = keyValues.brand.toUpperCase().trim();
            for (const b of ELEC_BRANDS) {
                if (bUpper.includes(b)) return b;
            }
        }
        const upper = (combinedText || "").toUpperCase();
        for (const b of ELEC_BRANDS) {
            if (new RegExp(`\\b${b}\\b`, 'i').test(upper)) {
                return b;
            }
        }
        return "Generic";
    }

    function extractRam(combinedText) {
        const match = (combinedText || "").match(/(\d{1,2})\s*(?:GB)?\s*(?:ram|memory)/i);
        if (match) {
            const val = parseInt(match[1], 10);
            if ([2, 4, 8, 16, 32, 64].includes(val)) return val;
        }
        return 8;
    }

    function extractStorage(combinedText) {
        const match = (combinedText || "").match(/(\d{2,4})\s*(?:GB|TB)\s*(?:ssd|hdd|nvme|storage|drive)?/i);
        if (match) {
            let val = parseInt(match[1], 10);
            if (match[0].toUpperCase().includes("TB")) val *= 1024;
            if ([128, 256, 512, 1024, 2048].includes(val)) return val;
        }
        return 256;
    }

    function parse(pageContext) {
        const { title = "", price = null, raw_text = "", key_values = {} } = pageContext;
        const scope = `${title} ${key_values.brand || ""} ${key_values.model || ""} ${raw_text}`;

        const subCat = extractSubCategory(scope);
        const brand = extractBrand(scope, key_values);
        let model = key_values.model || title.replace(new RegExp(`\\b${brand}\\b`, 'gi'), '').replace(/for sale.*$/i, '').trim() || "Standard Model";
        const ram = extractRam(scope);
        const storage = extractStorage(scope);

        const missingFields = [];
        if (!brand || brand === "Generic") missingFields.push("Brand");
        if (!price || isNaN(price) || price <= 0) missingFields.push("Listing Price");

        return {
            category: "electronics",
            valid: missingFields.length === 0,
            missing_fields: missingFields,
            error_message: missingFields.length > 0 ? `Missing required electronics details: [${missingFields.join(", ")}].` : null,
            data: {
                category: subCat,
                brand: brand,
                model: model,
                ram: ram,
                storage: storage,
                listed_price: price || null
            }
        };
    }

    return { parse: parse };
})();
