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

    function extractSubCategory(title, combinedText) {
        const titleLower = (title || "").toLowerCase();
        const lower = (combinedText || "").toLowerCase();
        
        // 1. Check title first (highest confidence)
        if (titleLower.includes("monitor") || titleLower.includes("monitors")) {
            return "monitor";
        }
        if (titleLower.includes("tablet") || titleLower.includes("ipad") || titleLower.includes("matepad") || titleLower.includes("mediapad") || titleLower.includes("galaxy tab") || titleLower.includes("mi pad")) {
            return "tablet";
        }
        if (titleLower.includes("tab ") || titleLower.endsWith("tab") || /\btab\b/.test(titleLower)) {
            return "tablet";
        }
        if (titleLower.includes("laptop") || titleLower.includes("notebook") || titleLower.includes("macbook") || titleLower.includes("thinkpad") || titleLower.includes("elitebook") || titleLower.includes("probook") || titleLower.includes("zenbook") || titleLower.includes("vivobook") || titleLower.includes("latitude") || titleLower.includes("inspiron") || titleLower.includes("vostro") || titleLower.includes("precision")) {
            return "laptop";
        }
        
        // 2. Check full text (lower confidence, order of specificity)
        if (lower.includes("monitor") || lower.includes("monitors")) {
            return "monitor";
        }
        if (lower.includes("tablet") || lower.includes("ipad") || lower.includes("matepad") || lower.includes("mediapad") || lower.includes("galaxy tab") || lower.includes("mi pad")) {
            return "tablet";
        }
        if (lower.includes("tab ") || lower.endsWith("tab") || /\btab\b/.test(lower)) {
            return "tablet";
        }
        if (lower.includes("laptop") || lower.includes("notebook") || lower.includes("macbook") || lower.includes("thinkpad") || lower.includes("elitebook") || lower.includes("probook") || lower.includes("zenbook") || lower.includes("vivobook") || lower.includes("latitude") || lower.includes("inspiron") || lower.includes("vostro") || lower.includes("precision")) {
            return "laptop";
        }
        
        // 3. Fallback for screen/display keywords in text
        if (lower.includes("display") || lower.includes("screen")) {
            if (lower.includes("ram") || lower.includes("storage") || lower.includes("ssd") || lower.includes("intel") || lower.includes("ryzen") || lower.includes("cpu") || lower.includes("core i")) {
                return "laptop";
            }
            return "monitor";
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

    function extractRam(combinedText, key_values) {
        if (key_values) {
            for (const key of ['ram', 'memory', 'ram (gb)']) {
                if (key_values[key]) {
                    const match = key_values[key].match(/(\d+)/);
                    if (match) {
                        const val = parseInt(match[1], 10);
                        if ([2, 4, 8, 16, 32, 64].includes(val)) return val;
                    }
                }
            }
        }
        const match = (combinedText || "").match(/(\d{1,2})\s*(?:GB)?\s*(?:ram|memory)/i);
        if (match) {
            const val = parseInt(match[1], 10);
            if ([2, 4, 8, 16, 32, 64].includes(val)) return val;
        }
        return 8;
    }

    function extractStorage(combinedText, key_values) {
        if (key_values) {
            for (const key of ['storage', 'storage capacity', 'hard drive', 'storage capacity (gb)']) {
                if (key_values[key]) {
                    const match = key_values[key].match(/(\d+)\s*(GB|TB)?/i);
                    if (match) {
                        let val = parseInt(match[1], 10);
                        const unit = match[2] ? match[2].toUpperCase() : '';
                        if (unit === 'TB' || val <= 4) val *= 1024;
                        if ([128, 256, 512, 1024, 2048].includes(val)) return val;
                    }
                }
            }
        }
        const match = (combinedText || "").match(/(\d{2,4})\s*(?:GB|TB)\s*(?:ssd|hdd|nvme|storage|drive)?/i);
        if (match) {
            let val = parseInt(match[1], 10);
            if (match[0].toUpperCase().includes("TB")) val *= 1024;
            if ([128, 256, 512, 1024, 2048].includes(val)) return val;
        }
        return 256;
    }

    function extractSize(combinedText, key_values) {
        if (key_values) {
            for (const key of ['size', 'screen size', 'display size']) {
                if (key_values[key]) {
                    const match = key_values[key].match(/(\d+)/);
                    if (match) return match[1] + " Inch";
                }
            }
        }
        const match = (combinedText || "").match(/(\d{2})\s*(?:inch|")/i);
        if (match) {
            return match[1] + " Inch";
        }
        return "24 Inch";
    }

    function extractHz(combinedText, key_values) {
        if (key_values) {
            for (const key of ['refresh rate', 'frequency']) {
                if (key_values[key]) {
                    const match = key_values[key].match(/(\d+)/);
                    if (match) return match[1] + "Hz";
                }
            }
        }
        const match = (combinedText || "").match(/(\d{2,3})\s*hz/i);
        if (match) {
            return match[1] + "Hz";
        }
        return "60Hz";
    }

    function extractResolution(combinedText, key_values) {
        if (key_values) {
            for (const key of ['resolution', 'display resolution']) {
                if (key_values[key]) {
                    const val = key_values[key].toUpperCase();
                    if (val.includes("4K") || val.includes("3840")) return "4K";
                    if (val.includes("2K") || val.includes("2560")) return "2K";
                    if (val.includes("FHD") || val.includes("1080")) return "FHD";
                }
            }
        }
        const lower = (combinedText || "").toLowerCase();
        if (lower.includes("4k") || lower.includes("uhd") || lower.includes("2160p")) return "4K";
        if (lower.includes("2k") || lower.includes("qhd") || lower.includes("1440p")) return "2K";
        return "FHD";
    }

    function parse(pageContext) {
        const { title = "", price = null, raw_text = "", key_values = {} } = pageContext;
        const scope = `${title} ${key_values.brand || ""} ${key_values.model || ""} ${raw_text}`;

        const subCat = extractSubCategory(title, scope);
        const brand = extractBrand(scope, key_values);
        let model = key_values.model || title.replace(new RegExp(`\\b${brand}\\b`, 'gi'), '').replace(/for sale.*$/i, '').trim() || "Standard Model";

        const missingFields = [];
        if (!brand || brand === "Generic") missingFields.push("Brand");
        if (!price || isNaN(price) || price <= 0) missingFields.push("Listing Price");

        const parsedData = {
            category: subCat,
            brand: brand,
            model: model,
            listed_price: price || null
        };

        if (subCat === "monitor") {
            parsedData.size = extractSize(scope, key_values);
            parsedData.refresh_rate = extractHz(scope, key_values);
            parsedData.resolution = extractResolution(scope, key_values);
        } else {
            const ram = extractRam(scope, key_values);
            const storage = extractStorage(scope, key_values);
            parsedData.ram = ram;
            parsedData.ram_gb = ram;
            parsedData.storage = storage;
            parsedData.storage_gb = storage;
        }

        return {
            category: "electronics",
            valid: missingFields.length === 0,
            missing_fields: missingFields,
            error_message: missingFields.length > 0 ? `Missing required electronics details: [${missingFields.join(", ")}].` : null,
            data: parsedData
        };
    }

    return { parse: parse };
})();
