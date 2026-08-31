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

    function extractBrand(combinedText, keyValues, title) {
        const titleUpper = (title || "").toUpperCase();
        
        // 1. Search title for series keywords first (highest accuracy)
        if (titleUpper.includes("THINKPAD") || titleUpper.includes("IDEAPAD") || titleUpper.includes("LEGION") || titleUpper.includes("THINKBOOK")) return "LENOVO";
        if (titleUpper.includes("MACBOOK")) return "APPLE";
        if (titleUpper.includes("VIVOBOOK") || titleUpper.includes("ZENBOOK") || titleUpper.includes("ROG") || titleUpper.includes("TUF")) return "ASUS";
        if (titleUpper.includes("LATITUDE") || titleUpper.includes("INSPIRON") || titleUpper.includes("VOSTRO") || titleUpper.includes("PRECISION") || titleUpper.includes("XPS") || titleUpper.includes("ALIENWARE")) return "DELL";
        if (titleUpper.includes("ELITEBOOK") || titleUpper.includes("PROBOOK") || titleUpper.includes("PAVILION") || titleUpper.includes("ENVY") || titleUpper.includes("SPECTRE") || titleUpper.includes("OMEN") || titleUpper.includes("VICTUS")) return "HP";
        if (titleUpper.includes("ASPIRE") || titleUpper.includes("SWIFT") || titleUpper.includes("NITRO") || titleUpper.includes("PREDATOR")) return "ACER";
        
        // 2. Search title for brand names directly
        for (const b of ELEC_BRANDS) {
            if (new RegExp(`\\b${b}\\b`, 'i').test(titleUpper)) {
                return b;
            }
        }
        
        // 3. Fallback to combinedText series mapping
        const upper = (combinedText || "").toUpperCase();
        if (upper.includes("THINKPAD") || upper.includes("IDEAPAD") || upper.includes("LEGION") || upper.includes("THINKBOOK")) return "LENOVO";
        if (upper.includes("MACBOOK")) return "APPLE";
        if (upper.includes("VIVOBOOK") || upper.includes("ZENBOOK") || upper.includes("ROG") || upper.includes("TUF")) return "ASUS";
        if (upper.includes("LATITUDE") || upper.includes("INSPIRON") || upper.includes("VOSTRO") || upper.includes("PRECISION") || upper.includes("XPS") || upper.includes("ALIENWARE")) return "DELL";
        if (upper.includes("ELITEBOOK") || upper.includes("PROBOOK") || upper.includes("PAVILION") || upper.includes("ENVY") || upper.includes("SPECTRE") || upper.includes("OMEN") || upper.includes("VICTUS")) return "HP";
        if (upper.includes("ASPIRE") || upper.includes("SWIFT") || upper.includes("NITRO") || upper.includes("PREDATOR")) return "ACER";

        // 4. Fallback to keyValues brand dropdown
        if (keyValues && keyValues.brand) {
            const bUpper = keyValues.brand.toUpperCase().trim();
            for (const b of ELEC_BRANDS) {
                if (bUpper.includes(b)) return b;
            }
        }
        
        // 5. Fallback to combinedText search
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
                    const match = String(key_values[key]).match(/(\d+)/);
                    if (match) {
                        const val = parseInt(match[1], 10);
                        if ([2, 4, 8, 16, 32, 64].includes(val)) return val;
                    }
                }
            }
        }
        
        const cleanText = (combinedText || "");
        
        // Pattern 1: 16GB RAM or 16 RAM
        const match1 = cleanText.match(/(\d{1,2})\s*(?:GB)?\s*(?:ram|memory)/i);
        if (match1) {
            const val = parseInt(match1[1], 10);
            if ([2, 4, 8, 16, 32, 64].includes(val)) return val;
        }
        
        // Pattern 2: RAM: 16 GB or RAM 16GB
        const match2 = cleanText.match(/(?:ram|memory)[^\d]{0,8}(\d{1,2})\s*(?:gb)?/i);
        if (match2) {
            const val = parseInt(match2[1], 10);
            if ([2, 4, 8, 16, 32, 64].includes(val)) return val;
        }
        return null;
    }

    function extractStorage(combinedText, key_values) {
        if (key_values) {
            for (const key of ['storage', 'storage capacity', 'hard drive', 'storage capacity (gb)']) {
                if (key_values[key]) {
                    const match = String(key_values[key]).match(/(\d+)\s*(GB|TB)?/i);
                    if (match) {
                        let val = parseInt(match[1], 10);
                        const unit = match[2] ? match[2].toUpperCase() : '';
                        if (unit === 'TB' || val <= 4) val *= 1024;
                        if ([128, 256, 512, 1024, 2048].includes(val)) return val;
                    }
                }
            }
        }
        
        const cleanText = (combinedText || "");
        
        // Scan cleanText for Pattern 1: e.g., 512GB SSD or 512 GB
        const regex1 = /(\d{2,4})\s*(GB|TB)(?:\s*(?:ssd|hdd|nvme|storage|drive|disk))?/gi;
        let match;
        while ((match = regex1.exec(cleanText)) !== null) {
            let val = parseInt(match[1], 10);
            const unit = match[2].toUpperCase();
            if (unit === 'TB' || val <= 4) val *= 1024;
            
            // Exclude if followed by RAM/Memory keyword within next 15 chars
            const afterMatch = cleanText.substring(match.index + match[0].length, match.index + match[0].length + 15).toLowerCase();
            if (afterMatch.includes("ram") || afterMatch.includes("memory")) {
                continue;
            }
            
            if ([128, 256, 512, 1024, 2048].includes(val)) return val;
        }
        
        // Scan cleanText for Pattern 2: e.g., Storage: 512 GB or SSD: 256GB
        const regex2 = /(?:storage|ssd|hdd|nvme|disk|drive)[^\d]{0,10}(\d{2,4})\s*(?:gb|tb)?/gi;
        while ((match = regex2.exec(cleanText)) !== null) {
            let val = parseInt(match[1], 10);
            if (match[0].toUpperCase().includes("TB") || val <= 4) val *= 1024;
            if ([128, 256, 512, 1024, 2048].includes(val)) return val;
        }
        return null;
    }

    function extractSize(combinedText, key_values) {
        if (key_values) {
            for (const key of ['size', 'screen size', 'display size']) {
                if (key_values[key]) {
                    const match = String(key_values[key]).match(/(\d+)/);
                    if (match) return match[1] + " Inch";
                }
            }
        }
        const match = (combinedText || "").match(/(\d{2})\s*(?:inch|")/i);
        if (match) {
            return match[1] + " Inch";
        }
        return null;
    }

    function extractHz(combinedText, key_values) {
        if (key_values) {
            for (const key of ['refresh rate', 'frequency']) {
                if (key_values[key]) {
                    const match = String(key_values[key]).match(/(\d+)/);
                    if (match) return match[1] + "Hz";
                }
            }
        }
        const match = (combinedText || "").match(/(\d{2,3})\s*hz/i);
        if (match) {
            return match[1] + "Hz";
        }
        return null;
    }

    function extractResolution(combinedText, key_values) {
        if (key_values) {
            for (const key of ['resolution', 'display resolution']) {
                if (key_values[key]) {
                    const val = String(key_values[key]).toUpperCase();
                    if (val.includes("4K") || val.includes("3840")) return "4K";
                    if (val.includes("2K") || val.includes("2560")) return "2K";
                    if (val.includes("FHD") || val.includes("1080")) return "FHD";
                }
            }
        }
        const lower = (combinedText || "").toLowerCase();
        if (lower.includes("4k") || lower.includes("uhd") || lower.includes("2160p")) return "4K";
        if (lower.includes("2k") || lower.includes("qhd") || lower.includes("1440p")) return "2K";
        if (lower.includes("fhd") || lower.includes("1080p") || lower.includes("1920")) return "FHD";
        return null;
    }

    function extractCpu(combinedText) {
        const upper = (combinedText || "").toUpperCase();
        const cpuPatterns = {
            'I9': /\bI9\b|CORE I9/i,
            'I7': /\bI7\b|CORE I7/i,
            'I5': /\bI5\b|CORE I5/i,
            'I3': /\bI3\b|CORE I3/i,
            'RYZEN 9': /RYZEN 9/i,
            'RYZEN 7': /RYZEN 7/i,
            'RYZEN 5': /RYZEN 5/i,
            'RYZEN 3': /RYZEN 3/i,
            'M1': /\bM1\b/i,
            'M2': /\bM2\b/i,
            'M3': /\bM3\b/i,
            'CELERON': /CELERON/i,
            'PENTIUM': /PENTIUM/i,
            'QUAD CORE': /QUAD CORE|QUAD-CORE/i
        };
        for (const [cpu, regex] of Object.entries(cpuPatterns)) {
            if (regex.test(upper)) {
                return cpu;
            }
        }
        return "Other";
    }

    function extractGeneration(combinedText) {
        const upper = (combinedText || "").toUpperCase();
        const match = upper.match(/(\d+)(?:ST|ND|RD|TH)?\s*(?:GEN|GENERATION)/i);
        if (match) {
            return parseInt(match[1], 10);
        }
        const yearMatch = upper.match(/(20\d{2})/);
        if (yearMatch) {
            const year = parseInt(yearMatch[1], 10);
            if (year >= 2020) return 12;
            if (year >= 2018) return 8;
            if (year >= 2015) return 5;
        }
        return 0;
    }

    function extractStorageType(combinedText, key_values) {
        if (key_values) {
            for (const key of ['storage', 'storage capacity', 'hard drive', 'storage capacity (gb)']) {
                if (key_values[key]) {
                    const val = key_values[key].toUpperCase();
                    if (val.includes("SSD") || val.includes("NVME") || val.includes("M.2")) return "SSD";
                    if (val.includes("HDD")) return "HDD";
                }
            }
        }
        const upper = (combinedText || "").toUpperCase();
        if (upper.includes("SSD") || upper.includes("NVME") || upper.includes("M.2")) {
            return "SSD";
        }
        return "HDD";
    }

    function parse(pageContext) {
        const { title = "", price = null, raw_text = "", key_values = {}, description = "" } = pageContext;
        const scope = `${title} ${description} ${key_values.brand || ""} ${key_values.model || ""} ${raw_text}`;

        const subCat = extractSubCategory(title, scope);
        const brand = extractBrand(scope, key_values, title);
        let model = key_values.model || title.replace(new RegExp(`\\b${brand}\\b`, 'gi'), '').replace(/for sale.*$/i, '').trim() || "Standard Model";

        if (subCat === "laptop") {
            const laptopSeries = {
                'DELL': ['LATITUDE', 'INSPIRON', 'VOSTRO', 'PRECISION', 'XPS', 'ALIENWARE', 'G15', 'G3', 'G5', 'G7'],
                'HP': ['ELITEBOOK', 'PROBOOK', 'PAVILION', 'SPECTRE', 'ENVY', 'OMEN', 'VICTUS', 'ZBOOK', 'NOTEBOOK', 'ELITE'],
                'LENOVO': ['THINKPAD', 'IDEAPAD', 'LEGION', 'YOGA', 'V15', 'THINKBOOK', 'T470', 'T480', 'T490', 'X1'],
                'ASUS': ['VIVOBOOK', 'ZENBOOK', 'ROG', 'TUF', 'EXPERTBOOK'],
                'ACER': ['ASPIRE', 'SWIFT', 'NITRO', 'PREDATOR', 'TRAVELMATE', 'SPIN'],
                'APPLE': ['MACBOOK PRO', 'MACBOOK AIR', 'MACBOOK'],
                'MSI': ['MODERN', 'STEALTH', 'KATANA', 'SWORD', 'CYBORG', 'GAMING']
            };
            const upperScope = scope.toUpperCase();
            let matchedSeries = "";
            
            const cleanBrandName = (brand || "").toUpperCase();
            if (laptopSeries[cleanBrandName]) {
                for (const ser of laptopSeries[cleanBrandName]) {
                    if (upperScope.includes(ser)) {
                        matchedSeries = ser;
                        break;
                    }
                }
            }
            
            if (!matchedSeries) {
                for (const brandKey in laptopSeries) {
                    for (const ser of laptopSeries[brandKey]) {
                        if (upperScope.includes(ser)) {
                            matchedSeries = ser;
                            break;
                        }
                    }
                    if (matchedSeries) break;
                }
            }
            
            if (matchedSeries) {
                model = matchedSeries.charAt(0) + matchedSeries.slice(1).toLowerCase();
                const lModel = model.toLowerCase();
                if (lModel === "macbook pro") model = "MacBook Pro";
                else if (lModel === "macbook air") model = "MacBook Air";
                else if (lModel === "expertbook") model = "ExpertBook";
                else if (lModel === "thinkbook") model = "ThinkBook";
                else if (lModel === "elitebook") model = "EliteBook";
                else if (lModel === "probook") model = "ProBook";
                else if (lModel === "ideapad") model = "IdeaPad";
                else if (lModel === "vivobook") model = "VivoBook";
                else if (lModel === "zenbook") model = "ZenBook";
            }
        } else if (subCat === "tablet") {
            const tabletSeries = {
                'APPLE': ['IPAD PRO', 'IPAD AIR', 'IPAD MINI', 'IPAD'],
                'SAMSUNG': ['GALAXY TAB S', 'GALAXY TAB A', 'GALAXY TAB'],
                'HUAWEI': ['MATEPAD', 'MEDIAPAD'],
                'XIAOMI': ['REDMI PAD', 'XIAOMI PAD', 'MI PAD'],
                'MICROSOFT': ['SURFACE PRO', 'SURFACE GO', 'SURFACE'],
                'AMAZON': ['KINDLE', 'FIRE HD', 'FIRE'],
                'REALME': ['REALME PAD']
            };
            const upperScope = scope.toUpperCase();
            let matchedSeries = "";
            
            const cleanBrandName = (brand || "").toUpperCase();
            if (tabletSeries[cleanBrandName]) {
                for (const ser of tabletSeries[cleanBrandName]) {
                    if (upperScope.includes(ser)) {
                        matchedSeries = ser;
                        break;
                    }
                }
            }
            
            if (!matchedSeries) {
                for (const brandKey in tabletSeries) {
                    for (const ser of tabletSeries[brandKey]) {
                        if (upperScope.includes(ser)) {
                            matchedSeries = ser;
                            break;
                        }
                    }
                    if (matchedSeries) break;
                }
            }
            
            if (matchedSeries) {
                if (matchedSeries === "IPAD PRO") model = "iPad Pro";
                else if (matchedSeries === "IPAD AIR") model = "iPad Air";
                else if (matchedSeries === "IPAD MINI") model = "iPad Mini";
                else if (matchedSeries === "IPAD") model = "iPad";
                else if (matchedSeries === "GALAXY TAB S") model = "Galaxy Tab S";
                else if (matchedSeries === "GALAXY TAB A") model = "Galaxy Tab A";
                else if (matchedSeries === "GALAXY TAB") model = "Galaxy Tab";
                else if (matchedSeries === "SURFACE PRO") model = "Surface Pro";
                else if (matchedSeries === "SURFACE GO") model = "Surface Go";
                else if (matchedSeries === "SURFACE") model = "Surface";
                else if (matchedSeries === "REDMI PAD") model = "Redmi Pad";
                else if (matchedSeries === "XIAOMI PAD") model = "Xiaomi Pad";
                else if (matchedSeries === "MI PAD") model = "Mi Pad";
                else if (matchedSeries === "MATEPAD") model = "MatePad";
                else if (matchedSeries === "MEDIAPAD") model = "MediaPad";
                else if (matchedSeries === "REALME PAD") model = "Realme Pad";
                else if (matchedSeries === "FIRE HD") model = "Fire HD";
                else {
                    model = matchedSeries.charAt(0) + matchedSeries.slice(1).toLowerCase();
                }
            }
        }

        const missingFields = [];
        if (!brand || brand === "Generic") missingFields.push("Brand");
        if (!model) missingFields.push("Model");
        if (!price || isNaN(price) || price <= 0) missingFields.push("Listing Price");

        const parsedData = {
            category: subCat,
            brand: brand || "",
            model: model || "",
            listed_price: price || null
        };

        if (subCat === "monitor") {
            const size = extractSize(scope, key_values);
            const hz = extractHz(scope, key_values);
            const res = extractResolution(scope, key_values);
            if (!size) missingFields.push("Screen Size");
            parsedData.size = size || "24 Inch";
            parsedData.refresh_rate = hz || "60Hz";
            parsedData.resolution = res || "FHD";
        } else {
            const ram = extractRam(scope, key_values);
            const storage = extractStorage(scope, key_values);
            parsedData.ram = ram;
            parsedData.ram_gb = ram;
            parsedData.storage = storage;
            parsedData.storage_gb = storage;
            
            if (subCat === "laptop") {
                parsedData.cpu = extractCpu(scope);
                parsedData.generation = extractGeneration(scope);
                parsedData.storageType = extractStorageType(scope, key_values);
                parsedData.storage_type = parsedData.storageType;
            }
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
