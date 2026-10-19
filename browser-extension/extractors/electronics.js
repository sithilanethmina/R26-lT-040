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

    function extractSubCategory(title, combinedText, keyValues = {}) {
        const titleLower = (title || "").toLowerCase();
        // Strip generic marketplace category headers like "computers & tablets" from text before checking subcategory
        const lower = (combinedText || "").toLowerCase().replace(/computers\s*(?:&|and)\s*tablets/gi, '');
        const itemType = (keyValues && (keyValues.item_type || keyValues["item type"]) ? String(keyValues.item_type || keyValues["item type"]).toLowerCase() : "");
        
        // 0. Check explicit desktop PC patterns (Not supported by mobile/laptop models)
        const isDesktopPattern = /\b(desktop pc|gaming pc|desktop computer|desktop computers|full set pc|full pc|cpu unit|gaming rig|all-in-one pc|all in one pc|aio pc|custom pc|workstation pc|pc tower|gaming desktop)\b/i.test(titleLower) || /\b(desktop pc|gaming pc|full set pc|gaming rig|cpu unit|aio pc)\b/i.test(lower) || itemType === "desktop" || itemType === "desktop computers";
        const isLaptopKeyword = titleLower.includes("laptop") || titleLower.includes("notebook") || titleLower.includes("macbook") || titleLower.includes("thinkpad") || titleLower.includes("elitebook") || titleLower.includes("probook") || titleLower.includes("zenbook") || titleLower.includes("vivobook") || titleLower.includes("latitude") || titleLower.includes("inspiron") || titleLower.includes("vostro") || titleLower.includes("precision") || titleLower.includes("tuf") || titleLower.includes("rog");
        const isTabletKeyword = /\b(tablet|tablets|ipad|galaxy\s*tab|tab|tabs|mediapad|matepad|mi\s*pad)\b/i.test(titleLower);
        const isMonitorKeyword = titleLower.includes("monitor") || titleLower.includes("monitors") || titleLower.includes("frameless display");

        if (isDesktopPattern && !isLaptopKeyword && !isTabletKeyword && !isMonitorKeyword) {
            return "unsupported_desktop";
        }

        // 1. Check title first (highest confidence)
        if (titleLower.includes("monitor") || titleLower.includes("monitors")) {
            return "monitor";
        }
        if (/\b(tablet|tablets|ipad|galaxy\s*tab|tab|tabs|matepad|mediapad|mi\s*pad|surface\s*pro|surface\s*go)\b/i.test(titleLower)) {
            return "tablet";
        }
        if (isLaptopKeyword) {
            return "laptop";
        }
        
        // 2. Check full text (lower confidence, order of specificity)
        if (lower.includes("monitor") || lower.includes("monitors")) {
            return "monitor";
        }
        if (lower.includes("ipad") || lower.includes("matepad") || lower.includes("mediapad") || lower.includes("galaxy tab") || lower.includes("mi pad") || /\btablet\b/i.test(lower)) {
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
        
        if (isDesktopPattern) {
            return "unsupported_desktop";
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
        // Match explicit generation string: e.g. "11th Gen", "10th Generation"
        const match = upper.match(/\b(\d{1,2})(?:ST|ND|RD|TH)?\s*(?:GEN|GENERATION)\b/i);
        if (match) {
            const val = parseInt(match[1], 10);
            if (val >= 1 && val <= 14) return val;
        }
        // Match processor part numbers like i5-1135G7 (11), i7-8550U (8), i5-12500H (12), i7-7700HQ (7)
        const procMatch = upper.match(/\b(?:CORE\s*)?I[3579][ -](\d{1,2})\d{2,3}[A-Z0-9]*\b/i);
        if (procMatch) {
            const val = parseInt(procMatch[1], 10);
            if (val >= 2 && val <= 14) return val;
        }
        // Match Ryzen series like Ryzen 5 5600H (5), Ryzen 7 4800H (4)
        const ryzenMatch = upper.match(/\bRYZEN\s*[3579]\s*(\d)\d{3}\b/i);
        if (ryzenMatch) {
            return parseInt(ryzenMatch[1], 10);
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
        if (upper.includes("SSD") || upper.includes("NVME") || upper.includes("M.2") || upper.includes("EMMC")) {
            return "SSD";
        }
        return "HDD";
    }

    function parse(pageContext) {
        const { title = "", price = null, raw_text = "", key_values = {}, description = "" } = pageContext;
        const scope = `${title} ${description} ${key_values.brand || ""} ${key_values.model || ""} ${raw_text}`;

        const subCat = extractSubCategory(title, scope, key_values);
        const brand = extractBrand(scope, key_values, title);
        let model = key_values.model || title.replace(new RegExp(`\\b${brand}\\b`, 'gi'), '').replace(/for sale.*$/i, '').trim() || "Standard Model";

        if (subCat === "laptop") {
            const laptopSeries = {
                'DELL': ['CHROMEBOOK', 'CROMEBOOK', 'LATITUDE', 'INSPIRON', 'VOSTRO', 'PRECISION', 'XPS', 'ALIENWARE', 'G15', 'G3', 'G5', 'G7'],
                'HP': ['CHROMEBOOK', 'CROMEBOOK', 'ELITEBOOK', 'PROBOOK', 'PAVILION', 'SPECTRE', 'ENVY', 'OMEN', 'VICTUS', 'ZBOOK', 'NOTEBOOK', 'ELITE'],
                'LENOVO': ['CHROMEBOOK', 'CROMEBOOK', 'THINKPAD', 'IDEAPAD', 'LEGION', 'YOGA', 'V15', 'THINKBOOK', 'T470', 'T480', 'T490', 'X1'],
                'ASUS': ['CHROMEBOOK', 'CROMEBOOK', 'VIVOBOOK', 'ZENBOOK', 'ROG', 'TUF', 'EXPERTBOOK'],
                'ACER': ['CHROMEBOOK', 'CROMEBOOK', 'ASPIRE', 'SWIFT', 'NITRO', 'PREDATOR', 'TRAVELMATE', 'SPIN', 'EXTENSA'],
                'APPLE': ['MACBOOK PRO', 'MACBOOK AIR', 'MACBOOK'],
                'MSI': ['MODERN', 'STEALTH', 'KATANA', 'SWORD', 'CYBORG', 'GAMING'],
                'SAMSUNG': ['CHROMEBOOK', 'CROMEBOOK', 'GALAXY BOOK']
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
                if (lModel.includes("chromebook") || lModel.includes("cromebook")) model = "Chromebook";
                else if (lModel === "macbook pro") model = "MacBook Pro";
                else if (lModel === "macbook air") model = "MacBook Air";
                else if (lModel === "expertbook") model = "ExpertBook";
                else if (lModel === "thinkbook") model = "ThinkBook";
                else if (lModel === "elitebook") model = "EliteBook";
                else if (lModel === "probook") model = "ProBook";
                else if (lModel === "ideapad") model = "IdeaPad";
                else if (lModel === "vivobook") model = "VivoBook";
                else if (lModel === "zenbook") model = "ZenBook";
            } else if (upperScope.includes("CHROMEBOOK") || upperScope.includes("CROMEBOOK")) {
                model = "Chromebook";
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

        // Targeted Ikman Ad Extraction Helpers (Isolated strictly to Electronics)
        function expandShowMoreSections() {
            try {
                const candidates = document.querySelectorAll('button, [role="button"], a, span');
                candidates.forEach(el => {
                    const txt = (el.innerText || el.textContent || '').trim().toLowerCase();
                    if (txt === 'show more' || txt.startsWith('show more') || txt === 'read more' || txt.startsWith('read more') || txt === 'තවත් පෙන්වන්න' || txt === 'மேலும் காட்டு') {
                        try { el.click(); } catch (e) {}
                    }
                });
                document.querySelectorAll('[class*="show-more"], [class*="showMore"], [data-testid*="show-more"]').forEach(el => {
                    try { el.click(); } catch (e) {}
                });
            } catch (e) {}
        }

        function extractIkmanLocationSubtitle(h1) {
            if (!h1) return "";
            let next = h1.nextElementSibling;
            while (next) {
                const txt = (next.innerText || "").trim();
                if (txt && (txt.includes("Posted on") || txt.includes("am,") || txt.includes("pm,"))) {
                    return txt;
                }
                next = next.nextElementSibling;
            }
            if (h1.parentElement) {
                const sub = h1.parentElement.querySelector('[class*="sub-title"], [class*="subtitle"], [data-testid*="subtitle"], p, span');
                if (sub && sub !== h1) {
                    const txt = (sub.innerText || "").trim();
                    if (txt && (txt.includes("Posted on") || txt.includes(","))) return txt;
                }
            }
            const allSubs = document.querySelectorAll('p, div, span');
            for (const el of allSubs) {
                const txt = (el.innerText || "").trim();
                if (txt.startsWith("Posted on ") && txt.length < 150) {
                    return txt;
                }
            }
            return "";
        }

        function parseLocationFromSubtitle(subtitle) {
            if (!subtitle) return "";
            const timeMatch = subtitle.match(/(?:am|pm|\d{2}:\d{2}),\s*([^,\n]+(?:,\s*[^,\n]+)?)/i);
            if (timeMatch && timeMatch[1]) {
                return timeMatch[1].trim();
            }
            const commaIdx = subtitle.lastIndexOf(",");
            if (commaIdx > 0 && commaIdx < subtitle.length - 1) {
                return subtitle.substring(subtitle.indexOf(",") + 1).trim();
            }
            return "";
        }

        const isIkman = typeof window !== "undefined" && window.location && window.location.hostname.includes("ikman.lk");
        let ikmanLocation = "";
        let ikmanLocationSubtitle = "";
        let ikmanDescription = description || "";
        let targetedRawText = raw_text || "";

        if (isIkman) {
            expandShowMoreSections();
            const h1El = document.querySelector('h1');
            if (h1El) {
                ikmanLocationSubtitle = extractIkmanLocationSubtitle(h1El);
                ikmanLocation = parseLocationFromSubtitle(ikmanLocationSubtitle);
            }
            const descEl = document.querySelector('[class*="description--"], [class*="description"], div[data-testid="ad-description"], .ad-description');
            if (descEl) {
                const dt = (descEl.innerText || "").trim();
                if (dt && dt.length > ikmanDescription.length) {
                    ikmanDescription = dt;
                }
            }
            targetedRawText = `Listing Title: ${title}\n` +
                (ikmanLocationSubtitle ? `Location Subtitle: ${ikmanLocationSubtitle}\n` : '') +
                (ikmanLocation ? `Detected Location: ${ikmanLocation}\n` : '') +
                (price ? `Asking Price: Rs. ${price}\n` : '') +
                `Ikman Defined Fields:\n` +
                Object.entries(key_values).map(([k, v]) => `  • ${k}: ${v}`).join('\n') +
                (ikmanDescription ? `\nSeller Description:\n${ikmanDescription}\n` : '');
        }

        const missingFields = [];
        if (!brand || brand === "Generic") missingFields.push("Brand");
        if (!model) missingFields.push("Model");
        if (!price || isNaN(price) || price <= 0) missingFields.push("Listing Price");

        const rawCond = (key_values.condition || "").toLowerCase();
        let isNew = false;
        // Priority 1: Ikman Defined Field Ground Truth (Used always wins over description claims)
        if (rawCond.includes("used") || rawCond.includes("second") || rawCond.includes("2nd") || rawCond.includes("refurbish") || rawCond.includes("recondition")) {
            isNew = false;
        } else if (rawCond.includes("brand new") || rawCond.includes("brand-new") || rawCond === "new" || rawCond.includes("sealed")) {
            isNew = true;
        } else if (!rawCond && (title.toLowerCase().includes("brand new") || title.toLowerCase().includes("sealed pack")) && !title.toLowerCase().includes("used")) {
            isNew = true;
        }
        const cleanCondition = isNew ? "Brand New" : "Used";

        if (subCat === "unsupported_desktop") {
            return {
                category: "unsupported",
                valid: false,
                missing_fields: [],
                error_message: "Desktop Computers, PC Towers, and Custom Rigs are currently not supported for automated valuation. FairPriceLK evaluates Laptops, Tablets, and Monitors.",
                data: {
                    title: title,
                    item_type: "Desktop Computer",
                    listed_price: price || null,
                    condition: cleanCondition,
                    location: ikmanLocation || key_values.location || ""
                }
            };
        }

        const domTree = {
            platform: isIkman ? "ikman" : "generic",
            title: title || "",
            location: ikmanLocation || key_values.location || "",
            location_subtitle: ikmanLocationSubtitle || "",
            price: price || null,
            brand: brand || "",
            model: model || "",
            condition: cleanCondition,
            breadcrumbs: pageContext.breadcrumbs || [],
            specification_table: key_values,
            description: ikmanDescription || description || "",
            url: typeof window !== "undefined" && window.location ? window.location.href : ""
        };

        const parsedData = {
            category: subCat,
            subcategory: subCat,
            brand: brand || "",
            model: model || "",
            title: title || "",
            description: ikmanDescription || description || "",
            raw_text: targetedRawText || raw_text || "",
            listed_price: price || null,
            condition: cleanCondition,
            location: ikmanLocation || key_values.location || "",
            dom_tree: domTree
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
            data: parsedData,
            dom_tree: domTree
        };
    }

    return { parse: parse };
})();
