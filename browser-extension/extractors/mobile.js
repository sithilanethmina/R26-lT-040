/**
 * FairPriceLK - Mobile Phone Category Extractor
 * Provides high-precision parsing of mobile phone listings from marketplace pages (Ikman, etc.)
 * Scrapes listing DOM for specs and enriches with known phone specifications
 * for feature-engineered fields required by the ML model.
 */

window.FairPriceLK_Extractors = window.FairPriceLK_Extractors || {};

window.FairPriceLK_Extractors.mobile = (function () {

    // ── Known Brands & Canonical Names ──────────────────────────────────────
    const CANONICAL_BRANDS = {
        "APPLE": "Apple", "SAMSUNG": "Samsung", "XIAOMI": "Xiaomi",
        "REDMI": "Xiaomi", "POCO": "Xiaomi", "ONEPLUS": "OnePlus",
        "GOOGLE": "Google", "PIXEL": "Google", "HUAWEI": "Huawei",
        "VIVO": "Vivo", "OPPO": "Oppo", "REALME": "Realme",
        "SONY": "Sony", "NOKIA": "Nokia", "MOTOROLA": "Motorola",
        "MOTO": "Motorola", "ASUS": "Asus", "HONOR": "Honor",
        "NOTHING": "Nothing", "INFINIX": "Infinix", "TECNO": "Tecno",
        "ITEL": "Itel", "LG": "LG", "HTC": "HTC", "LENOVO": "Lenovo",
        "ZTE": "ZTE"
    };
    const PHONE_BRANDS = Object.keys(CANONICAL_BRANDS);

    // ── Brand tier mapping (3=premium, 2=mid, 1=budget) ──────────────────────
    const BRAND_TIER_MAP = {
        "APPLE": 3, "SAMSUNG": 3, "GOOGLE": 3, "SONY": 3,
        "ONEPLUS": 2, "OPPO": 2, "VIVO": 2, "XIAOMI": 2,
        "REALME": 2, "HUAWEI": 2, "HONOR": 2, "NOTHING": 2,
        "MOTOROLA": 2, "NOKIA": 2, "ASUS": 2, "LG": 2, "HTC": 2,
        "REDMI": 2, "POCO": 2,
        "INFINIX": 1, "TECNO": 1, "ITEL": 1
    };

    // ── iPhone RAM lookup ────────────────────────────────────────────────────
    const IPHONE_RAM_GB = {
        "iphone 6": 1, "iphone 6 plus": 1, "iphone 6s": 2, "iphone 6s plus": 2,
        "iphone 7": 2, "iphone 7 plus": 3, "iphone 8": 2, "iphone 8 plus": 3,
        "iphone x": 3, "iphone xr": 3, "iphone xs": 4, "iphone xs max": 4,
        "iphone 11": 4, "iphone 11 pro": 4, "iphone 11 pro max": 4,
        "iphone se": 2, "iphone se 2": 3, "iphone se 3": 4,
        "iphone 12": 4, "iphone 12 mini": 4, "iphone 12 pro": 6, "iphone 12 pro max": 6,
        "iphone 13": 4, "iphone 13 mini": 4, "iphone 13 pro": 6, "iphone 13 pro max": 6,
        "iphone 14": 6, "iphone 14 plus": 6, "iphone 14 pro": 6, "iphone 14 pro max": 6,
        "iphone 15": 6, "iphone 15 plus": 6, "iphone 15 pro": 8, "iphone 15 pro max": 8,
        "iphone 16": 8, "iphone 16 plus": 8, "iphone 16 pro": 8, "iphone 16 pro max": 8,
        "iphone 16e": 8,
        "iphone 17": 8, "iphone 17 air": 12, "iphone air": 12,
        "iphone 17 pro": 12, "iphone 17 pro max": 12, "iphone 17e": 8
    };

    // ── iPhone release years (for phone_age_years) ───────────────────────────
    const IPHONE_RELEASE_YEAR = {
        "iphone 6": 2014, "iphone 6 plus": 2014, "iphone 6s": 2015, "iphone 6s plus": 2015,
        "iphone 7": 2016, "iphone 7 plus": 2016, "iphone 8": 2017, "iphone 8 plus": 2017,
        "iphone x": 2017, "iphone xr": 2018, "iphone xs": 2018, "iphone xs max": 2018,
        "iphone 11": 2019, "iphone 11 pro": 2019, "iphone 11 pro max": 2019,
        "iphone se": 2016, "iphone se 2": 2020, "iphone se 3": 2022,
        "iphone 12": 2020, "iphone 12 mini": 2020, "iphone 12 pro": 2020, "iphone 12 pro max": 2020,
        "iphone 13": 2021, "iphone 13 mini": 2021, "iphone 13 pro": 2021, "iphone 13 pro max": 2021,
        "iphone 14": 2022, "iphone 14 plus": 2022, "iphone 14 pro": 2022, "iphone 14 pro max": 2022,
        "iphone 15": 2023, "iphone 15 plus": 2023, "iphone 15 pro": 2023, "iphone 15 pro max": 2023,
        "iphone 16": 2024, "iphone 16 plus": 2024, "iphone 16 pro": 2024, "iphone 16 pro max": 2024,
        "iphone 16e": 2025,
        "iphone 17": 2025, "iphone 17 air": 2025, "iphone air": 2025,
        "iphone 17 pro": 2025, "iphone 17 pro max": 2025, "iphone 17e": 2025
    };

    // ── iPhone model tier (1–10 scale) ───────────────────────────────────────
    const IPHONE_MODEL_TIER = {
        "iphone 6": 2, "iphone 6 plus": 2, "iphone 6s": 3, "iphone 6s plus": 3,
        "iphone 7": 4, "iphone 7 plus": 4, "iphone 8": 5, "iphone 8 plus": 5,
        "iphone x": 6, "iphone xr": 5, "iphone xs": 6, "iphone xs max": 7,
        "iphone 11": 6, "iphone 11 pro": 7, "iphone 11 pro max": 8,
        "iphone se": 3, "iphone se 2": 4, "iphone se 3": 5,
        "iphone 12": 6, "iphone 12 mini": 6, "iphone 12 pro": 8, "iphone 12 pro max": 9,
        "iphone 13": 7, "iphone 13 mini": 7, "iphone 13 pro": 8, "iphone 13 pro max": 9,
        "iphone 14": 7, "iphone 14 plus": 7, "iphone 14 pro": 9, "iphone 14 pro max": 10,
        "iphone 15": 8, "iphone 15 plus": 8, "iphone 15 pro": 9, "iphone 15 pro max": 10,
        "iphone 16": 8, "iphone 16 plus": 8, "iphone 16 pro": 10, "iphone 16 pro max": 10,
        "iphone 16e": 7,
        "iphone 17": 9, "iphone 17 air": 9, "iphone air": 9,
        "iphone 17 pro": 10, "iphone 17 pro max": 10, "iphone 17e": 8
    };

    // ── Android model tier patterns → (regex, tier, isFlagship) ──────────────
    const ANDROID_TIER_PATTERNS = [
        // Samsung flagships
        [/galaxy\s*s2[3456]\s*ultra/i, 10, true],
        [/galaxy\s*s2[3456]\s*(?:plus|\+)/i, 9, true],
        [/galaxy\s*s2[3456](?!\s*ultra|\s*plus|\s*\+)/i, 8, true],
        [/galaxy\s*s2[012]\s*ultra/i, 9, true],
        [/galaxy\s*s2[012]\s*(?:plus|\+|fe)/i, 8, true],
        [/galaxy\s*s2[012](?!\s*ultra|\s*plus|\s*\+|\s*fe)/i, 7, true],
        [/galaxy\s*z\s*(?:fold|flip)/i, 9, true],
        [/galaxy\s*note\s*20\s*ultra/i, 9, true],
        [/galaxy\s*note\s*20/i, 8, true],
        // Samsung mid-range
        [/galaxy\s*a[5-9]\d/i, 5, false],
        [/galaxy\s*a[1-4]\d/i, 3, false],
        [/galaxy\s*m\d/i, 3, false],
        [/galaxy\s*f\d/i, 2, false],
        // OnePlus
        [/oneplus\s*1[2-5](?!\s*r)/i, 8, true],
        [/oneplus\s*1[01](?!\s*r)/i, 7, true],
        [/oneplus\s*(?:nord|1\dr)/i, 5, false],
        // Google Pixel
        [/pixel\s*[89]\s*pro/i, 9, true],
        [/pixel\s*[89]/i, 7, true],
        [/pixel\s*[67]\s*pro/i, 8, true],
        [/pixel\s*[67]a?/i, 6, false],
        // Xiaomi flagships
        [/mi\s*1[1-4]\s*ultra/i, 9, true],
        [/poco\s*f[5-9]/i, 7, true],
        [/poco\s*[xm]\d/i, 4, false],
        [/redmi\s*note\s*1[3-5]\s*pro/i, 5, false],
        [/redmi\s*note\s*\d/i, 4, false],
        [/redmi\s*1[2-9]c?/i, 3, false],
        [/redmi\s*a\d/i, 2, false],
        // Oppo/Vivo/Realme
        [/reno\s*1[0-2]\s*pro/i, 7, true],
        [/reno\s*\d/i, 5, false],
        [/realme\s*gt/i, 7, true],
        [/realme\s*\d+\s*(?:pro)?/i, 4, false],
        [/iqoo\s*neo/i, 6, true],
        [/iqoo\s*\d/i, 6, true],
        // Vivo tiers
        [/vivo\s*x\d+\s*(?:pro|ultra)/i, 8, true],
        [/vivo\s*x\d+/i, 7, true],
        [/vivo\s*v\d+\s*(?:pro|\+|plus)/i, 5, false],
        [/vivo\s*v\d+/i, 4, false],
        [/vivo\s*s\d+/i, 4, false],
        [/vivo\s*t\d+/i, 3, false],
        [/vivo\s*y\d+/i, 2, false],
        // Oppo budget
        [/oppo\s*a\d+/i, 3, false],
        [/oppo\s*f\d+/i, 4, false],
        // Nokia / Motorola
        [/nokia\s*[gx]\d+/i, 4, false],
        [/nokia\s*c\d+/i, 2, false],
        [/moto\s*g\d+/i, 4, false],
        [/moto\s*e\d+/i, 2, false],
        // Budget brands
        [/(?:infinix|tecno|itel)/i, 2, false],
    ];

    // ── iPhone-specific 5G/eSIM/dual-SIM defaults ────────────────────────────
    const IPHONE_5G_SUPPORT = {
        "iphone 12": true, "iphone 12 mini": true, "iphone 12 pro": true, "iphone 12 pro max": true,
        "iphone 13": true, "iphone 13 mini": true, "iphone 13 pro": true, "iphone 13 pro max": true,
        "iphone 14": true, "iphone 14 plus": true, "iphone 14 pro": true, "iphone 14 pro max": true,
        "iphone 15": true, "iphone 15 plus": true, "iphone 15 pro": true, "iphone 15 pro max": true,
        "iphone 16": true, "iphone 16 plus": true, "iphone 16 pro": true, "iphone 16 pro max": true,
        "iphone se 3": true, "iphone 16e": true,
        "iphone 17": true, "iphone 17 air": true, "iphone air": true,
        "iphone 17 pro": true, "iphone 17 pro max": true, "iphone 17e": true
    };
    const IPHONE_ESIM_SUPPORT = {
        "iphone xs": true, "iphone xs max": true, "iphone xr": true,
        "iphone 11": true, "iphone 11 pro": true, "iphone 11 pro max": true,
        "iphone se 2": true, "iphone se 3": true,
        "iphone 12": true, "iphone 12 mini": true, "iphone 12 pro": true, "iphone 12 pro max": true,
        "iphone 13": true, "iphone 13 mini": true, "iphone 13 pro": true, "iphone 13 pro max": true,
        "iphone 14": true, "iphone 14 plus": true, "iphone 14 pro": true, "iphone 14 pro max": true,
        "iphone 15": true, "iphone 15 plus": true, "iphone 15 pro": true, "iphone 15 pro max": true,
        "iphone 16": true, "iphone 16 plus": true, "iphone 16 pro": true, "iphone 16 pro max": true,
        "iphone 16e": true,
        "iphone 17": true, "iphone 17 air": true, "iphone air": true,
        "iphone 17 pro": true, "iphone 17 pro max": true, "iphone 17e": true
    };

    // ── DOM Scraping Helpers ─────────────────────────────────────────────────

    /**
     * Scrape ikman.lk or similar marketplace listing DOM for mobile phone specs.
     * Reads key-value pairs from spec tables, dl/dt/dd, divs, spans etc.
     */
    function scrapeListingDOM() {
        const result = {
            title: null,
            price: null,
            brand: null,
            model: null,
            condition: null,
            storage: null,
            ram: null,
            warranty: null,
            description: null,
            has_5g: null,
            dual_sim: null,
            has_esim: null,
            operating_system: null,
            network: null,
            battery_health: null
        };

        try {
            // 1. Title
            const h1 = document.querySelector('h1');
            if (h1) {
                result.title = h1.innerText.trim();
            }

            // 2. Price — find element containing "Rs." with ≥4 digits
            const leafEls = Array.from(document.querySelectorAll('div, span, p, td, strong, b, h2, h3'));
            for (const el of leafEls) {
                const directText = Array.from(el.childNodes)
                    .filter(n => n.nodeType === Node.TEXT_NODE)
                    .map(n => n.textContent)
                    .join('');
                const fullText = (el.innerText || el.textContent || '').trim();
                const targetText = directText.trim() || fullText;

                if ((targetText.includes('Rs.') || targetText.includes('Rs ') || targetText.includes('LKR')) && /\d{4,}/.test(targetText)) {
                    const match = targetText.match(/(?:Rs\.?|LKR)\s*([\d,]+)/i);
                    if (match) {
                        const num = parseInt(match[1].replace(/,/g, ''), 10);
                        if (num > 1000 && (!result.price || num > result.price)) {
                            result.price = num;
                            break;
                        }
                    }
                }
            }

            // 3. Spec table rows — match label → value pairs
            const tables = document.querySelectorAll('table');
            for (const table of tables) {
                const rows = table.querySelectorAll('tr');
                for (const row of rows) {
                    const cells = row.querySelectorAll('td, th');
                    if (cells.length < 2) continue;
                    const label = (cells[0].innerText || cells[0].textContent || '').trim().toUpperCase();
                    const value = (cells[1].innerText || cells[1].textContent || '').trim();
                    if (!label || !value) continue;

                    if (label === 'BRAND' || label === 'MAKE')              result.brand = value;
                    else if (label === 'MODEL')                              result.model = value;
                    else if (label === 'CONDITION')                          result.condition = value;
                    else if (label.includes('STORAGE') || label.includes('INTERNAL') || label.includes('ROM') || label === 'MEMORY')
                                                                             result.storage = value;
                    else if (label === 'RAM' || label.includes('RAM'))       result.ram = value;
                    else if (label.includes('WARRANTY'))                     result.warranty = value;
                    else if (label.includes('BATTERY') && (label.includes('HEALTH') || label.includes('CONDITION')))
                                                                             result.battery_health = value;
                    else if (label === 'BATTERY' || label === 'BH')          result.battery_health = value;
                    else if (label.includes('5G'))                           result.has_5g = value;
                    else if (label.includes('SIM'))                          result.dual_sim = value;
                    else if (label.includes('ESIM'))                         result.has_esim = value;
                    else if (label.includes('OPERATING') || label === 'OS')  result.operating_system = value;
                    else if (label === 'NETWORK' || label.includes('NETWORK')) result.network = value;
                }
            }

            // 4. dl/dt/dd or adjacent-span/div fallback for missing fields
            const labelEls = document.querySelectorAll('dt, th, .label, [class*="spec-label"], [class*="detail-label"]');
            labelEls.forEach(lEl => {
                const lText = (lEl.innerText || '').trim().toUpperCase();
                const vEl = lEl.nextElementSibling;
                if (!vEl) return;
                const vText = (vEl.innerText || '').trim();

                if (lText.includes('BRAND')    && !result.brand)     result.brand = vText;
                if (lText.includes('MODEL')    && !result.model)     result.model = vText;
                if (lText.includes('CONDITION')&& !result.condition) result.condition = vText;
                if (lText.includes('STORAGE')  && !result.storage)   result.storage = vText;
                if (lText.includes('RAM')      && !result.ram)       result.ram = vText;
                if (lText.includes('WARRANTY') && !result.warranty)  result.warranty = vText;
                if ((lText.includes('BATTERY') && (lText.includes('HEALTH') || lText.includes('CONDITION'))) && !result.battery_health)
                                                                    result.battery_health = vText;
                if ((lText === 'BATTERY' || lText === 'BH') && !result.battery_health)
                                                                    result.battery_health = vText;
            });

            // 5. Ikman-style adjacent div/span pairs
            const allTextEls = document.querySelectorAll('div, span, p, li');
            for (let i = 0; i < allTextEls.length; i++) {
                const el = allTextEls[i];
                const t = (el && el.innerText ? el.innerText : '').trim().toLowerCase();
                if (!t) continue;

                const nextEl = el.nextElementSibling;
                const nextText = nextEl && nextEl.innerText ? nextEl.innerText.trim() : '';

                if ((t === 'brand:' || t === 'brand') && nextText && !result.brand)     result.brand = nextText;
                if ((t === 'model:' || t === 'model') && nextText && !result.model)     result.model = nextText;
                if ((t === 'condition:' || t === 'condition') && nextText && !result.condition)
                                                                                         result.condition = nextText;
                if ((t === 'storage:' || t === 'internal storage:' || t === 'memory:' || t === 'memory') && nextText && !result.storage)
                                                                                         result.storage = nextText;
                if ((t === 'ram:' || t === 'ram') && nextText && !result.ram)             result.ram = nextText;
                if ((t === 'battery health:' || t === 'battery health' || t === 'battery condition:' || t === 'battery condition' || t === 'bh:' || t === 'bh') && nextText && !result.battery_health)
                                                                                         result.battery_health = nextText;
                if ((t === 'operating system:' || t === 'operating system' || t === 'os:' || t === 'os') && nextText && !result.operating_system)
                                                                                         result.operating_system = nextText;
                if ((t === 'network:' || t === 'network') && nextText && !result.network) result.network = nextText;
                if ((t === 'sim support:' || t === 'sim support' || t === 'sim:') && nextText && !result.dual_sim)
                                                                                         result.dual_sim = nextText;
            }

            // 6. Description
            const descBox = document.querySelector('.description') ||
                            document.querySelector('[class*="description"]') ||
                            document.querySelector('.morebox') ||
                            document.querySelector('.contentbox');
            if (descBox) {
                result.description = descBox.innerText.trim();
            }

        } catch (e) {
            console.warn('[FairPriceLK] Mobile DOM scrape error:', e);
        }

        return result;
    }

    // Known Android-only brands — if brand matches, it's DEFINITELY not an iPhone
    const ANDROID_ONLY_BRANDS = [
        "SAMSUNG", "XIAOMI", "REDMI", "POCO", "ONEPLUS", "GOOGLE",
        "HUAWEI", "VIVO", "OPPO", "REALME", "SONY", "NOKIA", "MOTOROLA",
        "ASUS", "HONOR", "NOTHING", "INFINIX", "TECNO", "ITEL", "LG", "HTC"
    ];

    // ── Field Extractors ─────────────────────────────────────────────────────

    /**
     * Determine phone type using structured data first (brand, OS field),
     * then fall back to TITLE-only keyword match.
     * NEVER uses full page raw_text (which may contain "iPhone" from sidebar ads).
     */
    function extractPhoneType(titleText, brand, scraped) {
        // Priority 1: Scraped "Operating System" field from listing specs
        if (scraped && scraped.operating_system) {
            const os = scraped.operating_system.toLowerCase();
            if (os.includes('android')) return 'android';
            if (os.includes('ios')) return 'iphone';
        }

        // Priority 2: Brand check — if it's a known Android-only brand, it's android
        if (brand) {
            const bUpper = brand.toUpperCase().trim();
            if (bUpper === 'APPLE') return 'iphone';
            for (const ab of ANDROID_ONLY_BRANDS) {
                if (bUpper.includes(ab)) return 'android';
            }
        }

        // Priority 3: Title-only check (NOT raw_text, to avoid sidebar ad contamination)
        const titleLower = (titleText || '').toLowerCase();
        if (titleLower.includes('iphone') || titleLower.includes('apple')) {
            return 'iphone';
        }

        // Default: android (most common in Sri Lankan market)
        return 'android';
    }

    function extractBrand(combinedText, scraped) {
        // Priority 1: scraped brand from DOM
        if (scraped && scraped.brand) {
            const bUpper = scraped.brand.toUpperCase().trim();
            for (const b of PHONE_BRANDS) {
                if (bUpper.includes(b)) return CANONICAL_BRANDS[b];
            }
            // Return the scraped brand capitalised even if not in our list
            return scraped.brand.trim();
        }

        // Priority 2: key_values brand
        const upper = (combinedText || "").toUpperCase();
        for (const b of PHONE_BRANDS) {
            if (new RegExp(`\\b${b}\\b`, 'i').test(upper)) {
                return CANONICAL_BRANDS[b];
            }
        }
        return null;
    }

    /**
     * Extract the specific phone model name from title + scraped data.
     * For iPhones, normalises to a canonical form like "iPhone 13 mini".
     * For Android, returns the scraped/title model string.
     */
    function extractModel(combinedText, scraped) {
        let model = null;

        // Priority 1: DOM scraped model field
        if (scraped && scraped.model && scraped.model.trim()) {
            model = scraped.model.trim();
        }

        // Priority 2: Try to extract from title/text
        if (!model) {
            // iPhone pattern: iPhone 13 mini, iPhone 14 Pro Max, iPhone SE 3, etc.
            const iPhoneMatch = (combinedText || "").match(
                /\b(iPhone\s*(?:SE\s*\d?|(?:1[0-7]|X[RSs]?)\s*(?:mini|Plus|Pro\s*Max|Pro)?(?:\s*Max)?))\b/i
            );
            if (iPhoneMatch) {
                model = iPhoneMatch[1].trim();
            }
        }

        if (!model) {
            // Android: try common patterns — Galaxy S24 Ultra, Redmi Note 13 Pro, etc.
            const androidMatch = (combinedText || "").match(
                /\b(Galaxy\s*(?:S|A|M|F|Z\s*(?:Fold|Flip)|Note)\s*\d+\s*(?:Ultra|Plus|\+|FE|Pro|Pro\s*Max|Max|Lite|5G)?)/i
            ) || (combinedText || "").match(
                /\b(Redmi\s*(?:Note\s*)?\d+[A-Za-z]?\s*(?:Pro|Plus|\+|5G)?)/i
            ) || (combinedText || "").match(
                /\b(Poco\s*[A-Z]\d+\s*(?:Pro|GT)?)/i
            ) || (combinedText || "").match(
                /\b(OnePlus\s*(?:Nord\s*)?(?:CE\s*)?\d*\s*(?:Pro|Ultra|T)?)/i
            ) || (combinedText || "").match(
                /\b(Pixel\s*\d+\s*(?:a|Pro)?)/i
            ) || (combinedText || "").match(
                /\b((?:Vivo|Oppo|Realme|Honor|Nothing)\s*[A-Z]?\d+[A-Za-z]*\s*(?:Pro|Plus|\+|5G|Lite|Ultra|s)?)/i
            ) || (combinedText || "").match(
                /\b(Nokia\s*(?:G|C|X)?\d+\s*(?:Plus|5G)?)/i
            ) || (combinedText || "").match(
                /\b(Huawei\s*(?:P|Mate|Nova|Y)\s*\d+\s*(?:Pro|Plus|Lite)?)/i
            );
            if (androidMatch) {
                model = androidMatch[1].trim();
            }
        }

        return model || null;
    }

    function extractStorage(combinedText, scraped) {
        // Priority 1: DOM scraped value
        if (scraped && scraped.storage) {
            const cleaned = scraped.storage.replace(/[^0-9TGBtgb]/gi, '');
            const m = cleaned.match(/(\d{1,4})\s*(?:GB|TB)?/i);
            if (m) {
                let num = parseInt(m[1], 10);
                if (/TB/i.test(scraped.storage)) num *= 1024;
                if ([16, 32, 64, 128, 256, 512, 1024].includes(num)) {
                    return num;
                }
            }
        }

        // Priority 2: Parse from combined text
        // Match patterns like "128GB", "256 GB", "128GB ROM", "128GB storage", "128gb internal"
        const storagePatterns = [
            /(\d{2,4})\s*(?:GB|gb)\s*(?:rom|storage|internal|ssd|memory)/i,
            /(?:storage|rom|internal|memory)\s*[:=]?\s*(\d{2,4})\s*(?:GB|gb)?/i,
            /(\d{2,4})\s*(?:GB|gb)\s*\/\s*\d+\s*(?:GB|gb)/i,       // "128GB / 6GB" (storage/RAM)
            /(\d{2,4})\s*(?:GB|gb)(?!\s*(?:ram))/i                   // general "128GB" not followed by "ram"
        ];

        for (const pat of storagePatterns) {
            const match = (combinedText || "").match(pat);
            if (match) {
                const num = parseInt(match[1], 10);
                if ([16, 32, 64, 128, 256, 512, 1024].includes(num)) {
                    return num;
                }
            }
        }

        return null;
    }

    function extractRam(combinedText, scraped, phoneType, modelName) {
        // Priority 1: DOM scraped value
        if (scraped && scraped.ram) {
            const m = scraped.ram.match(/(\d{1,2})/);
            if (m) {
                const val = parseInt(m[1], 10);
                if ([1, 2, 3, 4, 6, 8, 12, 16].includes(val)) {
                    return val;
                }
            }
        }

        // Priority 2: Parse from combined text — match "6GB RAM", "8 GB ram", "6/128"
        const ramPatterns = [
            /(\d{1,2})\s*(?:GB|gb)\s*(?:ram)/i,
            /(?:ram)\s*[:=]?\s*(\d{1,2})\s*(?:GB|gb)?/i,
            /(\d{1,2})\s*(?:GB|gb)\s*\/\s*\d{2,4}\s*(?:GB|gb)/i    // "6GB / 128GB" (RAM/storage)
        ];

        for (const pat of ramPatterns) {
            const match = (combinedText || "").match(pat);
            if (match) {
                const val = parseInt(match[1], 10);
                if ([1, 2, 3, 4, 6, 8, 12, 16].includes(val)) {
                    return val;
                }
            }
        }

        // Priority 3: iPhone RAM from known specs
        if (phoneType === 'iphone' && modelName) {
            const key = modelName.toLowerCase().trim();
            if (IPHONE_RAM_GB[key]) {
                return IPHONE_RAM_GB[key];
            }
            // Try fuzzy match
            for (const [k, v] of Object.entries(IPHONE_RAM_GB)) {
                if (key.includes(k) || k.includes(key)) return v;
            }
        }

        return null;
    }

    /**
     * Extract warranty in days strictly from explicit warranty field or description.
     * NEVER searches raw full-page text to avoid false positives (e.g. "posted 1 month ago").
     */
    function extractWarranty(scrapedWarranty, description, title) {
        // Priority 1: Scraped explicit spec field
        if (scrapedWarranty) {
            const clean = String(scrapedWarranty).toLowerCase().trim();
            if (clean === 'no' || clean === 'no warranty' || clean === 'none' || clean === '0' || clean === 'expired') return 0;
            const numMatch = clean.match(/(\d+)/);
            const num = numMatch ? parseInt(numMatch[1], 10) : 1;
            if (clean.includes('year') || clean.includes('yr')) return num * 365;
            if (clean.includes('month') || clean.includes('mon')) return num * 30;
            if (clean.includes('week')) return num * 7;
            if (clean.includes('day')) return num;
        }

        // Priority 2: Description & Title search with STRICT warranty keyword binding
        const text = `${title || ''} ${description || ''}`.toLowerCase();
        if (!text.trim()) return 0;

        // Negative check
        if (text.includes("no warranty") || text.includes("without warranty") || text.includes("warranty expired") || text.includes("out of warranty")) {
            return 0;
        }

        // Strict patterns requiring the word "warranty" attached to duration
        const yrMatch = text.match(/(\d+)\s*(?:year|yr)s?\s*(?:company\s*|shop\s*|seller\s*|apple\s*|agent\s*)?warranty/i);
        if (yrMatch) return parseInt(yrMatch[1], 10) * 365;
        if (text.includes("1 year warranty") || text.includes("one year warranty") || text.includes("year warranty")) return 365;

        const moMatch = text.match(/(\d+)\s*(?:month|mon)s?\s*(?:company\s*|shop\s*|seller\s*|checking\s*|check\s*|agent\s*)?warranty/i);
        if (moMatch) return parseInt(moMatch[1], 10) * 30;
        if (text.includes("6 month warranty") || text.includes("6 months warranty")) return 180;
        if (text.includes("3 month warranty") || text.includes("3 months warranty")) return 90;
        if (text.includes("1 month warranty") || text.includes("one month warranty") || text.includes("month warranty")) return 30;

        const dayMatch = text.match(/(\d+)\s*(?:day|days)\s*(?:checking\s*|check\s*|seller\s*|shop\s*)?warranty/i);
        if (dayMatch) return parseInt(dayMatch[1], 10);
        if (text.includes("checking warranty") || text.includes("check warranty")) return 7;

        return 0;
    }

    /**
     * Extract battery health percentage (primarily for iPhone second-hand market listings).
     * Returns a valid percentage between 50 and 100, or null if not specified.
     */
    function extractBatteryHealth(scrapedBH, description, title, phoneType) {
        if (scrapedBH) {
            const num = parseFloat(String(scrapedBH).replace(/[^0-9.]/g, ''));
            if (!isNaN(num) && num >= 50 && num <= 100) return num;
        }

        const text = `${title || ''} ${description || ''}`;
        if (!text.trim()) return null;

        // Ordered list of patterns — first match wins.
        // Covers: "Battery Health 87%", "BH: 85%", "Battery 92%", "88% battery health",
        //         "BH 87", "BH-85", "BH - 85", "battery - 85%", "bat health 92",
        //         "battery 85", "battary health 90", "btry 88"
        const patterns = [
            /\b(?:battery\s*health|bat\s*health|battary\s*health|bh)\s*[:=\-]?\s*(\d{2,3})\s*%/i,
            /\b(\d{2,3})\s*%\s*(?:battery\s*health|battery|bh)\b/i,
            /\b(?:battery|battary|btry)\s*[:=\-]?\s*(\d{2,3})\s*%/i,
            /\b(?:battery\s*health|bat\s*health|battary\s*health|bh)\s*[:=\-]?\s*(\d{2,3})\b/i,
            /\b(?:battery|battary|btry)\s*[:=\-]?\s*(\d{2,3})(?:\s|$|,|\.|\))/i
        ];

        for (const pat of patterns) {
            const m = text.match(pat);
            if (m) {
                const v = parseFloat(m[1]);
                if (v >= 50 && v <= 100) return v;
            }
        }

        return null;
    }

    function extractHas5G(combinedText, scraped, phoneType, modelName) {
        // Priority 1: scraped 5G field
        if (scraped && scraped.has_5g) {
            const v = scraped.has_5g.toLowerCase();
            if (v === 'yes' || v === 'true' || v === '1' || v.includes('5g')) return true;
            if (v === 'no' || v === 'false' || v === '0') return false;
        }

        // Priority 2: text mentions
        if (/\b5G\b/.test(combinedText || "")) return true;

        // Priority 3: iPhone known 5G support
        if (phoneType === 'iphone' && modelName) {
            const key = modelName.toLowerCase().trim();
            if (IPHONE_5G_SUPPORT[key]) return true;
            for (const k of Object.keys(IPHONE_5G_SUPPORT)) {
                if (key.includes(k) || k.includes(key)) return true;
            }
        }

        return false;
    }

    function extractDualSim(combinedText, scraped) {
        if (scraped && scraped.dual_sim) {
            const v = scraped.dual_sim.toLowerCase();
            if (v === 'yes' || v === 'true' || v === '1' || v.includes('dual')) return true;
            if (v === 'no' || v === 'false' || v === '0' || v.includes('single')) return false;
        }
        const lower = (combinedText || "").toLowerCase();
        if (lower.includes("dual sim") || lower.includes("dual-sim")) return true;
        if (lower.includes("single sim")) return false;
        return true; // Most modern phones have dual SIM
    }

    function extractEsim(combinedText, scraped, phoneType, modelName) {
        if (scraped && scraped.has_esim) {
            const v = scraped.has_esim.toLowerCase();
            if (v === 'yes' || v === 'true' || v === '1' || v.includes('esim')) return true;
            if (v === 'no' || v === 'false' || v === '0') return false;
        }
        if (/\besim\b/i.test(combinedText || "")) return true;

        // iPhone eSIM from known data
        if (phoneType === 'iphone' && modelName) {
            const key = modelName.toLowerCase().trim();
            if (IPHONE_ESIM_SUPPORT[key]) return true;
            for (const k of Object.keys(IPHONE_ESIM_SUPPORT)) {
                if (key.includes(k) || k.includes(key)) return true;
            }
        }
        return false;
    }

    // ── Feature Engineering (mirroring Python src/feature_engineering.py) ─────

    function computeModelTier(brand, model, phoneType) {
        if (phoneType === 'iphone' && model) {
            const key = model.toLowerCase().trim();
            if (IPHONE_MODEL_TIER[key]) return IPHONE_MODEL_TIER[key];
            for (const [k, v] of Object.entries(IPHONE_MODEL_TIER)) {
                if (key.includes(k) || k.includes(key)) return v;
            }
            return 5;
        }

        // Android: match against tier patterns (try both orderings)
        const textA = ((brand || '') + ' ' + (model || '')).toLowerCase();
        const textB = ((model || '') + ' ' + (brand || '')).toLowerCase();
        for (const [pattern, tier, _] of ANDROID_TIER_PATTERNS) {
            if (pattern.test(textA) || pattern.test(textB)) return tier;
        }
        return 3; // default mid-tier
    }

    function computeBrandTier(brand) {
        if (!brand) return 2;
        const upper = brand.toUpperCase().trim();
        return BRAND_TIER_MAP[upper] || 2;
    }

    function computePhoneAge(model, phoneType) {
        const currentYear = new Date().getFullYear();

        if (phoneType === 'iphone' && model) {
            const key = model.toLowerCase().trim();
            if (IPHONE_RELEASE_YEAR[key]) {
                return Math.max(0, currentYear - IPHONE_RELEASE_YEAR[key]);
            }
            for (const [k, year] of Object.entries(IPHONE_RELEASE_YEAR)) {
                if (key.includes(k) || k.includes(key)) {
                    return Math.max(0, currentYear - year);
                }
            }
        }

        // Android: try to extract year from model name
        const yearMatch = (model || '').match(/\b(20(?:1[5-9]|2[0-9]))\b/);
        if (yearMatch) {
            return Math.max(0, currentYear - parseInt(yearMatch[1], 10));
        }

        return 3.0; // default assumption (matches ML training pipeline)
    }

    function computeIsFlagship(brand, model, phoneType) {
        if (phoneType === 'iphone' && model) {
            const tier = computeModelTier(brand, model, phoneType);
            return tier >= 8 ? 1 : 0;
        }

        const textA = ((brand || '') + ' ' + (model || '')).toLowerCase();
        const textB = ((model || '') + ' ' + (brand || '')).toLowerCase();
        for (const [pattern, _, isFlag] of ANDROID_TIER_PATTERNS) {
            if (pattern.test(textA) || pattern.test(textB)) return isFlag ? 1 : 0;
        }
        return 0;
    }

    // ── Main parse() ─────────────────────────────────────────────────────────

    const NON_PHONE_PATTERNS = [
        /\b(smart\s*watch|smartwatch|watch|wrist\s*watch|fitness\s*band|wristband|watch\s*strap)\b/i,
        /\b(airpods|earbuds|earphones|headphones|headset|bluetooth\s*speaker|ear\s*buds|tws)\b/i,
        /\b(charger|charging\s*cable|power\s*adapter|data\s*cable|fast\s*charger|wireless\s*charger)\b/i,
        /\b(phone\s*case|back\s*cover|flip\s*cover|pouch|silicone\s*case|leather\s*case)\b/i,
        /\b(tempered\s*glass|screen\s*protector|lens\s*protector|gorilla\s*glass)\b/i,
        /\b(power\s*bank|powerbank|battery\s*pack)\b/i,
        /\b(sim\s*tray|housing|display\s*panel|touch\s*display|lcd\s*panel|spare\s*parts|battery\s*replacement)\b/i
    ];

    function parse(pageContext) {
        const { title = "", price = null, raw_text = "", key_values = {} } = pageContext;

        // Step 1: Scrape DOM for structured data
        let scraped = {};
        try { scraped = scrapeListingDOM(); }
        catch (e) { console.warn('[FairPriceLK] Mobile scrape failed:', e); }

        const scopeTitle = scraped.title || title;

        // Check if item is an accessory or smartwatch (NOT a mobile phone)
        const isNonPhone = NON_PHONE_PATTERNS.some(p => p.test(scopeTitle) || p.test(title));
        if (isNonPhone) {
            return {
                category: "unsupported",
                valid: false,
                is_unsupported_item: true,
                error_message: `This listing appears to be a Smart Watch / Accessory ("${scopeTitle || title}"), not a mobile phone. FairPriceLK mobile valuation is designed for Smartphones only.`,
                data: {
                    title: scopeTitle || title,
                    listed_price: scraped.price || price || null,
                    condition: scraped.condition || key_values.condition || null,
                    brand: null,
                    model: null,
                    item_type: "Smart Watch / Accessory"
                }
            };
        }

        // Build combined search scope from all sources
        const descText = scraped.description || "";
        const scope = `${scopeTitle} ${scraped.brand || key_values.brand || ""} ${scraped.model || key_values.model || ""} ${raw_text} ${descText}`;

        // Step 2: Extract brand FIRST (needed for accurate phone_type detection)
        const brand = extractBrand(scope, scraped) || extractBrand(scope, { brand: key_values.brand });

        // Step 3: Determine phone type using brand + OS field + title (NOT full raw_text)
        const phoneType = extractPhoneType(scopeTitle, brand, scraped);

        // Step 4: Extract remaining fields
        let model = extractModel(scope, scraped);
        if (!model && key_values.model) {
            model = key_values.model.trim();
        }
        if (!model && scopeTitle) {
            // Fallback: clean the title by removing brand and noise
            let cleaned = scopeTitle;
            if (brand) cleaned = cleaned.replace(new RegExp(`\\b${brand}\\b`, 'gi'), '').trim();
            cleaned = cleaned.replace(/for sale.*$/i, '').replace(/brand new.*$/i, '').replace(/used.*$/i, '').trim();
            if (cleaned.length > 2) model = cleaned;
        }

        const storage = extractStorage(scope, scraped);
        const ram = extractRam(scope, scraped, phoneType, model);

        // Warranty: only search description and scraped warranty field with strict warranty binding
        const warranty = extractWarranty(scraped.warranty, descText, scopeTitle);

        // Battery Health: only search description and scraped field (for iPhone)
        const batteryHealth = extractBatteryHealth(scraped.battery_health, descText, scopeTitle, phoneType);

        // 5G: check scraped network field first (e.g. "4G" → not 5G)
        let has5g = extractHas5G(scope, scraped, phoneType, model);
        if (scraped && scraped.network) {
            const net = scraped.network.toLowerCase();
            if (net.includes('5g')) has5g = true;
            else if (net.includes('4g') || net.includes('3g') || net.includes('2g')) has5g = false;
        }

        const dualSim = extractDualSim(scope, scraped);
        const hasEsim = extractEsim(scope, scraped, phoneType, model);

        // Step 5: Compute engineered features (matching Python pipeline)
        const modelTier = computeModelTier(brand, model, phoneType);
        const brandTier = computeBrandTier(brand);
        const phoneAge = computePhoneAge(model, phoneType);
        const isFlagship = computeIsFlagship(brand, model, phoneType);

        // Resolve final price
        const finalPrice = scraped.price || price || null;

        // Step 4: Validate
        const missingFields = [];
        if (!brand) missingFields.push("Brand");
        if (!model) missingFields.push("Model");
        if (!storage) missingFields.push("Storage (GB)");
        if (!ram && phoneType !== "iphone") missingFields.push("RAM (GB)");
        if (!finalPrice || isNaN(finalPrice) || finalPrice <= 0) missingFields.push("Listing Price");

        // Step 5: Return — data shape matches API PredictRequest
        return {
            category: "mobile",
            valid: missingFields.length === 0,
            missing_fields: missingFields,
            error_message: missingFields.length > 0
                ? `Could not auto-extract: [${missingFields.join(", ")}]. Please fill in below.`
                : null,
            data: {
                // ── API PredictRequest fields ────────────────────────────
                phone_type: phoneType,
                brand: brand || "",
                model: model || "",
                storage_gb: storage || 128,
                ram_gb: ram || (phoneType === 'iphone' ? 4 : 6),
                warranty_days: warranty,
                battery_health_percent: batteryHealth,
                dual_sim: dualSim,
                has_5g: has5g,
                has_esim: hasEsim,
                model_tier: modelTier,
                brand_tier: brandTier,
                phone_age_years: phoneAge,
                is_flagship: isFlagship,

                // ── Display / UI fields ──────────────────────────────────
                title: scopeTitle,
                listed_price: finalPrice,
                condition: scraped.condition || key_values.condition || null
            }
        };
    }

    return {
        parse: parse,
        scrapeListingDOM: scrapeListingDOM,
        PHONE_BRANDS: PHONE_BRANDS
    };
})();
