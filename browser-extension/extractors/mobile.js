/**
 * FairPriceLK - Mobile Phone Category Extractor
 * Provides high-precision parsing of mobile phone listings from marketplace pages (Ikman, etc.)
 * Scrapes listing DOM for user-visible specs (title, asking price, storage, RAM, battery health, warranty).
 * Relies on the Python backend as the Single Source of Truth (SSOT) for hardware specs and feature engineering.
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

    // Known Android-only brands — if brand matches, it's DEFINITELY not an iPhone
    const ANDROID_ONLY_BRANDS = [
        "SAMSUNG", "XIAOMI", "REDMI", "POCO", "ONEPLUS", "GOOGLE",
        "HUAWEI", "VIVO", "OPPO", "REALME", "SONY", "NOKIA", "MOTOROLA",
        "ASUS", "HONOR", "NOTHING", "INFINIX", "TECNO", "ITEL", "LG", "HTC"
    ];

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
            battery_health: null,
            dual_sim: null,
            has_5g: null,
            has_esim: null,
            operating_system: null,
            network: null,
            description: null
        };

        try {
            // 1. Title
            const h1 = document.querySelector('h1');
            if (h1) {
                result.title = h1.innerText.trim();
            }

            // 2. Price — find element containing "Rs." with >=4 digits
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

            // 3. Spec table rows — match label -> value pairs
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
                    else if (/\b(STORAGE|INTERNAL|ROM|MEMORY)\b/i.test(label)) result.storage = value;
                    else if (/\bRAM\b/i.test(label) && !/\b(FRAME|CAMERA|PANORAMA|PROGRAM)\b/i.test(label)) result.ram = value;
                    else if (/\bWARRANTY\b/i.test(label))                     result.warranty = value;
                    else if (/\b(BATTERY|BH)\b/i.test(label))                result.battery_health = value;
                    else if (/\b5G\b/i.test(label))                           result.has_5g = value;
                    else if (/\bSIM\b/i.test(label))                          result.dual_sim = value;
                    else if (/\bESIM\b/i.test(label))                         result.has_esim = value;
                    else if (/\b(OPERATING|OS)\b/i.test(label))              result.operating_system = value;
                    else if (/\bNETWORK\b/i.test(label))                      result.network = value;
                }
            }

            // 4. dl/dt/dd or adjacent-span/div fallback for missing fields
            const labelEls = document.querySelectorAll('dt, th, .label, [class*="spec-label"], [class*="detail-label"]');
            labelEls.forEach(lEl => {
                const lText = (lEl.innerText || '').trim().toUpperCase();
                const vEl = lEl.nextElementSibling;
                if (!vEl) return;
                const vText = (vEl.innerText || '').trim();

                if (/\bBRAND\b/i.test(lText) && !result.brand)           result.brand = vText;
                if (/\bMODEL\b/i.test(lText) && !result.model)           result.model = vText;
                if (/\bCONDITION\b/i.test(lText) && !result.condition)   result.condition = vText;
                if (/\b(STORAGE|INTERNAL|ROM|MEMORY)\b/i.test(lText) && !result.storage) result.storage = vText;
                if (/\bRAM\b/i.test(lText) && !/\b(FRAME|CAMERA|PANORAMA|PROGRAM)\b/i.test(lText) && !result.ram) result.ram = vText;
                if (/\bWARRANTY\b/i.test(lText) && !result.warranty)     result.warranty = vText;
                if (/\b(BATTERY|BH)\b/i.test(lText) && !result.battery_health) result.battery_health = vText;
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

    // ── Field Extractors ─────────────────────────────────────────────────────

    function extractPhoneType(titleText, brand, scraped) {
        if (scraped && scraped.operating_system) {
            const os = scraped.operating_system.toLowerCase();
            if (os.includes('android')) return 'android';
            if (os.includes('ios')) return 'iphone';
        }

        if (brand) {
            const bUpper = brand.toUpperCase().trim();
            if (bUpper === 'APPLE') return 'iphone';
            for (const ab of ANDROID_ONLY_BRANDS) {
                if (bUpper.includes(ab)) return 'android';
            }
        }

        const titleLower = (titleText || '').toLowerCase();
        if (titleLower.includes('iphone') || titleLower.includes('apple')) {
            return 'iphone';
        }

        return 'android';
    }

    function extractBrand(combinedText, scraped) {
        if (scraped && scraped.brand) {
            const bUpper = scraped.brand.toUpperCase().trim();
            for (const b of PHONE_BRANDS) {
                if (bUpper.includes(b)) return CANONICAL_BRANDS[b];
            }
            return scraped.brand.trim();
        }

        const upper = (combinedText || "").toUpperCase();
        for (const b of PHONE_BRANDS) {
            if (new RegExp(`\\b${b}\\b`, 'i').test(upper)) {
                return CANONICAL_BRANDS[b];
            }
        }
        return null;
    }

    function extractModel(combinedText, scraped) {
        let model = null;

        if (scraped && scraped.model && scraped.model.trim()) {
            model = scraped.model.trim();
        }

        if (!model) {
            const iPhoneMatch = (combinedText || "").match(
                /\b(iPhone\s*(?:SE\s*\d?|(?:1[0-7]|X[RSs]?)\s*(?:mini|Plus|Pro\s*Max|Pro)?(?:\s*Max)?))\b/i
            );
            if (iPhoneMatch) {
                model = iPhoneMatch[1].trim();
            }
        }

        if (!model) {
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

        const storagePatterns = [
            /(\d{2,4})\s*(?:GB|gb)\s*(?:rom|storage|internal|ssd|memory)/i,
            /(?:storage|rom|internal|memory)\s*[:=]?\s*(\d{2,4})\s*(?:GB|gb)?/i,
            /(\d{2,4})\s*(?:GB|gb)\s*\/\s*\d+\s*(?:GB|gb)/i,
            /(\d{2,4})\s*(?:GB|gb)(?!\s*(?:ram))/i
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
        if (scraped && scraped.ram) {
            const m = scraped.ram.match(/\b(\d{1,2})\b/);
            if (m) {
                const val = parseInt(m[1], 10);
                if ([1, 2, 3, 4, 6, 8, 12, 16, 24].includes(val)) {
                    return val;
                }
            }
        }

        // Strict RAM patterns requiring explicit 'RAM' or 'X GB RAM' or 'X GB / Y GB'
        const ramPatterns = [
            /\b(\d{1,2})\s*gb\s*ram\b/i,
            /\bram\s*[:=]\s*(\d{1,2})\s*(?:gb)?\b/i,
            /\b(\d{1,2})\s*gb\s*\/\s*\d{2,4}\s*gb\b/i
        ];

        for (const pat of ramPatterns) {
            const match = (combinedText || "").match(pat);
            if (match) {
                const val = parseInt(match[1], 10);
                if ([1, 2, 3, 4, 6, 8, 12, 16, 24].includes(val)) {
                    return val;
                }
            }
        }

        return null;
    }

    function extractWarranty(scrapedWarranty, description, title) {
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

        const text = `${title || ''} ${description || ''}`.toLowerCase();
        if (!text.trim()) return 0;

        if (text.includes("no warranty") || text.includes("without warranty") || text.includes("warranty expired") || text.includes("out of warranty")) {
            return 0;
        }

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

    function extractBatteryHealth(scrapedBH, description, title, phoneType) {
        if (scrapedBH) {
            const num = parseFloat(String(scrapedBH).replace(/[^0-9.]/g, ''));
            if (!isNaN(num) && num >= 50 && num <= 100) return num;
        }

        const text = `${title || ''} ${description || ''}`;
        if (!text.trim()) return null;

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

    function extractHas5G(combinedText, scraped) {
        if (scraped && scraped.has_5g) {
            const v = String(scraped.has_5g).toLowerCase();
            if (v === 'yes' || v === 'true' || v === '1' || v.includes('5g')) return true;
            if (v === 'no' || v === 'false' || v === '0') return false;
        }
        if (scraped && scraped.network) {
            const net = scraped.network.toLowerCase();
            if (net.includes('5g')) return true;
            if (net.includes('4g') || net.includes('3g') || net.includes('2g')) return false;
        }
        if (/\b5G\b/i.test(combinedText || "")) return true;
        return null; // Let backend auto-resolve
    }

    function extractDualSim(combinedText, scraped) {
        if (scraped && scraped.dual_sim) {
            const v = String(scraped.dual_sim).toLowerCase();
            if (v === 'yes' || v === 'true' || v === '1' || v.includes('dual')) return true;
            if (v === 'no' || v === 'false' || v === '0' || v.includes('single')) return false;
        }
        const lower = (combinedText || "").toLowerCase();
        if (lower.includes("dual sim") || lower.includes("dual-sim")) return true;
        if (lower.includes("single sim")) return false;
        return null; // Let backend auto-resolve
    }

    function extractEsim(combinedText, scraped) {
        if (scraped && scraped.has_esim) {
            const v = String(scraped.has_esim).toLowerCase();
            if (v === 'yes' || v === 'true' || v === '1' || v.includes('esim')) return true;
            if (v === 'no' || v === 'false' || v === '0') return false;
        }
        if (/\besim\b/i.test(combinedText || "")) return true;
        return null; // Let backend auto-resolve
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
            const titleDisplay = (scopeTitle || title) ? `"${scopeTitle || title}"` : "This listing";
            return {
                category: "unsupported",
                valid: false,
                is_unsupported_item: true,
                error_message: `${titleDisplay} appears to be a mobile accessory or wearable, not a smartphone. FairPriceLK mobile valuation is designed for Smartphones only.`,
                data: {
                    title: scopeTitle || title,
                    listed_price: scraped.price || price || null,
                    condition: scraped.condition || key_values.condition || null,
                    brand: null,
                    model: null,
                    item_type: "Smart Watch / Phone Accessory"
                }
            };
        }

        // Build combined search scope from all sources
        const descText = scraped.description || "";
        const scope = `${scopeTitle} ${scraped.brand || key_values.brand || ""} ${scraped.model || key_values.model || ""} ${raw_text} ${descText}`;

        // Step 2: Extract brand
        const brand = extractBrand(scope, scraped) || extractBrand(scope, { brand: key_values.brand });

        // Step 3: Determine phone type
        const phoneType = extractPhoneType(scopeTitle, brand, scraped);

        // Step 4: Extract model and visible specs
        let model = extractModel(scope, scraped);
        if (!model && key_values.model) {
            model = key_values.model.trim();
        }
        if (!model && scopeTitle) {
            let cleaned = scopeTitle;
            if (brand) cleaned = cleaned.replace(new RegExp(`\\b${brand}\\b`, 'gi'), '').trim();
            cleaned = cleaned.replace(/for sale.*$/i, '').replace(/brand new.*$/i, '').replace(/used.*$/i, '').trim();
            if (cleaned.length > 2) model = cleaned;
        }

        const storage = extractStorage(scope, scraped);
        const ram = extractRam(scope, scraped, phoneType, model);
        const warranty = extractWarranty(scraped.warranty, descText, scopeTitle);
        const batteryHealth = extractBatteryHealth(scraped.battery_health, descText, scopeTitle, phoneType);
        const has5g = extractHas5G(scope, scraped);
        const dualSim = extractDualSim(scope, scraped);
        const hasEsim = extractEsim(scope, scraped);

        // Resolve final price
        const finalPrice = scraped.price || price || null;

        // Step 5: Validate required minimum fields (Brand, Model, Listed Price)
        const missingFields = [];
        if (!brand) missingFields.push("Brand");
        if (!model) missingFields.push("Model");
        if (!finalPrice || isNaN(finalPrice) || finalPrice <= 0) missingFields.push("Listing Price");

        return {
            category: "mobile",
            valid: missingFields.length === 0,
            missing_fields: missingFields,
            error_message: missingFields.length > 0
                ? `Could not auto-extract: [${missingFields.join(", ")}]. Please fill in below.`
                : null,
            data: {
                // ── API PredictRequest fields (Backend enriches any nulls) ──
                phone_type: phoneType,
                brand: brand || "",
                model: model || "",
                storage_gb: storage,
                ram_gb: ram,
                warranty_days: warranty,
                battery_health_percent: batteryHealth,
                dual_sim: dualSim,
                has_5g: has5g,
                has_esim: hasEsim,

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
