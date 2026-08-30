/**
 * FairPriceLK - Master Extractors Dispatcher
 * Automatically detects the listing type on marketplace pages and invokes the proper extractor.
 */

window.FairPriceLK_Extractors = window.FairPriceLK_Extractors || {};

window.FairPriceLK_Extractors.index = (function () {

    function scrapePageDom() {
        const result = {
            title: "",
            price: null,
            key_values: {},
            raw_text: "",
            url: window.location.href
        };

        // 1. Extract Title
        const h1 = document.querySelector('h1');
        if (h1) {
            result.title = h1.innerText.trim();
        }

        // 2. Extract Price
        // Check ikman price elements first
        const priceSelectors = [
            '[class*="price"]',
            'div[data-testid="price"]',
            'span[data-testid="price"]',
            'h2', 'h3', 'strong'
        ];

        let foundPrice = null;
        let targetPriceElement = null;
        for (const sel of priceSelectors) {
            const elements = document.querySelectorAll(sel);
            for (const el of elements) {
                const text = el.innerText || "";
                if ((text.includes("Rs") || text.includes("LKR")) && text.replace(/[^0-9]/g, '').length >= 3) {
                    const cleanNum = parseFloat(text.replace(/[^0-9]/g, ''));
                    if (cleanNum > 100) {
                        foundPrice = cleanNum;
                        targetPriceElement = el;
                        break;
                    }
                }
            }
            if (foundPrice) break;
        }

        // Fallback scan all text nodes for "Rs"
        if (!foundPrice) {
            const allElements = Array.from(document.querySelectorAll('div, span, p, td')).filter(el =>
                el && el.innerText && (el.innerText.includes('Rs') || el.innerText.includes('LKR')) && el.children.length === 0
            );
            if (allElements.length > 0) {
                const cleanNum = parseFloat(allElements[0].innerText.replace(/[^0-9]/g, ''));
                if (cleanNum > 100) {
                    foundPrice = cleanNum;
                    targetPriceElement = allElements[0];
                }
            }
        }
        result.price = foundPrice;
        result.price_element = targetPriceElement;

        // 3. Extract Key-Value Spec Pairs (Ikman attribute list, tables, dl/dt/dd)
        const keyValues = {};

        // Ikman specific attribute items / classes
        document.querySelectorAll('[class*="word-break"], [class*="item-property"], [class*="meta-item"], [class*="attribute"], [data-testid*="attribute"]').forEach(el => {
            const text = (el.innerText || "").trim();
            if (text.includes(":") && text.length < 120) {
                const parts = text.split(":");
                const k = parts[0].toLowerCase().trim();
                const v = parts.slice(1).join(":").trim();
                if (k && v && !keyValues[k]) keyValues[k] = v;
            }
        });

        // Scan dl/dt/dd and tables
        document.querySelectorAll('tr, dl, div, li').forEach(row => {
            const text = (row.innerText || "").trim();
            if (text.includes(":") && text.length < 120) {
                const parts = text.split(":");
                const k = parts[0].toLowerCase().trim();
                const v = parts.slice(1).join(":").trim();
                if (k && v && !keyValues[k]) keyValues[k] = v;
            }
        });

        // Scan sequential divs/spans often used by ikman
        const allText = document.querySelectorAll('div, span, p, li, strong, b');
        let fullCollectedText = "";

        for (let i = 0; i < allText.length; i++) {
            const el = allText[i];
            const t = (el && el.innerText ? el.innerText : "").trim();
            if (!t) continue;

            if (fullCollectedText.length < 4000) {
                fullCollectedText += t + " ";
            }

            const lower = t.toLowerCase();
            const nextEl = el.nextElementSibling;
            const nextText = nextEl && nextEl.innerText ? nextEl.innerText.trim() : "";

            if (lower === 'brand:' || lower === 'brand') keyValues.brand = keyValues.brand || nextText;
            if (lower === 'model:' || lower === 'model') keyValues.model = keyValues.model || nextText;
            if (lower === 'condition:' || lower === 'condition' || lower === 'තත්ත්වය:' || lower === 'තත්වය:' || lower === 'நிலை:') keyValues.condition = keyValues.condition || nextText;
            if (lower === 'edition:' || lower === 'edition') keyValues.edition = keyValues.edition || nextText;
            if (lower === 'trim / edition:' || lower === 'trim/edition:') keyValues.variant = keyValues.variant || nextText;
            if (lower === 'year of manufacture:' || lower === 'year:') keyValues.year = keyValues.year || nextText;
        }

        // Direct condition fallback scan if still not found
        if (!keyValues.condition) {
            document.querySelectorAll('div, span, p, li').forEach(el => {
                if (keyValues.condition) return;
                const text = (el.innerText || "").trim();
                const lower = text.toLowerCase();
                if (lower.startsWith("condition:") || lower.startsWith("condition :")) {
                    keyValues.condition = text.replace(/^condition\s*:\s*/i, '').trim();
                } else if (lower.startsWith("තත්ත්වය:") || lower.startsWith("තත්වය:")) {
                    keyValues.condition = text.replace(/^තත්?ත්වය\s*:\s*/i, '').trim();
                } else if (lower.startsWith("நிலை:")) {
                    keyValues.condition = text.replace(/^நிலை\s*:\s*/i, '').trim();
                }
            });
        }

        // 4. Extract Breadcrumbs / Category Tags
        const breadcrumbs = [];
        document.querySelectorAll('nav[aria-label*="breadcrumb"] a, ol[class*="breadcrumb"] a, ul[class*="breadcrumb"] a, [class*="breadcrumb"] a, a[href*="/ads/sri-lanka/"]').forEach(a => {
            const text = (a.innerText || a.textContent || '').trim();
            if (text && !breadcrumbs.includes(text)) breadcrumbs.push(text);
        });
        result.breadcrumbs = breadcrumbs;

        result.key_values = keyValues;
        result.raw_text = fullCollectedText;

        return result;
    }

    const NON_PHONE_PATTERNS = [
        /\b(smart\s*watch|smartwatch|watch|wrist\s*watch|fitness\s*band|wristband|watch\s*strap)\b/i,
        /\b(airpods|earbuds|earphones|headphones|headset|bluetooth\s*speaker|ear\s*buds|tws)\b/i,
        /\b(charger|charging\s*cable|power\s*adapter|data\s*cable|fast\s*charger|wireless\s*charger)\b/i,
        /\b(phone\s*case|back\s*cover|flip\s*cover|pouch|silicone\s*case|leather\s*case)\b/i,
        /\b(tempered\s*glass|screen\s*protector|lens\s*protector|gorilla\s*glass)\b/i,
        /\b(power\s*bank|powerbank|battery\s*pack)\b/i,
        /\b(sim\s*tray|housing|display\s*panel|touch\s*display|lcd\s*panel|spare\s*parts|battery\s*replacement)\b/i
    ];

    function detectCategory(pageContext) {
        const url = (pageContext.url || "").toLowerCase();
        const title = (pageContext.title || "").toLowerCase();
        const bText = (pageContext.breadcrumbs || []).join(" ").toLowerCase();
        const text = `${pageContext.title} ${pageContext.raw_text}`.toLowerCase();

        // 0. Check for explicit Accessories / Smart Watches (NOT supported mobile phones)
        const isAccessoryBreadcrumb = bText.includes("accessories") || bText.includes("wearables") || bText.includes("smart watch") || bText.includes("audio");
        const isNonPhoneTitle = NON_PHONE_PATTERNS.some(p => p.test(title));

        if (isNonPhoneTitle || isAccessoryBreadcrumb) {
            // If it's a vehicle or computer hardware, let those proceed
            if (!url.includes("riyasewana.com/buy/") && !url.includes("cars") && !url.includes("graphic-card") && !url.includes("laptop")) {
                if (isNonPhoneTitle || !bText.includes("mobile phones")) {
                    return "unsupported";
                }
            }
        }

        // 1. URL based detection
        // Riyasewana.com vehicle listings always follow /buy/<slug> pattern
        if (url.includes("riyasewana.com/buy/")) {
            return "vehicle";
        }

        if (url.includes("computer-accessories") || url.includes("graphic-card") || url.includes("vga") || url.includes("gpu")) {
            return "gpu";
        }
        if (url.includes("mobile-phones") || url.includes("mobile_phones")) {
            return isNonPhoneTitle ? "unsupported" : "mobile";
        }
        if (url.includes("cars") || url.includes("vehicles") || url.includes("van") || url.includes("suv") || url.includes("auto")) {
            return "vehicle";
        }
        if (url.includes("laptop") || url.includes("computer") || url.includes("monitor") || url.includes("tablet") || url.includes("electronics")) {
            if (text.includes("rtx") || text.includes("gtx") || text.includes("rx ") || text.includes("graphics card") || text.includes("vga card") || text.includes("geforce")) {
                return "gpu";
            }
            return "electronics";
        }

        // 2. Keyword heuristic detection on TITLE first
        if (/\b(rtx|gtx|rx\s*\d{3,4}|graphics card|vga card|geforce|radeon)\b/i.test(title)) {
            return "gpu";
        }
        if (/\b(iphone|samsung galaxy|redmi|poco|oneplus|pixel|android phone|mobile phone|huawei|vivo|oppo|realme|nokia|infinix|tecno)\b/i.test(title)) {
            return isNonPhoneTitle ? "unsupported" : "mobile";
        }
        if (/\b(toyota|suzuki|corolla|aqua|alto|honda|nissan|wagon r|prius|axio|premio|vezel|vitz|land cruiser|prado|dolphin|hiace)\b/i.test(title)) {
            return "vehicle";
        }
        if (/\b(laptop|macbook|thinkpad|notebook|dell monitor|curved monitor|ipad)\b/i.test(title)) {
            return "electronics";
        }

        // 3. Fallback check on full text (only if not an accessory)
        if (isNonPhoneTitle) {
            return "unsupported";
        }

        if (/\b(iphone|samsung galaxy|redmi note|oneplus|google pixel)\b/i.test(text)) {
            return "mobile";
        }
        if (/\b(rtx|gtx|geforce|radeon)\b/i.test(text)) {
            return "gpu";
        }
        if (/\b(toyota|suzuki|corolla|aqua|alto)\b/i.test(text)) {
            return "vehicle";
        }

        return "unsupported";
    }

    function isBrandNewCondition(conditionStr, titleStr, rawText, keyValues) {
        const cond = String(conditionStr || "").toLowerCase().trim();
        const title = String(titleStr || "").toLowerCase().trim();

        // 1. Direct condition keywords
        const brandNewKeywords = [
            "new", "brand new", "brand-new", "brandnew", "sealed", "sealed pack",
            "box pack", "box packed", "unopened", "unregistered", "brand new sealed",
            "100% new", "100% brand new", "අලුත්", "අලුත්ම", "නොපැදවූ", "புதியது", "புதிய"
        ];

        if (brandNewKeywords.some(kw => cond === kw || cond.startsWith(kw))) {
            return true;
        }

        if ((cond.includes("brand new") || cond.includes("brand-new") || cond.includes("brandnew") || cond.includes("sealed")) &&
            !cond.includes("used") && !cond.includes("refurbished") && !cond.includes("reconditioned")) {
            return true;
        }

        // 2. Check keyValues
        if (keyValues && typeof keyValues === "object") {
            for (const [k, v] of Object.entries(keyValues)) {
                const kLow = String(k).toLowerCase();
                const vLow = String(v).toLowerCase().trim();
                if (kLow.includes("condition") || kLow.includes("තත්ව") || kLow.includes("நிலை")) {
                    if (brandNewKeywords.some(kw => vLow === kw || vLow.startsWith(kw))) {
                        return true;
                    }
                    if ((vLow.includes("new") || vLow.includes("sealed") || vLow.includes("unregistered")) &&
                        !vLow.includes("used") && !vLow.includes("refurbish") && !vLow.includes("recondition")) {
                        return true;
                    }
                }
            }
        }

        // 3. Fallback scan DOM elements on ikman.lk
        try {
            const condNodes = Array.from(document.querySelectorAll('div, span, li, p, td')).filter(el => {
                const t = (el.innerText || "").trim().toLowerCase();
                return t.startsWith("condition:") || t.startsWith("condition :") || t.startsWith("තත්ත්වය:") || t.startsWith("තත්වය:") || t.startsWith("நிலை:");
            });
            for (const el of condNodes) {
                const t = (el.innerText || "").toLowerCase();
                if ((t.includes("new") || t.includes("sealed") || t.includes("අලුත්") || t.includes("புதிய")) &&
                    !t.includes("used") && !t.includes("recondition") && !t.includes("refurbish")) {
                    return true;
                }
            }
        } catch (e) {}

        // 4. Title analysis (ruling out "like brand new", "same as brand new", "used like brand new")
        const isLikeBrandNew = /\b(like\s+brand\s*new|as\s+brand\s*new|same\s+as\s+brand\s*new|used\s+like|99%\s*condition|mint\s*condition)\b/i.test(title);
        if (!isLikeBrandNew) {
            const titleBrandNewPattern = /\b(brand\s*new|brandnew|sealed\s*pack|sealed\s*box|box\s*pack|box\s*packed|company\s*seal|unregistered|100%\s*brand\s*new)\b/i;
            if (titleBrandNewPattern.test(title) && !title.includes("used") && !title.includes("second hand") && !title.includes("2nd hand")) {
                return true;
            }
        }

        return false;
    }

    function extractAll() {
        const pageContext = scrapePageDom();
        const category = detectCategory(pageContext);
        const isBrandNew = isBrandNewCondition(
            pageContext.key_values ? pageContext.key_values.condition : "",
            pageContext.title,
            pageContext.raw_text,
            pageContext.key_values
        );

        if (category === "unsupported") {
            const isNonPhone = NON_PHONE_PATTERNS.some(p => p.test(pageContext.title || ""));
            const itemType = isNonPhone ? "Smart Watch / Accessory" : "Non-Phone / Unsupported Product";
            return {
                category: "unsupported",
                valid: false,
                is_unsupported_item: true,
                is_brand_new: isBrandNew,
                error_message: `This listing was detected as a ${itemType} ("${pageContext.title || 'Item'}"), not a mobile phone. FairPriceLK is specifically built for Mobile Phones (Smartphones), GPUs, Vehicles, and Computer Hardware.`,
                data: {
                    title: pageContext.title,
                    listed_price: pageContext.price,
                    condition: pageContext.key_values ? pageContext.key_values.condition : null,
                    item_type: itemType
                },
                pageContext: pageContext
            };
        }

        const extractor = window.FairPriceLK_Extractors[category];

        if (extractor && typeof extractor.parse === "function") {
            const parsed = extractor.parse(pageContext);
            parsed.pageContext = pageContext;
            parsed.is_brand_new = isBrandNew || (parsed.data ? isBrandNewCondition(parsed.data.condition, parsed.data.title || pageContext.title, pageContext.raw_text, pageContext.key_values) : false);
            if (parsed.data && !parsed.data.condition && pageContext.key_values && pageContext.key_values.condition) {
                parsed.data.condition = pageContext.key_values.condition;
            }
            return parsed;
        }

        // Fallback generic extraction
        return {
            category: category,
            valid: false,
            is_brand_new: isBrandNew,
            missing_fields: ["Model", "Price"],
            error_message: "Could not safely parse listing specifications.",
            data: {
                title: pageContext.title,
                listed_price: pageContext.price,
                condition: pageContext.key_values ? pageContext.key_values.condition : null
            },
            pageContext: pageContext
        };
    }

    return {
        scrapePageDom: scrapePageDom,
        detectCategory: detectCategory,
        isBrandNewCondition: isBrandNewCondition,
        extractAll: extractAll
    };
})();
