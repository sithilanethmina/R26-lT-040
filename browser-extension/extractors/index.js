/**
 * FairPriceLK - Master Extractors Dispatcher
 * Automatically detects the listing type on marketplace pages and invokes the proper extractor.
 */

window.FairPriceLK_Extractors = window.FairPriceLK_Extractors || {};

window.FairPriceLK_Extractors.index = (function () {

    function extractJsonLdAndMeta() {
        const extracted = {
            title: "",
            price: null,
            brand: "",
            category: "",
            condition: "",
            description: ""
        };

        // 1. Check JSON-LD scripts (<script type="application/ld+json">)
        try {
            const scripts = document.querySelectorAll('script[type="application/ld+json"]');
            for (const script of scripts) {
                const text = (script.innerText || script.textContent || "").trim();
                if (!text) continue;
                const json = JSON.parse(text);
                const items = Array.isArray(json) ? json : (json['@graph'] || [json]);

                for (const item of items) {
                    if (!item) continue;
                    const type = String(item['@type'] || '').toLowerCase();
                    if (type.includes('product') || type.includes('vehicle') || type.includes('car') || type.includes('offer') || type.includes('individualproduct')) {
                        if (item.name && !extracted.title) extracted.title = item.name.trim();
                        if (item.description && !extracted.description) extracted.description = item.description.trim();
                        if (item.brand) {
                            if (typeof item.brand === 'string') extracted.brand = item.brand.trim();
                            else if (item.brand.name) extracted.brand = item.brand.name.trim();
                        }
                        if (item.category && !extracted.category) {
                            extracted.category = typeof item.category === 'string' ? item.category : '';
                        }
                        if (item.itemCondition && !extracted.condition) {
                            extracted.condition = String(item.itemCondition).replace(/https?:\/\/schema\.org\//i, '').replace(/condition/i, '').trim();
                        }
                        // Check offer prices
                        if (item.offers) {
                            const offers = Array.isArray(item.offers) ? item.offers : [item.offers];
                            for (const off of offers) {
                                if (off && (off.price || off.lowPrice)) {
                                    const p = parseFloat(String(off.price || off.lowPrice).replace(/[^0-9.]/g, ''));
                                    if (p > 100 && !extracted.price) extracted.price = p;
                                }
                            }
                        } else if (item.price) {
                            const p = parseFloat(String(item.price).replace(/[^0-9.]/g, ''));
                            if (p > 100 && !extracted.price) extracted.price = p;
                        }
                    }
                }
            }
        } catch (e) {
            // Non-blocking fallback
        }

        // 2. Check OpenGraph / meta tags
        try {
            const ogTitle = document.querySelector('meta[property="og:title"], meta[name="twitter:title"]');
            if (ogTitle && ogTitle.content && !extracted.title) {
                extracted.title = ogTitle.content.trim();
            }

            const ogPrice = document.querySelector('meta[property="product:price:amount"], meta[property="og:price:amount"], meta[name="price"]');
            if (ogPrice && ogPrice.content && !extracted.price) {
                const p = parseFloat(ogPrice.content.replace(/[^0-9.]/g, ''));
                if (p > 100) extracted.price = p;
            }

            const ogDesc = document.querySelector('meta[property="og:description"], meta[name="description"]');
            if (ogDesc && ogDesc.content && !extracted.description) {
                extracted.description = ogDesc.content.trim();
            }
        } catch (e) {
            // Non-blocking fallback
        }

        return extracted;
    }

    function scrapePageDom() {
        const metaData = extractJsonLdAndMeta();

        const result = {
            title: metaData.title || "",
            price: metaData.price || null,
            key_values: {},
            raw_text: metaData.description ? metaData.description + " " : "",
            url: window.location.href,
            meta_category: metaData.category || ""
        };

        // 1. Extract Title from H1 if not found via JSON-LD/meta
        if (!result.title) {
            const h1 = document.querySelector('h1');
            if (h1) {
                result.title = h1.innerText.trim();
            }
        }

        // 2. Extract Price if not already extracted from structured metadata
        let foundPrice = result.price;
        let targetPriceElement = null;

        const priceSelectors = [
            '[class*="price"]',
            'div[data-testid="price"]',
            'span[data-testid="price"]',
            'h2', 'h3', 'strong'
        ];

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
            if (foundPrice && targetPriceElement) break;
        }

        // Fallback scan leaf text nodes for "Rs"
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
        if (metaData.brand) keyValues.brand = metaData.brand;
        if (metaData.condition) keyValues.condition = metaData.condition;

        const parseLineForKeyValue = (rawLine) => {
            if (!rawLine) return;
            const line = rawLine.trim();
            const colonIdx = line.indexOf(":");
            if (colonIdx > 0 && colonIdx < line.length - 1 && line.length < 100) {
                const k = line.substring(0, colonIdx).toLowerCase().trim();
                const v = line.substring(colonIdx + 1).trim();
                if (k && v && !keyValues[k]) {
                    // Strip any trailing labels or newlines
                    keyValues[k] = v.split(/\r?\n/)[0].trim();
                }
            }
        };

        // Scan attribute elements and split multi-line blocks line by line
        document.querySelectorAll('[class*="word-break"], [class*="item-property"], [class*="meta-item"], [class*="attribute"], [data-testid*="attribute"], tr, dl, div, li, p').forEach(el => {
            const fullText = (el.innerText || "").trim();
            if (fullText.includes(":")) {
                const lines = fullText.split(/\r?\n/);
                for (const line of lines) {
                    parseLineForKeyValue(line);
                }
            }
        });

        // Scan sequential divs/spans often used by marketplaces
        const allText = document.querySelectorAll('div, span, p, li, strong, b');
        let fullCollectedText = result.raw_text;

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
            if (lower === 'item type:' || lower === 'item type' || lower === 'item-type:' || lower === 'item-type' || lower === 'භාණ්ඩ වර්ගය:' || lower === 'භාණ්ඩ වර්ගය' || lower === 'උපාංග වර්ගය:' || lower === 'උපාංග වර්ගය' || lower === 'ප්‍රභේදය:' || lower === 'ප්‍රභේදය' || lower === 'பொருள் வகை:' || lower === 'பொருள் வகை') keyValues.item_type = keyValues.item_type || nextText;
            if (lower === 'device type:' || lower === 'device type' || lower === 'type of item:' || lower === 'type of item' || lower === 'category:' || lower === 'category' || lower === 'sub-category:' || lower === 'subcategory:') keyValues.item_type = keyValues.item_type || nextText;
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

    // Explicit complete computer systems / host devices patterns that MUST NOT be misclassified as standalone GPUs
    const COMPLETE_SYSTEM_PATTERNS = [
        /\b(laptop|notebook|macbook|gaming laptop|ultrabook|chromebook)\b/i,
        /\b(thinkpad|ideapad|legion|loq|thinkbook|vivobook|zenbook|expertbook|flow|zephyrus|predator|helios|nitro\s*\d*|aspire|swift|spin|travelmate|alienware|omen|victus|pavilion|envy|spectre|latitude|inspiron|vostro|precision|elitebook|probook|zbook|katana|cyborg|thin|bravo|sword|stealth|raider|titan|modern|prestige|summit|gf63|gf65|gf75|gp66|gp76|gl65|gl75|pulse|vector|crosshair|delta)\b/i,
        /\b(i[3579](?:-?\d{2,5}[a-z]*)?|core\s*i[3579]|ryzen\s*[3579]|core\s*ultra|intel\s*core|amd\s*ryzen)\b/i,
        /\b(desktop pc|gaming pc|computer system|full set pc|full unit|complete set|cpu unit|workstation|all-in-one pc|aio pc|gaming rig)\b/i,
        /\b(playstation|ps4|ps5|xbox|nintendo switch|console)\b/i
    ];

    // Explicit standalone GPU identifiers
    const STANDALONE_GPU_PATTERNS = [
        /\b(graphics\s*card|graphic\s*card|vga\s*card|video\s*card|display\s*card|gpu\s*only|card\s*only)\b/i
    ];

    function getStructuredItemTypeValue(keyValues) {
        if (!keyValues || typeof keyValues !== "object") return "";
        for (const [k, v] of Object.entries(keyValues)) {
            const kLow = String(k).toLowerCase().trim();
            if (
                kLow === "item type" || kLow === "item_type" || kLow === "item-type" ||
                kLow === "type of item" || kLow === "device type" || kLow === "device_type" ||
                kLow === "category" || kLow === "sub-category" || kLow === "subcategory" ||
                kLow.includes("item type") || kLow.includes("item-type") ||
                kLow.includes("භාණ්ඩ වර්ගය") || kLow.includes("උපාංග වර්ගය") || kLow.includes("ප්‍රභේදය") ||
                kLow.includes("பொருள் வகை")
            ) {
                if (v && typeof v === "string") return v.toLowerCase().trim();
            }
        }
        return "";
    }

    function detectCategory(pageContext) {
        const url = (pageContext.url || "").toLowerCase();
        const title = (pageContext.title || "").toLowerCase();
        const bText = (pageContext.breadcrumbs || []).join(" ").toLowerCase();
        const metaCat = (pageContext.meta_category || "").toLowerCase();
        const text = `${pageContext.title} ${pageContext.raw_text}`.toLowerCase();
        const itemTypeVal = getStructuredItemTypeValue(pageContext.key_values);

        // 0. Check for explicit Mobile Accessories / Smart Watches (NOT supported mobile phones)
        const isPhoneAccessoryBreadcrumb = (bText.includes("mobile accessories") || bText.includes("phone accessories") || bText.includes("wearables") || bText.includes("smart watch") || bText.includes("audio")) && !bText.includes("computer") && !bText.includes("graphic");
        const isNonPhoneTitle = NON_PHONE_PATTERNS.some(p => p.test(title));

        if (isNonPhoneTitle || isPhoneAccessoryBreadcrumb) {
            // If it's a vehicle or computer hardware/laptop/gpu, let those proceed
            if (!url.includes("riyasewana.com/buy/") && !url.includes("cars") && !url.includes("graphic-card") && !url.includes("laptop") && !bText.includes("laptop") && !bText.includes("graphic") && !bText.includes("vga")) {
                if (isNonPhoneTitle || !bText.includes("mobile phones")) {
                    return "unsupported";
                }
            }
        }

        // 1. STRUCTURED ITEM TYPE / ATTRIBUTE CHECK (Highest Priority - 100% Deterministic)
        if (itemTypeVal) {
            // GPU / VGA Cards
            if (/\b(graphic|graphics|vga|video\s*card|display\s*card|gpu|ග්‍රැෆික්|கிராபிக்)\b/i.test(itemTypeVal)) {
                return "gpu";
            }
            // Laptops / Computers / Monitors / Tablets / Computer Accessories
            if (/\b(laptop|laptops|notebook|notebooks|desktop|desktops|monitor|monitors|tablet|tablets|macbook|computer\s*accessories|computer|computers|hard\s*drive|ram|motherboard|processor|cpu|casing|power\s*supply|ups|sound\s*card|mouse|keyboard|networking|software|ලැප්ටොප්|මොනිටර්|පරිගණක|கணினி|மடிக்கணினி)\b/i.test(itemTypeVal)) {
                return "electronics";
            }
            // Mobile phones
            if (/\b(mobile\s*phone|mobile\s*phones|smartphone|smartphones|mobile|දුරකථන|ජංගම|கைபேසි)\b/i.test(itemTypeVal)) {
                return isNonPhoneTitle ? "unsupported" : "mobile";
            }
            // Vehicles
            if (/\b(car|cars|van|vans|motorbike|motorbikes|motorcycle|motorcycles|scooter|scooters|three\s*wheel|three\s*wheelers|auto|suv|suvs|truck|trucks|lorry|lorries|bus|buses|tractor|tractors|vehicle|vehicles)\b/i.test(itemTypeVal)) {
                return "vehicle";
            }
        }

        // 2. BREADCRUMBS & META CATEGORY (High Priority)
        if (bText.includes("graphic card") || bText.includes("graphic cards") || bText.includes("video card") || bText.includes("vga card") || bText.includes("vga") || metaCat.includes("graphic card") || bText.includes("ග්‍රැෆික්") || bText.includes("கிராபிக்")) {
            return "gpu";
        }
        if (bText.includes("laptops") || bText.includes("laptop computers") || bText.includes("desktop computers") || bText.includes("monitors") || (bText.includes("tablets") && !bText.includes("computers & tablets")) || (bText.includes("computer accessories") && !bText.includes("graphic") && !bText.includes("vga")) || metaCat.includes("laptop") || bText.includes("ලැප්ටොප්") || bText.includes("මොනිටර්") || bText.includes("පරිගණක") || bText.includes("மடிக்கணினி")) {
            return "electronics";
        }
        if (bText.includes("mobile phones") || metaCat.includes("phone") || bText.includes("දුරකථන") || bText.includes("கைபேසි")) {
            return isNonPhoneTitle ? "unsupported" : "mobile";
        }
        if (bText.includes("cars") || bText.includes("vans") || bText.includes("motorbikes") || bText.includes("vehicles") || metaCat.includes("vehicle") || metaCat.includes("car")) {
            return "vehicle";
        }

        // 3. URL BASED DETECTION
        if (url.includes("riyasewana.com/buy/")) {
            return "vehicle";
        }
        if (url.includes("graphic-card") || url.includes("graphic-cards") || url.includes("vga") || url.includes("/vga-")) {
            return "gpu";
        }
        if (url.includes("/laptops") || url.includes("-laptops-") || url.includes("/monitors") || url.includes("-monitors-") || url.includes("/desktop-computers") || url.includes("/computer-accessories") || url.includes("/computers-tablets")) {
            return "electronics";
        }
        if (url.includes("mobile-phones") || url.includes("mobile_phones")) {
            return isNonPhoneTitle ? "unsupported" : "mobile";
        }
        if (url.includes("cars") || url.includes("vehicles") || url.includes("/van-") || url.includes("/suv-") || url.includes("/auto-")) {
            return "vehicle";
        }

        // 4. TITLE & KEYWORD HEURISTICS (Fallback when metadata/breadcrumbs missing)
        const isCompleteSystemTitle = COMPLETE_SYSTEM_PATTERNS.some(p => p.test(title));
        const isExplicitGpuTitle = STANDALONE_GPU_PATTERNS.some(p => p.test(title));
        const hasGpuModelInTitle = /\b(rtx\s*\d{3,4}|gtx\s*\d{3,4}|rx\s*\d{3,4}|gt\s*\d{3,4}|geforce|radeon|arc\s*a\d{3})\b/i.test(title);

        // If title explicitly represents a Laptop, Desktop PC, or Complete System (even if GPU is mentioned) -> ELECTRONICS
        if (isCompleteSystemTitle && !isExplicitGpuTitle) {
            return "electronics";
        }

        // If title has a GPU model or GPU keywords and NOT marked as a complete system -> GPU
        if ((hasGpuModelInTitle || isExplicitGpuTitle) && !isCompleteSystemTitle) {
            return "gpu";
        }

        if (/\b(iphone|samsung galaxy|redmi|poco|oneplus|pixel|android phone|mobile phone|huawei|vivo|oppo|realme|nokia|infinix|tecno)\b/i.test(title)) {
            return isNonPhoneTitle ? "unsupported" : "mobile";
        }
        if (/\b(toyota|suzuki|corolla|aqua|alto|honda|nissan|wagon r|prius|axio|premio|vezel|vitz|land cruiser|prado|dolphin|hiace)\b/i.test(title)) {
            return "vehicle";
        }
        if (/\b(laptop|macbook|thinkpad|notebook|dell monitor|curved monitor|ipad|tab\b|tablet|matepad|mediapad|vostro|latitude|inspiron|elitebook|probook|zenbook|vivobook|thinkbook|ideapad|gaming laptop)\b/i.test(title)) {
            return "electronics";
        }

        // 5. FALLBACK CHECK ON FULL TEXT
        if (isNonPhoneTitle) {
            return "unsupported";
        }

        if (COMPLETE_SYSTEM_PATTERNS.some(p => p.test(text))) {
            return "electronics";
        }
        if (/\b(graphics card|graphic card|vga card|geforce|radeon)\b/i.test(text)) {
            return "gpu";
        }
        if (/\b(iphone|samsung galaxy|redmi note|oneplus|google pixel)\b/i.test(text)) {
            return "mobile";
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
            const itemType = isNonPhone ? "Smart Watch / Phone Accessory" : "Unsupported Product Category";
            const titleDisplay = pageContext.title ? `"${pageContext.title}"` : "This listing";
            const errorMessage = isNonPhone
                ? `${titleDisplay} appears to be a mobile accessory or wearable, not a supported smartphone. FairPriceLK currently supports Mobile Phones (Smartphones), Graphics Cards (GPUs), Vehicles, and Computer Hardware.`
                : `${titleDisplay} is not in a supported category. FairPriceLK currently provides price valuation for Mobile Phones, Graphics Cards (GPUs), Vehicles, and Computer Hardware (Laptops/Monitors).`;

            return {
                category: "unsupported",
                valid: false,
                is_unsupported_item: true,
                is_brand_new: isBrandNew,
                error_message: errorMessage,
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
