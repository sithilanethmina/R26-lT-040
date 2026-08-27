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
        
        // Scan dl/dt/dd and tables
        document.querySelectorAll('tr, dl, div, li').forEach(row => {
            const text = (row.innerText || "").trim();
            if (text.includes(":") && text.length < 120) {
                const parts = text.split(":");
                const k = parts[0].toLowerCase().trim();
                const v = parts.slice(1).join(":").trim();
                if (k && v) keyValues[k] = v;
            }
        });

        // Scan sequential divs/spans often used by ikman
        const allText = document.querySelectorAll('div, span, p, li');
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

            if (lower === 'brand:' || lower === 'brand') keyValues.brand = nextText;
            if (lower === 'model:' || lower === 'model') keyValues.model = nextText;
            if (lower === 'condition:' || lower === 'condition') keyValues.condition = nextText;
            if (lower === 'edition:' || lower === 'edition') keyValues.edition = nextText;
            if (lower === 'trim / edition:' || lower === 'trim/edition:') keyValues.variant = nextText;
            if (lower === 'year of manufacture:' || lower === 'year:') keyValues.year = nextText;
        }

        result.key_values = keyValues;
        result.raw_text = fullCollectedText;

        return result;
    }

    function detectCategory(pageContext) {
        const url = (pageContext.url || "").toLowerCase();
        const text = `${pageContext.title} ${pageContext.raw_text}`.toLowerCase();

        // 1. URL based detection

        // Riyasewana.com vehicle listings always follow /buy/<slug> pattern
        if (url.includes("riyasewana.com/buy/")) {
            return "vehicle";
        }

        if (url.includes("computer-accessories") || url.includes("graphic-card") || url.includes("vga") || url.includes("gpu")) {
            return "gpu";
        }
        if (url.includes("mobile-phone") || url.includes("phones") || url.includes("mobile_phones")) {
            return "mobile";
        }
        if (url.includes("cars") || url.includes("vehicles") || url.includes("van") || url.includes("suv") || url.includes("auto")) {
            return "vehicle";
        }
        if (url.includes("laptop") || url.includes("computer") || url.includes("monitor") || url.includes("tablet") || url.includes("electronics")) {
            // Further distinguish GPU vs Electronics
            if (text.includes("rtx") || text.includes("gtx") || text.includes("rx ") || text.includes("graphics card") || text.includes("vga card") || text.includes("geforce")) {
                return "gpu";
            }
            return "electronics";
        }

        // 2. Keyword heuristic detection
        if (/\b(rtx|gtx|rx\s*\d{3,4}|graphics card|vga card|geforce|radeon)\b/i.test(text)) {
            return "gpu";
        }
        if (/\b(iphone|samsung galaxy|redmi|poco|oneplus|pixel|android phone|mobile phone)\b/i.test(text)) {
            return "mobile";
        }
        if (/\b(toyota|suzuki|corolla|aqua|alto|honda|nissan|hybrid|automatic transmission)\b/i.test(text)) {
            return "vehicle";
        }
        if (/\b(laptop|macbook|thinkpad|notebook|dell monitor|curved monitor|ipad)\b/i.test(text)) {
            return "electronics";
        }

        return "gpu"; // Default fallback
    }

    function extractAll() {
        const pageContext = scrapePageDom();
        const category = detectCategory(pageContext);
        const extractor = window.FairPriceLK_Extractors[category];

        if (extractor && typeof extractor.parse === "function") {
            const parsed = extractor.parse(pageContext);
            parsed.pageContext = pageContext;
            return parsed;
        }

        // Fallback generic extraction
        return {
            category: category,
            valid: false,
            missing_fields: ["Model", "Price"],
            error_message: "Could not safely parse listing specifications.",
            data: {
                title: pageContext.title,
                listed_price: pageContext.price
            },
            pageContext: pageContext
        };
    }

    return {
        scrapePageDom: scrapePageDom,
        detectCategory: detectCategory,
        extractAll: extractAll
    };
})();
