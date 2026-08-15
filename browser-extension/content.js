// This script is injected into the marketplace page to extract details.

function extractIkmanDetails() {
    let details = {
        title: "",
        price: "",
        brand: "",
        model: "",
        vram: "",
        category: "unknown",
        raw_text: ""
    };

    // 1. Extract Title
    const h1 = document.querySelector('h1');
    if (h1) {
        details.title = h1.innerText.trim();
    }

    // 2. Extract Price
    const priceElements = Array.from(document.querySelectorAll('div, span, p')).filter(el => 
        el && el.innerText && el.innerText.includes('Rs') && el.children && el.children.length === 0
    );
    
    if (priceElements.length > 0) {
        const rawPrice = priceElements[0].innerText || '';
        const numericPrice = rawPrice.replace(/[^0-9]/g, '');
        if (numericPrice) {
            details.price = numericPrice;
        }
    }

    // 3. Extract key-value properties
    const allTextElements = document.querySelectorAll('div, span, li');
    let fullText = "";
    
    for (let i = 0; i < allTextElements.length; i++) {
        const el = allTextElements[i];
        const text = (el && el.innerText ? el.innerText : "").toLowerCase().trim();
        if (!text) continue;
        
        // Collect some text for heuristic category matching
        if (fullText.length < 2000) {
            fullText += text + " ";
        }

        let nextText = "";
        if (el.nextElementSibling && el.nextElementSibling.innerText) {
            nextText = el.nextElementSibling.innerText.trim();
        }

        if (text === 'brand:' || text === 'brand') {
            details.brand = nextText || "Unknown";
        }
        
        if (text === 'model:' || text === 'model') {
            details.model = nextText || "Unknown";
        }
    }
    
    details.raw_text = fullText;

    // 4. Try to infer category
    const url = window.location.href.toLowerCase();
    
    // Check URL or breadcrumbs first
    if (url.includes('mobile-phone') || url.includes('phones')) {
        details.category = 'mobile';
    } else if (url.includes('car') || url.includes('vehicle') || url.includes('van') || url.includes('suv')) {
        details.category = 'vehicle';
    } else if (url.includes('computer') || url.includes('laptop') || url.includes('electronic')) {
        // Distinguish between laptop/monitor/tablet vs GPU
        if (url.includes('laptop') || fullText.includes('laptop')) {
            details.category = 'electronics';
        } else if (fullText.includes('gpu') || fullText.includes('graphics card') || fullText.includes('vga') || fullText.includes('gtx') || fullText.includes('rtx') || fullText.includes('rx ')) {
            details.category = 'gpu';
        } else {
            details.category = 'electronics';
        }
    } else {
        // Fallback heuristic based on title/text
        const titleLower = details.title.toLowerCase();
        if (titleLower.includes('iphone') || titleLower.includes('samsung galaxy') || titleLower.includes('redmi')) {
            details.category = 'mobile';
        } else if (titleLower.includes('toyota') || titleLower.includes('suzuki') || titleLower.includes('honda')) {
            details.category = 'vehicle';
        } else if (titleLower.includes('rtx') || titleLower.includes('gtx') || titleLower.includes('rx ') || titleLower.includes('graphics card')) {
            details.category = 'gpu';
        } else if (titleLower.includes('laptop') || titleLower.includes('monitor') || titleLower.includes('tablet') || titleLower.includes('ipad')) {
            details.category = 'electronics';
        }
    }

    // 5. GPU specific fallbacks
    const titleLower = details.title.toLowerCase();
    const vramMatch = titleLower.match(/(\d+)\s*gb/);
    if (vramMatch) {
        details.vram = vramMatch[1];
    }

    if (details.category === 'gpu' && !details.model) {
        const commonModels = ['rtx 3060 ti', 'rtx 3060', 'rtx 3070', 'rtx 4090', 'rx 580', 'gtx 1650', 'gtx 1050 ti'];
        for (const m of commonModels) {
            if (titleLower.includes(m)) {
                details.model = m.toUpperCase();
                break;
            }
        }
    }

    return details;
}

// Listen for messages from the popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "extract_details") {
        try {
            const data = extractIkmanDetails();
            sendResponse({ success: true, data: data });
        } catch (error) {
            console.error("Extraction error:", error);
            sendResponse({ success: false, error: error.message });
        }
    }
    return true; // Keep message channel open for async response
});
