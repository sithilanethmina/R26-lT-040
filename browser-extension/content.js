// This script is injected into the marketplace page to extract details.

function extractIkmanDetails() {
    let details = {
        title: "",
        price: "",
        brand: "",
        model: "",
        vram: "",
        category: "unknown",
        vehicle_type: "cars",
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
        if (url.includes('suv') || details.title.toLowerCase().includes('suv') || fullText.includes('suv')) {
            details.vehicle_type = 'suvs';
        }
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
            const url = window.location.href.toLowerCase();
            let data;
            if (url.includes('riyasewana.com/buy')) {
                data = extractRiyasewanaDetails();
            } else {
                data = extractIkmanDetails();
            }
            sendResponse({ success: true, data: data });
        } catch (error) {
            console.error("Extraction error:", error);
            sendResponse({ success: false, error: error.message });
        }
    }
    return true; // Keep message channel open for async response
});

// ---- Riyasewana.com extractor ----
function extractRiyasewanaDetails() {
    let details = {
        title: "",
        price: "",
        brand: "",
        model: "",
        year: "",
        mileage: "",
        transmission: "",
        fuel_type: "",
        engine_cc: "",
        category: "vehicle",
        vehicle_type: "cars",
        raw_text: ""
    };

    // 1. Title
    const h1 = document.querySelector('h1');
    if (h1) {
        details.title = h1.innerText.trim();
        if (details.title.toLowerCase().includes('suv')) {
            details.vehicle_type = 'suvs';
        }
    }

    // 2. Price — Riyasewana shows "Rs. 4,500,000" in various elements
    const allEls = Array.from(document.querySelectorAll('div, span, p, strong, b'));
    for (const el of allEls) {
        const txt = el.innerText || '';
        if (txt.match(/Rs\.?\s*[\d,]+/) && el.children.length === 0) {
            const numeric = txt.replace(/[^0-9]/g, '');
            if (numeric && numeric.length > 3) {
                details.price = numeric;
                break;
            }
        }
    }

    // 3. Spec table — Riyasewana renders specs as <div class="detail-row"> (new) or <tr> (old)
    const extractSpec = (label, value) => {
        if (label.includes('MAKE'))             details.brand        = value;
        else if (label.includes('MODEL'))       details.model        = value;
        else if (label.includes('YEAR'))        details.year         = value;
        else if (label.includes('GEAR'))        details.transmission = value;
        else if (label.includes('FUEL TYPE'))   details.fuel_type    = value;
        else if (label.includes('MILEAGE'))     details.mileage      = value.replace(/[^0-9]/g, '');
        else if (label.includes('ENGINE (CC)')) details.engine_cc    = value.replace(/[^0-9]/g, '');
        else if (label.includes('BODY TYPE') && value.toUpperCase().includes('SUV')) details.vehicle_type = 'suvs';
    };

    const detailRows = document.querySelectorAll('.detail-row');
    if (detailRows.length > 0) {
        detailRows.forEach(row => {
            const labelEl = row.querySelector('.detail-label');
            const valueEl = row.querySelector('.detail-value');
            if (labelEl && valueEl) {
                extractSpec(labelEl.innerText.toUpperCase().trim(), valueEl.innerText.trim());
            }
        });
    } else {
        const rows = document.querySelectorAll('tr');
        rows.forEach(row => {
            const cells = row.querySelectorAll('td, th');
            if (cells.length >= 2) {
                extractSpec(cells[0].innerText.toUpperCase().trim(), cells[1].innerText.trim());
            }
        });
    }

    // 4. Normalise Transmission → Automatic / Manual
    if (details.transmission) {
        const tl = details.transmission.toLowerCase();
        if (tl.includes('auto')) {
            details.transmission = 'Automatic';
        } else if (tl.includes('manual') || tl.includes('tip') || tl.includes('stick')) {
            details.transmission = 'Manual';
        }
    }

    // 5. Normalise Fuel Type → Petrol / Diesel / Hybrid / Electric
    if (details.fuel_type) {
        const fl = details.fuel_type.toLowerCase();
        if (fl.includes('petrol') || fl.includes('gasoline')) {
            details.fuel_type = 'Petrol';
        } else if (fl.includes('diesel')) {
            details.fuel_type = 'Diesel';
        } else if (fl.includes('hybrid')) {
            details.fuel_type = 'Hybrid';
        } else if (fl.includes('electric')) {
            details.fuel_type = 'Electric';
        }
    }

    // 6. Title-case Brand (e.g. "SUZUKI" → "Suzuki")
    if (details.brand) {
        details.brand = details.brand.charAt(0).toUpperCase() + details.brand.slice(1).toLowerCase();
    }

    details.raw_text = document.body.innerText.slice(0, 2000);
    return details;
}
