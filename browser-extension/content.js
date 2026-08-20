/**
 * FairPriceLK - Content Script & On-Page Floating Valuation Widget
 * Runs modular extractors on marketplace listings, injects live on-page widget,
 * contacts API gateway directly, and provides manual input fallback if extraction has missing fields.
 */

(function () {
    // Avoid double injection
    if (window.__fairPriceLkInjected) return;
    window.__fairPriceLkInjected = true;

    function getApiBase() {
        if (typeof CONFIG !== 'undefined' && CONFIG.API_BASE_URL) return CONFIG.API_BASE_URL;
        if (window.CONFIG && window.CONFIG.API_BASE_URL) return window.CONFIG.API_BASE_URL;
        return 'http://127.0.0.1:8000';
    }

    let currentExtraction = null;
    let widgetRoot = null;
    let inlineBadgeRoot = null;
    let isExpanded = false;
    let cachedPrediction = null;

    // --- High-level extraction runner ---
    function runExtraction() {
        if (!window.FairPriceLK_Extractors || !window.FairPriceLK_Extractors.index) {
            console.warn("FairPriceLK extractors not loaded yet.");
            return null;
        }
        currentExtraction = window.FairPriceLK_Extractors.index.extractAll();
        return currentExtraction;
    }

    // --- Create UI Elements ---
    function isItemDetailPage() {
        const href = window.location.href.toLowerCase();
        
        // 1. Must NOT be an aggregate category or search results listing page
        if (href.includes('/ads/') || href.includes('/ads?')) {
            // ikman detail pages have /en/ad/ or /si/ad/ (singular 'ad'), search feeds have /ads/ (plural 'ads')
            return false;
        }

        // 2. Must be an individual ad detail page or contain a clear item price + header
        const hasAdPattern = href.includes('/ad/') || 
                             href.includes('/item/') || 
                             href.includes('/product/') ||
                             href.includes('/view/');

        const hasListingHeader = document.querySelector('h1') && (
            document.querySelector('[class*="price"]') || 
            document.querySelector('div[data-testid="price"]') ||
            document.querySelector('span[data-testid="price"]')
        );

        return hasAdPattern || hasListingHeader;
    }

    function initOnPageWidget() {
        if (document.getElementById('fairpricelk-widget-root')) return;
        
        if (!isItemDetailPage()) return;

        widgetRoot = document.createElement('div');
        widgetRoot.id = 'fairpricelk-widget-root';
        document.body.appendChild(widgetRoot);

        runExtraction();
        renderWidget();
        renderInlineBadge();

        // If extraction is 100% valid, automatically trigger evaluation
        if (currentExtraction && currentExtraction.valid) {
            triggerPrediction();
        }
    }

    function renderInlineBadge() {
        // Find price element from extraction
        const priceEl = currentExtraction && currentExtraction.pageContext ? currentExtraction.pageContext.price_element : null;
        if (!priceEl) return;

        if (!inlineBadgeRoot) {
            inlineBadgeRoot = document.createElement('span');
            inlineBadgeRoot.className = 'fplk-inline-badge-container';
            inlineBadgeRoot.id = 'fairpricelk-inline-root';
            // Insert right next to price
            priceEl.insertAdjacentElement('afterend', inlineBadgeRoot);
        }

        const ext = currentExtraction || { valid: false };
        let tagClass = 'neutral';
        let tagText = 'Evaluating';
        let valText = 'Checking market...';

        if (cachedPrediction && cachedPrediction.evaluation) {
            tagClass = cachedPrediction.evaluation.badge_class || 'fair';
            tagText = cachedPrediction.evaluation.verdict || 'FAIR';
            if (cachedPrediction.fair_market_range && cachedPrediction.fair_market_range.lower_price_lkr) {
                valText = `Est. Rs ${Math.round(cachedPrediction.fair_market_range.lower_price_lkr / 1000)}k–${Math.round(cachedPrediction.fair_market_range.upper_price_lkr / 1000)}k`;
            } else if (cachedPrediction.predicted_price) {
                valText = `Est. Rs ${Math.round(cachedPrediction.predicted_price / 1000)}k`;
            }
        } else if (!ext.valid) {
            tagClass = 'neutral';
            tagText = 'Details Needed';
            valText = 'FairPriceLK';
        }

        const iconUrl = chrome.runtime.getURL('icon.png');

        inlineBadgeRoot.innerHTML = `
            <div class="fplk-inline-badge" title="FairPriceLK Price Intelligence - Click for details">
                <span class="fplk-inline-brand">
                    <img src="${iconUrl}" width="14" height="14" alt="FairPriceLK" style="border-radius: 3px; object-fit: contain; vertical-align: middle;">
                    FairPriceLK
                </span>
                <span class="fplk-inline-divider"></span>
                <span class="fplk-inline-val">${valText}</span>
                <span class="fplk-inline-tag ${tagClass}">${tagText}</span>
            </div>
        `;

        inlineBadgeRoot.onclick = (e) => {
            e.stopPropagation();
            isExpanded = true;
            renderWidget();
        };
    }

    function renderWidget() {
        if (!widgetRoot) return;

        const ext = currentExtraction || { valid: false, category: 'gpu', data: {} };
        const data = ext.data || {};
        const cat = ext.category || 'gpu';
        const iconUrl = chrome.runtime.getURL('icon.png');

        widgetRoot.innerHTML = `
            ${isExpanded ? `
                <div class="fplk-card-backdrop" id="fplk-backdrop"></div>
                <div class="fplk-card" id="fplk-card-container">
                    <div class="fplk-header">
                        <div class="fplk-brand-wrap">
                            <img src="${iconUrl}" width="20" height="20" alt="FairPriceLK Logo" style="border-radius: 4px; object-fit: contain;">
                            <h3>FairPriceLK Valuation</h3>
                        </div>
                        <div class="fplk-header-controls">
                            <button class="fplk-icon-btn" id="fplk-reextract-btn" title="Re-scan Page">
                                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>
                            </button>
                            <button class="fplk-icon-btn" id="fplk-close-btn" title="Close">
                                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                            </button>
                        </div>
                    </div>

                    <div class="fplk-body">
                        <!-- Product Identity -->
                        <div class="fplk-product-identity">
                            <div class="fplk-product-title">${data.title || data.model || "Marketplace Listing"}</div>
                            <div class="fplk-product-meta">
                                <span class="fplk-product-tag">${cat.toUpperCase()}</span>
                                ${data.brand ? `<span>Brand: <strong>${data.brand}</strong></span>` : ''}
                                ${data.vram_gb ? `<span>· VRAM: <strong>${data.vram_gb} GB</strong></span>` : ''}
                                ${data.model_year ? `<span>· Year: <strong>${data.model_year}</strong></span>` : ''}
                            </div>
                        </div>

                        ${!ext.valid ? `
                            <div class="fplk-verdict-box error">
                                <div class="fplk-verdict-header">
                                    <span class="fplk-verdict-tag">Information Needed</span>
                                </div>
                                <div class="fplk-verdict-body">
                                    ${ext.error_message || "Could not automatically identify full specifications. Please check or confirm details below."}
                                </div>
                            </div>
                        ` : ''}

                        ${cachedPrediction ? renderPredictionResult(cachedPrediction, data.listed_price) : ''}

                        <!-- Manual Adjustment Section -->
                        <details class="fplk-form-section">
                            <summary class="fplk-form-title" style="cursor: pointer;">
                                <span>Refine Detected Details</span>
                                <span style="font-size: 10px; color: #71717A;">Click to edit</span>
                            </summary>

                            <div style="margin-top: 10px; display: flex; flex-direction: column; gap: 8px;">
                                ${renderCategoryFormFields(cat, data)}

                                <div class="fplk-form-group">
                                    <label class="fplk-label">Listing Asking Price (LKR)</label>
                                    <input type="number" class="fplk-input" id="fplk-input-price" value="${data.listed_price || ''}" placeholder="e.g. 75000">
                                </div>

                                <button class="fplk-btn" id="fplk-eval-btn">
                                    Recalculate Market Value
                                </button>
                            </div>
                        </details>

                        <div class="fplk-footer">
                            <span>FairPriceLK Intelligence</span>
                            <span>Engine: Local Active</span>
                        </div>
                    </div>
                </div>
            ` : ''}
        `;

        renderInlineBadge();
        attachEventListeners();
    }

    function renderPredictionResult(pred, listedPrice) {
        if (!pred) return '';

        const hasRange = pred.fair_market_range && pred.fair_market_range.lower_price_lkr;
        const lower = hasRange ? pred.fair_market_range.lower_price_lkr : (pred.predicted_price * 0.9);
        const upper = hasRange ? pred.fair_market_range.upper_price_lkr : (pred.predicted_price * 1.1);
        const pointPrice = pred.predicted_price || pred.price || ((lower + upper) / 2);
        
        // Evaluate using universal engine
        let fairness = null;
        const category = (currentExtraction && currentExtraction.category) ? currentExtraction.category : 'gpu';
        const itemDetails = (currentExtraction && currentExtraction.data) ? currentExtraction.data : {};
        if (window.FairPriceLK_Fairness) {
            fairness = window.FairPriceLK_Fairness.evaluate(
                listedPrice, 
                pointPrice, 
                lower, 
                upper, 
                category, 
                itemDetails
            );
        }

        const badgeCls = fairness ? fairness.badgeClass : (pred.evaluation && pred.evaluation.badge_class) || 'fair';
        const verdictTitle = fairness ? fairness.badgeText : (pred.evaluation && pred.evaluation.verdict) || 'Fair Market Price';
        const scoreVal = fairness && fairness.score !== null ? fairness.score : (pred.evaluation && pred.evaluation.fairness_score);
        const adviceText = fairness ? fairness.advice : (pred.evaluation && pred.evaluation.description);
        const actionAdvice = fairness ? fairness.actionAdvice : null;
        const negotiationTarget = fairness ? fairness.negotiationTarget : null;

        // Calculate visual marker position (0% - 100%)
        let markerPos = 50;
        if (listedPrice && upper > lower) {
            const span = (upper - lower) * 1.5;
            const minBound = lower - (span * 0.25);
            markerPos = Math.max(4, Math.min(96, ((listedPrice - minBound) / span) * 100));
        }

        return `
            <!-- Price Comparison Grid -->
            <div class="fplk-price-grid">
                <div class="fplk-price-card">
                    <span class="fplk-price-label">Asking Price</span>
                    <span class="fplk-price-val">${listedPrice ? `Rs. ${Number(listedPrice).toLocaleString('en-LK')}` : 'Not Specified'}</span>
                </div>
                <div class="fplk-price-card primary">
                    <span class="fplk-price-label">Est. Market Range</span>
                    <span class="fplk-price-val" style="font-size: 15px;">Rs. ${Math.round(lower/1000)}k – ${Math.round(upper/1000)}k</span>
                </div>
            </div>

            <!-- Visual Price Position Gauge -->
            ${listedPrice ? `
                <div class="fplk-visual-range-container">
                    <div class="fplk-visual-range-labels">
                        <span>Low: Rs. ${Math.round(lower/1000)}k</span>
                        <span>Mid: Rs. ${Math.round(pointPrice/1000)}k</span>
                        <span>High: Rs. ${Math.round(upper/1000)}k</span>
                    </div>
                    <div class="fplk-range-track">
                        <div class="fplk-range-fill" style="left: 15%; width: 70%;"></div>
                        <div class="fplk-range-marker ${badgeCls}" style="left: ${markerPos}%;"></div>
                    </div>
                    <div class="fplk-gauge-subtext">
                        ● Asking Price: <strong>Rs. ${Number(listedPrice).toLocaleString('en-LK')}</strong> 
                        ${fairness && fairness.diffPercent ? `(${fairness.diffPercent > 0 ? '+' : ''}${fairness.diffPercent}%)` : ''}
                    </div>
                </div>
            ` : ''}

            <!-- Fairness & Verdict Card -->
            <div class="fplk-verdict-box ${badgeCls}">
                <div class="fplk-verdict-header">
                    <span class="fplk-verdict-tag">${verdictTitle}</span>
                    ${scoreVal !== undefined && scoreVal !== null ? `<span class="fplk-score-pill">Score: <strong>${scoreVal}/100</strong></span>` : ''}
                </div>
                <div class="fplk-verdict-body">
                    ${adviceText || 'Estimated using second-hand market distribution and hardware specifications.'}
                </div>
                ${actionAdvice ? `
                    <div class="fplk-action-advice">
                        <strong>💡 Recommendation:</strong> ${actionAdvice}
                    </div>
                ` : ''}
                ${negotiationTarget ? `
                    <div class="fplk-negotiation-badge">
                        🎯 Target Counter-Offer: <strong>${negotiationTarget}</strong>
                    </div>
                ` : ''}
            </div>
        `;
    }

    function renderCategoryFormFields(category, data) {
        if (category === 'gpu') {
            const models = (window.FairPriceLK_Extractors && window.FairPriceLK_Extractors.gpu) ? 
                            window.FairPriceLK_Extractors.gpu.CANONICAL_MODELS : [];
            const brands = (window.FairPriceLK_Extractors && window.FairPriceLK_Extractors.gpu) ? 
                            window.FairPriceLK_Extractors.gpu.KNOWN_BRANDS : ["Any", "ASUS", "MSI", "GIGABYTE", "ZOTAC"];

            const currentModel = data.model || "";
            const currentBrand = data.brand || "Any";
            const currentVram = data.vram_gb || 8;

            return `
                <div class="fplk-form-group full-width">
                    <label class="fplk-label">GPU Model</label>
                    <input type="text" class="fplk-input ${!currentModel ? 'invalid' : ''}" id="fplk-input-gpu-model" value="${currentModel}" placeholder="e.g. RTX 3060, RX 580, GTX 1660 SUPER" list="fplk-gpu-models-list">
                    <datalist id="fplk-gpu-models-list">
                        ${models.map(m => `<option value="${m}">`).join('')}
                    </datalist>
                </div>
                <div class="fplk-form-grid">
                    <div class="fplk-form-group">
                        <label class="fplk-label">VRAM (GB)</label>
                        <select class="fplk-select" id="fplk-select-gpu-vram">
                            ${[1, 2, 3, 4, 6, 8, 10, 11, 12, 16, 20, 24].map(v => 
                                `<option value="${v}" ${Number(currentVram) === v ? 'selected' : ''}>${v} GB</option>`
                            ).join('')}
                        </select>
                    </div>
                    <div class="fplk-form-group">
                        <label class="fplk-label">Brand</label>
                        <select class="fplk-select" id="fplk-select-gpu-brand">
                            <option value="Any">Any</option>
                            ${brands.map(b => `<option value="${b}" ${currentBrand.toUpperCase() === b.toUpperCase() ? 'selected' : ''}>${b}</option>`).join('')}
                        </select>
                    </div>
                </div>
            `;
        } else if (category === 'mobile') {
            return `
                <div class="fplk-form-grid">
                    <div class="fplk-form-group">
                        <label class="fplk-label">OS / Type</label>
                        <select class="fplk-select" id="fplk-mobile-type">
                            <option value="android" ${data.phone_type === 'android' ? 'selected' : ''}>Android</option>
                            <option value="iphone" ${data.phone_type === 'iphone' ? 'selected' : ''}>iPhone</option>
                        </select>
                    </div>
                    <div class="fplk-form-group">
                        <label class="fplk-label">Brand</label>
                        <input type="text" class="fplk-input" id="fplk-mobile-brand" value="${data.brand || ''}" placeholder="e.g. Samsung">
                    </div>
                </div>
                <div class="fplk-form-group full-width">
                    <label class="fplk-label">Model</label>
                    <input type="text" class="fplk-input" id="fplk-mobile-model" value="${data.model || ''}" placeholder="e.g. Galaxy S21">
                </div>
                <div class="fplk-form-grid">
                    <div class="fplk-form-group">
                        <label class="fplk-label">Storage (GB)</label>
                        <input type="number" class="fplk-input" id="fplk-mobile-storage" value="${data.storage_gb || 128}">
                    </div>
                    <div class="fplk-form-group">
                        <label class="fplk-label">RAM (GB)</label>
                        <input type="number" class="fplk-input" id="fplk-mobile-ram" value="${data.ram_gb || 6}">
                    </div>
                </div>
            `;
        } else if (category === 'vehicle') {
            return `
                <div class="fplk-form-group full-width">
                    <label class="fplk-label">Vehicle Model</label>
                    <select class="fplk-select" id="fplk-vehicle-model">
                        <option value="Toyota Corolla" ${data.model === 'Toyota Corolla' ? 'selected' : ''}>Toyota Corolla</option>
                        <option value="Toyota Aqua" ${data.model === 'Toyota Aqua' ? 'selected' : ''}>Toyota Aqua</option>
                        <option value="Suzuki Alto" ${data.model === 'Suzuki Alto' ? 'selected' : ''}>Suzuki Alto</option>
                    </select>
                </div>
                <div class="fplk-form-grid">
                    <div class="fplk-form-group">
                        <label class="fplk-label">Year</label>
                        <input type="number" class="fplk-input" id="fplk-vehicle-year" value="${data.model_year || 2015}">
                    </div>
                    <div class="fplk-form-group">
                        <label class="fplk-label">Variant</label>
                        <input type="text" class="fplk-input" id="fplk-vehicle-variant" value="${data.variant || '121'}">
                    </div>
                </div>
            `;
        } else {
            return `
                <div class="fplk-form-grid">
                    <div class="fplk-form-group">
                        <label class="fplk-label">Brand</label>
                        <input type="text" class="fplk-input" id="fplk-elec-brand" value="${data.brand || ''}" placeholder="e.g. Dell">
                    </div>
                    <div class="fplk-form-group">
                        <label class="fplk-label">Model</label>
                        <input type="text" class="fplk-input" id="fplk-elec-model" value="${data.model || ''}" placeholder="e.g. XPS 13">
                    </div>
                </div>
            `;
        }
    }

    function attachEventListeners() {
        const backdrop = document.getElementById('fplk-backdrop');
        if (backdrop) {
            backdrop.addEventListener('click', () => {
                isExpanded = false;
                renderWidget();
            });
        }

        const closeBtn = document.getElementById('fplk-close-btn');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                isExpanded = false;
                renderWidget();
            });
        }

        const reextractBtn = document.getElementById('fplk-reextract-btn');
        if (reextractBtn) {
            reextractBtn.addEventListener('click', () => {
                runExtraction();
                if (currentExtraction && currentExtraction.valid) {
                    triggerPrediction();
                } else {
                    renderWidget();
                }
            });
        }

        const evalBtn = document.getElementById('fplk-eval-btn');
        if (evalBtn) {
            evalBtn.addEventListener('click', () => {
                readFormInputsIntoData();
                triggerPrediction();
            });
        }
    }

    function readFormInputsIntoData() {
        if (!currentExtraction) currentExtraction = { category: 'gpu', data: {} };
        const cat = currentExtraction.category || 'gpu';
        const d = currentExtraction.data || {};

        const priceEl = document.getElementById('fplk-input-price');
        if (priceEl && priceEl.value) d.listed_price = parseFloat(priceEl.value);

        if (cat === 'gpu') {
            const mEl = document.getElementById('fplk-input-gpu-model');
            const vEl = document.getElementById('fplk-select-gpu-vram');
            const bEl = document.getElementById('fplk-select-gpu-brand');

            if (mEl) d.model = mEl.value.trim();
            if (vEl) d.vram_gb = parseFloat(vEl.value);
            if (bEl) d.brand = bEl.value;
            d.stock = 'In Stock';
        } else if (cat === 'mobile') {
            const tEl = document.getElementById('fplk-mobile-type');
            const bEl = document.getElementById('fplk-mobile-brand');
            const mEl = document.getElementById('fplk-mobile-model');
            const sEl = document.getElementById('fplk-mobile-storage');
            const rEl = document.getElementById('fplk-mobile-ram');

            if (tEl) d.phone_type = tEl.value;
            if (bEl) d.brand = bEl.value.trim();
            if (mEl) d.model = mEl.value.trim();
            if (sEl) d.storage_gb = parseFloat(sEl.value) || 128;
            if (rEl) d.ram_gb = parseFloat(rEl.value) || 6;
            d.warranty_days = 0;
        } else if (cat === 'vehicle') {
            const mEl = document.getElementById('fplk-vehicle-model');
            const yEl = document.getElementById('fplk-vehicle-year');
            const vEl = document.getElementById('fplk-vehicle-variant');

            if (mEl) d.model = mEl.value;
            if (yEl) d.model_year = parseInt(yEl.value, 10) || 2015;
            if (vEl) d.variant = vEl.value.trim();
            d.transmission = 'Automatic';
            d.fuel_type = 'Petrol';
        } else if (cat === 'electronics') {
            const bEl = document.getElementById('fplk-elec-brand');
            const mEl = document.getElementById('fplk-elec-model');

            if (bEl) d.brand = bEl.value.trim();
            if (mEl) d.model = mEl.value.trim();
            d.category = 'laptop';
            d.ram = 8;
            d.storage = 256;
        }

        currentExtraction.data = d;
    }

    async function triggerPrediction() {
        if (!currentExtraction || !currentExtraction.data) return;
        const cat = currentExtraction.category || 'gpu';
        const payload = currentExtraction.data;

        const evalBtn = document.getElementById('fplk-eval-btn');
        if (evalBtn) {
            evalBtn.innerHTML = `<span class="fplk-spinner"></span> Checking ML Valuation...`;
            evalBtn.disabled = true;
        }

        // Show loading in inline badge
        if (inlineBadgeRoot) {
            const iconUrl = chrome.runtime.getURL('icon.png');
            inlineBadgeRoot.innerHTML = `
                <div class="fplk-inline-badge" title="Valuating...">
                    <span class="fplk-inline-brand">
                        <img src="${iconUrl}" width="14" height="14" alt="FairPriceLK" style="border-radius: 3px; object-fit: contain; vertical-align: middle;">
                        Valuating...
                    </span>
                    <span class="fplk-inline-spinner"></span>
                </div>
            `;
        }

        // Send message to background script to bypass Chrome HTTPS -> HTTP Mixed-Content restriction
        chrome.runtime.sendMessage({
            action: "predict_price",
            category: cat,
            payload: payload
        }, (response) => {
            if (chrome.runtime.lastError) {
                console.warn("Runtime message error:", chrome.runtime.lastError);
                handlePredictionFailure(chrome.runtime.lastError.message);
                return;
            }

            if (!response || !response.success) {
                const errMsg = response ? response.error : "No response from local server";
                handlePredictionFailure(errMsg);
                return;
            }

            cachedPrediction = response.data;
            currentExtraction.valid = true;
            renderWidget();
        });
    }

    function handlePredictionFailure(errMsg) {
        cachedPrediction = null;
        if (currentExtraction) {
            currentExtraction.valid = false;
            const isFailedToFetch = errMsg && (errMsg.includes("Failed to fetch") || errMsg.includes("NetworkError"));
            currentExtraction.error_message = isFailedToFetch 
                ? `Cannot connect to local backend at ${getApiBase()}. Please ensure start_all.py is running.`
                : `Prediction error: ${errMsg}`;
        }
        renderWidget();
    }

    // --- Message Listener for Extension Popup (Preserves 100% compatibility) ---
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
        if (request.action === "extract_details") {
            try {
                const extraction = runExtraction();
                const d = extraction ? extraction.data : {};
                sendResponse({
                    success: true,
                    data: {
                        title: d.title || "",
                        price: d.listed_price ? String(d.listed_price) : "",
                        brand: d.brand || "",
                        model: d.model || "",
                        vram: d.vram_gb ? String(d.vram_gb) : "",
                        category: extraction ? extraction.category : "gpu",
                        valid: extraction ? extraction.valid : false,
                        error_message: extraction ? extraction.error_message : null
                    }
                });
            } catch (err) {
                sendResponse({ success: false, error: err.message });
            }
        }
        return true;
    });

    // Initialize on DOM load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initOnPageWidget);
    } else {
        initOnPageWidget();
    }

})();
