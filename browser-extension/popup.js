document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const extractBtn = document.getElementById('extractBtn');
    const predictBtn = document.getElementById('predictBtn');
    const categorySelect = document.getElementById('categorySelect');
    const priceInput = document.getElementById('priceInput');
    const extractionStatus = document.getElementById('extractionStatus');
    const errorSection = document.getElementById('errorSection');
    const resultSection = document.getElementById('resultSection');
    
    // Result Elements
    const predictedPriceEl = document.getElementById('predictedPrice');
    const fairnessBadge = document.getElementById('fairnessBadge');
    const priceDiffEl = document.getElementById('priceDiff');
    const modelUsedName = document.getElementById('modelUsedName');

    // Status Footer
    const serverStatusDot = document.getElementById('serverStatusDot');
    const serverStatusText = document.getElementById('serverStatusText');

    // Gateway URL (Single source of truth from config.js)
    const GATEWAY_URL = CONFIG.API_BASE_URL;

    // Default GPU Metadata
    let GPU_METADATA = {
        models: [],
        brands: ["Any", "ASUS", "MSI", "GIGABYTE", "ZOTAC", "GALAX", "PALIT", "SAPPHIRE", "ASROCK", "POWERCOLOR", "COLORFUL", "INNO3D", "PNY", "EVGA", "EMTEK", "NVIDIA", "AMD"],
        manufacturers: ["Any", "NVIDIA", "AMD", "Intel"]
    };

    // --- Normalization (Matches Streamlit gpu_price_predictor.pipeline.normalize_model) ---
    function normalizeGpuModel(raw) {
        if (!raw) return "";
        let text = String(raw).replace(/\s+/g, " ").trim().toUpperCase();
        if (text === "UNKNOWN") return "UNKNOWN";
        
        text = text.replace(/-/g, " ");
        text = text.replace(/\bGEFORCE\b/g, "");
        text = text.replace(/\bRADEON\b/g, "");
        text = text.replace(/\bNVIDIA\b/g, "");
        text = text.replace(/\bAMD\b/g, "");
        text = text.replace(/\bINTEL ARC\b/g, "ARC");
        text = text.replace(/([A-Z]+)\s*(\d{3,4})([A-Z]*)/g, "$1 $2 $3");
        text = text.replace(/\b(TI|XT|XTX|SUPER)\b/g, " $1");
        text = text.replace(/\s+/g, " ").trim();
        return text;
    }

    // --- Load GPU Metadata (Local JSON with live Gateway Sync) ---
    async function initGpuMetadata() {
        try {
            const localRes = await fetch('gpu_metadata.json').catch(() => null);
            if (localRes && localRes.ok) {
                GPU_METADATA = await localRes.json();
            }
        } catch (e) {
            console.warn("Local gpu_metadata.json load failed:", e);
        }

        try {
            const liveRes = await fetch(`${GATEWAY_URL}/api/gpu/metadata`, { signal: AbortSignal.timeout(2000) }).catch(() => null);
            if (liveRes && liveRes.ok) {
                const liveData = await liveRes.json();
                if (liveData && liveData.models && liveData.models.length > 0) {
                    GPU_METADATA = liveData;
                }
            }
        } catch (e) {
            // keep local metadata
        }

        // Ensure every model entry has valid properties
        if (GPU_METADATA && Array.isArray(GPU_METADATA.models)) {
            GPU_METADATA.models.forEach(m => {
                if (m) {
                    m.model = m.model || "";
                    m.normalized = m.normalized || normalizeGpuModel(m.model);
                    m.manufacturer = m.manufacturer || "NVIDIA";
                    m.default_vram = m.default_vram || 8.0;
                    m.valid_vrams = m.valid_vrams || [m.default_vram];
                    m.brands = m.brands || [];
                }
            });
        }

        populateGpuBrands();
        setupGpuAutocomplete();
    }

    function populateGpuBrands(brandsList) {
        const brandSelect = document.getElementById('gpuBrandSelect');
        if (!brandSelect) return;
        
        const currentVal = brandSelect.value;
        const list = (brandsList && brandsList.length > 0) ? brandsList : (GPU_METADATA.brands || ["Any"]);
        
        brandSelect.innerHTML = '';
        const unique = Array.from(new Set(["Any", ...list]));
        unique.forEach(b => {
            const opt = document.createElement('option');
            opt.value = b;
            opt.textContent = b;
            brandSelect.appendChild(opt);
        });

        if (unique.includes(currentVal)) {
            brandSelect.value = currentVal;
        } else {
            brandSelect.value = "Any";
        }
    }

    function updateGpuVramOptions(modelEntry, preferredVram) {
        const vramSelect = document.getElementById('gpuVramSelect');
        if (!vramSelect) return;

        vramSelect.innerHTML = '';
        const vrams = modelEntry && modelEntry.valid_vrams && modelEntry.valid_vrams.length > 0
            ? modelEntry.valid_vrams
            : [modelEntry?.default_vram || 8.0];

        vrams.forEach(v => {
            const opt = document.createElement('option');
            opt.value = String(v);
            opt.textContent = `${v % 1 === 0 ? Math.round(v) : v} GB`;
            vramSelect.appendChild(opt);
        });

        if (preferredVram && vrams.map(Number).includes(Number(preferredVram))) {
            vramSelect.value = String(preferredVram);
        } else if (modelEntry && modelEntry.default_vram) {
            vramSelect.value = String(modelEntry.default_vram);
        } else if (vrams.length > 0) {
            vramSelect.value = String(vrams[0]);
        }
    }

    function setupGpuAutocomplete() {
        const modelInput = document.getElementById('gpuModelInput');
        const suggestionsBox = document.getElementById('gpuModelSuggestions');
        const modelError = document.getElementById('gpuModelError');
        const manufacturerSelect = document.getElementById('gpuManufacturerSelect');
        if (!modelInput || !suggestionsBox) return;

        let activeIndex = -1;

        function renderSuggestions(matches) {
            suggestionsBox.innerHTML = '';
            activeIndex = -1;

            if (!matches || matches.length === 0) {
                suggestionsBox.classList.add('hidden');
                return;
            }

            matches.slice(0, 10).forEach((item) => {
                const div = document.createElement('div');
                div.className = 'suggestion-item';
                div.innerHTML = `
                    <span>${item.model}</span>
                    <span class="suggestion-meta">${item.manufacturer || 'GPU'} · ${item.default_vram || 8}GB</span>
                `;
                div.addEventListener('mousedown', (e) => {
                    e.preventDefault();
                    selectModel(item);
                });
                suggestionsBox.appendChild(div);
            });

            suggestionsBox.classList.remove('hidden');
        }

        function selectModel(modelEntry) {
            modelInput.value = modelEntry.model;
            modelInput.classList.remove('input-invalid');
            if (modelError) modelError.classList.add('hidden');
            suggestionsBox.classList.add('hidden');

            updateGpuVramOptions(modelEntry);
            populateGpuBrands(modelEntry.brands);

            if (modelEntry.manufacturer && manufacturerSelect) {
                manufacturerSelect.value = modelEntry.manufacturer;
            }
        }

        function validateModelInput() {
            const val = modelInput.value.trim();
            if (!val) {
                modelInput.classList.remove('input-invalid');
                if (modelError) modelError.classList.add('hidden');
                return null;
            }

            const normVal = normalizeGpuModel(val);
            const match = (GPU_METADATA.models || []).find(m => {
                if (!m || !m.model) return false;
                const mNorm = m.normalized || normalizeGpuModel(m.model);
                return m.model.toUpperCase() === val.toUpperCase() || (mNorm && mNorm === normVal);
            });

            if (match) {
                modelInput.classList.remove('input-invalid');
                if (modelError) modelError.classList.add('hidden');
                modelInput.value = match.model;
                updateGpuVramOptions(match);
                populateGpuBrands(match.brands);
                if (match.manufacturer && manufacturerSelect) {
                    manufacturerSelect.value = match.manufacturer;
                }
                return match;
            } else {
                modelInput.classList.add('input-invalid');
                if (modelError) {
                    modelError.textContent = `Unrecognized GPU model: "${val}"`;
                    modelError.classList.remove('hidden');
                }
                return null;
            }
        }

        modelInput.addEventListener('input', (e) => {
            const query = e.target.value.trim().toUpperCase();
            const normQuery = normalizeGpuModel(query);

            if (!query) {
                renderSuggestions([]);
                modelInput.classList.remove('input-invalid');
                if (modelError) modelError.classList.add('hidden');
                return;
            }

            const selectedManuf = manufacturerSelect ? manufacturerSelect.value : 'Any';

            const matches = (GPU_METADATA.models || []).filter(m => {
                if (!m || !m.model) return false;
                if (selectedManuf !== 'Any' && m.manufacturer !== selectedManuf) {
                    return false;
                }
                const mUpper = (m.model || '').toUpperCase();
                const mNorm = m.normalized || normalizeGpuModel(m.model);
                return mUpper.includes(query) || (mNorm && mNorm.includes(normQuery)) || (normQuery && normQuery.includes(mNorm));
            });

            renderSuggestions(matches);
        });

        modelInput.addEventListener('keydown', (e) => {
            const items = suggestionsBox.querySelectorAll('.suggestion-item');
            if (items.length === 0 || suggestionsBox.classList.contains('hidden')) return;

            if (e.key === 'ArrowDown') {
                e.preventDefault();
                activeIndex = (activeIndex + 1) % items.length;
                updateActiveItem(items);
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                activeIndex = (activeIndex - 1 + items.length) % items.length;
                updateActiveItem(items);
            } else if (e.key === 'Enter') {
                e.preventDefault();
                if (activeIndex >= 0 && items[activeIndex]) {
                    items[activeIndex].dispatchEvent(new MouseEvent('mousedown'));
                } else if (items.length > 0) {
                    items[0].dispatchEvent(new MouseEvent('mousedown'));
                }
            } else if (e.key === 'Escape') {
                suggestionsBox.classList.add('hidden');
            }
        });

        function updateActiveItem(items) {
            items.forEach((it, idx) => {
                if (idx === activeIndex) {
                    it.classList.add('active');
                    it.scrollIntoView({ block: 'nearest' });
                } else {
                    it.classList.remove('active');
                }
            });
        }

        modelInput.addEventListener('blur', () => {
            setTimeout(() => {
                suggestionsBox.classList.add('hidden');
                validateModelInput();
            }, 200);
        });

        if (manufacturerSelect) {
            manufacturerSelect.addEventListener('change', () => {
                const currentVal = modelInput.value.trim();
                if (currentVal) {
                    validateModelInput();
                }
            });
        }
    }

    // --- Check Gateway Health ---
    async function checkHealth() {
        try {
            const res = await fetch(`${GATEWAY_URL}/api/health`, { signal: AbortSignal.timeout(2000) });
            if (res.ok) {
                serverStatusDot.className = 'dot online';
                serverStatusText.innerText = 'Server Connected';
            } else {
                throw new Error();
            }
        } catch {
            serverStatusDot.className = 'dot offline';
            serverStatusText.innerText = 'Server Offline (localhost:8000)';
        }
    }
    checkHealth();
    initGpuMetadata();

    // --- Category Switching ---
    const forms = {
        'gpu': document.getElementById('gpuForm'),
        'mobile': document.getElementById('mobileForm'),
        'vehicle': document.getElementById('vehicleForm'),
        'electronics': document.getElementById('electronicsForm')
    };

    categorySelect.addEventListener('change', (e) => {
        const selected = e.target.value;
        Object.values(forms).forEach(f => f.classList.add('hidden'));
        if (forms[selected]) {
            forms[selected].classList.remove('hidden');
        }
    });

    // --- Extraction ---
    extractBtn.addEventListener('click', async () => {
        setLoading(extractBtn, true, 'Extracting...');
        hideMessages();
        
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            
            await chrome.scripting.executeScript({
                target: { tabId: tab.id },
                files: [
                    'config.js',
                    'extractors/gpu.js',
                    'extractors/mobile.js',
                    'extractors/vehicle.js',
                    'extractors/electronics.js',
                    'extractors/index.js',
                    'content.js'
                ]
            }).catch(() => {});

            chrome.tabs.sendMessage(tab.id, { action: "extract_details" }, (response) => {
                setLoading(extractBtn, false, '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3"/></svg> Extract');
                
                if (chrome.runtime.lastError) {
                    showError("Could not connect to the page. Refresh and try again.");
                    return;
                }

                if (response && response.success) {
                    populateForm(response.data);
                    showStatus("Details extracted from page.");
                } else {
                    showError("Extraction failed.");
                }
            });
        } catch (err) {
            setLoading(extractBtn, false, 'Extract');
            showError("Error accessing page.");
        }
    });

    // --- Prediction ---
    predictBtn.addEventListener('click', async () => {
        hideMessages();
        resultSection.classList.add('hidden');
        
        const category = categorySelect.value;
        const listedPrice = parseFloat(priceInput.value);
        const payload = buildPayload(category);

        if (!payload) return; // Validation failed inside buildPayload

        setLoading(predictBtn, true, 'Checking...');

        try {
            const response = await fetch(`${GATEWAY_URL}/api/${category}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errData = await response.json().catch(()=>({}));
                throw new Error(errData.detail || "Gateway error. Ensure services are running.");
            }

            const data = await response.json();
            displayResults(data, listedPrice, category);
            
        } catch (err) {
            showError(err.message);
        } finally {
            setLoading(predictBtn, false, 'Check Price');
        }
    });

    // --- Helpers ---
    function buildPayload(category) {
        if (category === 'gpu') {
            const modelInput = document.getElementById('gpuModelInput');
            const modelVal = modelInput ? modelInput.value.trim() : "";
            if (!modelVal) { 
                showError("GPU Model is required."); 
                if (modelInput) modelInput.classList.add('input-invalid');
                return null; 
            }

            const normVal = normalizeGpuModel(modelVal);
            const matched = (GPU_METADATA.models || []).find(m => {
                if (!m || !m.model) return false;
                const mNorm = m.normalized || normalizeGpuModel(m.model);
                return m.model.toUpperCase() === modelVal.toUpperCase() || (mNorm && mNorm === normVal);
            });

            if (!matched) {
                showError(`Unknown GPU model: "${modelVal}". Please select a recognized model.`);
                if (modelInput) modelInput.classList.add('input-invalid');
                return null; 
            }

            const vramSelect = document.getElementById('gpuVramSelect');
            const vramVal = parseFloat(vramSelect ? vramSelect.value : "8");
            if (isNaN(vramVal) || vramVal <= 0) {
                showError("Valid VRAM (GB) is required.");
                return null;
            }

            const brandSelect = document.getElementById('gpuBrandSelect');
            const brandVal = brandSelect ? brandSelect.value : 'Any';

            const manufSelect = document.getElementById('gpuManufacturerSelect');
            const manufVal = manufSelect ? manufSelect.value : 'Any';

            const listedPrice = parseFloat(priceInput.value);

            return {
                model: matched.model,
                vram_gb: vramVal,
                brand: brandVal || 'Any',
                manufacturer: manufVal || 'Any',
                stock: 'In Stock',
                listed_price: (!isNaN(listedPrice) && listedPrice > 0) ? listedPrice : null
            };
        } else if (category === 'mobile') {
            const brand = document.getElementById('mobileBrandInput').value.trim();
            const model = document.getElementById('mobileModelInput').value.trim();
            if (!brand || !model) { showError("Brand and Model are required."); return null; }
            return {
                phone_type: document.getElementById('mobileTypeSelect').value,
                brand: brand,
                model: model,
                storage_gb: parseFloat(document.getElementById('mobileStorageInput').value) || 128,
                ram_gb: parseFloat(document.getElementById('mobileRamInput').value) || 6,
                warranty_days: parseFloat(document.getElementById('mobileWarrantyInput').value) || 0
            };
        } else if (category === 'vehicle') {
            const variant = document.getElementById('vehicleVariantInput').value.trim();
            if (!variant) { showError("Variant is required."); return null; }
            return {
                model: document.getElementById('vehicleModelSelect').value,
                model_year: parseInt(document.getElementById('vehicleYearInput').value) || 2015,
                variant: variant,
                transmission: document.getElementById('vehicleTransmissionSelect').value,
                fuel_type: document.getElementById('vehicleFuelSelect').value
            };
        } else if (category === 'electronics') {
            const cat = document.getElementById('elecCategorySelect').value;
            const brand = document.getElementById('elecBrandInput').value.trim();
            const model = document.getElementById('elecModelInput').value.trim();
            if (!brand || !model) { showError("Brand and Model are required."); return null; }
            return {
                category: cat,
                brand: brand,
                model: model,
                ram: parseFloat(document.getElementById('elecRamInput').value) || 8,
                storage: parseFloat(document.getElementById('elecStorageInput').value) || 256
            };
        }
        return null;
    }

    function populateForm(data) {
        if (!data) return;
        if (data.price) priceInput.value = data.price;
        
        if (data.category && forms[data.category]) {
            categorySelect.value = data.category;
            categorySelect.dispatchEvent(new Event('change'));
        }
        
        const cat = categorySelect.value;
        if (cat === 'gpu') {
            const gpuInput = document.getElementById('gpuModelInput');
            const vramSelect = document.getElementById('gpuVramSelect');
            const brandSelect = document.getElementById('gpuBrandSelect');
            const manufSelect = document.getElementById('gpuManufacturerSelect');

            const rawModelCandidate = data.model || data.title || "";
            const normCandidate = normalizeGpuModel(rawModelCandidate);

            // 1. Try exact normalized match first
            let matched = (GPU_METADATA.models || []).find(m => {
                if (!m || !m.model) return false;
                const mNorm = m.normalized || normalizeGpuModel(m.model);
                return mNorm === normCandidate || (data.model && m.model.toUpperCase() === data.model.toUpperCase());
            });

            // 2. Try longest substring match (so "RX 6600 XT" matches before "RX 6600")
            if (!matched && (normCandidate || data.title)) {
                const titleNorm = normCandidate || normalizeGpuModel(data.title);
                const sortedByLen = [...(GPU_METADATA.models || [])].sort((a, b) => (b.model || '').length - (a.model || '').length);
                matched = sortedByLen.find(m => {
                    if (!m || !m.model) return false;
                    const mNorm = m.normalized || normalizeGpuModel(m.model);
                    return titleNorm && mNorm && titleNorm.includes(mNorm);
                });
            }

            if (matched) {
                if (gpuInput) {
                    gpuInput.value = matched.model;
                    gpuInput.classList.remove('input-invalid');
                }
                const modelErr = document.getElementById('gpuModelError');
                if (modelErr) modelErr.classList.add('hidden');

                updateGpuVramOptions(matched, data.vram);
                populateGpuBrands(matched.brands);

                if (data.brand && brandSelect && Array.isArray(matched.brands) && matched.brands.map(b => b.toUpperCase()).includes(data.brand.toUpperCase())) {
                    brandSelect.value = data.brand.toUpperCase();
                }

                if (matched.manufacturer && manufSelect) {
                    manufSelect.value = matched.manufacturer;
                }

                const vramVal = vramSelect ? vramSelect.value : "";
                const brandVal = brandSelect ? brandSelect.value : "";
                showStatus(`Auto-detected ${matched.model} (${vramVal}GB, ${brandVal})`);
            } else if (data.model && gpuInput) {
                gpuInput.value = data.model;
            }
        } else if (cat === 'mobile') {
            if (data.brand) document.getElementById('mobileBrandInput').value = data.brand;
            if (data.model) document.getElementById('mobileModelInput').value = data.model;
            const text = ((data.title || "")).toLowerCase();
            if (text.includes("iphone") || text.includes("apple")) {
                document.getElementById('mobileTypeSelect').value = "iphone";
            } else {
                document.getElementById('mobileTypeSelect').value = "android";
            }
        } else if (cat === 'vehicle') {
            const text = ((data.title || "")).toLowerCase();
            if (text.includes("aqua")) document.getElementById('vehicleModelSelect').value = "Toyota Aqua";
            if (text.includes("alto")) document.getElementById('vehicleModelSelect').value = "Suzuki Alto";
        } else if (cat === 'electronics') {
            if (data.brand) document.getElementById('elecBrandInput').value = data.brand;
            if (data.model) document.getElementById('elecModelInput').value = data.model;
        }
    }

    function displayResults(data, listedPrice, category) {
        if (!data) {
            showError("Prediction failed. Try adjusting inputs.");
            return;
        }

        const verdictDescEl = document.getElementById('verdictDescription');
        const displayListedPrice = document.getElementById('displayListedPrice');
        const detectedCategoryTag = document.getElementById('detectedCategoryTag');
        const detectedModelTitle = document.getElementById('detectedModelTitle');
        const verdictBanner = document.getElementById('verdictBanner');
        const rangeMarker = document.getElementById('rangeMarker');

        if (detectedCategoryTag) detectedCategoryTag.innerText = category.toUpperCase();
        if (detectedModelTitle) {
            const m = data.metadata ? (data.metadata.model_name || data.metadata.model) : (document.getElementById('gpuModelInput')?.value || category.toUpperCase());
            detectedModelTitle.innerText = m;
        }

        if (displayListedPrice) {
            displayListedPrice.innerText = listedPrice ? `Rs. ${listedPrice.toLocaleString('en-LK')}` : 'Not Specified';
        }

        if (verdictDescEl) {
            verdictDescEl.innerText = "";
            verdictDescEl.classList.add('hidden');
        }

        let lowerPrice = 0;
        let upperPrice = 0;
        let pointPrice = 0;
        let modelUsed = "";

        if (category === 'gpu' || (data.fair_market_range && data.fair_market_range.lower_price_lkr)) {
            lowerPrice = data.lower_price || (data.fair_market_range ? data.fair_market_range.lower_price_lkr : 0);
            upperPrice = data.upper_price || (data.fair_market_range ? data.fair_market_range.upper_price_lkr : 0);
            pointPrice = data.predicted_price || 0;
            modelUsed = (data.best_model_used || "Random Forest").replace('_', ' ').toUpperCase();
        } else if (category === 'mobile') {
            pointPrice = data.predicted_price || 0;
            lowerPrice = data.fair_market_range ? data.fair_market_range.lower_price_lkr : Math.round(pointPrice * 0.9);
            upperPrice = data.fair_market_range ? data.fair_market_range.upper_price_lkr : Math.round(pointPrice * 1.1);
            modelUsed = "Mobile Model (" + (data.phone_type || "Standard") + ")";
        } else if (category === 'vehicle') {
            if (data.predictions && data.predictions.length > 0) {
                pointPrice = data.predictions[0].predictedPrice || 0;
                modelUsed = data.predictions[0].name || "Vehicle Model";
            } else {
                pointPrice = data.predicted_price || 0;
                modelUsed = "Vehicle Model";
            }
            lowerPrice = data.fair_market_range ? data.fair_market_range.lower_price_lkr : Math.round(pointPrice * 0.92);
            upperPrice = data.fair_market_range ? data.fair_market_range.upper_price_lkr : Math.round(pointPrice * 1.08);
        } else if (category === 'electronics') {
            pointPrice = typeof data.price === 'string' ? parseFloat(data.price.replace(/[^0-9.]/g, '')) : (data.price || 0);
            modelUsed = data.model_name || "Electronics Model";
            lowerPrice = data.fair_market_range ? data.fair_market_range.lower_price_lkr : Math.round(pointPrice * 0.9);
            upperPrice = data.fair_market_range ? data.fair_market_range.upper_price_lkr : Math.round(pointPrice * 1.1);
        }

        // Display Estimated Range / Point Price
        if (upperPrice > lowerPrice && lowerPrice > 0) {
            predictedPriceEl.innerText = `Rs. ${Math.round(lowerPrice/1000)}k – ${Math.round(upperPrice/1000)}k`;
        } else if (pointPrice > 0) {
            predictedPriceEl.innerText = `Rs. ${Math.round(pointPrice).toLocaleString('en-LK')}`;
        } else {
            predictedPriceEl.innerText = "--";
        }

        // Calculate visual marker position
        if (rangeMarker && listedPrice && upperPrice > lowerPrice) {
            const span = (upperPrice - lowerPrice) * 1.5;
            const minBound = lowerPrice - (span * 0.25);
            const markerPos = Math.max(4, Math.min(96, ((listedPrice - minBound) / span) * 100));
            rangeMarker.style.left = `${markerPos}%`;
        }

        let modelFooterText = `Model: ${modelUsed} · Pt Est: Rs. ${pointPrice.toLocaleString('en-LK', {maximumFractionDigits: 0})}`;
        if (data.metadata && data.metadata.limited_data_warning) {
            modelFooterText += " (Limited Data)";
        }
        modelUsedName.innerText = modelFooterText;
        resultSection.classList.remove('hidden');

        // Evaluate Fairness
        let fairness = null;
        if (window.FairPriceLK_Fairness) {
            fairness = window.FairPriceLK_Fairness.evaluate(listedPrice, pointPrice, lowerPrice, upperPrice, category);
        }

        if (fairness && fairness.status === "OK") {
            if (verdictBanner) verdictBanner.className = 'verdict-banner ' + fairness.badgeClass;
            fairnessBadge.innerText = fairness.badgeText;
            priceDiffEl.innerText = fairness.score !== null ? `Score: ${fairness.score}/100` : '';

            if (verdictDescEl) {
                let descHtml = fairness.advice;
                if (fairness.actionAdvice) {
                    descHtml += `<div style="margin-top: 6px; font-weight: 500; font-size: 11px;">💡 ${fairness.actionAdvice}</div>`;
                }
                if (fairness.negotiationTarget) {
                    descHtml += `<div style="margin-top: 4px; font-weight: 600; color: #1e3a8a; font-size: 11px;">🎯 Counter-Offer: ${fairness.negotiationTarget}</div>`;
                }
                verdictDescEl.innerHTML = descHtml;
                verdictDescEl.classList.remove('hidden');
            }
        } else if (listedPrice && listedPrice > 0) {
            priceDiffEl.innerText = `Asking: Rs. ${listedPrice.toLocaleString('en-LK')}`;
            fairnessBadge.innerText = "Evaluated";
        } else {
            priceDiffEl.innerText = "Specify price to score";
            if (verdictBanner) verdictBanner.className = 'verdict-banner neutral';
            fairnessBadge.innerText = "Price Missing";
        }
    }

    function setLoading(btn, isLoading, text) {
        btn.innerHTML = text;
        btn.disabled = isLoading;
    }

    function showError(msg) {
        errorSection.innerText = msg;
        errorSection.classList.remove('hidden');
    }

    function showStatus(msg) {
        extractionStatus.innerText = msg;
        extractionStatus.classList.remove('hidden');
    }

    function hideMessages() {
        errorSection.classList.add('hidden');
        extractionStatus.classList.add('hidden');
    }
});
