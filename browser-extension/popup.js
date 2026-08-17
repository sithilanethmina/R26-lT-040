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

    // Vehicle API URL — point directly to local vehicle microservice to bypass gateway timeout/IPv6 issues
    const VEHICLE_DEV_URL = 'http://127.0.0.1:8003';

    // Default GPU Metadata
    let GPU_METADATA = {
        models: [],
        brands: ["Any", "ASUS", "MSI", "GIGABYTE", "ZOTAC", "GALAX", "PALIT", "SAPPHIRE", "ASROCK", "POWERCOLOR", "COLORFUL", "INNO3D", "PNY", "EVGA", "EMTEK", "NVIDIA", "AMD"],
        manufacturers: ["Any", "NVIDIA", "AMD", "Intel"]
    };

    // Vehicle metadata state
    let VEHICLE_METADATA = { brands: [], models_by_brand: {} };
    let CURRENT_VEHICLE_TYPE = 'cars'; // 'cars' | 'suvs'

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
            console.warn("Live GPU metadata fetch failed, keeping local fallback:", e);
            // Keep local metadata — do NOT rethrow
        }

        // Ensure every model entry has valid properties — isolated so one bad entry
        // cannot crash the rest of the extension startup.
        try {
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
        } catch (e) {
            console.warn("GPU model normalisation failed, autocomplete may be limited:", e);
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
            // Wait up to 6000ms. The gateway waits 2000ms *per downstream service* in some cases.
            // If downstream services are slow, gateway can take ~2-3 seconds to reply.
            const res = await fetch(`${GATEWAY_URL}/api/health`, { signal: AbortSignal.timeout(6000) });
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
    initVehicleMetadata('cars');

    // --- Vehicle Metadata & Dynamic Dropdowns ---
    async function initVehicleMetadata(vehicleType, isRetry = false, retryCount = 0) {
        CURRENT_VEHICLE_TYPE = vehicleType;
        const endpoint = vehicleType === 'suvs'
            ? `${VEHICLE_DEV_URL}/metadata/suv`
            : `${VEHICLE_DEV_URL}/metadata/cars`;

        const brandSelect = document.getElementById('vehicleBrandSelect');
        const modelSelect = document.getElementById('vehicleModelSelect');
        if (!isRetry) {
            if (brandSelect) brandSelect.innerHTML = '<option value="">Loading...</option>';
            if (modelSelect) modelSelect.innerHTML = '<option value="">Loading...</option>';
        }

        try {
            const res = await fetch(endpoint, { signal: AbortSignal.timeout(5000) });

            // 503 = service still starting up — schedule automatic retries (max 3)
            if (res.status === 503) {
                if (retryCount < 3) {
                    console.warn(`Vehicle metadata returned 503 — service starting up. Retrying in 8s (attempt ${retryCount + 1}/3)…`);
                    if (brandSelect) brandSelect.innerHTML = `<option value="">Service loading, retrying (${retryCount + 1}/3)…</option>`;
                    if (modelSelect) modelSelect.innerHTML = '<option value=""></option>';
                    setTimeout(() => initVehicleMetadata(vehicleType, true, retryCount + 1), 8000);
                    return; 
                } else {
                    console.warn(`Vehicle metadata returned 503 — gave up after 3 retries.`);
                    throw new Error("Service Unavailable");
                }
            }

            if (!res.ok) throw new Error(`Vehicle metadata request failed with status ${res.status}`);
            const data = await res.json();

            // Normalise both JSON shapes into a single internal structure:
            //   Backend may return { "brands": [...], "models": { "Brand": [...] } }
            //                   OR { "brands": [...], "models_by_brand": { "Brand": [...] } }
            if (data && Array.isArray(data.brands)) {
                VEHICLE_METADATA = {
                    brands: data.brands,
                    // prefer 'models', fall back to 'models_by_brand'
                    models_by_brand: data.models || data.models_by_brand || {}
                };
            } else {
                VEHICLE_METADATA = { brands: [], models_by_brand: {} };
            }
        } catch (e) {
            console.warn('Vehicle metadata fetch failed — dropdowns will be empty:', e);
            VEHICLE_METADATA = { brands: [], models_by_brand: {} };
            // Do NOT rethrow — let other category initialisations continue
        }

        populateVehicleBrands();
    }

    function populateVehicleBrands() {
        const brandSelect = document.getElementById('vehicleBrandSelect');
        if (!brandSelect) return;

        const brands = VEHICLE_METADATA.brands || [];
        brandSelect.innerHTML = '';

        if (brands.length === 0) {
            brandSelect.innerHTML = '<option value="">No data available</option>';
            return;
        }

        // brands may be plain strings or objects with a 'name' key
        brands.forEach(b => {
            const val = typeof b === 'string' ? b : (b.name || String(b));
            const opt = document.createElement('option');
            opt.value = val;
            opt.textContent = val;
            brandSelect.appendChild(opt);
        });

        // Trigger model population for the first brand
        populateVehicleModels(brandSelect.value);
    }

    function populateVehicleModels(brand) {
        const modelSelect = document.getElementById('vehicleModelSelect');
        if (!modelSelect) return;

        modelSelect.innerHTML = '';

        const byBrand = VEHICLE_METADATA.models_by_brand || {};
        const models = byBrand[brand] || [];

        if (models.length === 0) {
            modelSelect.innerHTML = '<option value="">No models available</option>';
            return;
        }

        models.forEach(m => {
            const val = typeof m === 'string' ? m : (m.name || String(m));
            const opt = document.createElement('option');
            opt.value = val;
            opt.textContent = val;
            modelSelect.appendChild(opt);
        });
    }

    function updateVehicleTypeUI(vehicleType) {
        const engineCCGroup = document.getElementById('vehicleEngineCCGroup');
        if (engineCCGroup) engineCCGroup.classList.remove('hidden'); // Keep visible for BOTH Cars and SUVs
        if (vehicleType === 'suvs') {
            predictBtn.textContent = 'Predict SUV Price';
        } else {
            predictBtn.textContent = 'Predict Car Price';
        }
    }

    // Listen to vehicle type changes
    const vehicleTypeSelect = document.getElementById('vehicleTypeSelect');
    if (vehicleTypeSelect) {
        vehicleTypeSelect.addEventListener('change', (e) => {
            const vt = e.target.value;
            updateVehicleTypeUI(vt);
            initVehicleMetadata(vt);
        });
        // Set initial state: Cars selected, engine CC hidden
        updateVehicleTypeUI('cars');
    }

    // Listen to vehicle brand changes to refresh models
    const vehicleBrandSelect = document.getElementById('vehicleBrandSelect');
    if (vehicleBrandSelect) {
        vehicleBrandSelect.addEventListener('change', (e) => {
            populateVehicleModels(e.target.value);
        });
    }

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
        // Set a meaningful button label for every category
        const btnLabels = {
            gpu:         'Predict GPU Price',
            mobile:      'Predict Mobile Price',
            electronics: 'Check Electronics Price'
        };
        if (selected === 'vehicle') {
            const vt = document.getElementById('vehicleTypeSelect');
            updateVehicleTypeUI(vt ? vt.value : 'cars');
        } else {
            predictBtn.textContent = btnLabels[selected] || 'Check Price';
        }
    });

    // Fire once on load to set the initial button label for the default category (gpu)
    categorySelect.dispatchEvent(new Event('change'));

    // --- Extraction ---
    extractBtn.addEventListener('click', async () => {
        setLoading(extractBtn, true, 'Extracting...');
        hideMessages();
        
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            
            await chrome.scripting.executeScript({
                target: { tabId: tab.id },
                files: ['content.js']
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
            // For vehicle, route to the correct sub-type endpoint
            let predictEndpoint = `${GATEWAY_URL}/api/${category}/predict`;
            if (category === 'vehicle') {
                const vt = document.getElementById('vehicleTypeSelect');
                const vtVal = vt ? vt.value : 'cars';
                predictEndpoint = vtVal === 'suvs'
                    ? `${VEHICLE_DEV_URL}/api/predict/suv`
                    : `${VEHICLE_DEV_URL}/api/predict`;
            }

            const response = await fetch(predictEndpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errData = await response.json().catch(()=>({}));
                throw new Error(errData.detail || "Gateway error. Ensure services are running.");
            }

            const data = await response.json();
            
            let predictedPrice = 0;
            let modelUsed = "";
            
            if (category === 'gpu') {
                predictedPrice = data.predicted_price;
                modelUsed = (data.best_model_used || "GPU Model").replace('_', ' ').toUpperCase();
            } else if (category === 'mobile') {
                predictedPrice = data.predicted_price;
                modelUsed = "Mobile Model (" + data.phone_type + ")";
            } else if (category === 'vehicle') {
                // Both /api/vehicle/predict and /api/suv/predict return a flat
                // PredictResponse: { predicted_price, model_used, confidence, ... }
                predictedPrice = data.predicted_price;
                const confidence = data.confidence ? ` · ${data.confidence} confidence` : '';
                modelUsed = (data.model_used || 'Vehicle Model') + confidence;
            } else if (category === 'electronics') {
                if (typeof data.price === 'string') {
                    predictedPrice = parseFloat(data.price.replace(/[^0-9.]/g, ''));
                } else {
                    predictedPrice = data.price;
                }
                modelUsed = data.model_name || "Electronics Model";
            }

            displayResults(predictedPrice, listedPrice, modelUsed);
            
        } catch (err) {
            showError(err.message);
        } finally {
            // Restore correct button text based on active category
            const activeCat = categorySelect.value;
            if (activeCat === 'vehicle') {
                const vt = document.getElementById('vehicleTypeSelect');
                updateVehicleTypeUI(vt ? vt.value : 'cars');
            } else {
                setLoading(predictBtn, false, 'Check Price');
            }
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

            return {
                model: matched.model,
                vram_gb: vramVal,
                brand: brandVal || 'Any',
                manufacturer: manufVal || 'Any',
                stock: 'In Stock'
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
            const vt = document.getElementById('vehicleTypeSelect');
            const vehicleType = vt ? vt.value : 'cars';
            const brandVal = document.getElementById('vehicleBrandSelect')?.value || '';
            const modelVal = document.getElementById('vehicleModelSelect')?.value || '';
            const variant = document.getElementById('vehicleVariantInput').value.trim();
            const mileageVal = parseInt(document.getElementById('vehicleMileageInput')?.value) || 0;

            if (!brandVal) { showError("Please select a Brand."); return null; }
            if (!modelVal) { showError("Please select a Model."); return null; }

            const basePayload = {
                brand: brandVal,
                model: modelVal,
                model_year: parseInt(document.getElementById('vehicleYearInput').value) || 2015,
                mileage_km: mileageVal,
                variant: variant,
                transmission: document.getElementById('vehicleTransmissionSelect').value,
                fuel_type: document.getElementById('vehicleFuelSelect').value
            };

            const engineCC = parseInt(document.getElementById('vehicleEngineCCInput')?.value);
            if (engineCC && engineCC > 0) {
                basePayload.engine_cc = engineCC;
            } else if (vehicleType === 'suvs') {
                showError("Engine Capacity (CC) is required for SUVs."); 
                return null;
            }

            return basePayload;
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

    async function populateForm(data) {
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

            let matched = (GPU_METADATA.models || []).find(m => {
                if (!m || !m.model) return false;
                const mNorm = m.normalized || normalizeGpuModel(m.model);
                return (normCandidate && mNorm && normCandidate.includes(mNorm)) || 
                       (mNorm && mNorm === normCandidate) ||
                       (data.model && m.model.toUpperCase() === data.model.toUpperCase());
            });

            if (!matched && data.title) {
                const titleNorm = normalizeGpuModel(data.title);
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
            // -- Step 0: ensure the category dropdown shows 'vehicle' and the form is visible --
            if (categorySelect.value !== 'vehicle') {
                categorySelect.value = 'vehicle';
                categorySelect.dispatchEvent(new Event('change'));
            }

            // -- Step 0.5: set vehicle type and wait for metadata if it changed --
            const vtSelect = document.getElementById('vehicleTypeSelect');
            if (data.vehicle_type && vtSelect && vtSelect.value !== data.vehicle_type) {
                vtSelect.value = data.vehicle_type;
                updateVehicleTypeUI(data.vehicle_type);
                await initVehicleMetadata(data.vehicle_type);
            }

            // -- Step 1: set Brand and dispatch 'change' to trigger model list population --
            const bs = document.getElementById('vehicleBrandSelect');
            if (data.brand && bs) {
                // Try exact match first, then case-insensitive
                const options = Array.from(bs.options);
                const match = options.find(o => o.value === data.brand)
                           || options.find(o => o.value.toLowerCase() === data.brand.toLowerCase());
                if (match) {
                    bs.value = match.value;
                } else {
                    bs.value = data.brand; // set anyway; populateVehicleModels will handle gracefully
                }
                bs.dispatchEvent(new Event('change')); // triggers populateVehicleModels
            }

            // -- Step 2: after a short delay (model list repopulates async), set all other fields --
            setTimeout(() => {
                // Model
                if (data.model) {
                    const ms = document.getElementById('vehicleModelSelect');
                    if (ms) {
                        const opts = Array.from(ms.options);
                        let mMatch = opts.find(o => o.value === data.model)
                                  || opts.find(o => o.value.toLowerCase() === data.model.toLowerCase());
                        if (!mMatch) {
                            // Fallback: check if extracted model contains option value, or vice versa
                            mMatch = opts.find(o => data.model.toLowerCase().includes(o.value.toLowerCase())) ||
                                     opts.find(o => o.value.toLowerCase().includes(data.model.toLowerCase()));
                        }
                        ms.value = mMatch ? mMatch.value : data.model;
                    }
                }
                // Year
                if (data.year) {
                    const yi = document.getElementById('vehicleYearInput');
                    if (yi) yi.value = parseInt(data.year) || '';
                }
                // Mileage
                if (data.mileage) {
                    const mi = document.getElementById('vehicleMileageInput');
                    if (mi) mi.value = parseInt(data.mileage) || '';
                }
                // Transmission
                if (data.transmission) {
                    const ts = document.getElementById('vehicleTransmissionSelect');
                    if (ts) ts.value = data.transmission;
                }
                // Fuel type
                if (data.fuel_type) {
                    const fs = document.getElementById('vehicleFuelSelect');
                    if (fs) fs.value = data.fuel_type;
                }
                // Engine CC
                if (data.engine_cc) {
                    const eci = document.getElementById('vehicleEngineCCInput');
                    if (eci) eci.value = parseInt(data.engine_cc) || '';
                }
                // Price
                if (data.price) {
                    priceInput.value = data.price;
                }

                // -- Step 3: auto-trigger prediction --
                const predictBtn = document.getElementById('predictBtn');
                if (predictBtn && !predictBtn.disabled) {
                    predictBtn.click();
                }
            }, 350); // 350 ms gives populateVehicleModels time to render
        } else if (cat === 'electronics') {
            if (data.brand) document.getElementById('elecBrandInput').value = data.brand;
            if (data.model) document.getElementById('elecModelInput').value = data.model;
        }
    }

    function displayResults(predictedPrice, listedPrice, modelUsed) {
        if (!predictedPrice) {
            showError("Prediction failed. Try adjusting inputs.");
            return;
        }

        predictedPriceEl.innerText = predictedPrice.toLocaleString('en-LK', {maximumFractionDigits: 0});
        modelUsedName.innerText = "Model used: " + modelUsed;
        
        resultSection.classList.remove('hidden');

        if (listedPrice && !isNaN(listedPrice) && listedPrice > 0) {
            const diff = listedPrice - predictedPrice;
            const diffPercent = (diff / predictedPrice) * 100;
            
            const diffFormatted = Math.abs(diff).toLocaleString('en-LK', {maximumFractionDigits: 0});
            if (diff > 0) {
                priceDiffEl.innerText = `+Rs.${diffFormatted} (${diffPercent.toFixed(1)}%)`;
            } else {
                priceDiffEl.innerText = `-Rs.${diffFormatted} (${Math.abs(diffPercent).toFixed(1)}%)`;
            }

            fairnessBadge.className = 'badge'; // reset
            if (diffPercent > 15) {
                fairnessBadge.innerText = "OVERPRICED";
                fairnessBadge.classList.add('overpriced');
            } else if (diffPercent < -25) {
                fairnessBadge.innerText = "SCAM RISK";
                fairnessBadge.classList.add('scam');
            } else {
                fairnessBadge.innerText = "FAIR PRICE";
                fairnessBadge.classList.add('fair');
            }
        } else {
            priceDiffEl.innerText = "No listing price provided";
            fairnessBadge.className = 'badge hidden';
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
