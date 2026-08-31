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
    if (typeof CONFIG !== "undefined" && CONFIG.API_BASE_URL)
      return CONFIG.API_BASE_URL;
    if (window.CONFIG && window.CONFIG.API_BASE_URL)
      return window.CONFIG.API_BASE_URL;
    return "http://127.0.0.1:8000";
  }

  let currentExtraction = null;
  let embeddedCardRoot = null;
  let cachedPrediction = null;

  // --- High-level extraction runner ---
  function runExtraction() {
    if (
      !window.FairPriceLK_Extractors ||
      !window.FairPriceLK_Extractors.index
    ) {
      console.warn("FairPriceLK extractors not loaded yet.");
      return null;
    }
    currentExtraction = window.FairPriceLK_Extractors.index.extractAll();
    return currentExtraction;
  }

  // --- Create UI Elements ---
  function isItemDetailPage() {
    const href = window.location.href.toLowerCase();
    const hostname = window.location.hostname.toLowerCase();

    // The on-site embedded DOM card is explicitly restricted to ikman.lk and riyasewana.com
    const isIkman = hostname === "ikman.lk" || hostname.endsWith(".ikman.lk");
    const isRiyasewana = hostname === "riyasewana.com" || hostname.endsWith(".riyasewana.com");

    if (!isIkman && !isRiyasewana) {
      return false;
    }

    // 1. Must NOT be an aggregate category or search results listing page
    if (href.includes("/ads/") || href.includes("/ads?")) {
      // ikman detail pages have /en/ad/ or /si/ad/ (singular 'ad'), search feeds have /ads/ (plural 'ads')
      return false;
    }

    // 2. Riyasewana.com vehicle listing pages use /buy/<slug> pattern
    if (isRiyasewana) {
      return href.includes("/buy/");
    }

    // 3. Ikman.lk detail page check
    if (isIkman) {
      const hasAdPattern =
        href.includes("/ad/") ||
        href.includes("/en/ad/") ||
        href.includes("/si/ad/");

      const hasListingHeader =
        document.querySelector("h1") &&
        (document.querySelector('[class*="price"]') ||
          document.querySelector('div[data-testid="price"]') ||
          document.querySelector('span[data-testid="price"]'));

      return hasAdPattern || Boolean(hasListingHeader);
    }

    return false;
  }

  function findInsertionTarget() {
    // Priority 1: Price element or price container
    const priceEl =
      currentExtraction && currentExtraction.pageContext
        ? currentExtraction.pageContext.price_element
        : null;
    if (priceEl) {
      // If the price element is inside a container, insert after price or container
      const container =
        priceEl.closest('div[data-testid="price-section"]') ||
        priceEl.closest(".price-section") ||
        priceEl.closest("div") ||
        priceEl;
      return { element: container, position: "afterend" };
    }

    // Priority 2: Ikman / General marketplace ad details section or h1
    const titleEl = document.querySelector("h1");
    if (titleEl) {
      return { element: titleEl, position: "afterend" };
    }

    // Fallback: main content container or body
    const main =
      document.querySelector("main") ||
      document.querySelector("article") ||
      document.body;
    return { element: main, position: "beforeend" };
  }

  function isBrandNewCondition(conditionStr, titleStr = "", rawText = "", keyValues = {}) {
    if (window.FairPriceLK_Extractors && window.FairPriceLK_Extractors.index && window.FairPriceLK_Extractors.index.isBrandNewCondition) {
      return window.FairPriceLK_Extractors.index.isBrandNewCondition(conditionStr, titleStr, rawText, keyValues);
    }
    const cond = (conditionStr || "").toLowerCase().trim();
    const title = (titleStr || "").toLowerCase().trim();
    if (
      cond.includes("brand new") ||
      cond === "new" ||
      cond.startsWith("new") ||
      cond.includes("brand-new") ||
      cond.includes("unregistered") ||
      cond.includes("sealed") ||
      cond.includes("අලුත්") ||
      cond.includes("புதிய")
    ) {
      return true;
    }
    if (title.includes("brand new") && !cond.includes("used")) {
      return true;
    }
    return false;
  }

  function initOnPageWidget() {
    if (!isItemDetailPage()) return;

    runExtraction();
    renderEmbeddedCard();

    const data =
      currentExtraction && currentExtraction.data ? currentExtraction.data : {};
    const ext = currentExtraction || {};
    const isBrandNew =
      ext.is_brand_new !== undefined
        ? ext.is_brand_new
        : isBrandNewCondition(data.condition, data.title, ext.pageContext ? ext.pageContext.raw_text : "", ext.pageContext ? ext.pageContext.key_values : {});
    const isUnsupported =
      ext.category === "unsupported" || ext.is_unsupported_item;

    // If extraction is valid, NOT Brand New, and NOT an unsupported item, automatically trigger evaluation
    if (
      ext.valid &&
      !isBrandNew &&
      !isUnsupported
    ) {
      triggerPrediction();
    }
  }

  function renderEmbeddedCard(manualEstimateRequested = false) {
    if (!isItemDetailPage()) return;
    const target = findInsertionTarget();
    if (!target || !target.element) return;

    if (!embeddedCardRoot) {
      // Check if already in DOM
      const existing = document.getElementById("fairpricelk-embedded-card");
      if (existing) {
        embeddedCardRoot = existing;
      } else {
        embeddedCardRoot = document.createElement("div");
        embeddedCardRoot.id = "fairpricelk-embedded-card";
        embeddedCardRoot.className = "fplk-embedded-container";
        target.element.insertAdjacentElement(target.position, embeddedCardRoot);
      }
    }

    const ext = currentExtraction || {
      valid: false,
      category: "gpu",
      data: {},
    };
    const data = ext.data || {};
    const cat = ext.category || "gpu";
    const isUnsupported = cat === "unsupported" || ext.is_unsupported_item;
    const isBrandNew =
      ext.is_brand_new !== undefined
        ? ext.is_brand_new
        : isBrandNewCondition(data.condition, data.title, ext.pageContext ? ext.pageContext.raw_text : "", ext.pageContext ? ext.pageContext.key_values : {});
    const iconUrl = chrome.runtime.getURL("icon.png");

    embeddedCardRoot.innerHTML = `
            <div class="fplk-embedded-box">
                <!-- Header with branding and rescan -->
                <div class="fplk-embedded-header">
                    <div class="fplk-brand-wrap">
                        <img src="${iconUrl}" width="20" height="20" alt="FairPriceLK Logo" style="border-radius: 4px; object-fit: contain;">
                        <span class="fplk-brand-name">FairPriceLK</span>
                        <span class="fplk-category-pill" style="${isUnsupported ? "background:#FEE2E2; color:#991B1B; border-color:#FECACA;" : ""}">${isUnsupported ? "UNSUPPORTED" : cat.toUpperCase()}</span>
                    </div>
                    <div class="fplk-header-controls">
                        <button class="fplk-embedded-btn-icon" id="fplk-embedded-reextract-btn" title="Re-scan listing">
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>
                            <span>Re-scan</span>
                        </button>
                    </div>
                </div>

                <div class="fplk-embedded-body">
                    <!-- Extracted Listing Details -->
                    <div class="fplk-extracted-details">
                        <div class="fplk-extracted-title">
                            <strong>${data.title || data.model || "Marketplace Listing"}</strong>
                        </div>
                        <div class="fplk-extracted-tags">
                            ${data.listed_price ? `<span class="fplk-extracted-tag price">Asking: Rs. ${Number(data.listed_price).toLocaleString("en-LK")}</span>` : ""}
                            ${data.item_type ? `<span class="fplk-extracted-tag" style="background:#FEE2E2; color:#991B1B; border-color:#FECACA;">Type: <strong>${data.item_type}</strong></span>` : ""}
                            ${!isUnsupported && data.phone_type ? `<span class="fplk-extracted-tag">Type: <strong>${data.phone_type === "iphone" ? "iPhone" : "Android"}</strong></span>` : ""}
                            ${!isUnsupported && data.vehicle_type ? `<span class="fplk-extracted-tag">Category: <strong>${data.vehicle_type === "suvs" ? "SUV" : data.vehicle_type === "vans" ? "Van" : "Car"}</strong></span>` : ""}
                            ${!isUnsupported && (data.brand || data.make) ? `<span class="fplk-extracted-tag">Make: <strong>${data.brand || data.make}</strong></span>` : ""}
                            ${!isUnsupported && data.model ? `<span class="fplk-extracted-tag">Model: <strong>${data.model}</strong></span>` : ""}
                            ${!isUnsupported && data.vram_gb ? `<span class="fplk-extracted-tag">VRAM: <strong>${data.vram_gb} GB</strong></span>` : ""}
                            ${!isUnsupported && data.storage_gb ? `<span class="fplk-extracted-tag">Storage: <strong>${data.storage_gb} GB</strong></span>` : ""}
                            ${!isUnsupported && data.ram_gb ? `<span class="fplk-extracted-tag">RAM: <strong>${data.ram_gb} GB</strong></span>` : ""}
                            ${!isUnsupported && data.battery_health_percent ? `<span class="fplk-extracted-tag">Battery Health: <strong>${data.battery_health_percent}%</strong></span>` : ""}
                            ${!isUnsupported && data.warranty_days ? `<span class="fplk-extracted-tag">Warranty: <strong>${data.warranty_days} days</strong></span>` : ""}
                            ${!isUnsupported && (data.model_year || data.year) ? `<span class="fplk-extracted-tag">Year: <strong>${data.model_year || data.year}</strong></span>` : ""}
                            ${!isUnsupported && data.variant ? `<span class="fplk-extracted-tag">Variant: <strong>${data.variant}</strong></span>` : ""}
                            ${!isUnsupported && (data.engine_cc || data.engineCC) ? `<span class="fplk-extracted-tag">Engine CC: <strong>${data.engine_cc || data.engineCC}</strong></span>` : ""}
                            ${!isUnsupported && (data.mileage || data.mileage_km) ? `<span class="fplk-extracted-tag">Mileage: <strong>${Number(data.mileage || data.mileage_km).toLocaleString("en-LK")} km</strong></span>` : ""}
                            ${!isUnsupported && (data.gear || data.transmission) ? `<span class="fplk-extracted-tag">Gear: <strong>${data.gear || data.transmission}</strong></span>` : ""}
                            ${!isUnsupported && (data.fuelType || data.fuel_type) ? `<span class="fplk-extracted-tag">Fuel: <strong>${data.fuelType || data.fuel_type}</strong></span>` : ""}
                            ${data.condition ? `<span class="fplk-extracted-tag ${isBrandNew ? "price" : ""}">Condition: <strong>${data.condition}</strong></span>` : ""}
                        </div>
                    </div>

                    <!-- Unsupported Item Notice -->
                    ${
                      isUnsupported
                        ? `
                        <div class="fplk-verdict-box warning" style="background:#FEF2F2; border:1px solid #FECACA; color:#991B1B;">
                            <div class="fplk-verdict-header">
                                <span class="fplk-verdict-tag" style="background:#FEE2E2; color:#991B1B; font-weight:700;">⚠️ CATEGORY NOT SUPPORTED</span>
                            </div>
                            <div class="fplk-verdict-body" style="font-size: 12.5px; color: #7F1D1D; line-height: 1.5; margin-top: 4px;">
                                ${ext.error_message || `This listing ("${data.title || "Listing"}") is not in a supported category. FairPriceLK provides valuation for <strong>Mobile Phones</strong>, <strong>Graphics Cards (GPUs)</strong>, <strong>Vehicles</strong>, and <strong>Computer Hardware (Laptops/Monitors)</strong>.`}
                            </div>
                        </div>
                    `
                        : ""
                    }

                    <!-- Brand New Notice -->
                    ${
                      !isUnsupported && isBrandNew && !manualEstimateRequested
                        ? `
                        <div class="fplk-verdict-box warning">
                            <div class="fplk-verdict-header">
                                <span class="fplk-verdict-tag" style="background:#FDE68A; color:#92400E; font-weight:700;">⚠️ BRAND NEW ITEM</span>
                            </div>
                            <div class="fplk-verdict-body" style="font-size: 13px; color: #78350F; line-height: 1.5; margin-top: 4px;">
                                FairPriceLK only predicts prices for <strong>used / second-hand</strong> items. Valuation is unavailable for brand new listings.
                            </div>
                        </div>
                    `
                        : ""
                    }

                    <!-- Evaluation or Missing/Invalid info -->
                    ${
                      !isUnsupported &&
                      !ext.valid &&
                      !cachedPrediction &&
                      !isBrandNew
                        ? `
                        <div class="fplk-verdict-box ${ext.error_message && (ext.error_message.includes('not match') || ext.error_message.includes('Unrecognized') || ext.error_message.includes('conflicting')) ? 'warning' : 'neutral'}">
                            <div class="fplk-verdict-header">
                                <span class="fplk-verdict-tag" style="${ext.error_message && (ext.error_message.includes('not match') || ext.error_message.includes('Unrecognized') || ext.error_message.includes('conflicting')) ? 'background:#FEE2E2; color:#991B1B; font-weight:700;' : ''}">
                                    ${ext.error_message && (ext.error_message.includes('not match') || ext.error_message.includes('Unrecognized') || ext.error_message.includes('conflicting')) ? '⚠️ SPECIFICATION ISSUE' : 'Listing Details Detected'}
                                </span>
                            </div>
                            <div class="fplk-verdict-body" style="${ext.error_message && (ext.error_message.includes('not match') || ext.error_message.includes('Unrecognized') || ext.error_message.includes('conflicting')) ? 'font-size:12.5px; color:#7F1D1D; line-height:1.5; margin-top:4px;' : ''}">
                                ${ext.error_message || "Could not automatically identify all specifications for valuation. You can refine details below."}
                            </div>
                        </div>
                    `
                        : ""
                    }

                    ${!isUnsupported && cachedPrediction && (!isBrandNew || manualEstimateRequested) ? renderPredictionResult(cachedPrediction, data.listed_price) : ""}

                    <!-- Manual Refine Form (Collapsible/Accordion) - Only for supported Used items -->
                    ${
                      !isUnsupported && !isBrandNew
                        ? `
                    <details class="fplk-form-section">
                        <summary class="fplk-form-title" style="cursor: pointer;">
                            <span>Refine Details / Manual Estimate</span>
                            <span style="font-size: 10px; color: #71717A;">Click to adjust</span>
                        </summary>

                        <div style="margin-top: 10px; display: flex; flex-direction: column; gap: 8px;">
                            ${renderCategoryFormFields(cat, data)}

                            <div class="fplk-form-group">
                                <label class="fplk-label">Listing Asking Price (LKR)</label>
                                <input type="number" class="fplk-input" id="fplk-input-price" value="${data.listed_price || ""}" placeholder="e.g. 75000">
                            </div>

                            <button class="fplk-btn" id="fplk-eval-btn">
                                Recalculate Market Valuation
                            </button>
                        </div>
                    </details>
                    `
                        : ""
                    }
                </div>
            </div>
        `;

    attachEventListeners();
  }

  function renderPredictionResult(pred, listedPrice) {
    if (!pred) return "";

    const hasRange =
      pred.fair_market_range && pred.fair_market_range.lower_price_lkr;
    const lower = hasRange
      ? pred.fair_market_range.lower_price_lkr
      : pred.predicted_price * 0.9;
    const upper = hasRange
      ? pred.fair_market_range.upper_price_lkr
      : pred.predicted_price * 1.1;
    const pointPrice =
      pred.predicted_price || pred.price || (lower + upper) / 2;

    // Evaluate using universal engine
    let fairness = null;
    const category =
      currentExtraction && currentExtraction.category
        ? currentExtraction.category
        : "gpu";
    const itemDetails =
      currentExtraction && currentExtraction.data ? currentExtraction.data : {};
    if (window.FairPriceLK_Fairness) {
      fairness = window.FairPriceLK_Fairness.evaluate(
        listedPrice,
        pointPrice,
        lower,
        upper,
        category,
        itemDetails,
      );
    }

    const badgeCls = fairness
      ? fairness.badgeClass
      : (pred.evaluation && pred.evaluation.badge_class) || "fair";
    const verdictTitle = fairness
      ? fairness.badgeText
      : (pred.evaluation && pred.evaluation.verdict) || "Fair Market Price";
    const scoreVal =
      fairness && fairness.score !== null
        ? fairness.score
        : pred.evaluation && pred.evaluation.fairness_score;
    const adviceText = fairness
      ? fairness.advice
      : pred.evaluation && pred.evaluation.description;
    const actionAdvice = fairness ? fairness.actionAdvice : null;
    const negotiationTarget = fairness ? fairness.negotiationTarget : null;

    return `
            <!-- Price Range & Score Grid -->
            <div class="fplk-price-grid">
                <div class="fplk-price-card primary">
                    <span class="fplk-price-label">PREDICTED PRICE</span>
                    <span class="fplk-price-val">Rs. ${Math.round(pointPrice).toLocaleString("en-LK")}</span>
                </div>
                <div class="fplk-price-card score-card">
                    <span class="fplk-price-label">Fairness Score</span>
                    <span class="fplk-price-val fplk-score-val ${badgeCls}">${scoreVal !== undefined && scoreVal !== null ? `${scoreVal}/100` : "N/A"}</span>
                    <span class="fplk-price-sublabel">${verdictTitle}</span>
                </div>
            </div>

            <!-- Fairness Description & Advice Card -->
            <div class="fplk-verdict-box ${badgeCls}">
                <div class="fplk-verdict-header">
                    <span class="fplk-verdict-tag">${verdictTitle}</span>
                    ${scoreVal !== undefined && scoreVal !== null ? `<span class="fplk-score-pill">Score: <strong>${scoreVal}/100</strong></span>` : ""}
                </div>
                <div class="fplk-verdict-body">
                    ${adviceText || "Estimated based on second-hand market distribution and hardware specifications."}
                </div>
            </div>

            <!-- Score Calculation Breakdown (Collapsible Accordion) -->
            ${
              fairness && fairness.breakdown
                ? `
                <details class="fplk-breakdown-details">
                    <summary class="fplk-breakdown-summary">
                        <span class="fplk-breakdown-summary-title">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20V10M18 20V4M6 20v-4"/></svg>
                            Score Breakdown & How It's Calculated
                        </span>
                        <span class="fplk-breakdown-summary-hint">View factors</span>
                    </summary>
                    <div class="fplk-breakdown-body">
                        <div class="fplk-formula-card">
                            <div class="fplk-formula-title">FORMULA STEP</div>
                            <div class="fplk-formula-math">${fairness.breakdown.formulaExplanation}</div>
                        </div>

                        <div class="fplk-factors-list">
                            ${(fairness.breakdown.factors || [])
                              .map((f) => {
                                const pillClass = (f.impact || "neutral")
                                  .toLowerCase()
                                  .replace(/\s+/g, "-");
                                return `
                                    <div class="fplk-factor-card">
                                        <div class="fplk-factor-header">
                                            <span class="fplk-factor-name">${f.name}</span>
                                            <span class="fplk-factor-tag ${pillClass}">${f.value}</span>
                                        </div>
                                        <div class="fplk-factor-desc">${f.desc}</div>
                                    </div>
                                `;
                              })
                              .join("")}
                        </div>
                    </div>
                </details>
            `
                : ""
            }
        `;
  }

  function renderCategoryFormFields(category, data) {
    if (category === "gpu") {
      const models =
        window.FairPriceLK_Extractors && window.FairPriceLK_Extractors.gpu
          ? window.FairPriceLK_Extractors.gpu.CANONICAL_MODELS
          : [];
      const brands =
        window.FairPriceLK_Extractors && window.FairPriceLK_Extractors.gpu
          ? window.FairPriceLK_Extractors.gpu.KNOWN_BRANDS
          : ["Any", "ASUS", "MSI", "GIGABYTE", "ZOTAC"];

      const currentModel = data.model || "";
      const currentBrand = data.brand || "Any";
      const currentVram = data.vram_gb || 8;

      return `
                <div class="fplk-form-group full-width">
                    <label class="fplk-label">GPU Model</label>
                    <input type="text" class="fplk-input ${!currentModel ? "invalid" : ""}" id="fplk-input-gpu-model" value="${currentModel}" placeholder="e.g. RTX 3060, RX 580, GTX 1660 SUPER" list="fplk-gpu-models-list">
                    <datalist id="fplk-gpu-models-list">
                        ${models.map((m) => `<option value="${m}">`).join("")}
                    </datalist>
                </div>
                <div class="fplk-form-grid">
                    <div class="fplk-form-group">
                        <label class="fplk-label">VRAM (GB)</label>
                        <select class="fplk-select" id="fplk-select-gpu-vram">
                            ${[1, 2, 3, 4, 6, 8, 10, 11, 12, 16, 20, 24]
                              .map(
                                (v) =>
                                  `<option value="${v}" ${Number(currentVram) === v ? "selected" : ""}>${v} GB</option>`,
                              )
                              .join("")}
                        </select>
                    </div>
                    <div class="fplk-form-group">
                        <label class="fplk-label">Brand</label>
                        <select class="fplk-select" id="fplk-select-gpu-brand">
                            <option value="Any">Any</option>
                            ${brands.map((b) => `<option value="${b}" ${currentBrand.toUpperCase() === b.toUpperCase() ? "selected" : ""}>${b}</option>`).join("")}
                        </select>
                    </div>
                </div>
            `;
    } else if (category === "mobile") {
      return `
                <div class="fplk-form-grid">
                    <div class="fplk-form-group">
                        <label class="fplk-label">OS / Type</label>
                        <select class="fplk-select" id="fplk-mobile-type">
                            <option value="android" ${data.phone_type === "android" ? "selected" : ""}>Android</option>
                            <option value="iphone" ${data.phone_type === "iphone" ? "selected" : ""}>iPhone</option>
                        </select>
                    </div>
                    <div class="fplk-form-group">
                        <label class="fplk-label">Brand</label>
                        <input type="text" class="fplk-input" id="fplk-mobile-brand" value="${data.brand || ""}" placeholder="e.g. Samsung">
                    </div>
                </div>
                <div class="fplk-form-group full-width">
                    <label class="fplk-label">Model</label>
                    <input type="text" class="fplk-input" id="fplk-mobile-model" value="${data.model || ""}" placeholder="e.g. Galaxy S21">
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
                <div class="fplk-form-grid">
                    <div class="fplk-form-group">
                        <label class="fplk-label">Battery Health (%) <span style="color:#71717A;font-size:10px;">(iPhone)</span></label>
                        <input type="number" class="fplk-input" id="fplk-mobile-battery" value="${data.battery_health_percent !== null && data.battery_health_percent !== undefined ? data.battery_health_percent : ""}" placeholder="e.g. 87" min="50" max="100">
                    </div>
                    <div class="fplk-form-group">
                        <label class="fplk-label">Warranty (Days)</label>
                        <input type="number" class="fplk-input" id="fplk-mobile-warranty" value="${data.warranty_days || 0}" min="0">
                    </div>
                </div>
            `;
    } else if (category === "vehicle") {
      const currentMake = data.brand || data.make || "";
      const currentModel = data.model || "";
      const currentYear = data.model_year || data.year || 2015;
      const currentMileage = data.mileage_km || data.mileage || "";
      const currentGear = data.transmission || data.gear || "Automatic";
      const currentFuel = data.fuel_type || data.fuelType || "Petrol";
      const currentCC = data.engine_cc || "";
      const currentVariant = data.variant || "Standard";
      return `
                <div class="fplk-form-grid">
                    <div class="fplk-form-group">
                        <label class="fplk-label">Make (Brand)</label>
                        <input type="text" class="fplk-input" id="fplk-vehicle-make" value="${currentMake}" placeholder="e.g. Toyota">
                    </div>
                    <div class="fplk-form-group">
                        <label class="fplk-label">Model</label>
                        <input type="text" class="fplk-input" id="fplk-vehicle-model" value="${currentModel}" placeholder="e.g. Aqua">
                    </div>
                </div>
                <div class="fplk-form-grid">
                    <div class="fplk-form-group">
                        <label class="fplk-label">Year</label>
                        <input type="number" class="fplk-input" id="fplk-vehicle-year" value="${currentYear}" min="1980" max="2026">
                    </div>
                    <div class="fplk-form-group">
                        <label class="fplk-label">Mileage (km)</label>
                        <input type="number" class="fplk-input" id="fplk-vehicle-mileage" value="${currentMileage}" placeholder="e.g. 75000">
                    </div>
                </div>
                <div class="fplk-form-grid">
                    <div class="fplk-form-group">
                        <label class="fplk-label">Transmission</label>
                        <select class="fplk-select" id="fplk-vehicle-transmission">
                            <option value="Automatic" ${currentGear === "Automatic" ? "selected" : ""}>Automatic</option>
                            <option value="Manual" ${currentGear === "Manual" ? "selected" : ""}>Manual</option>
                        </select>
                    </div>
                    <div class="fplk-form-group">
                        <label class="fplk-label">Fuel Type</label>
                        <select class="fplk-select" id="fplk-vehicle-fuel">
                            <option value="Petrol"   ${currentFuel === "Petrol" ? "selected" : ""}>Petrol</option>
                            <option value="Diesel"   ${currentFuel === "Diesel" ? "selected" : ""}>Diesel</option>
                            <option value="Hybrid"   ${currentFuel === "Hybrid" ? "selected" : ""}>Hybrid</option>
                            <option value="Electric" ${currentFuel === "Electric" ? "selected" : ""}>Electric</option>
                        </select>
                    </div>
                </div>
                <div class="fplk-form-grid">
                    <div class="fplk-form-group">
                        <label class="fplk-label">Engine (CC) <span style="color:#71717A;font-size:10px;">(optional)</span></label>
                        <input type="number" class="fplk-input" id="fplk-vehicle-enginecc" value="${currentCC}" placeholder="e.g. 1500">
                    </div>
                    <div class="fplk-form-group">
                        <label class="fplk-label">Variant <span style="color:#71717A;font-size:10px;">(optional)</span></label>
                        <input type="text" class="fplk-input" id="fplk-vehicle-variant" value="${currentVariant}" placeholder="e.g. G Grade">
                    </div>
                </div>
            `;
    } else {
      return `
                <div class="fplk-form-grid">
                    <div class="fplk-form-group">
                        <label class="fplk-label">Brand</label>
                        <input type="text" class="fplk-input" id="fplk-elec-brand" value="${data.brand || ""}" placeholder="e.g. Dell">
                    </div>
                    <div class="fplk-form-group">
                        <label class="fplk-label">Model</label>
                        <input type="text" class="fplk-input" id="fplk-elec-model" value="${data.model || ""}" placeholder="e.g. XPS 13">
                    </div>
                </div>
            `;
    }
  }

  function attachEventListeners() {
    const reextractBtn = document.getElementById("fplk-embedded-reextract-btn");
    if (reextractBtn) {
      reextractBtn.addEventListener("click", () => {
        cachedPrediction = null;
        runExtraction();
        const d =
          currentExtraction && currentExtraction.data
            ? currentExtraction.data
            : {};
        const isBrandNew = isBrandNewCondition(d.condition, d.title);
        if (currentExtraction && currentExtraction.valid && !isBrandNew) {
          triggerPrediction();
        } else {
          renderEmbeddedCard();
        }
      });
    }

    const evalBtn = document.getElementById("fplk-eval-btn");
    if (evalBtn) {
      evalBtn.addEventListener("click", () => {
        readFormInputsIntoData();
        triggerPrediction(true);
      });
    }
  }

  function readFormInputsIntoData() {
    if (!currentExtraction) currentExtraction = { category: "gpu", data: {} };
    const cat = currentExtraction.category || "gpu";
    const d = currentExtraction.data || {};

    const priceEl = document.getElementById("fplk-input-price");
    if (priceEl && priceEl.value) d.listed_price = parseFloat(priceEl.value);

    if (cat === "gpu") {
      const mEl = document.getElementById("fplk-input-gpu-model");
      const vEl = document.getElementById("fplk-select-gpu-vram");
      const bEl = document.getElementById("fplk-select-gpu-brand");

      if (mEl) d.model = mEl.value.trim();
      if (vEl) d.vram_gb = parseFloat(vEl.value);
      if (bEl) d.brand = bEl.value;
      d.stock = "In Stock";
    } else if (cat === "mobile") {
      const tEl = document.getElementById("fplk-mobile-type");
      const bEl = document.getElementById("fplk-mobile-brand");
      const mEl = document.getElementById("fplk-mobile-model");
      const sEl = document.getElementById("fplk-mobile-storage");
      const rEl = document.getElementById("fplk-mobile-ram");

      if (tEl) d.phone_type = tEl.value;
      if (bEl) d.brand = bEl.value.trim();
      if (mEl) d.model = mEl.value.trim();
      if (sEl) d.storage_gb = parseFloat(sEl.value) || 128;
      if (rEl) d.ram_gb = parseFloat(rEl.value) || 6;

      const bhEl = document.getElementById("fplk-mobile-battery");
      const wEl = document.getElementById("fplk-mobile-warranty");
      if (bhEl && bhEl.value) {
        const bh = parseFloat(bhEl.value);
        d.battery_health_percent =
          !isNaN(bh) && bh >= 50 && bh <= 100 ? bh : null;
      } else {
        d.battery_health_percent = null;
      }
      if (wEl && wEl.value) {
        d.warranty_days = parseFloat(wEl.value) || 0;
      } else {
        d.warranty_days = 0;
      }

      // Re-derive engineered features using the extractor's helpers if available
      const mobileExt =
        window.FairPriceLK_Extractors && window.FairPriceLK_Extractors.mobile;
      if (mobileExt && mobileExt.parse) {
        // Trigger a fresh parse to recompute features based on updated brand/model
        const freshParsed = mobileExt.parse({
          title: d.model || "",
          price: d.listed_price,
          raw_text: "",
          key_values: { brand: d.brand, model: d.model },
        });
        if (freshParsed && freshParsed.data) {
          d.model_tier = freshParsed.data.model_tier;
          d.brand_tier = freshParsed.data.brand_tier;
          d.phone_age_years = freshParsed.data.phone_age_years;
          d.is_flagship = freshParsed.data.is_flagship;
          d.has_5g = freshParsed.data.has_5g;
          d.has_esim = freshParsed.data.has_esim;
          d.dual_sim = freshParsed.data.dual_sim;
        }
      }
    } else if (cat === "vehicle") {
      const makeEl = document.getElementById("fplk-vehicle-make");
      const mEl = document.getElementById("fplk-vehicle-model");
      const yEl = document.getElementById("fplk-vehicle-year");
      const miEl = document.getElementById("fplk-vehicle-mileage");
      const txEl = document.getElementById("fplk-vehicle-transmission");
      const fuelEl = document.getElementById("fplk-vehicle-fuel");
      const ccEl = document.getElementById("fplk-vehicle-enginecc");
      const vEl = document.getElementById("fplk-vehicle-variant");

      if (makeEl) {
        d.brand = makeEl.value.trim();
        d.make = d.brand;
      }
      if (mEl) d.model = mEl.value.trim();
      if (yEl) d.model_year = parseInt(yEl.value, 10) || 2015;
      if (miEl && miEl.value) {
        const km = parseInt(miEl.value, 10);
        d.mileage_km = !isNaN(km) && km > 0 ? km : null;
        d.mileage = d.mileage_km;
      }
      if (txEl) {
        d.transmission = txEl.value;
        d.gear = d.transmission;
      }
      if (fuelEl) {
        d.fuel_type = fuelEl.value;
        d.fuelType = d.fuel_type;
      }
      if (ccEl && ccEl.value) {
        const cc = parseInt(ccEl.value, 10);
        d.engine_cc = !isNaN(cc) && cc > 0 ? cc : null;
      }
      if (vEl) d.variant = vEl.value.trim() || "Standard";
      d.year = d.model_year;
    } else if (cat === "electronics") {
      const bEl = document.getElementById("fplk-elec-brand");
      const mEl = document.getElementById("fplk-elec-model");

      if (bEl) d.brand = bEl.value.trim();
      if (mEl) d.model = mEl.value.trim();
      d.category = "laptop";
      d.ram = 8;
      d.storage = 256;
    }

    currentExtraction.data = d;
  }

  async function triggerPrediction(manualOverride = false) {
    if (!currentExtraction || !currentExtraction.data) return;
    const cat = currentExtraction.category || "gpu";
    const originalData = currentExtraction.data;

    const evalBtn = document.getElementById("fplk-eval-btn");
    if (evalBtn) {
      evalBtn.innerHTML = `<span class="fplk-spinner"></span> Calculating Valuation...`;
      evalBtn.disabled = true;
    }

    let payloadForFetch = { ...originalData };

    // Strip listed_price from the backend payload (used only for frontend fairness calculation)
    delete payloadForFetch.listed_price;

    let subpath = "";
    if (cat === "mobile") {
      // Strictly limit payload to the fields the Mobile API PredictRequest expects
      payloadForFetch = {
        phone_type: originalData.phone_type || "android",
        brand: originalData.brand || "",
        model: originalData.model || "",
        storage_gb: originalData.storage_gb || 128,
        ram_gb: originalData.ram_gb || 6,
        warranty_days:
          originalData.warranty_days !== undefined &&
          originalData.warranty_days !== null
            ? originalData.warranty_days
            : 0,
        battery_health_percent:
          originalData.battery_health_percent !== undefined &&
          originalData.battery_health_percent !== null
            ? originalData.battery_health_percent
            : null,
        dual_sim: originalData.dual_sim ? true : false,
        has_5g: originalData.has_5g ? true : false,
        has_esim: originalData.has_esim ? true : false,
        model_tier:
          originalData.model_tier !== undefined ? originalData.model_tier : 3,
        brand_tier:
          originalData.brand_tier !== undefined ? originalData.brand_tier : 2,
        phone_age_years:
          originalData.phone_age_years !== undefined
            ? originalData.phone_age_years
            : 3.0,
        is_flagship:
          originalData.is_flagship !== undefined ? originalData.is_flagship : 0,
      };
    } else if (cat === "vehicle") {
      const vType = originalData.vehicle_type || "cars";

      if (vType === "suvs") {
        subpath = "suv";
      } else if (vType === "vans") {
        subpath = "van";
      } else {
        // For standard cars, strictly limit the payload to only the 7 expected fields
        payloadForFetch = {
          brand: originalData.brand,
          model: originalData.model,
          variant: originalData.variant,
          model_year: originalData.model_year,
          mileage_km: originalData.mileage_km,
          fuel_type: originalData.fuel_type,
          transmission: originalData.transmission,
        };
      }
    }

    try {
      chrome.runtime.sendMessage(
        {
          action: "predict_price",
          category: cat,
          subpath: subpath,
          payload: payloadForFetch,
        },
        (response) => {
          if (chrome.runtime.lastError) {
            console.error("Extension message error:", chrome.runtime.lastError);
            handlePredictionFailure(
              `Extension error: ${chrome.runtime.lastError.message}`,
              manualOverride,
            );
            return;
          }

          if (!response || !response.success) {
            handlePredictionFailure(
              response
                ? response.error
                : "Unknown error from background script",
              manualOverride,
            );
            return;
          }

          if (response.data && response.data.confidence === "Unknown") {
            cachedPrediction = null;
            currentExtraction.valid = false;
            currentExtraction.error_message = "Market valuation is temporarily not supported for this vehicle model.";
            const evalBtn = document.getElementById("fplk-eval-btn");
            if (evalBtn) {
              evalBtn.innerHTML = "Recalculate Market Valuation";
              evalBtn.disabled = false;
            }
            renderEmbeddedCard(manualOverride);
            return;
          }

          cachedPrediction = response.data;
          currentExtraction.valid = true;
          renderEmbeddedCard(manualOverride);
        },
      );
    } catch (err) {
      console.error("Message dispatch error:", err);
      handlePredictionFailure(`${err.name}: ${err.message}`, manualOverride);
    }
  }

  function handlePredictionFailure(errMsg, manualOverride = false) {
    cachedPrediction = null;
    if (currentExtraction) {
      currentExtraction.valid = false;
      const isFailedToFetch =
        errMsg &&
        (errMsg.includes("Failed to fetch") || errMsg.includes("NetworkError"));
      currentExtraction.error_message = isFailedToFetch
        ? `Cannot connect to local backend at ${getApiBase()}. Please ensure start_all.py is running.`
        : `${errMsg}`;
    }
    renderEmbeddedCard(manualOverride);
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
            storage: d.storage_gb ? String(d.storage_gb) : "",
            ram: d.ram_gb ? String(d.ram_gb) : "",
            phone_type: d.phone_type || "android",
            warranty_days: d.warranty_days || 0,
            vram: d.vram_gb ? String(d.vram_gb) : "",
            condition: d.condition || "",
            item_type: d.item_type || null,
            category: extraction ? extraction.category : "unsupported",
            valid: extraction ? extraction.valid : false,
            is_unsupported_item: extraction
              ? extraction.is_unsupported_item ||
                extraction.category === "unsupported"
              : false,
            is_brand_new: extraction ? Boolean(extraction.is_brand_new) : false,
            error_message: extraction ? extraction.error_message : null,
          },
        });
      } catch (err) {
        sendResponse({ success: false, error: err.message });
      }
    }
    return true;
  });

  // Initialize on DOM load
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initOnPageWidget);
  } else {
    initOnPageWidget();
  }
})();
