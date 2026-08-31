/**
 * FairPriceLK - Vehicle Category Extractor (Riyasewana.com + Generic Fallback)
 * Primary: Scrapes all listing fields directly from the riyasewana.com DOM table.
 * Fallback: Uses pageContext heuristics for other marketplace pages (ikman, etc.).
 *
 * Extracted fields mapped to PredictRequest API schema:
 *   brand, model, variant, model_year, mileage_km, fuel_type, transmission
 */

window.FairPriceLK_Extractors = window.FairPriceLK_Extractors || {};

window.FairPriceLK_Extractors.vehicle = (function () {

    const SUPPORTED_SUV_BRANDS = ['toyota', 'suzuki', 'peugeot', 'nissan', 'mitsubishi', 'micro', 'mg', 'kia', 'hyundai', 'honda', 'daihatsu', 'bmw', 'audi'];

    // ── Riyasewana DOM Scraper ────────────────────────────────────────────────
    /**
     * Reads the listing details <table> on riyasewana.com.
     * Each <tr> has two <td>s: label (col 1) and value (col 2).
     * Also attempts to find the asking price from any element containing "Rs.".
     */
    function scrapeRiyasewana() {
        const result = {
            price:     null,
            year:      null,
            mileage:   null,
            make:      null,
            model:     null,
            gear:      null,
            fuelType:  null,
            engineCC:  null,
            condition: null,
            title:     null,
            description: null,
            vehicle_type: 'cars',
            variant:   null,
            body_type: null
        };

        try {
            // 1. Title
            const h1 = document.querySelector('h1');
            if (h1) {
                result.title = h1.innerText.trim();
                const titleUp = result.title.toUpperCase();
                if (titleUp.includes('SUV')) result.vehicle_type = 'suvs';
                else if (titleUp.includes('VAN')) result.vehicle_type = 'vans';
            }

            // 1b. Description
            let optionsText = "";
            let detailsText = "";
            const headers = Array.from(document.querySelectorAll('h2, h3, h4, h5, div.options, div.more, div.more-card-title'));
            for (const h of headers) {
                const title = (h.innerText || h.textContent || '').trim().toUpperCase();
                if (title === 'OPTIONS' || title === 'MORE DETAILS' || title === 'DESCRIPTION') {
                    // Collect text from all next siblings until another header
                    let collected = [];
                    let sibling = h.nextElementSibling;
                    while (sibling) {
                        if (sibling.tagName && (sibling.tagName.match(/^H[1-6]$/) || sibling.className === 'more-card-title')) break;
                        const t = (sibling.innerText || sibling.textContent || '').replace(/\s+/g, ' ').trim();
                        if (t) collected.push(t);
                        sibling = sibling.nextElementSibling;
                    }
                    // If sibling iteration got nothing, maybe it's inside a parent wrapper
                    if (collected.length === 0 && h.parentElement) {
                        const parentText = (h.parentElement.innerText || h.parentElement.textContent || '').replace(h.innerText || h.textContent, '').replace(/\s+/g, ' ').trim();
                        if (parentText) collected.push(parentText);
                    }
                    
                    if (collected.length > 0) {
                        if (title.includes('OPTION')) {
                            optionsText = collected.join(' ');
                        } else {
                            detailsText = collected.join(' ');
                        }
                    }
                }
            }
            
            // Independent Fallbacks
            if (!optionsText) {
                const optionsBox = document.querySelector('#options, .options-list, .options');
                if (optionsBox) optionsText = (optionsBox.innerText || optionsBox.textContent || '').replace(/\s+/g, ' ').trim();
            }
            if (!detailsText) {
                const moreBox = document.querySelector('.morebox, .contentbox, .description, .more-card-body');
                if (moreBox) detailsText = (moreBox.innerText || moreBox.textContent || '').replace(/\s+/g, ' ').trim();
            }
            
            let descParts = [];
            if (optionsText) descParts.push("Options: " + optionsText);
            if (detailsText) descParts.push("Details: " + detailsText);
            
            result.description = descParts.join(' | ') || "";

            // 2. Price — find leaf node containing "Rs." with ≥4 digits
            const leafEls = Array.from(document.querySelectorAll('div, span, p, td, strong, b, h2, h3'));
            for (const el of leafEls) {
                // Skip containers that have non-text children
                const directText = Array.from(el.childNodes)
                    .filter(n => n.nodeType === Node.TEXT_NODE)
                    .map(n => n.textContent)
                    .join('');
                const fullText = (el.innerText || el.textContent || '').trim();
                const targetText = directText.trim() || fullText;

                if ((targetText.includes('Rs.') || targetText.includes('Rs ')) && /\d{4,}/.test(targetText)) {
                    const match = targetText.match(/Rs\.?\s*([\d,]+)/i);
                    if (match) {
                        const digits = match[1].replace(/,/g, '');
                        const num = parseInt(digits, 10);
                        if (num > 10000 && (!result.price || num > result.price)) {
                            result.price = num;
                            break;
                        }
                    }
                }
            }

            // 3. Spec table rows — match label → value pairs
            const tables = document.querySelectorAll('table');
            for (const table of tables) {
                const rows = table.querySelectorAll('tr');
                for (const row of rows) {
                    const cells = row.querySelectorAll('td');
                    if (cells.length < 2) continue;

                    const label = (cells[0].innerText || cells[0].textContent || '').trim().toUpperCase();
                    const value = (cells[1].innerText || cells[1].textContent || '').trim();

                    if (!label || !value) continue;

                    if (label === 'YEAR' || label.startsWith('YEAR'))                    result.year      = value;
                    else if (label === 'MILEAGE' || label.startsWith('MILEAGE'))         result.mileage   = value;
                    else if (label === 'MAKE')                                            result.make      = value;
                    else if (label === 'MODEL')                                           result.model     = value;
                    else if (label === 'GEAR' || label.includes('GEAR'))                 result.gear      = value;
                    else if (label === 'FUEL TYPE' || label.includes('FUEL'))            result.fuelType  = value;
                    else if (label.includes('ENGINE'))                                    result.engineCC  = value;
                    else if (label === 'CONDITION' || label.includes('CONDITION'))       result.condition = value;
                    else if (label === 'EDITION' || label === 'TRIM' || label === 'VARIANT') result.variant = value;
                    else if (label === 'BODY TYPE' || label.includes('BODY')) {
                        result.body_type = value;
                        const bt = value.toUpperCase();
                        if (bt.includes('SUV')) result.vehicle_type = 'suvs';
                        else if (bt.includes('VAN')) result.vehicle_type = 'vans';
                    }
                }
            }

            // 4. dl/dt/dd or adjacent-span fallback for missing fields
            if (!result.make || !result.model) {
                const labelEls = document.querySelectorAll('dt, th, .label, [class*="spec-label"], [class*="detail-label"]');
                labelEls.forEach(lEl => {
                    const lText = (lEl.innerText || '').trim().toUpperCase();
                    const vEl   = lEl.nextElementSibling;
                    if (!vEl) return;
                    const vText = (vEl.innerText || '').trim();

                    if (lText.includes('MAKE')    && !result.make)      result.make      = vText;
                    if (lText.includes('MODEL')   && !result.model)     result.model     = vText;
                    if (lText.includes('YEAR')    && !result.year)      result.year      = vText;
                    if (lText.includes('MILEAGE') && !result.mileage)   result.mileage   = vText;
                    if (lText.includes('GEAR')    && !result.gear)      result.gear      = vText;
                    if (lText.includes('FUEL')    && !result.fuelType)  result.fuelType  = vText;
                    if (lText.includes('ENGINE')  && !result.engineCC)  result.engineCC  = vText;
                    if (lText.includes('COND')    && !result.condition) result.condition = vText;
                });
            }

        } catch (e) {
            console.warn('[FairPriceLK] Riyasewana DOM scrape error:', e);
        }

        return result;
    }

    // ── Field Cleaners ────────────────────────────────────────────────────────

    /** "75,000 km" → 75000 */
    function cleanMileage(raw) {
        if (!raw) return null;
        const digits = String(raw).replace(/[^0-9]/g, '');
        const num = parseInt(digits, 10);
        return (!isNaN(num) && num > 0) ? num : null;
    }

    /** "2015" / "Year: 2015" → 2015 */
    function cleanYear(raw) {
        if (!raw) return null;
        const match = String(raw).match(/\b(19\d\d|20[0-2]\d)\b/);
        if (match) {
            const y = parseInt(match[1], 10);
            if (y >= 1980 && y <= 2026) return y;
        }
        return null;
    }

    /** "800 cc" / "1500" / "1.5L" → numeric cc value */
    function cleanEngineCC(raw) {
        if (!raw) return null;
        const str = String(raw);
        const ccMatch = str.match(/(\d{3,4})\s*(?:cc|CC)?/);
        if (ccMatch) {
            const v = parseInt(ccMatch[1], 10);
            if (v >= 50 && v <= 9000) return v;
        }
        const litreMatch = str.match(/(\d+\.\d+)\s*(?:l|L)/);
        if (litreMatch) return Math.round(parseFloat(litreMatch[1]) * 1000);
        return null;
    }

    /** "Auto" / "Automatic" → "Automatic", "Manual" → "Manual" */
    function cleanGear(raw) {
        if (!raw) return null;
        const lower = String(raw).toLowerCase();
        if (lower.includes('auto')) return 'Automatic';
        if (lower.includes('manual') || lower === 'man') return 'Manual';
        return raw.trim();
    }

    /** Normalise fuel type string */
    function cleanFuelType(raw) {
        if (!raw) return null;
        const lower = String(raw).toLowerCase();
        if (lower.includes('hybrid'))               return 'Hybrid';
        if (lower.includes('diesel'))               return 'Diesel';
        if (lower.includes('petrol') || lower.includes('gasoline')) return 'Petrol';
        if (lower.includes('electric'))             return 'Electric';
        return raw.trim();
    }

    // ── Generic Heuristic Helpers (non-Riyasewana fallback) ──────────────────

    function guessMakeModel(text) {
        const lower = (text || '').toLowerCase();
        if (lower.includes('aqua'))                                          return { make: 'Toyota',  model: 'Aqua' };
        if (lower.includes('alto'))                                          return { make: 'Suzuki',  model: 'Alto' };
        if (lower.includes('corolla')) {
            if (lower.includes('141')) return { make: 'Toyota', model: 'Corolla 141' };
            if (lower.includes('121')) return { make: 'Toyota', model: 'Corolla 121' };
            return { make: 'Toyota', model: 'Corolla' };
        }
        if (lower.includes('vitz'))                                          return { make: 'Toyota',  model: 'Vitz' };
        if (lower.includes('fit') && lower.includes('honda'))               return { make: 'Honda',   model: 'Fit' };
        if (lower.includes('swift') && lower.includes('suzuki'))            return { make: 'Suzuki',  model: 'Swift' };
        if (lower.includes('axio') || lower.includes('fielder'))            return { make: 'Toyota',  model: 'Axio' };
        if (lower.includes('allion') || lower.includes('premio'))           return { make: 'Toyota',  model: 'Allion' };
        if (lower.includes('prius'))                                         return { make: 'Toyota',  model: 'Prius' };
        return null;
    }

    function guessYearFromText(text) {
        const match = (text || '').match(/\b(19\d\d|20[0-2]\d)\b/);
        if (match) {
            const y = parseInt(match[1], 10);
            if (y >= 1980 && y <= 2026) return y;
        }
        return 2015;
    }

    function guessTransmissionFromText(text) {
        const lower = (text || '').toLowerCase();
        if (lower.includes('manual')) return 'Manual';
        return 'Automatic';
    }

    function guessFuelFromText(text) {
        const lower = (text || '').toLowerCase();
        if (lower.includes('hybrid'))  return 'Hybrid';
        if (lower.includes('diesel'))  return 'Diesel';
        return 'Petrol';
    }

    /** Derive a sensible API variant value from available text */
    function deriveVariant(model, scopeText) {
        const lower = (scopeText || '').toLowerCase();
        const m = (model || '').toLowerCase();

        if (m.includes('aqua')) {
            if (lower.includes('s grade')) return 'S Grade';
            if (lower.includes('l grade')) return 'L Grade';
            return 'G Grade';
        }
        if (m.includes('alto')) {
            if (lower.includes('660')) return '660';
            return '800';
        }
        if (m.includes('corolla 121') || m.includes('corolla 141')) {
            return ''; // Variant is empty for 121 / 141 since they are now separate models
        }
        if (m.includes('corolla')) {
            return 'Standard';
        }
        if (m.includes('vitz')) {
            if (lower.includes('ksp90')) return 'KSP90';
            return 'Standard';
        }
        return 'Standard';
    }

    /** Guess brand from model name when make is missing */
    function guessBrandFromModel(model) {
        const lower = (model || '').toLowerCase();
        const BRAND_KEYWORDS = {
            Toyota: ['corolla', 'aqua', 'vitz', 'prius', 'axio', 'allion', 'fielder', 'hilux', 'fortuner', 'rav4', 'land cruiser'],
            Suzuki: ['alto', 'swift', 'wagon', 'baleno', 'ignis', 'vitara', 'jimny'],
            Honda: ['fit', 'vezel', 'civic', 'accord', 'cr-v', 'hr-v', 'jazz'],
            Nissan: ['march', 'note', 'leaf', 'tiida', 'almera', 'navara', 'patrol'],
            Mitsubishi: ['lancer', 'outlander', 'pajero', 'montero', 'eclipse', 'colt'],
            Mazda: ['axela', 'atenza', 'demio', 'cx-5', 'cx-3'],
        };
        for (const [brand, keywords] of Object.entries(BRAND_KEYWORDS)) {
            if (keywords.some(k => lower.includes(k))) return brand;
        }
        return '';
    }

    // ── Main parse() ─────────────────────────────────────────────────────────

    function parse(pageContext) {
        const url = (pageContext && pageContext.url)
            ? pageContext.url.toLowerCase()
            : window.location.href.toLowerCase();

        const isRiyasewana = url.includes('riyasewana.com');

        // ── Step 1: Scrape raw values ─────────────────────────────────────
        let scraped = {};
        if (isRiyasewana) {
            try { scraped = scrapeRiyasewana(); }
            catch (e) { console.warn('[FairPriceLK] scrapeRiyasewana failed:', e); }
        }

        const kv   = (pageContext && pageContext.key_values) || {};
        const title = scraped.title || (pageContext && pageContext.title) || '';
        const rawText = (pageContext && pageContext.raw_text) || '';
        const titleScope = `${title} ${scraped.make || ''} ${scraped.model || ''} ${scraped.variant || ''} ${rawText}`;

        // ── Step 2: Resolve and clean each field ──────────────────────────

        // --- Unsupported Vehicle Check ---
        const unsupportedRegex = /\b(motorcycle|motorbike|scooter|lorry|truck|bus|crew\s*cab|double\s*cab|three\s*wheel|tuk\s*tuk|tractor|heavy\s*machinery|excavator|bicycle)\b/i;
        let isUnsupported = false;

        // Check H1 Title
        if (unsupportedRegex.test(title)) {
            isUnsupported = true;
        }

        // Check Breadcrumbs
        if (!isUnsupported && pageContext && pageContext.breadcrumbs) {
            const breadcrumbText = pageContext.breadcrumbs.join(' ');
            if (unsupportedRegex.test(breadcrumbText)) {
                isUnsupported = true;
            }
        }

        // Check Body Type from scraped data or key_values
        if (!isUnsupported) {
            const bodyType = scraped.body_type || (pageContext && pageContext.key_values && (pageContext.key_values['body type'] || pageContext.key_values['body_type'] || pageContext.key_values['body'])) || '';
            if (unsupportedRegex.test(bodyType)) {
                isUnsupported = true;
            }
        }

        if (isUnsupported) {
            return {
                category: 'unsupported',
                valid: false,
                is_unsupported_item: true,
                error_message: "FairPriceLK currently supports Cars, SUVs, and Vans. Valuations for commercial vehicles and bikes are not available.",
                data: {
                    title: title,
                    listed_price: scraped.price || (pageContext && pageContext.price) || null,
                    item_type: "Unsupported Vehicle"
                },
                pageContext: pageContext
            };
        }
        // ----------------------------------

        // Price
        const price = scraped.price || (pageContext && pageContext.price) || null;

        // Year
        const year = cleanYear(scraped.year)
            || cleanYear(kv.year)
            || guessYearFromText(titleScope);

        // Mileage (null = model will estimate it)
        const mileage = cleanMileage(scraped.mileage) || cleanMileage(kv.mileage) || null;

        // Make & Model
        let make  = (scraped.make  || kv.brand || '').trim();
        let model = (scraped.model || kv.model || '').trim();

        // If model or title indicates Corolla 121 or 141, adjust model accordingly
        const combinedModelCheck = `${model} ${scraped.variant || ''} ${titleScope}`.toLowerCase();
        if (combinedModelCheck.includes('corolla')) {
            if (combinedModelCheck.includes('141')) {
                model = 'Corolla 141';
            } else if (combinedModelCheck.includes('121')) {
                model = 'Corolla 121';
            } else if (!model) {
                model = 'Corolla';
            }
        }

        // If both are empty, try compound-name heuristic
        if (!make && !model) {
            const guessed = guessMakeModel(titleScope);
            if (guessed) { make = guessed.make; model = guessed.model; }
        } else if (!make && model) {
            make = guessBrandFromModel(model);
        }

        // Gear / Transmission
        const gear = cleanGear(scraped.gear)
            || cleanGear(kv.transmission)
            || guessTransmissionFromText(titleScope);

        // Fuel Type
        const fuelType = cleanFuelType(scraped.fuelType)
            || cleanFuelType(kv.fuel_type)
            || guessFuelFromText(titleScope);

        // Engine CC (optional; used for SUV/Van sub-routes)
        const engineCC = cleanEngineCC(scraped.engineCC)
            || cleanEngineCC(kv.engine_cc)
            || (function() {
                const ccMatch = titleScope.match(/\b([1-9]\d{2,3})\s*(?:cc|CC)\b/i);
                if (ccMatch) return parseInt(ccMatch[1], 10);
                const lMatch = titleScope.match(/\b(\d+\.\d+)\s*(?:l|L|litre|liters)\b/i);
                if (lMatch) return Math.round(parseFloat(lMatch[1]) * 1000);
                return null;
            })()
            || null;

        // Condition (informational only)
        const condition = (scraped.condition || kv.condition || '').trim() || null;

        // Variant — empty string if 121 or 141, otherwise derived normally
        let variant = '';
        if (model === 'Corolla 121' || model === 'Corolla 141') {
            variant = '';
        } else {
            variant = scraped.variant || deriveVariant(model, titleScope);
        }

        // ── Step 3: Validate ─────────────────────────────────────────────
        
        // --- Unsupported SUV Brands Check ---
        const vehicleType = scraped.vehicle_type || 'cars';
        if (vehicleType === 'suvs' && make && !SUPPORTED_SUV_BRANDS.includes(make.toLowerCase())) {
            return {
                category: 'unsupported',
                valid: false,
                is_unsupported_item: true,
                error_message: "FairPriceLK currently does not support price predictions for " + make + " SUVs.",
                data: {
                    title: title,
                    listed_price: price,
                    item_type: "Unsupported SUV Brand"
                },
                pageContext: pageContext
            };
        }

        const missingFields = [];
        if (!make)    missingFields.push('Make');
        if (!model)   missingFields.push('Model');
        if (!year)    missingFields.push('Year');
        if (!fuelType)missingFields.push('Fuel Type');
        if (!gear)    missingFields.push('Gear/Transmission');
        if (!price || isNaN(price) || price <= 0) missingFields.push('Listing Price');

        // ── Step 4: Return ────────────────────────────────────────────────
        return {
            category: 'vehicle',
            valid: missingFields.length === 0,
            missing_fields: missingFields,
            error_message: missingFields.length > 0
                ? `Could not auto-extract: [${missingFields.join(', ')}]. Please fill in below.`
                : null,
            data: {
                // ── PredictRequest API fields ─────────────────────────────
                brand:        make        || '',
                model:        model       || '',
                variant:      variant,          // empty string for 121/141
                model_year:   year        || 2015,
                mileage_km:   mileage,          
                fuel_type:    fuelType    || 'Petrol',
                transmission: gear        || 'Automatic',
                engine_cc:    engineCC,         
                vehicle_type: scraped.vehicle_type || 'cars',
                description:  scraped.description || (pageContext && pageContext.description) || null,

                // ── Display / UI fields ───────────────────────────────────
                title:        title,
                condition:    condition,
                listed_price: price,
                make:         make        || '',
                year:         year        || 2015,
                mileage:      mileage,
                gear:         gear        || 'Automatic',
                fuelType:     fuelType    || 'Petrol',
            }
        };
    }

    // ── Public API ────────────────────────────────____________________________
    return {
        parse: parse,
        scrapeRiyasewana: scrapeRiyasewana     
    };

})();