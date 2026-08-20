/**
 * FairPriceLK - Vehicle Category Extractor
 * Parses Vehicle listings (Toyota Corolla, Aqua, Alto, etc.) from marketplace pages.
 */

window.FairPriceLK_Extractors = window.FairPriceLK_Extractors || {};

window.FairPriceLK_Extractors.vehicle = (function () {
    const TARGET_MODELS = [
        "Toyota Corolla",
        "Toyota Aqua",
        "Suzuki Alto"
    ];

    function extractModel(combinedText) {
        const lower = (combinedText || "").toLowerCase();
        if (lower.includes("aqua")) return "Toyota Aqua";
        if (lower.includes("alto")) return "Suzuki Alto";
        if (lower.includes("corolla") || lower.includes("121") || lower.includes("141")) return "Toyota Corolla";
        return "Toyota Corolla";
    }

    function extractYear(combinedText) {
        const match = (combinedText || "").match(/\b(19\d\d|20[0-2]\d)\b/);
        if (match) {
            const y = parseInt(match[1], 10);
            if (y >= 1990 && y <= 2026) return y;
        }
        return 2015;
    }

    function extractTransmission(combinedText) {
        const lower = (combinedText || "").toLowerCase();
        if (lower.includes("manual")) return "Manual";
        return "Automatic";
    }

    function extractFuel(combinedText) {
        const lower = (combinedText || "").toLowerCase();
        if (lower.includes("hybrid")) return "Hybrid";
        if (lower.includes("diesel")) return "Diesel";
        return "Petrol";
    }

    function parse(pageContext) {
        const { title = "", price = null, raw_text = "", key_values = {} } = pageContext;
        const scope = `${title} ${key_values.model || ""} ${raw_text}`;

        const model = extractModel(scope);
        const year = extractYear(scope);
        const transmission = extractTransmission(scope);
        const fuel = extractFuel(scope);
        const variant = key_values.variant || (model === "Toyota Aqua" ? "G Grade" : (model === "Suzuki Alto" ? "800" : "121"));

        const missingFields = [];
        if (!price || isNaN(price) || price <= 0) missingFields.push("Listing Price");

        return {
            category: "vehicle",
            valid: missingFields.length === 0,
            missing_fields: missingFields,
            error_message: missingFields.length > 0 ? `Missing required vehicle details: [${missingFields.join(", ")}].` : null,
            data: {
                model: model,
                model_year: year,
                variant: variant,
                transmission: transmission,
                fuel_type: fuel,
                listed_price: price || null
            }
        };
    }

    return { parse: parse };
})();
