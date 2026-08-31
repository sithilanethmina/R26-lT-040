/**
 * FairPriceLK - Universal Fairness Calculator & Explainability Engine
 * Evaluates listing asking price relative to predicted fair market value.
 * Supports GPU, Mobile, Vehicle, and Electronics with tailored risk advice.
 */

window.FairPriceLK_Fairness = (function () {

    /**
     * Compute comprehensive fairness metrics and advice.
     * @param {number} listedPrice - Seller asking price in LKR
     * @param {number} pointPrice - Model's predicted point estimate in LKR
     * @param {number} lowerPrice - Fair market range lower bound in LKR
     * @param {number} upperPrice - Fair market range upper bound in LKR
     * @param {string} category - "vehicle" | "mobile" | "gpu" | "electronics"
     * @param {object} itemDetails - Optional extracted details (e.g. mileage, year, storage, condition)
     * @returns {object} Fairness assessment with score, badge, colors, and actionable advice.
     */
    function evaluate(listedPrice, pointPrice, lowerPrice, upperPrice, category = "gpu", itemDetails = {}) {
        if (!listedPrice || listedPrice <= 0) {
            return {
                status: "NO_PRICE",
                score: null,
                tier: "NO_PRICE",
                badgeText: "Price Not Listed",
                badgeClass: "neutral",
                color: "#71717A",
                diffPercent: 0,
                diffLkr: 0,
                headline: "No Asking Price Provided",
                advice: "Enter or detect a seller asking price to calculate market fairness.",
                actionAdvice: null,
                negotiationTarget: null,
                isSuspicious: false
            };
        }

        const fairMid = pointPrice || ((lowerPrice + upperPrice) / 2.0);
        if (!fairMid || fairMid <= 0) {
            return {
                status: "NO_BASELINE",
                score: null,
                tier: "NO_BASELINE",
                badgeText: "Market Baseline Unavailable",
                badgeClass: "neutral",
                color: "#71717A",
                diffPercent: 0,
                diffLkr: 0,
                headline: "Awaiting Model Baseline",
                advice: "Listing details are insufficient to determine a fair market baseline.",
                actionAdvice: null,
                negotiationTarget: null,
                isSuspicious: false
            };
        }

        const diffLkr = listedPrice - fairMid;
        const diffPercent = (diffLkr / fairMid) * 100.0;
        const hasBounds = Boolean(lowerPrice > 0 && upperPrice > 0 && upperPrice > lowerPrice);
        const halfWidth = hasBounds ? (upperPrice - lowerPrice) / 2.0 : (fairMid * 0.10);
        const isWithinRange = hasBounds 
            ? (listedPrice >= lowerPrice && listedPrice <= upperPrice)
            : Math.abs(diffPercent) <= 10;

        // ── 1. Continuous Price Fairness Index (0 to 100) ──────────────────────
        // - Peaks at 100 when price equals fair midpoint
        // - High (80 to 100) within the empirical conformal interval [lower, upper]
        // - Smoothly decays outside the interval
        // 1. Standard Continuous Math (Used directly for non-vehicles, and as Base for vehicles)
        let score = 80;
        if (isWithinRange) {
            const distFromMid = Math.abs(listedPrice - fairMid);
            score = Math.round(100 - (distFromMid / Math.max(halfWidth, 1)) * 20);
            score = Math.min(100, Math.max(80, score));
        } else if (listedPrice < (lowerPrice || fairMid * 0.9)) {
            const excessDist = (lowerPrice || fairMid * 0.9) - listedPrice;
            const decay = Math.pow(excessDist / Math.max(halfWidth, 1), 1.3);
            score = Math.max(5, Math.min(79, Math.round(80 - (decay * 25))));
        } else {
            const excessDist = listedPrice - (upperPrice || fairMid * 1.1);
            const decay = Math.pow(excessDist / Math.max(halfWidth, 1), 1.3);
            score = Math.max(5, Math.min(79, Math.round(80 - (decay * 25))));
        }

        // 2. VEHICLE SPECIFIC LOGIC (NLP Modifier + 95 Cap)
        let nlpModifier = 0;
        let nlpExplanation = "";

        if (category === "vehicle") {
            let baseScore = Math.min(95, score); // Cap base at 95 for vehicles
            
            const isExtremeUnderprice = hasBounds ? (listedPrice < (lowerPrice - 1.5 * halfWidth)) : (diffPercent < -35);
            const isExtremeOverprice = hasBounds ? (listedPrice > (upperPrice + 1.2 * halfWidth)) : (diffPercent > 25);

            if (itemDetails && itemDetails.nlp_score !== undefined) {
                const nlp = itemDetails.nlp_score;
                if (nlp >= 65) {
                    nlpModifier = 5;
                    nlpExplanation = "Good description provided (+5 points).";
                } else if (nlp < 45 && nlp > 0) {
                    nlpModifier = -15;
                    nlpExplanation = "High-risk keywords detected in description (-15 points).";
                } else if (nlp > 0) {
                    nlpExplanation = "Standard description detected.";
                }

                // GATING RULE: No positive bonus for suspicious pricing
                if (isExtremeUnderprice || isExtremeOverprice) {
                    nlpModifier = Math.min(0, nlpModifier);
                    if (nlp >= 65) {
                        nlpExplanation = "Good description detected, but bonus disabled due to high-risk price variance.";
                    }
                }
            }
            score = Math.max(5, Math.min(95, baseScore + nlpModifier)); // Final vehicle score
        }

        let tier = "FAIR_PRICE";
        let badgeText = "Fair Market Price";
        let badgeClass = "fair";
        let color = "#2563EB";
        let headline = "Within Normal Market Range";
        let advice = "This asking price matches current market trends for this specification.";
        let actionAdvice = "Good standard price. If negotiating, a modest 3-5% discount is typical.";
        let isSuspicious = false;

        // ── 2. Statistically Anchored Tiering ─────────────────────────────────
        const isExtremeUnderprice = hasBounds 
            ? (listedPrice < (lowerPrice - 1.5 * halfWidth))
            : (diffPercent < -35);

        const isBelowBound = hasBounds
            ? (listedPrice < lowerPrice)
            : (diffPercent < -10);

        const isSlightlyAbove = hasBounds
            ? (listedPrice > upperPrice && listedPrice <= (upperPrice + 1.2 * halfWidth))
            : (diffPercent > 10 && diffPercent <= 25);

        const isExtremeOverprice = hasBounds
            ? (listedPrice > (upperPrice + 1.2 * halfWidth))
            : (diffPercent > 25);

        // TIER 1: Unusually Low (Extreme Underpricing / Hardware Fault Risk)
        if (isExtremeUnderprice) {
            tier = "SUSPICIOUS_LOW";
            badgeText = "⚠️ Suspiciously Low Price";
            badgeClass = "suspicious";
            color = "#DC2626";
            isSuspicious = true;
            headline = `Listed ~${Math.abs(Math.round(diffPercent))}% below estimated market price`;
            
            const categoryRisks = {
                gpu: "The asking price is unusually low compared to normal market rates. Second-hand listings priced this low usually indicate seller-noted hardware faults (e.g., faulty VRAM, dead video ports, thermal issues), ex-mining wear, or parts-only sales.",
                mobile: "The asking price is unusually low compared to normal market rates. This often indicates disclosed defects (e.g. cracked screen, FaceID/TouchID issues, degraded battery) or network/carrier locks.",
                vehicle: "The vehicle asking price is far below typical market rates. This typically reflects down-payment lease listings or major accident history.",
                electronics: "Hardware priced far below normal market rates commonly indicates internal component damage or missing accessories."
            };
            advice = categoryRisks[category] || "The asking price is unusually low compared to normal market rates. Please inspect the listing description carefully for disclosed defects or parts-only condition.";
        }
        // TIER 2: High Value Deal (Below Lower Prediction Bound)
        else if (isBelowBound) {
            tier = "GREAT_DEAL";
            badgeText = "🟢 Great Deal";
            badgeClass = "great-deal";
            color = "#16A34A";
            headline = `Listed ~${Math.abs(Math.round(diffPercent))}% below market midpoint`;
            advice = "Competitively priced below the typical market range. Attractive value opportunity provided standard functional checks pass.";
        }
        // TIER 3: Fair Market Price (Within Calibrated Range)
        else if (isWithinRange) {
            tier = "FAIR_PRICE";
            badgeText = "🔵 Fair Price";
            badgeClass = "fair";
            color = "#2563EB";
            headline = diffPercent < 0 
                ? `Priced ~${Math.abs(Math.round(diffPercent))}% below midpoint (inside fair range)`
                : `Priced ~${Math.round(diffPercent)}% within fair market range`;
            advice = "Asking price matches the fair market value for this specification.";
        }
        // TIER 4: Slightly Above Baseline Range
        else if (isSlightlyAbove) {
            tier = "SLIGHTLY_HIGH";
            badgeText = "🟡 Slightly Above Average";
            badgeClass = "high";
            color = "#D97706";
            headline = `Listed ~${Math.round(diffPercent)}% above expected market price`;
            
            const categoryHighNote = {
                gpu: "Asking price is slightly above average. Often justifiable if remaining company/agent warranty, original box, or mint condition is provided.",
                mobile: "Asking price is slightly above average. Often accompanied by remaining official warranty or complete original accessories.",
                vehicle: "Asking price is slightly above average. Often reflects verified low mileage or agent service records.",
                electronics: "Priced slightly above average. Often justified by active agent warranty or pristine cosmetic condition."
            };
            advice = categoryHighNote[category] || "Asking price is somewhat above the typical market range. Room for price negotiation.";
        }
        // TIER 5: Significantly Overpriced
        else {
            tier = "OVERPRICED";
            badgeText = "🔴 Overpriced";
            badgeClass = "overpriced";
            color = "#EF4444";
            headline = `Listed ~${Math.round(diffPercent)}% above market price`;
            
            const categoryOverpricedNote = {
                gpu: "Asking price significantly exceeds the expected second-hand market range. Negotiation advised unless substantial active warranty is included.",
                mobile: "Asking price significantly exceeds second-hand market rates. Noticeable markup.",
                vehicle: "Asking price is significantly above the baseline market rate.",
                electronics: "Asking price is significantly above verified second-hand market listings."
            };
            advice = categoryOverpricedNote[category] || "Asking price is significantly higher than normal second-hand market value.";
        }

        // Breakdown & Calculation Factor Details
        let factors = [];
        let formulaText = "";

        if (category === "vehicle" && itemDetails && itemDetails.nlp_score !== undefined && itemDetails.nlp_score > 0) {
            factors.push({
                name: "Description Analysis (NLP)",
                impact: nlpModifier > 0 ? "Favorable" : (nlpModifier < 0 ? "Caution" : "Neutral"),
                value: itemDetails.nlp_score + "/100",
                desc: nlpExplanation + " Verdict: " + (itemDetails.nlp_verdict || "N/A")
            });
        }

        if (isExtremeUnderprice) {
            formulaText = `Unusually Low Price: Listed well below estimated market range [${formatLKR(lowerPrice)}]. Score: ${score}/100`;
            factors.push({
                name: "Market Price Range",
                impact: "Outside",
                value: "Well Below Range",
                desc: `Listed below estimated market lower range of ${formatLKR(lowerPrice || (fairMid * 0.85))}.`
            });
            factors.push({
                name: "Condition & Risk Check",
                impact: "Caution",
                value: "Verify Description",
                desc: "Prices this low usually indicate seller-noted defects, hardware issues, or down-payment terms."
            });
        } else if (isBelowBound) {
            formulaText = `Great Deal: Priced below typical market price (${formatLKR(lowerPrice)}). Score: ${score}/100`;
            factors.push({
                name: "Market Comparison",
                impact: "Positive",
                value: `${Math.abs(Math.round(diffPercent))}% Below Mid`,
                desc: `Competitively priced below the expected market baseline of ${formatLKR(fairMid)}.`
            });
            factors.push({
                name: "Market Range",
                impact: "Favorable",
                value: "Below Average Range",
                desc: `Fair market range: ${formatLKR(lowerPrice)} – ${formatLKR(upperPrice)}.`
            });
        } else if (isWithinRange) {
            formulaText = `Fair Market Price: Inside expected market range [${formatLKR(lowerPrice)} – ${formatLKR(upperPrice)}]. Score: ${score}/100`;
            factors.push({
                name: "Market Price Range",
                impact: "Optimal",
                value: "Within Fair Range",
                desc: `Comfortably within the estimated fair market range (${formatLKR(lowerPrice)} – ${formatLKR(upperPrice)}).`
            });
            factors.push({
                name: "Market Alignment",
                impact: "Neutral",
                value: `${diffPercent >= 0 ? '+' : ''}${Math.round(diffPercent)}% of Mid`,
                desc: `Aligns with verified second-hand market listings (Midpoint: ${formatLKR(fairMid)}).`
            });
        } else if (isSlightlyAbove) {
            formulaText = `Slightly Above Average: Listed above upper market range (${formatLKR(upperPrice)}). Score: ${score}/100`;
            factors.push({
                name: "Market Price Range",
                impact: "Above Range",
                value: `+${Math.round(diffPercent)}% vs Mid`,
                desc: `Exceeds the upper market range of ${formatLKR(upperPrice)}. Verify whether warranty or mint condition justifies the price.`
            });
        } else {
            formulaText = `High Price: Listed well above typical market value. Score: ${score}/100`;
            factors.push({
                name: "Market Comparison",
                impact: "High Price",
                value: `+${Math.round(diffPercent)}% Premium`,
                desc: `Significantly above fair market midpoint of ${formatLKR(fairMid)} (Difference: +${formatLKR(diffLkr)}).`
            });
        }

        const breakdown = {
            askingPrice: listedPrice,
            fairMidpoint: Math.round(fairMid),
            lowerBound: Math.round(lowerPrice),
            upperBound: Math.round(upperPrice),
            diffPercent: Math.round(diffPercent * 10) / 10,
            diffLkr: Math.round(diffLkr),
            formulaExplanation: formulaText,
            isWithinRange: isWithinRange,
            factors: factors
        };

        return {
            status: "OK",
            score: score,
            tier: tier,
            badgeText: badgeText,
            badgeClass: badgeClass,
            color: color,
            diffPercent: Math.round(diffPercent * 10) / 10,
            diffLkr: Math.round(diffLkr),
            headline: headline,
            advice: advice,
            actionAdvice: null,
            negotiationTarget: null,
            isSuspicious: isSuspicious,
            fairMidpoint: Math.round(fairMid),
            breakdown: breakdown
        };
    }

    function formatLKR(amount) {
        if (!amount || isNaN(amount)) return "Rs. --";
        if (amount >= 100000) {
            return `Rs. ${(amount / 100000).toFixed(2).replace(/\.00$/, '')} Lakh`;
        }
        return `Rs. ${Math.round(amount).toLocaleString('en-LK')}`;
    }

    return {
        evaluate: evaluate,
        formatLKR: formatLKR
    };
})();
