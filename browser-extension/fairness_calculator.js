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

        let score = 75;
        let tier = "FAIR_PRICE";
        let badgeText = "Fair Market Price";
        let badgeClass = "fair";
        let color = "#2563EB";
        let headline = "Within Normal Market Range";
        let advice = "This asking price matches current market trends for this specification.";
        let actionAdvice = "Good standard price. If negotiating, a modest 3-5% discount is typical.";
        let isSuspicious = false;

        // TIER 1: Unusually Low (Potential Defects, Parts-Only, Down-Payment Trick)
        if (diffPercent < -35) {
            tier = "SUSPICIOUS_LOW";
            badgeText = "⚠️ Suspiciously Low";
            badgeClass = "suspicious";
            color = "#DC2626";
            score = Math.max(15, Math.min(40, Math.round(50 + (diffPercent * 0.4))));
            isSuspicious = true;
            headline = `Listed ~${Math.abs(Math.round(diffPercent))}% below market average`;
            
            const categoryRisks = {
                gpu: "GPU priced far below market baseline. Sellers usually price items this low due to defects mentioned in the description (e.g., artifacting, dead display ports, overheating, fan noise, or BIOS issues). Check the listing description carefully.",
                mobile: "Unusually low mobile price. Often indicates defects mentioned in description (e.g. cracked display, FaceID/TouchID failure, battery service) or carrier/iCloud locks.",
                vehicle: "Vehicle listing far below market. Often represents leasing down-payment amounts or salvage/accident history.",
                electronics: "Hardware priced far below baseline may have unrepairable component faults or missing accessories."
            };
            advice = categoryRisks[category] || "Price is unusually far below typical second-hand market listings. Check listing description for noted defects or issues.";
        }
        // TIER 2: Great Deal (-35% to -10%)
        else if (diffPercent < -10) {
            tier = "GREAT_DEAL";
            badgeText = "🟢 Great Deal";
            badgeClass = "great-deal";
            color = "#16A34A";
            score = Math.round(85 + ((-diffPercent - 10) / 25) * 15);
            score = Math.min(100, Math.max(85, score));
            headline = `Listed ~${Math.abs(Math.round(diffPercent))}% below fair market value`;
            advice = "Competitively priced below typical market average for similar items.";
        }
        // TIER 3: Fair Market Price (-10% to +10%)
        else if (diffPercent <= 10) {
            tier = "FAIR_PRICE";
            badgeText = "🔵 Fair Price";
            badgeClass = "fair";
            color = "#2563EB";
            score = Math.round(84 - (Math.abs(diffPercent) / 10) * 14);
            headline = diffPercent < 0 
                ? `Priced ~${Math.abs(Math.round(diffPercent))}% below average`
                : `Priced ~${Math.round(diffPercent)}% within fair market average`;
            advice = "Asking price is well within standard second-hand market expectations.";
        }
        // TIER 4: Slightly Overpriced (+10% to +25%)
        else if (diffPercent <= 25) {
            tier = "SLIGHTLY_HIGH";
            badgeText = "🟡 Slightly Overpriced";
            badgeClass = "high";
            color = "#D97706";
            score = Math.round(69 - ((diffPercent - 10) / 15) * 19);
            headline = `Listed ~${Math.round(diffPercent)}% above expected market average`;
            
            const categoryHighNote = {
                gpu: "Asking price is above baseline. Second-hand GPUs in this range often command a premium if the seller provides remaining company/agent warranty, full box, or brand-new condition.",
                mobile: "Asking price is above baseline. Often due to remaining official warranty or mint condition with original accessories.",
                vehicle: "Asking price is above average. Often associated with low genuine mileage or comprehensive service history.",
                electronics: "Priced above average baseline. Often justified if agent warranty remains or accessories are included."
            };
            advice = categoryHighNote[category] || "Asking price is somewhat higher than comparable verified baseline listings.";
        }
        // TIER 5: Significantly Overpriced (> +25%)
        else {
            tier = "OVERPRICED";
            badgeText = "🔴 Overpriced";
            badgeClass = "overpriced";
            color = "#EF4444";
            score = Math.max(10, Math.round(49 - Math.min(39, (diffPercent - 25) * 0.8)));
            headline = `Listed ~${Math.round(diffPercent)}% above fair market value`;
            
            const categoryOverpricedNote = {
                gpu: "Asking price is significantly above baseline market rate. Check whether the listing includes extensive remaining official warranty, sealed packaging, or high-tier aftermarket cooling.",
                mobile: "Asking price is significantly above market rate. Likely includes brand-new condition or long active warranty.",
                vehicle: "Asking price is significantly above baseline market rate.",
                electronics: "Asking price is significantly above typical baseline listings."
            };
            advice = categoryOverpricedNote[category] || "Asking price is significantly higher than typical baseline market value.";
        }

        // Breakdown & Calculation Factor Details
        let factors = [];
        let formulaText = "";

        const isWithinRange = (lowerPrice > 0 && upperPrice > 0)
            ? (listedPrice >= lowerPrice && listedPrice <= upperPrice)
            : Math.abs(diffPercent) <= 10;

        if (diffPercent < -35) {
            formulaText = `Underprice Anomaly: Base 50 - (${Math.abs(Math.round(diffPercent))} * 0.4) = ${score}/100`;
            factors.push({
                name: "Market Baseline Comparison",
                impact: "Negative",
                value: `${Math.round(diffPercent)}% vs. Midpoint`,
                desc: `Asking price is significantly below fair market average (${formatLKR(fairMid)}).`
            });
            factors.push({
                name: "Defect / Condition Note",
                impact: "Penalty",
                value: "Check Description",
                desc: "Items priced this far below baseline typically have faults/defects noted in description (e.g. ports, thermals, fans, or display issues)."
            });
            factors.push({
                name: "Range Confidence",
                impact: "Outside",
                value: "Below Lower Bound",
                desc: `Listed below estimated market lower boundary of ${formatLKR(lowerPrice || (fairMid * 0.85))}.`
            });
        } else if (diffPercent < -10) {
            formulaText = `Great Deal Calculation: Base 85 + ((${Math.abs(Math.round(diffPercent))} - 10) / 25 * 15) = ${score}/100`;
            factors.push({
                name: "Market Baseline Comparison",
                impact: "Positive",
                value: `${Math.abs(Math.round(diffPercent))}% Below Mid`,
                desc: `Competitively priced below the expected market baseline of ${formatLKR(fairMid)}.`
            });
            factors.push({
                name: "Market Range Fit",
                impact: "Favorable",
                value: isWithinRange ? "Within Bounds" : "Near Lower Bound",
                desc: `Fair market interval: ${formatLKR(lowerPrice)} – ${formatLKR(upperPrice)}.`
            });
            factors.push({
                name: "Market Valuation Tier",
                impact: "Low",
                value: "High Value",
                desc: "Price is attractive relative to standard market distribution."
            });
        } else if (diffPercent <= 10) {
            formulaText = `Fair Range Valuation: Base 84 - (${Math.abs(Math.round(diffPercent))} / 10 * 14) = ${score}/100`;
            factors.push({
                name: "Market Baseline Comparison",
                impact: "Neutral",
                value: `${diffPercent >= 0 ? '+' : ''}${Math.round(diffPercent)}% of Mid`,
                desc: `Matches typical validated seller prices (Market midpoint: ${formatLKR(fairMid)}).`
            });
            factors.push({
                name: "Market Range Fit",
                impact: "Optimal",
                value: "Inside Range",
                desc: `Firmly situated within estimated confidence range (${formatLKR(lowerPrice)} – ${formatLKR(upperPrice)}).`
            });
            factors.push({
                name: "Market Distribution",
                impact: "Normal",
                value: "Standard Range",
                desc: "Asking price aligns with average second-hand market listings."
            });
        } else if (diffPercent <= 25) {
            formulaText = `Above Baseline Deduction: Base 69 - ((${Math.round(diffPercent)} - 10) / 15 * 19) = ${score}/100`;
            factors.push({
                name: "Market Baseline Comparison",
                impact: "Negative",
                value: `+${Math.round(diffPercent)}% Above Mid`,
                desc: `Exceeds the predicted fair value baseline of ${formatLKR(fairMid)}.`
            });
            factors.push({
                name: "Warranty / Condition Premium",
                impact: "Outside",
                value: "Check Warranty",
                desc: "Premium prices in this tier are commonly accompanied by remaining company/agent warranty, original packaging, or mint condition."
            });
            factors.push({
                name: "Market Range Fit",
                impact: "Outside",
                value: "Above Upper Bound",
                desc: `Priced above standard baseline upper confidence interval (${formatLKR(upperPrice)}).`
            });
        } else {
            formulaText = `High Premium Deduction: Base 49 - ((${Math.round(diffPercent)} - 25) * 0.8) = ${score}/100`;
            factors.push({
                name: "Market Baseline Comparison",
                impact: "Heavy Penalty",
                value: `+${Math.round(diffPercent)}% Premium`,
                desc: `Significantly above fair market midpoint of ${formatLKR(fairMid)} (Difference: +${formatLKR(diffLkr)}).`
            });
            factors.push({
                name: "Warranty & Extras Check",
                impact: "Outlier",
                value: "High Markup",
                desc: "Unless accompanied by substantial active warranty, brand-new condition, or rare custom editions, this price carries a steep markup."
            });
            factors.push({
                name: "Market Range Fit",
                impact: "Outlier",
                value: "Extreme Deviation",
                desc: `Far exceeds standard secondary market upper boundary of ${formatLKR(upperPrice)}.`
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
