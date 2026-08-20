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

        // TIER 1: Unusually Low (Potential Risk, Down-Payment Trick, or Scam)
        if (diffPercent < -35) {
            tier = "SUSPICIOUS_LOW";
            badgeText = "⚠️ Suspiciously Low";
            badgeClass = "suspicious";
            color = "#DC2626";
            // Anomaly penalty: cap score low
            score = Math.max(15, Math.min(40, Math.round(50 + (diffPercent * 0.4))));
            isSuspicious = true;
            headline = `Listed ~${Math.abs(Math.round(diffPercent))}% below market average`;
            
            const categoryRisks = {
                vehicle: "Car/Bike listings this far below market often represent leasing down-payment amounts, salvage titles, or advance-deposit scams.",
                mobile: "Unusually low mobile prices often indicate iCloud/MDM lock, defective displays/FaceID, or advance payment scams.",
                gpu: "Severe GPU discounts frequently indicate heavily degraded ex-mining cards, modified BIOS, or counterfeit chips.",
                electronics: "Hardware priced far below baseline may have unrepairable motherboard issues or missing critical accessories."
            };
            advice = categoryRisks[category] || "Price is unusually far below typical second-hand market listings.";
            actionAdvice = "Do NOT send advance bank deposits. Insist on in-person inspection and thorough hardware/document verification.";
        }
        // TIER 2: Great Deal (-35% to -10%)
        else if (diffPercent < -10) {
            tier = "GREAT_DEAL";
            badgeText = "🟢 Great Deal";
            badgeClass = "great-deal";
            color = "#16A34A";
            // Score from 85 to 100
            score = Math.round(85 + ((-diffPercent - 10) / 25) * 15);
            score = Math.min(100, Math.max(85, score));
            headline = `Listed ~${Math.abs(Math.round(diffPercent))}% below fair market value`;
            advice = "Competitively priced below typical market average for similar items.";
            actionAdvice = "High value opportunity! Items priced this well tend to sell quickly.";
        }
        // TIER 3: Fair Market Price (-10% to +10%)
        else if (diffPercent <= 10) {
            tier = "FAIR_PRICE";
            badgeText = "🔵 Fair Price";
            badgeClass = "fair";
            color = "#2563EB";
            // Score from 70 to 84
            score = Math.round(84 - (Math.abs(diffPercent) / 10) * 14);
            headline = diffPercent < 0 
                ? `Priced ~${Math.abs(Math.round(diffPercent))}% below average`
                : `Priced ~${Math.round(diffPercent)}% within fair market average`;
            advice = "Asking price is well within standard second-hand market expectations.";
            actionAdvice = "Fair valuation. A counter-offer around " + formatLKR(lowerPrice || (fairMid * 0.93)) + " may save extra cash.";
        }
        // TIER 4: Slightly Overpriced (+10% to +25%)
        else if (diffPercent <= 25) {
            tier = "SLIGHTLY_HIGH";
            badgeText = "🟡 Slightly Overpriced";
            badgeClass = "high";
            color = "#D97706";
            // Score from 50 to 69
            score = Math.round(69 - ((diffPercent - 10) / 15) * 19);
            headline = `Listed ~${Math.round(diffPercent)}% above expected market average`;
            advice = "Asking price is somewhat higher than comparable verified listings.";
            actionAdvice = `Suggest offering ${formatLKR(fairMid)} – ${formatLKR(fairMid * 1.05)} to bring it down to fair market rate.`;
        }
        // TIER 5: Significantly Overpriced (> +25%)
        else {
            tier = "OVERPRICED";
            badgeText = "🔴 Overpriced";
            badgeClass = "overpriced";
            color = "#EF4444";
            // Score below 50
            score = Math.max(10, Math.round(49 - Math.min(39, (diffPercent - 25) * 0.8)));
            headline = `Listed ~${Math.round(diffPercent)}% above fair market value`;
            advice = "Asking price is significantly higher than market reality.";
            actionAdvice = `Substantial negotiation recommended. Fair value sits around ${formatLKR(fairMid)}.`;
        }

        // Suggested Target Offer (Target between Lower Bound and Fair Midpoint)
        const targetLow = Math.round((lowerPrice || (fairMid * 0.9)) / 500) * 500;
        const targetHigh = Math.round(fairMid / 500) * 500;
        const negotiationTarget = (diffPercent > 5) ? `Rs. ${targetLow.toLocaleString('en-LK')} – Rs. ${targetHigh.toLocaleString('en-LK')}` : null;

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
            actionAdvice: actionAdvice,
            negotiationTarget: negotiationTarget,
            isSuspicious: isSuspicious,
            fairMidpoint: Math.round(fairMid)
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
