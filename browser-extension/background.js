/**
 * FairPriceLK - Extension Background Service Worker
 * Handles network requests with elevated extension privileges to bypass webpage CSP / Mixed Content restrictions
 */

importScripts("config.js");

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "predict_price") {
        const category = request.category || "gpu";
        const payload = request.payload || {};
        let apiBase = "http://127.0.0.1:8000";
        if (typeof CONFIG !== "undefined" && CONFIG.API_BASE_URL) {
            apiBase = CONFIG.API_BASE_URL;
        } else if (typeof self !== "undefined" && self.CONFIG && self.CONFIG.API_BASE_URL) {
            apiBase = self.CONFIG.API_BASE_URL;
        }

        const endpoint = `${apiBase}/api/${category}/predict`;
        console.log(`[FairPriceLK Background] Fetching ${endpoint} with payload:`, payload);

        fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        })
        .then(async (res) => {
            if (!res.ok) {
                const errJson = await res.json().catch(() => ({}));
                sendResponse({ success: false, error: errJson.detail || `Server error (${res.status})` });
            } else {
                const data = await res.json();
                console.log("[FairPriceLK Background] Prediction received:", data);
                sendResponse({ success: true, data: data });
            }
        })
        .catch((err) => {
            console.error("[FairPriceLK Background] Fetch error:", err);
            sendResponse({ success: false, error: `${err.name}: ${err.message}` });
        });

        // Return true to indicate asynchronous response
        return true;
    }
});
