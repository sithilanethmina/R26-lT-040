/**
 * FairPriceLK - Extension Background Service Worker
 * Handles network requests with elevated extension privileges to bypass webpage CSP / Mixed Content restrictions,
 * and captures high-coverage Full-Page tab screenshots for Gemini multimodal extraction.
 */

importScripts("config.js");

function captureSingleTab(windowId) {
    return new Promise((resolve) => {
        chrome.tabs.captureVisibleTab(windowId, { format: "jpeg", quality: 75 }, (dataUrl) => {
            if (chrome.runtime.lastError || !dataUrl) {
                resolve(null);
            } else {
                resolve(dataUrl);
            }
        });
    });
}

/**
 * Captures full-page / high-coverage screenshot by scrolling and stitching vertical slices.
 */
async function captureFullPageScreenshot(tabId, windowId) {
    try {
        if (!tabId) {
            return await captureSingleTab(windowId);
        }

        // 1. Query page dimensions from the tab
        const dimResults = await chrome.scripting.executeScript({
            target: { tabId: tabId },
            func: () => ({
                totalHeight: Math.max(document.documentElement.scrollHeight, document.body.scrollHeight, window.innerHeight),
                viewportHeight: window.innerHeight,
                viewportWidth: window.innerWidth,
                originalScrollY: window.scrollY
            })
        });

        if (!dimResults || !dimResults[0] || !dimResults[0].result) {
            return await captureSingleTab(windowId);
        }

        const { totalHeight, viewportHeight, originalScrollY } = dimResults[0].result;

        // If page fits in one screen, single capture is sufficient
        if (totalHeight <= viewportHeight * 1.15) {
            return await captureSingleTab(windowId);
        }

        // Capture up to 3 slices (covers 100% of standard marketplace ad pages)
        const maxSlices = 3;
        const step = Math.floor(viewportHeight * 0.95);
        const positions = [];
        for (let y = 0; y < totalHeight && positions.length < maxSlices; y += step) {
            positions.push(y);
        }

        const slices = [];
        for (const pos of positions) {
            await chrome.scripting.executeScript({
                target: { tabId: tabId },
                func: (scrollY) => window.scrollTo({ top: scrollY, behavior: 'instant' }),
                args: [pos]
            });
            await new Promise(r => setTimeout(r, 120)); // Brief pause for repaint
            const sliceDataUrl = await captureSingleTab(windowId);
            if (sliceDataUrl) {
                slices.push(sliceDataUrl);
            }
        }

        // Restore original scroll position immediately
        await chrome.scripting.executeScript({
            target: { tabId: tabId },
            func: (origY) => window.scrollTo({ top: origY, behavior: 'instant' }),
            args: [originalScrollY]
        });

        if (slices.length === 0) return null;
        if (slices.length === 1) return slices[0];

        // Stitch slices vertically using OffscreenCanvas
        const bitmaps = [];
        for (const sliceDataUrl of slices) {
            const resp = await fetch(sliceDataUrl);
            const blob = await resp.blob();
            const bmp = await createImageBitmap(blob);
            bitmaps.push(bmp);
        }

        const imgWidth = bitmaps[0].width;
        const sliceHeight = bitmaps[0].height;
        const totalCanvasHeight = sliceHeight * bitmaps.length;

        const offscreen = new OffscreenCanvas(imgWidth, totalCanvasHeight);
        const ctx = offscreen.getContext('2d');

        for (let i = 0; i < bitmaps.length; i++) {
            ctx.drawImage(bitmaps[i], 0, i * sliceHeight);
        }

        const stitchedBlob = await offscreen.convertToBlob({ type: 'image/jpeg', quality: 75 });
        const buffer = await stitchedBlob.arrayBuffer();
        let binary = '';
        const bytes = new Uint8Array(buffer);
        const chunkSize = 8192;
        for (let i = 0; i < bytes.byteLength; i += chunkSize) {
            binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize));
        }
        const base64 = btoa(binary);
        const fullDataUrl = `data:image/jpeg;base64,${base64}`;

        console.log(`[FairPriceLK Background] Stitched ${slices.length} slices into full-page screenshot (${Math.round(fullDataUrl.length / 1024)} KB)`);
        return fullDataUrl;

    } catch (e) {
        console.warn("[FairPriceLK Background] Full page capture error, falling back to viewport:", e);
        return await captureSingleTab(windowId);
    }
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    const tabId = sender && sender.tab && sender.tab.id ? sender.tab.id : null;
    const windowId = sender && sender.tab && sender.tab.windowId ? sender.tab.windowId : null;

    // 1. Dedicated Screenshot Capture Action
    if (request.action === "capture_screenshot") {
        captureFullPageScreenshot(tabId, windowId).then((dataUrl) => {
            if (!dataUrl) {
                sendResponse({ success: false, error: "Capture failed" });
            } else {
                sendResponse({ success: true, image_base64: dataUrl });
            }
        });
        return true; // async
    }

    // 2. Price Prediction Action (supports text and/or screenshot)
    if (request.action === "predict_price") {
        const category = request.category || "gpu";
        const payload = request.payload || {};
        let apiBase = "http://127.0.0.1:8000";
        if (typeof CONFIG !== "undefined" && CONFIG.API_BASE_URL) {
            apiBase = CONFIG.API_BASE_URL;
        } else if (typeof self !== "undefined" && self.CONFIG && self.CONFIG.API_BASE_URL) {
            apiBase = self.CONFIG.API_BASE_URL;
        }

        const executeFetch = (finalPayload) => {
            const subpath = request.subpath ? `/${request.subpath}` : '';
            const endpoint = `${apiBase}/api/${category}/predict${subpath}`;
            console.log(`[FairPriceLK Background] Fetching ${endpoint} with payload (image attached: ${Boolean(finalPayload.image_base64)})`);

            fetch(endpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(finalPayload)
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
        };

        // If with_screenshot is requested or category is electronics, capture full page automatically
        if ((request.with_screenshot || request.category === "electronics") && !payload.image_base64) {
            captureFullPageScreenshot(tabId, windowId).then((dataUrl) => {
                if (dataUrl) {
                    payload.image_base64 = dataUrl;
                }
                executeFetch(payload);
            });
        } else {
            executeFetch(payload);
        }

        // Return true to indicate asynchronous response
        return true;
    }
});
