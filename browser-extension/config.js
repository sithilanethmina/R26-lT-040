/**
 * Centralized Configuration for FairPriceLK Browser Extension
 */
const rootScope = typeof globalThis !== 'undefined' ? globalThis : (typeof self !== 'undefined' ? self : window);

var CONFIG = rootScope.CONFIG || {
    // Live Deployed Backend API Gateway URL (Heroku)
    // API_BASE_URL: 'https://fairpricelk-api-8cc091b2d27f.herokuapp.com',

    // Local Development URL (uncomment for local development testing)
    API_BASE_URL: 'http://127.0.0.1:8000',
};
rootScope.CONFIG = CONFIG;

// Export for module bundlers / node if applicable
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CONFIG;
}
