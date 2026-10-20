/**
 * Centralized Configuration for FairPriceLK Browser Extension
 */
const rootScope = typeof globalThis !== 'undefined' ? globalThis : (typeof self !== 'undefined' ? self : window);

var CONFIG = rootScope.CONFIG || {
    // Local Development URL (point to running local API gateway)
    API_BASE_URL: 'http://127.0.0.1:8000',

    // Remote Deployed Backend API Gateway URL (commented out for local testing)
    // API_BASE_URL: 'http://161.118.224.233:8000',
};
rootScope.CONFIG = CONFIG;

// Export for module bundlers / node if applicable
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CONFIG;
}
