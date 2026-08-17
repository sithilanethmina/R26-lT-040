/**
 * Centralized Configuration for FairPriceLK Browser Extension
 */
const CONFIG = {
    // Live Deployed Backend API Gateway URL (Heroku)
    API_BASE_URL: 'https://fairpricelk-api-8cc091b2d27f.herokuapp.com',

    // Local Development URL (uncomment for local development testing)
    API_BASE_URL: 'http://localhost:8000',
};

// Export for module bundlers / node if applicable
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CONFIG;
}
