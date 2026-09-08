/**
 * backend/src/config.js
 * ─────────────────────
 * Loads and validates all environment variables required by the Express backend.
 * Fails fast at startup if required variables are absent so misconfigurations
 * surface immediately rather than inside a request handler.
 */

'use strict';

require('dotenv').config({ path: require('path').resolve(__dirname, '../../.env') });

const REQUIRED_VARS = ['MONGO_URI', 'FASTAPI_URL'];

/**
 * Validates that all required env vars are present.
 * Logs a warning for optional vars that aren't set yet (e.g. GEMINI_API_KEY).
 */
function validateEnv() {
  const missing = REQUIRED_VARS.filter((key) => !process.env[key]);
  if (missing.length > 0) {
    console.error(
      `[config] ❌  Missing required environment variables: ${missing.join(', ')}\n` +
      '         Copy .env.example to .env and fill in the values.'
    );
    process.exit(1);
  }

  if (!process.env.GEMINI_API_KEY || process.env.GEMINI_API_KEY === 'your_gemini_api_key_here') {
    console.warn('[config] ℹ️   GEMINI_API_KEY not set — expected until Phase 11.');
  }
}

validateEnv();

module.exports = {
  PORT: parseInt(process.env.BACKEND_PORT || '4000', 10),
  MONGO_URI: process.env.MONGO_URI,
  FASTAPI_URL: process.env.FASTAPI_URL || 'http://localhost:8000',
  GEMINI_API_KEY: process.env.GEMINI_API_KEY || '',
  GEMINI_MODEL: process.env.GEMINI_MODEL || 'gemini-2.5-flash',
};
