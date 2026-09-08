/**
 * backend/src/index.js
 * ─────────────────────
 * Express application entry point — Phase 0 stub.
 *
 * Phase 0 exposes:
 *   GET /api/health   — liveness probe; pings the FastAPI ML service
 *
 * Later phases will add routers for:
 *   /api/incidents    (Phase 12+)
 *   /api/predict      (Phase 12+)
 *   /api/recommend    (Phase 12+)
 *
 * Run:  npm run dev
 */

'use strict';

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const mongoose = require('mongoose');
const config = require('./config');

const app = express();

// ── Middleware ────────────────────────────────────────────────────────────────
app.use(cors({
  origin: ['http://localhost:3000'], // Next.js dev server
}));
app.use(express.json());

// ── Routes ────────────────────────────────────────────────────────────────────

/**
 * GET /api/health
 * Liveness probe used by the Next.js dashboard and monitoring tools.
 * Pings the FastAPI ML service and reports MongoDB connection state.
 */
app.get('/api/health', async (req, res) => {
  const result = {
    status: 'ok',
    service: 'backend',
    mongo: mongoose.connection.readyState === 1 ? 'connected' : 'disconnected',
    mlService: null,
  };

  try {
    const mlResponse = await axios.get(`${config.FASTAPI_URL}/health`, { timeout: 3000 });
    result.mlService = mlResponse.data;
  } catch (err) {
    // ML service is unreachable — report but do not hard-fail.
    // The core backend must remain usable if the ML service is down (fail-safe requirement).
    result.mlService = { status: 'unreachable', error: err.message };
    result.status = 'degraded';
  }

  const httpStatus = result.status === 'ok' ? 200 : 207;
  res.status(httpStatus).json(result);
});

// ── Database connection ───────────────────────────────────────────────────────
async function connectDb() {
  try {
    await mongoose.connect(config.MONGO_URI);
    console.log('[backend] ✅  MongoDB connected');
  } catch (err) {
    // Log and continue — MongoDB is not required for Phase 0 health checks.
    console.warn(`[backend] ⚠️   MongoDB connection failed: ${err.message}`);
    console.warn('[backend]      Start MongoDB or update MONGO_URI in .env to resolve.');
  }
}

// ── Start ─────────────────────────────────────────────────────────────────────
async function start() {
  await connectDb();
  app.listen(config.PORT, () => {
    console.log(`[backend] 🚀  Listening on http://localhost:${config.PORT}`);
    console.log(`[backend]      ML service: ${config.FASTAPI_URL}`);
    console.log(`[backend]      Health:     http://localhost:${config.PORT}/api/health`);
  });
}

start();

module.exports = app; // exported for testing
