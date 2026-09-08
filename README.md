# Hybrid Threat Analysis and Cyber Defence Decision Support System

> An Explainable Hybrid AI and RAG-Based Approach to Cybersecurity Alert Triage and Decision Support

---

## Overview

A web-based cybersecurity decision-support platform that assists security analysts in
detecting, understanding, prioritising, and responding to suspicious network activity.

**Architecture at a glance:**

```
Network Flows (CIC-IDS2017)
        │
        ▼
 ML Threat Engine (Random Forest / XGBoost)
        │
        ▼
  SHAP Explainability + Security Rule Engine
        │
        ▼
   Event Correlation → Risk Scoring
        │
        ▼
  RAG Knowledge Retrieval (MITRE ATT&CK / NIST)
        │
        ▼
  Gemini LLM → Grounded Analyst Recommendations
        │
        ▼
     Next.js Analyst Dashboard
```

**Core principle:** The ML system determines what the data indicates. The RAG/LLM layer
is a secondary decision-support tool — it never invents evidence not present in context.

---

## Technology Stack

| Layer | Technology |
|---|---|
| ML / XAI / RAG service | Python 3.14 · FastAPI · LightGBM (RF baseline) · XGBoost · SHAP · sentence-transformers · Chroma |
| Backend orchestration | Node.js 22 · Express 4 · Mongoose · Axios |
| Frontend | Next.js 14 (App Router) · TypeScript |
| Database | MongoDB (local dev) · MongoDB Atlas (production) |
| LLM | Google Gemini 2.5 Flash via Gemini API |
| Dataset | CIC-IDS2017 (8-day flow-level CSVs, ~884 MB) |
| Deployment | Vercel (frontend) · Render (backend/ML) · MongoDB Atlas |

---

## Quick Start

### Prerequisites

- Python 3.14+
- Node.js 22+
- Git

### 1 — Clone & configure

```powershell
# Copy the environment template and fill in your values
Copy-Item .env.example .env
# Edit .env — add your GEMINI_API_KEY and confirm DATASET_DIR
```

### 2 — Python ML service

```powershell
cd ml_service
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
uvicorn main:app --reload --port 8000
# Health check: http://localhost:8000/health
```

### 3 — Node backend

```powershell
cd backend
npm install
npm run dev
# Health check: http://localhost:4000/api/health
```

### 4 — Next.js frontend

```powershell
cd frontend
npm install
npm run dev
# Open: http://localhost:3000
```

---

## Project Structure

```
├── .env.example          # Environment template (copy to .env)
├── .gitignore
├── README.md
├── config/
│   └── paths.py          # Canonical filesystem paths (imported by all Python modules)
├── ml_service/           # Python / FastAPI — ML, XAI, RAG
│   ├── requirements.txt
│   ├── venv/             # git-ignored
│   ├── main.py
│   └── src/
│       ├── config.py
│       └── utils/
│           └── memory_guard.py
├── backend/              # Node.js / Express — orchestration & persistence
│   ├── package.json
│   └── src/
│       ├── index.js
│       └── config.js
├── frontend/             # Next.js 14 — analyst dashboard
├── models/               # Saved model files — git-ignored, versioned filenames
├── artifacts/            # Evaluation outputs — git-ignored
└── notebooks/            # Jupyter EDA notebooks
```

---

## Development Phases

| # | Phase | Status |
|---|---|---|
| 0 | Environment Setup | 🔄 In Progress |
| 1 | Dataset Acquisition & Exploration | ⬜ |
| 2 | Data Cleaning | ⬜ |
| 3 | Feature Engineering & Train/Test Split | ⬜ |
| 4 | ML Baseline: Random Forest | ⬜ |
| 5 | ML Comparison: XGBoost | ⬜ |
| 6 | Explainability (SHAP) | ⬜ |
| 7 | Security Rule Engine | ⬜ |
| 8 | Event Correlation | ⬜ |
| 9 | Risk Scoring | ⬜ |
| 10 | RAG Knowledge Base | ⬜ |
| 11 | Gemini Integration | ⬜ |
| 12 | Backend Integration | ⬜ |
| 13 | Dashboard | ⬜ |
| 14 | Testing & Evaluation | ⬜ |
| 15 | Deployment & Documentation | ⬜ |

---

## Hardware Constraint

Development machine: **~8 GB RAM**. All data loading uses chunked reads. Models are
trained sequentially (RF → evaluate → save → free memory → XGBoost). Never load
multiple full DataFrames simultaneously.

**Empirically confirmed (2026-09-07):** loading the full cleaned dataset (2,520,798 rows)
consumed 1.778 GB RSS against ~0.860 GB free RAM; system Available RAM spiked to 28.8 MB.
Phases 3–4 must use chunked reads and/or intermediate Parquet files — never a single
full-dataset DataFrame in memory. Random Forest is implemented via LightGBM
(boosting_type='rf') for histogram-binned, memory-efficient training; XGBoost uses
tree_method='hist' for the same reason.

---

## References

- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Chen, T., & Guestrin, C. (2016). XGBoost. *KDD 2016*.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*.
- Lewis et al. (2020). Retrieval-augmented generation. *NeurIPS 33*.
- Sharafaldin et al. (2018). CIC-IDS2017 dataset. *ICISSP 2018*.
- MITRE Corporation. MITRE ATT&CK. https://attack.mitre.org
- NIST SP 800-61 Rev. 2. Computer Security Incident Handling Guide.
- Google. Gemini API Documentation. https://ai.google.dev/gemini-api/docs
