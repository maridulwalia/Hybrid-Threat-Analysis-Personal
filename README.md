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
├── RESEARCH_PAPER_TRACKER.md   # Living methodology & results document
├── config/
│   └── paths.py          # Canonical filesystem paths (imported by all Python modules)
├── ml_service/           # Python / FastAPI — ML, XAI, RAG
│   ├── requirements.txt
│   ├── venv/             # git-ignored
│   ├── main.py
│   └── src/
│       ├── config.py
│       ├── feature_engineer.py   # Phase 3 — 3-pass streaming split pipeline
│       ├── train_rf.py           # Phase 4 — Random Forest baseline training & evaluation
│       ├── train_xgb.py          # Phase 5 — XGBoost comparative model training & evaluation
│       └── utils/
│           └── memory_guard.py
├── backend/              # Node.js / Express — orchestration & persistence
│   ├── package.json
│   └── src/
│       ├── index.js
│       └── config.js
├── frontend/             # Next.js 14 — analyst dashboard
├── models/               # Saved model files — rf_baseline_category_v1.*, xgb_baseline_category_v1.*
├── artifacts/            # Evaluation outputs & metrics
│   ├── cleaned/          # Phase 2: 8 cleaned CSVs (2,520,798 flows)
│   ├── splits/           # Phase 3: X/y Parquet splits + scaler/encoder artifacts
│   ├── rf_metrics_v1.json# Phase 4: Machine-readable Random Forest evaluation metrics
│   ├── rf_metrics_v1.md  # Phase 4: Formatted Markdown evaluation report
│   ├── xgb_metrics_v1.json# Phase 5: Machine-readable XGBoost evaluation metrics
│   ├── xgb_metrics_v1.md # Phase 5: Formatted Markdown evaluation report
│   └── model_selection_decision_v1.md # Phase 5: Primary classifier selection audit
└── notebooks/            # Jupyter verification & analysis notebooks
    ├── 01_eda_placeholder.ipynb
    ├── 02_data_cleaning.ipynb
    ├── 03_feature_engineering.ipynb
    ├── 04_random_forest_baseline.ipynb
    └── 05_xgboost_comparison.ipynb
```

---

## Development Phases

| # | Phase | Status |
|---|---|---|
| 0 | Environment Setup | ✅ Completed |
| 1 | Dataset Acquisition & Exploration | ✅ Completed |
| 2 | Data Cleaning | ✅ Completed |
| 3 | Feature Engineering & Train/Test Split | ✅ Completed |
| 4 | ML Baseline: Random Forest | ✅ Completed |
| 5 | ML Comparison: XGBoost | ✅ Completed |
| 6 | Explainability (SHAP) | ⬜ Not started |
| 7 | Security Rule Engine | ⬜ Not started |
| 8 | Event Correlation | ⬜ Not started |
| 9 | Risk Scoring | ⬜ Not started |
| 10 | RAG Knowledge Base | ⬜ Not started |
| 11 | Gemini Integration | ⬜ Not started |
| 12 | Backend Integration | ⬜ Not started |
| 13 | Dashboard | ⬜ Not started |
| 14 | Testing & Evaluation | ⬜ Not started |
| 15 | Deployment & Documentation | ⬜ Not started |

---

## Hardware Constraint

Development machine: **~8 GB RAM**. All data loading uses chunked reads. Models are
trained sequentially (RF → evaluate → save → free memory → XGBoost). Never load
multiple full DataFrames simultaneously.

**Empirically confirmed (2026-09-06, check_ram.py):** loading the full cleaned dataset
(2,520,798 rows × 79 columns, float64) consumed 1.595 GB deep / 1.778 GB RSS delta
against ~0.860 GB free RAM; Task Manager confirmed Available RAM spiked to 28.8 MB.
Phases 3+ use chunked Parquet reads and never hold a full-dataset DataFrame in memory.

**Phase 3 measured results (2026-09-08):** 3-pass streaming pipeline processed 2,520,798
rows across 8 CSVs. Train: 2,016,638 rows · Test: 504,160 rows (80/20 stratified,
15 attack categories, seed=42). Peak RSS: 459.45 MB / 500 MB limit. Elapsed: 136.76 s.
Outputs: 76-feature float32 Parquet files in `artifacts/splits/`.

**Phase 4 measured results (2026-09-08):** Random Forest baseline (LightGBM native RF mode,
100 trees, `bagging_fraction=0.8`, `bagging_freq=1`, `feature_fraction=0.8`, `class_weight=None`).
Trained on 2,016,638 flows in 140.35 s; evaluated on 504,160 flows in 19.29 s (38.26 μs/sample).
Multiclass Accuracy: 98.41% · Weighted F1: 0.9899 · Binary Attack Detection Rate (TPR): 99.94% (85,097/85,148 attacks caught) ·
Binary FPR: 1.77% (7,402/419,012 benign alerts) · Binary F1: 0.9580 · ROC-AUC: 0.9917.
Training peak resident RSS: 531.28 MB (well within budget). Saved to `models/` and `artifacts/`.

**Phase 5 measured results (2026-09-10):** XGBoost comparative model (`tree_method='hist'`,
200 trees, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`).
Trained on 2,016,638 flows in 957.00 s; evaluated on 504,160 flows in 7.42 s (14.71 μs/sample, 2.6× faster than RF).
Multiclass Accuracy: 99.86% · Weighted F1: 0.9985 · Macro F1: 0.8613 (+0.222 vs RF).
Binary Attack Detection Rate (TPR): 99.80% (84,979/85,148 attacks caught) ·
Binary FPR: 0.095% (399/419,012 benign false alerts — 18.5× lower false alarm rate than RF) ·
Binary F1: 0.9967 · ROC-AUC: 0.9999. Training peak resident RSS: 136.6 MB.
Formally selected as Primary Classifier for Phase 6 (SHAP) in `artifacts/model_selection_decision_v1.md`.

Random Forest is implemented via LightGBM (`boosting_type='rf'`) for histogram-binned,
memory-efficient training; XGBoost uses `tree_method='hist'` for the same reason.

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
