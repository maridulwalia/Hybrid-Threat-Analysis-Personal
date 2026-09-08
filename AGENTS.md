# AGENTS.md — Hybrid Threat Analysis and Cyber Defence Decision Support System

## Project Overview

A web-based cybersecurity decision-support system. ML detects and classifies
suspicious network activity; SHAP explains predictions; deterministic rules
and event correlation enrich detection; a transparent risk score prioritizes
incidents; a RAG layer retrieves trusted cybersecurity knowledge; an LLM
converts evidence + retrieved knowledge into grounded analyst guidance.

**Core principle:** The ML system determines what the data indicates. The
RAG/LLM system is a secondary, decision-support layer — never the primary
detector. It must never invent evidence not present in the supplied context.

**Fail-safe requirement:** The core ML/XAI/rules/risk pipeline must remain
fully functional if the RAG/LLM layer is unavailable. Do not introduce hard
dependencies from Phases 1–9 on Phases 10–11.

---

## Finalized Technology Decisions (do not re-litigate without explicit instruction)

| Component | Decision |
|---|---|
| Primary dataset | CIC-IDS2017 (flow-level CSVs, 8 days) |
| Primary ML model | Random Forest (baseline) — implemented via LightGBM (boosting_type='rf') |
| Comparative ML model | XGBoost (tree_method='hist' for memory efficiency; trained sequentially, never concurrently with RF) |
| Explainability | SHAP / TreeExplainer |
| Vector database | Chroma (local, persistent) |
| Embedding model | all-MiniLM-L6-v2 via sentence-transformers (CPU-only, offline) |
| LLM | Google Gemini API (Gemini 2.5 Flash as initial candidate) |
| Backend orchestration | Node.js / Express |
| ML/XAI/RAG service | Python / FastAPI |
| Frontend | Next.js |
| Database | MongoDB |
| Deployment | Vercel (frontend), Render (backend/ML), MongoDB Atlas (DB) |

**Do not introduce:** LLM fine-tuning, training a foundation model, local
LLM as a hard requirement (Ollama optional only), FAISS instead of Chroma
without discussion, real-time streaming infrastructure, River (online/incremental
learning library — evaluated 2026-09-08 as a potential approach for Phase 14
unsampled-stream training; rejected: introduces a new training paradigm, new
dependency, and new evaluation methodology not required by the core research
question, and Phase 14/CSE-CIC-IDS2018 work is itself deferred future work;
do not install, implement, or reference River anywhere in the active pipeline),
or a second primary dataset without explicit approval. *(Evaluated 2026-09-06:
Proposed multi-dataset expansion with CSE-CIC-IDS2018, UNSW-NB15, NSL-KDD, and
CERT was audited and formally rejected for the active pipeline to preserve the
~8 GB RAM budget, avoid feature schema mismatch, and prevent scope paralysis;
relegated strictly to Phase 14 extended evaluation / future work).*

*(Evaluated 2026-09-07: RAM diagnostic (check_ram.py) showed loading the full
cleaned dataset (2,520,798 × 79) into a pandas DataFrame consumes 1.778 GB RSS
against ~0.860 GB available headroom, with system Available RAM spiking down to
28.8 MB during load — confirmed unsafe for standard in-memory
scikit-learn RandomForestClassifier.fit(). Algorithm choice is unchanged:
Random Forest baseline will be implemented using LightGBM's native RF mode
(boosting_type='rf'), which produces an equivalent bagged-tree ensemble via
histogram-binned, memory-efficient training. XGBoost comparative model will use
tree_method='hist' for the same reason. Implementation library changed for
hardware feasibility; no algorithmic decision has changed.
**Phase 4 implementation note — LightGBM RF mode requires explicit bagging
parameters or training will silently fall back to gradient boosting:**
`lgb.LGBMClassifier(boosting_type='rf', bagging_fraction=0.8, bagging_freq=1,
feature_fraction=0.8, ...)`. `bagging_freq` must be ≥ 1 (enables row-level
bagging) and both fractions must be strictly < 1.0. Without these, RF mode is
not active regardless of `boosting_type`.*)

*(Evaluated 2026-09-08: RAM diagnostic (check_ram.py, run 2026-09-06) confirmed
the full cleaned dataset at float64 (2,520,798 rows × 79 cols) consumes 1.595 GB
deep / 1.778 GB RSS delta against ~0.860 GB free RAM; Task Manager showed
Available RAM spike to 28.8 MB during load. Six engineering decisions recorded:
(1) RF implemented via LightGBM boosting_type='rf' — already recorded 2026-09-07,
no change. (2) XGBoost tree_method='hist' — already recorded 2026-09-07, no change.
(3) All ML feature columns cast to float32 (not float64) from Phase 3 onward;
for this dataset's value ranges, float32's ~7 significant digits is sufficient
and roughly halves in-memory size with no meaningful accuracy tradeoff for
tree-based models. (4) Phase 3+ intermediate split artifacts written as Parquet
(binary, column-oriented, 3–10× smaller than equivalent CSV with faster load
times); Phase 2 cleaned CSVs in artifacts/cleaned/ are frozen and unaffected.
(5) CSE-CIC-IDS2018 full-load evaluated on paper: at ~16.2M rows, estimated
footprint is ~10.2 GB at float64 and ~5.1 GB at float32, both exceeding available
RAM (~0.86–1.2 GB free at idle); confirmed that Phase 14, if ever undertaken,
requires chunked streaming with class-stratified sampling to a bounded subset
(e.g. ~500K–1M rows) before any in-memory training — never a full in-memory
load at any precision. (6) River rejected — see "Do not introduce" above.)*

---

## Hardware Constraint

Development machine has **~8 GB RAM**. This is a hard constraint, not a
suggestion:

- Load large CSVs with `usecols` and chunking, never all at once.
- Random Forest and XGBoost are trained **sequentially**: train → evaluate →
  save → release memory → train next.
- SHAP explanations are generated for a **representative sample**, never the
  full test set.
- Avoid loading multiple full-size DataFrames simultaneously.
- **Empirically confirmed unsafe (2026-09-06, check_ram.py):** loading the full
  cleaned dataset (2,520,798 rows × 79 columns) into a single pandas DataFrame
  consumed 1.595 GB (deep) / 1.778 GB RSS delta against ~0.860 GB free RAM;
  Task Manager confirmed Available RAM spiked to 28.8 MB during load, with a
  sustained second-lowest point around 200 MB. Phases 3 and 4 must **not** load
  the full cleaned dataset into a single DataFrame. Use chunked reads and/or
  write intermediate feature data to disk and load only what is needed at each
  step.
- **float32 standard precision (Phase 3 onward):** All ML feature columns are
  cast to float32 during Phase 3 feature engineering and all subsequent phases.
  float32's ~7 significant digits is sufficient for this dataset's value ranges
  (packet counts, byte counts, durations, rates) and roughly halves in-memory
  size relative to float64, with no meaningful accuracy tradeoff for tree-based
  models. This is a hard engineering constraint, not a suggestion.
- **Parquet as standard format for Phase 3+ intermediate artifacts:** Phase 3
  train/test split outputs (X_train, X_test, y_train, y_test) are written as
  Parquet files. Parquet is binary, column-oriented, and typically 3–10× smaller
  on disk than equivalent CSV, with faster load times and lower parsing overhead.
  **The Phase 2 cleaned CSVs in `artifacts/cleaned/` are frozen and unaffected
  — do not convert or replace them.**

---

## Dataset Notes (CIC-IDS2017)

- 8 daily CSV files. Monday = pure benign. Tuesday = FTP-Patator/SSH-Patator
  brute force (**use this as the ground-truth validation case for event
  correlation**). Wednesday = DoS/Heartbleed. Thursday = web attacks/
  infiltration. Friday = botnet/PortScan/DDoS.
- Known data-quality issues: leading whitespace in column names, `Infinity`/
  `NaN` values in flow-byte-rate columns, some duplicate flows. Must be
  cleaned before feature engineering.
- Keep raw `Source IP` / `Destination IP` / `Timestamp` in a separate
  reference table — they're dropped from the ML feature set but required
  downstream for event correlation.

---

## Correlation Strategy

- Group by `(Source IP, Destination IP, Destination Port)`.
- Sliding time window, starting at 5 minutes, tuned against the dataset's
  actual inter-arrival-time statistics.
- Only ML/rule-flagged flows enter correlation — never benign traffic.
- Validate specifically against Tuesday's labeled brute-force capture before
  considering this phase done.

## Risk Scoring Formula
Risk Score = (w1 × ML_confidence × 100)
+ (w2 × severity_weight[attack_category])
+ (w3 × log(1 + correlated_flow_count))
+ (w4 × rule_match_bonus)


Weights (w1–w4) must be documented once calibrated against real evaluation
results — never hard-coded before Phase 9's calibration step.

---

## Development Phases (build in this order; do not skip ahead)

0. **Environment Setup** — repo structure, Python venv, Node/Next.js scaffold,
   shared config for dataset paths / model paths / API keys.
1. **Dataset Acquisition & Exploration** — load CIC-IDS2017 in chunks, class
   distribution per day, document data-quality issues.
2. **Data Cleaning** — fix column whitespace, handle Infinity/NaN, dedupe,
   separate identifier columns from feature columns.
3. **Feature Engineering & Train/Test Split** — encode labels (binary +
   attack-category), scale features, stratified split, save split artifacts.
4. **ML Baseline: Random Forest** — train, evaluate (accuracy, precision,
   recall, F1, ROC-AUC, FPR, training time), save model + metrics.
5. **ML Comparison: XGBoost** — train after releasing RF from memory,
   identical split, same metrics, documented model-selection decision.
6. **Explainability (SHAP)** — TreeExplainer on selected model, representative
   sample only, sanity-check feature relevance.
7. **Security Rule Engine** — deterministic rules independent of ML (e.g.,
   connection-count thresholds per source IP/dest port/time window).
8. **Event Correlation** — implement strategy above, validate against
   Tuesday's ground truth.
9. **Risk Scoring** — implement formula above, calibrate severity weights
   per attack category present in the dataset.
10. **RAG Knowledge Base** — chunk MITRE ATT&CK/NIST docs relevant to
    CIC-IDS2017's attack categories, embed with MiniLM, load into Chroma,
    test retrieval in isolation before touching the LLM.
11. **Gemini Integration** — structured prompt (evidence + SHAP + correlation
    + risk + retrieved chunks) → Gemini → recommendation. Explicitly test
    that output doesn't cite anything outside the supplied context.
12. **Backend Integration** — FastAPI behind Node/Express, MongoDB
    persistence, verify graceful degradation if RAG/LLM layer fails.
13. **Dashboard** — Next.js views: overview, incident list, incident detail.
14. **Testing & Evaluation** — full metric suite across ML, SHAP, correlation,
    RAG, LLM output, and overall decision-support comparison.
15. **Deployment & Documentation** — Vercel/Render/Atlas, final report.

**Dependency rule:** Phases 1–6 are the non-negotiable core. Phases 7–9 are
the hybrid decision-support layer. Phases 10–11 are additive enhancements —
build them last, and never let them block the phases before them.

---

## General Coding Conventions

- Save every trained model, metric report, and evaluation artifact to disk
  with a clear versioned filename — never overwrite silently.
- No invented performance numbers anywhere (code, comments, docs) before
  actual evaluation has run.
- Prefer explicit, documented functions over "magic" pipelines — this
  project's evaluation criteria include auditability, not just accuracy.

---

## Confirmed Environment (verified 2026-08-28)

| Item | Value |
|---|---|
| Python | 3.14.7 (system PATH) |
| Node.js | v22.22.3 |
| npm | 10.9.8 |
| Git | **v2.54.0.windows.1** — installed, repo initialized on `master` branch |
| Git executable | `C:\Users\marid.CS-DEVELOPER\AppData\Local\Programs\Git\bin\git.exe` |
| Git PATH note | Not on system PATH yet — use full exe path or restart shell to pick up PATH update |
| Repo state | `git init` done; no commits yet; `.gitignore` not yet created |
| OS | Windows (PowerShell — use `;` not `&&` to chain commands) |
| Shell separator note | PowerShell does not accept `&&`; use `;` or separate `Run` calls |

---

## Dataset Location (confirmed 2026-08-28)

All 8 CIC-IDS2017 CSV files are already present — no download needed:

```
d:\Hybrid Threat Analysis and Cyber Defence Support System\
    MachineLearningCSV\
        MachineLearningCVE\
            Monday-WorkingHours.pcap_ISCX.csv            (~177 MB)
            Tuesday-WorkingHours.pcap_ISCX.csv           (~135 MB)
            Wednesday-workingHours.pcap_ISCX.csv         (~225 MB)
            Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv  (~52 MB)
            Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv (~83 MB)
            Friday-WorkingHours-Morning.pcap_ISCX.csv    (~58 MB)
            Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv   (~77 MB)
            Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv       (~77 MB)
```

Total on-disk: ~884 MB. The `.zip` archive (`MachineLearningCSV.zip`, ~235 MB) is also
present in the root but the extracted files are the ones to use.

---

## Phase 0 — Repo Layout Decision

The scaffold planned in the 2026-08-28 session:

```
<project root>/
├── .env.example
├── .gitignore
├── README.md
├── config/
│   └── paths.py              # single source of truth for all FS paths
├── ml_service/               # Python / FastAPI
│   ├── requirements.txt
│   ├── venv/                 # git-ignored
│   ├── main.py               # FastAPI stub (GET /health)
│   └── src/
│       ├── config.py
│       └── utils/
│           └── memory_guard.py   # safe_read_csv(), assert_memory_headroom()
├── backend/                  # Node.js / Express
│   ├── package.json
│   └── src/
│       ├── index.js          # GET /api/health stub
│       └── config.js
├── frontend/                 # Next.js 14, App Router, TypeScript
│   └── src/app/page.tsx
├── models/                   # git-ignored; versioned filenames only
├── artifacts/                # git-ignored; evaluation outputs
└── notebooks/
    └── 01_eda_placeholder.ipynb
```

---

## Phase 0 — Pinned Python Dependencies

```
# Verified Python 3.14 compatible versions (confirmed 2026-08-29).
# All packages have pre-built cp314 wheels — no C compiler required.
fastapi==0.141.1
uvicorn[standard]==0.52.4
pandas==3.0.5
numpy==2.5.2
scikit-learn==1.9.0
lightgbm==4.7.0                # RF baseline via boosting_type='rf'; cp314 wheel confirmed (added 2026-09-07)
xgboost==3.4.1
shap==0.52.0
pyarrow==25.0.1                # Parquet engine (Phase 3+); cp314-cp314-win_amd64 confirmed (added 2026-09-08)
sentence-transformers==3.3.*   # pulls torch CPU-only
chromadb==0.6.*
python-dotenv==1.2.3
psutil==7.2.2                  # required by memory_guard.py
```

Install with:
```
pip install --only-binary :all: -r ml_service/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

> **Note (2026-08-29):** Original pins (pandas 2.2, numpy 1.26, scikit-learn 1.5, shap 0.46)
> have no pre-built wheels for Python 3.14 and fail to compile from source without
> Visual Studio. Bumped to the nearest versions that ship py3.14 wheels.
> APIs are identical for the ML pipeline — no code changes required.

> **Note (2026-09-08) — Parquet engine resolved:** `pyarrow==25.0.1` pinned.
> Wheel `pyarrow-25.0.1-cp314-cp314-win_amd64.whl` confirmed on PyPI, installed
> into `ml_service/venv`, and smoke-tested (`pyarrow.parquet.ParquetWriter`
> available). FLAG comment removed from `requirements.txt`. Phase 3 implementation
> may now begin — no outstanding prerequisite blockers remain.

---

## Open Questions (all resolved 2026-08-29)

1. ~~**Git** — Install Git for Windows before scaffolding, or skip VCS for now?~~
   ✅ **Resolved (2026-08-29):** Git v2.54.0 installed; repo initialized on `master`.
   Exe at `C:\Users\marid.CS-DEVELOPER\AppData\Local\Programs\Git\bin\git.exe`.
   Next step: create `.gitignore` before first commit so CSVs and venv are never staged.
2. ~~**venv location** — `ml_service\venv\` (default) or elsewhere?~~
   ✅ **Resolved (2026-08-29):** Use recommended default — `ml_service\venv\`.
3. ~~**Gemini API key** — Available to put in `.env` now, or placeholder only?~~
   ✅ **Resolved (2026-08-29):** Key is available; will be written to `.env` (git-ignored).
4. ~~**Next.js** — Next.js 14, App Router, TypeScript confirmed?~~
   ✅ **Resolved (2026-08-29):** Next.js 14 with App Router + TypeScript confirmed.

**All questions resolved — Phase 0 execution approved.**

---

## Phase Completion Log

| Phase | Status | Date completed | Notes |
|---|---|---|---|
| 0 — Environment Setup | ✅ Completed | 2026-08-29 | Full scaffold verified: FastAPI, Express 5, Next.js 16 (Turbopack), Git initialized & initial commit |
| 1 — Dataset Acquisition & Exploration | ✅ Completed | 2026-09-04 | 8 CSVs streamed (2,830,743 flows, 79 cols). Verified schema consistency, 128-bit hash duplicate tracking (308,381 duplicates / 10.89%), distinct NaN (+1,358 in Flow Bytes/s) & +inf (+2,867 in Flow Packets/s, +1,509 in Flow Bytes/s), 65 columns with whitespace defects, class distributions & encoding artifacts documented in `artifacts/eda_summary.json` & `.md`. |
| 2 — Data Cleaning | ✅ Completed | 2026-09-05 | pipeline v2.0. Read 2,830,743 flows → removed 2,867 NaN/inf rows + 307,078 duplicates (255,236 within-file, 51,842 cross-file) → 2,520,798 clean unique flows written to `artifacts/cleaned/`. All 8 files passed independent zero-defect verification. Label normalizations: 2,180 Web Attack `\ufffd` labels corrected. Column schema: 76 ML features + 1 identifier (Destination Port) + 1 redundant (Fwd Header Length.1) + 1 target (Label) = 79. Peak RSS: 399.77 MB / 500 MB. SHA-256 source hashes prove raw CSVs unmodified. |
| 3 — Feature Engineering & Split | ✅ Completed | 2026-09-08 | 3-pass streaming pipeline (feature_engineer.py v1.0). 2,520,798 rows → train=2,016,638 / test=504,160 (80/20 stratified on category label). 76 float32 features, Parquet/snappy. Peak RSS: 459.5 MB / 500 MB limit. Elapsed: 136.8 s. 11 artifacts in `artifacts/splits/`. `.env` path vars corrected to Personal workspace (root cause of initial wrong-path write). |
| 4 — ML Baseline: Random Forest | ⬜ Not started | — | See Hardware Constraint & Technology Decisions updates 2026-09-08: LightGBM RF mode (boosting_type='rf'), explicit bagging params required. |
| 5 — ML Comparison: XGBoost | ⬜ Not started | — | |
| 6 — Explainability (SHAP) | ⬜ Not started | — | |
| 7 — Security Rule Engine | ⬜ Not started | — | |
| 8 — Event Correlation | ⬜ Not started | — | |
| 9 — Risk Scoring | ⬜ Not started | — | |
| 10 — RAG Knowledge Base | ⬜ Not started | — | |
| 11 — Gemini Integration | ⬜ Not started | — | |
| 12 — Backend Integration | ⬜ Not started | — | |
| 13 — Dashboard | ⬜ Not started | — | |
| 14 — Testing & Evaluation | ⬜ Not started | — | |
| 15 — Deployment & Documentation | ⬜ Not started | — | |