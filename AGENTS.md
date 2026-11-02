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
| Vector database | Chroma (local, persistent — chromadb==1.5.9 via Rust abi3 engine) |
| Embedding model | all-MiniLM-L6-v2 via sentence-transformers==3.3.1 (CPU-only, offline) |
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

## Risk Scoring Formula & Serving Schema

### Risk Scoring Formulation (Calibrated Convex Combination)
The transparent incident risk score is formulated as a normalized convex combination in $[0, 100]$:

$$\text{Risk Score} = w_1 \cdot C_{\text{ML}} + w_2 \cdot \text{Sev} + w_3 \cdot S_{\text{vol}} + w_4 \cdot B_{\text{rule}} \in [0, 100]$$

where $\sum_{i=1}^4 w_i = 1.0$, and the constituent sub-scores are defined in $[0, 100]$:
1. $C_{\text{ML}} = 100 \times \bar{p}_{\text{ML}}$: Mean XGBoost prediction confidence across all flows in the incident.
2. $\text{Sev} = \text{severity\_weight}[\text{primary\_threat}]$: Calibrated expert severity tier for the incident's plurality predicted attack class (Critical=100, High=80, Medium=60, Low=40, Info=20).
3. $S_{\text{vol}} = 100 \times \min\left(1.0, \frac{\log(1 + N_{\text{corr}})}{\log(1 + N_{\text{ref}})}\right)$: Log-scaled incident volume with reference saturation at $N_{\text{ref}} = 25$ flows (incidents with $N_{\text{corr}} \ge 25$ receive $S_{\text{vol}} = 100$).
4. $B_{\text{rule}} = 100 \times \mathbb{I}(\text{has\_rule\_hit} \land \text{is\_ml\_threat})$: Deterministic rule bonus under **Variant B** (requires at least one constituent flow to be flagged as a threat by ML before awarding the rule bonus, neutralizing spurious false campaign inflation).

**Frozen Calibrated Weights (`artifacts/risk/risk_weights_v1.json`):**
$$w_1 = 0.30, \quad w_2 = 0.10, \quad w_3 = 0.30, \quad w_4 = 0.30, \quad N_{\text{ref}} = 25.00$$

### Operational Incident Serving Schema (Phase 10+ Ingestion)
Downstream phases (Phase 10 RAG Knowledge Base, Phase 11 Gemini Decision Support, Phase 12 Backend) consume incident records strictly through the operational serving schema. To prevent data leakage and enforce decision-support grounding, **zero ground-truth labels or test-set telemetry** (`incident_label`, `true_label`, `majority_gt_label`, `n_attack_gt`, `purity`, etc.) are exposed. The schema below is **ILLUSTRATIVE (Schema v1.1)** and arithmetically verified against frozen weights and severity tables:

```json
{
  "schema_version": "1.1",
  "incident_id": 9999,
  "representative_flow_row_id": 123456,
  "source_ip": "172.16.0.1",
  "destination_ip": "192.168.10.50",
  "destination_port": 80,
  "start_time": 1499252400.0,
  "end_time": 1499252700.0,
  "duration_s": 300.0,
  "flow_count": 28,
  "flow_count_cap_note": "Volume saturates at N_ref=25 flows (S_vol=100.00)",
  "primary_threat": "DoS Hulk",
  "severity_tier": "Low / Volumetric Availability",
  "severity_value": 40.0,
  "mean_ml_confidence": 0.9982,
  "mean_p_attack": 0.9982,
  "has_rule_hit": false,
  "shap_top_features": null,
  "risk_score": 63.95,
  "sub_scores": {
    "c_ml_contrib": 29.95,
    "sev_contrib": 4.00,
    "vol_contrib": 30.00,
    "rule_contrib": 0.00
  }
}
```

> [!IMPORTANT]
> **Operational Presentation Requirement:** Downstream dashboards (Phase 13) and LLM prompt templates (Phase 11) **must present `severity_tier` alongside `risk_score`**. Because the risk score balances volume ($w_3=0.30$) and severity ($w_2=0.10$), low-volume critical exploits (e.g., Infiltration, SQL Injection) receive scores around ~49.4, comparable to volumetric scans (~47.2). Presenting the semantic severity tier ensures analysts never mistake the composite operational risk score for an intrinsic impact severity ranking.

### Methodological Disclosures & Operating Assumptions
1. **Analyst-Assigned Severity Table:** The severity lookup table (`SEVERITY_TABLE_PRIMARY`) uses **analyst-assigned heuristic severity tiers** (Critical=100, High=80, Medium=60, Low / Volumetric Availability=40, Reconnaissance=20, Zero=0), reflecting domain impact prioritization rather than formulas derived directly from NIST SP 800-61 or MITRE ATT&CK.
2. **Retrospective / Batch Incident Risk:** Incident scoring occurs retrospectively after sessionization closes an incident window ($G=300\text{ s}$ inactivity timeout), as $N_{\text{corr}}$ and duration require session completion.
3. **Volume Saturation:** The volume component saturates at $N_{\text{ref}}=25$ flows, preventing multi-thousand flow volumetric floods (e.g., DoS/DDoS) from mathematically drowning out low-volume targeted attacks.
4. **Non-Identification of $w_2, w_4$ by Binary AUC:** On this dataset, binary attack-vs-false AUC is largely insensitive to $w_2$ and $w_4$ because PortScan dominates incident volume ($>93\%$) and XGBoost already flags all flows matched by R104. The frozen weights ($w_1=0.30, w_2=0.10, w_3=0.30, w_4=0.30$) represent a pre-registered v1 profile with frozen tie-break selections.
5. **Score is Not an Intrinsic Severity Ranking:** High composite scores ($>68$) are attained exclusively by FTP-Patator (max 97.99) and SSH-Patator (max 97.98) due to the deterministic rule bonus ($B_{\text{rule}}=30.00$). Critical threats without rule hits (Infiltration: 49.39, Heartbleed: 44.38, SQLi: 49.36) score similarly to volumetric reconnaissance (PortScan: 47.17) and floods (DDoS: 47.23).
6. **Between-Cohort Volume Effect:** The $+0.021$ AUC gain of the composite score over ML-confidence alone is a macro volume effect across cohorts: within multi-flow ($N \ge 2$), gain is $+0.01023$ across full test ($+0.00000$ on report half); within singletons ($N=1$), risk score has slightly lower AUC ($-0.00657$ full test / $-0.00428$ report half) due to volume-scaling compression.
7. **$K_2$ vs $K_3$ Non-Comparability:** $K_2$ collapses horizontal scans into large multi-thousand flow incidents where $N_{\text{ref}}=25$ saturates immediately. $K_2$ AUC (0.97640) cannot be directly compared to $K_3$ AUC (0.96613).
8. **Shared Temporal Campaigns:** Calibration and report splits are stratified 50/50 across the 3,296 incidents; both halves share the same 8 calendar days, so temporal campaign co-occurrence is preserved.
9. **Report-Half Composition Scope:** The report-half attack incidents consist strictly of 1,051 PortScan, 66 Botnet, 1 FTP-Patator, and 1 Infiltration (total 1,119 attacks); DoS, DDoS, and Web attack categories have 0 attack incidents in the report half.
10. **Generalization Performance:** Exact measured values: Calibration AUC = 0.96059, Report AUC = 0.96613 (95% CI: $[0.95747, 0.97373]$, $\Delta = +0.00554$).

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
sentence-transformers==3.3.1   # CPU torch (torch==2.15.0+cpu via PyTorch index, installed 2026-11-02)
chromadb==1.5.9                # Rust-based CFFI abi3 engine (installed 2026-11-02; upgraded from 0.6.* because chroma-hnswlib lacks cp314 wheels)
python-dotenv==1.2.3
psutil==7.2.2                  # required by memory_guard.py
```

Install with:
```
pip install --only-binary :all: -c ml_service/constraints.txt chromadb==1.5.9 sentence-transformers==3.3.1 torch --extra-index-url https://download.pytorch.org/whl/cpu
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

> **Note (2026-11-02) — ChromaDB and Sentence-Transformers on Python 3.14 resolved:**
> `chromadb==0.6.*` pinned previously fails with `--only-binary :all:` because it
> requires `chroma-hnswlib==0.7.6`, which has zero cp314 binary wheels. `chromadb>=1.0`
> (specifically `chromadb==1.5.9`) compiles its vector search engine into a Rust CFFI
> `abi3` wheel (`cp39-abi3-win_amd64.whl`), removing `chroma-hnswlib` as a dependency.
> `torch==2.15.0+cpu` has official cp314 Windows wheels on `download.pytorch.org/whl/cpu`.
> `sentence-transformers==3.3.1` and `chromadb==1.5.9` installed with `--only-binary :all:`
> against `constraints.txt` locking all 50 pre-existing packages. Zero regression on ML pipeline.

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
| 4 — ML Baseline: Random Forest | ✅ Completed | 2026-09-08 | LightGBM RF mode (boosting_type='rf', bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, 100 trees). Train=2,016,638 / Test=504,160. Multiclass Acc=98.41%, Weighted F1=0.9899, Macro F1=0.6392. Binary Detection Rate=99.94% (85,097/85,148 attacks caught), FPR=1.77% (7,402/419,012 benign false alerts), Binary F1=0.9580, ROC-AUC=0.9917. Train time=140.35 s, inference latency=38.26 μs/sample, fit RSS=531.3 MB. Saved to models/ and artifacts/. |
| 5 — ML Comparison: XGBoost | ✅ Completed | 2026-09-10 | XGBoost histogram boosting (`tree_method='hist'`, 200 trees, max_depth=6, lr=0.1). Train=2,016,638 / Test=504,160. Multiclass Acc=99.86%, Weighted F1=0.9985, Macro F1=0.8613 (+0.222 vs RF). Binary Detection Rate=99.80% (84,979/85,148 attacks caught), FPR=0.095% (399/419,012 benign false alarms — 18.5× lower than RF), Binary F1=0.9967, ROC-AUC=0.9999. Train time=957.00 s, inference latency=14.71 μs/sample (2.6× faster than RF), fit RSS=136.6 MB. Formally selected as Primary Classifier for Phase 6 (SHAP) in `artifacts/model_selection_decision_v1.md`. All 5 research notebooks enriched with visual analytics. |
| 6 — Explainability (SHAP) | ✅ Completed | 2026-11-02 | TreeExplainer on selected XGBoost model (`tree_path_dependent`, no background set). Sample: 1,793 test flows (per-class cap: 150) preserving rare threats. Additivity check passed (max diff: 7.63e-06). Macro-averaged global ranking (top: Flow IAT Max, Max Packet Length, Init_Win_bytes_backward). Domain hypotheses confirmed; rare classes (Heartbleed, Infiltration, SQLi) disclosed as anecdotal. 7 plots in `artifacts/shap_plots/` (macro bar, heatmap, 4 TP waterfalls, 1 error case). Artifacts: `shap_values_v1.npz` (7.8 MB), `shap_summary_v1.json`, `.md`. Reusable `top_k_contributions()` helper for Phase 11. Notebook 06 created. Peak RSS: 587.1 MB. Elapsed: 2.39 s. |
| 7 — Security Rule Engine | ✅ Completed | 2026-11-02 | Deterministic rule engine on raw unscaled flows + Destination Port. Row alignment verified (100% match on 504,160 test rows). Calibrated on train split with strict 50% precision floor. R104_PORT_BRUTE_FORCE active (Train Prec=64.85%, Test Prec=63.85%, TPR=1.64%, FPR=0.189%, 1,399 TP, 792 FP). Measured latency=0.25 μs/sample. Complementarity matrix against XGBoost: Cat A (both)=1,399, Cat B (ML only)=83,979, Cat C (rules only)=792, Cat D (neither)=417,990. MachineLearningCVE absence of IP/Timestamp formally documented. Artifacts: rule_definitions_v1.json, rule_metrics_v1.json, .md, Notebook 07 created. |
| 8 — Event Correlation | ✅ Completed | 2026-11-02 | Ground-truth-blind sliding-gap sessionizer ($G=300\text{ s}$). Tuesday Acceptance Gate passed 5/5 (FTP frag=2, SSH frag=1, Purity=100.0%, Coverage=99.945%; noted as easiest case). Primary test partition ($N=504,160$): compressed 86,170 raw alert flows into 3,296 incidents (96.17% reduction, 26.1× compression factor; 96.05% dominated by Wed DoS, PortScan, and DDoS). Multi-flow purity ($N \ge 2$): 96.33% (all-inc: 96.41%), consistent coverage: 95.36%. Monday benign control produced 167 false incidents (150 singletons, 17 multi-flow; all 17 were RULE_ONLY:R104, confirming false alarms aggregate into spurious campaigns; benign false positives are not reduced by correlation). Cat C attack flows = 0; hybrid ingestion added +682 false incidents. PortScan ablation ($K_3$ vs $K_2$): 32.9× collapse (2,172 -> 66 incidents, 100.0% purity). Sliding gap proved superior to fixed 300s tumbling bins (3,296 vs 4,524 incidents). Artifacts: ml_flags_v1.parquet, incident_summary_v1.json, correlation_report_v1.md, preregistration_v1.md, Notebook 08 created. Peak RSS: 416.72 MB / 500 MB budget. |
| 9 — Risk Scoring | ✅ Completed | 2026-11-02 | Transparent incident risk scoring ($[0, 100]$) with convex sub-score decomposition ($w_1 C_{\text{ML}} + w_2 \text{Sev} + w_3 S_{\text{vol}} + w_4 B_{\text{rule}}$). Pre-registered 50/50 incident split on test partition ($3,296$ incidents). Frozen weights: $w_1=0.30, w_2=0.10, w_3=0.30, w_4=0.30$, $N_{\text{ref}}=25.00$, Variant B (requires ML flag). Evaluated on unseen report half ($1,644$ incidents: 1,119 attacks, 525 false): ROC-AUC=0.96613 (95% CI: [0.95747, 0.97373]; Calib=0.96059), Precision@10/25/50/100/200/300/400/500=100.00% (zero false alarms in top 538; first false alarm at rank #539). Report-half attack composition rescoped to effectively PortScan (1,051) + Botnet (66), with single incidents for FTP (1) and Infiltration (1), and 0 for DoS/DDoS/Web/SSH/Heartbleed. Sub-cohorts: N>=2 multi-flow AUC=0.99985 (ceiling), N=1 singleton AUC=0.90454. Cluster bootstrap yields 12 clusters (only 4 attack clusters, inferentially non-meaningful; incident bootstrap anti-conservative). Operating-point menu provided on calibration half (w2 policy parameter). Severity table is analyst-assigned heuristic tiers. K2 sensitivity AUC=0.97640. Peak RSS: 377.54 MB. Operational serving schema validated with zero ground-truth leakage for Phase 10+. |
| 10 — RAG Knowledge Base | Stage B executed - closure pending | 2026-11-02 | Stage B executed; Phase 10 closure pass underway (descriptive test reporting, max_seq_length audit, mapped retrieval evaluation, baseline realignment, v2 docs and artifacts). |
| 11 — Gemini Integration | ⬜ Not started | — | |
| 12 — Backend Integration | ⬜ Not started | — | |
| 13 — Dashboard | ⬜ Not started | — | |
| 14 — Testing & Evaluation | ⬜ Not started | — | |
| 15 — Deployment & Documentation | ⬜ Not started | — | |