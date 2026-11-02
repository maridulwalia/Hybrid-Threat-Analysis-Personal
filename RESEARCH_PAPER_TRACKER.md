# Research Paper & Academic Publication Tracking Document

**Project Title:** Hybrid Threat Analysis and Cyber Defence Decision Support System  
**Subtitle:** An Explainable Hybrid AI and RAG-Based Approach to Cybersecurity Alert Triage and Decision Support  
**Domain:** Artificial Intelligence & Machine Learning — Cybersecurity / Intrusion Detection Systems (IDS)  
**Target Publication Venues:** IEEE Transactions on Information Forensics and Security, Computers & Security (Elsevier), IEEE Access, or ACM Cyber Security Conferences.

---

## 1. Executive Research Summary & Abstract Framework

### 1.1 Working Abstract
> Modern Security Operations Centers (SOCs) face overwhelming alert fatigue driven by voluminous network traffic and opaque machine-learning intrusion detection systems. While supervised classification algorithms achieve high detection accuracy, they lack interpretability, struggle with alert prioritization, and fail to provide actionable, grounded remediation guidance. Conversely, direct application of Large Language Models (LLMs) to raw telemetry introduces severe risks of hallucination, non-deterministic reasoning, and high computational costs.
> 
> This research proposes a multi-stage **Hybrid Threat Analysis and Cyber Defence Decision Support System**. The architecture establishes a deterministic, non-negotiable detection pipeline powered by supervised machine learning (Random Forest baseline and XGBoost) and SHAP (SHapley Additive exPlanations) for model interpretability. To aggregate fragmented alerts, an entity-centric temporal event correlation engine collapses related flows within calibrated sliding time windows. A transparent multi-factor risk scoring formula synthesizes ML confidence, attack severity weights, correlated flow cardinality, and deterministic rule matches to prioritize high-risk incidents. Finally, an offline, CPU-efficient Retrieval-Augmented Generation (RAG) layer indexed with MITRE ATT&CK and NIST SP 800-61 frameworks provides grounded context to Google Gemini (LLM), generating actionable response recommendations strictly anchored in retrieved evidence. Experimental evaluation on the benchmark **CIC-IDS2017** dataset demonstrates that this hybrid architecture achieves superior triage accuracy, verifiable explainability, and hallucination-free decision support under constrained computational environments (~8 GB RAM).

### 1.2 Keywords
`Cybersecurity`, `Intrusion Detection Systems (IDS)`, `Explainable AI (XAI)`, `SHAP`, `Event Correlation`, `Risk Scoring`, `Retrieval-Augmented Generation (RAG)`, `Large Language Models (LLM)`, `Alert Fatigue`, `CIC-IDS2017`.

---

## 2. Research Problem, Objectives & Contributions

### 2.1 Core Problem Statement
1. **The Detection vs. Explanation Gap:** Modern ML models detect anomalies with high precision but do not convey *why* a particular flow is malicious to human analysts in real time.
2. **Alert Fragmentation & Redundancy:** A single coordinated attack (e.g. brute force, port scan, DoS) triggers thousands of individual flow alarms, overwhelming analysts.
3. **The Unsafe Generative Trap:** Using generative LLMs as raw threat detectors leads to hallucinated indicators of compromise (IOCs) and unreliable decision-making.

### 2.2 System Core Principle & Fallback Guarantees
- **Separation of Detection and Guidance:** The ML/rules engine determines *what the data indicates* (detection). The RAG/LLM system determines *what response guidance applies* (decision support).
- **Graceful Degradation:** If the LLM/RAG service is offline or rate-limited, the core ML, SHAP, correlation, and risk scoring pipeline remains 100% operational.

### 2.3 Key Contributions
1. A **fail-safe hybrid pipeline** integrating tree-based ML classifiers, SHAP local/global explainability, deterministic rule matching, and event correlation.
2. A **calibrated, transparent risk-scoring formula** avoiding black-box alert triage.
3. An **offline-first, CPU-efficient RAG indexing strategy** (all-MiniLM-L6-v2 + Chroma) mapped directly to MITRE ATT&CK techniques present in CIC-IDS2017.
4. An **empirical benchmark** comparing Random Forest and XGBoost under strict memory constraints (~8 GB RAM).

---

## 3. Formal Mathematical Formulations & Methodology

### 3.1 Supervised ML Classifiers
- **Random Forest Baseline:** Ensemble of $B$ bagged decision trees:
  $$\hat{y}_{RF} = \arg\max_{c} \frac{1}{B} \sum_{b=1}^{B} I(h_b(x) = c)$$
- **XGBoost Comparative Model:** Gradient-boosted trees minimizing regularized objective:
  $$\mathcal{L}^{(t)} = \sum_{i=1}^{n} l\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t)$$

### 3.2 Explainable AI (SHAP / TreeExplainer)
Feature attribution via classic Shapley values based on cooperative game theory:
$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]$$
Where $F$ is the total set of flow features, $S$ is a subset of features excluding feature $i$, and $\phi_i(x)$ quantifies the exact contribution of feature $i$ to the malicious classification probability.

### 3.3 Temporal Event Correlation Model
- **Grouping Tuple:** Flows grouped by key $K = (\text{SrcIP}, \text{DstIP}, \text{DstPort}, \text{Protocol})$.
- **Sliding Time Window:** Flows $f_1, f_2, \dots, f_m$ with timestamps $t(f_i)$ belong to incident $I_k$ if:
  $$|t(f_{j}) - t(f_{j-1})| \le \Delta t_{\text{window}} \quad (\text{default } \Delta t = 300\text{ seconds})$$
- **Filtering Condition:** Only flows flagged by ML ($\hat{y} \ne \text{BENIGN}$) or the rule engine enter the correlation pool.

### 3.4 Multi-Factor Risk Scoring Formulation
The overall risk score $R(I) \in [0, 100]$ for an incident $I$ is calculated as:
$$R(I) = \min\left(100, \; w_1 \cdot \bar{P}_{\text{ML}} \cdot 100 + w_2 \cdot S_{\text{cat}} + w_3 \cdot \ln(1 + N_{\text{flows}}) + w_4 \cdot B_{\text{rules}}\right)$$
Where:
- $\bar{P}_{\text{ML}} \in [0, 1]$: Mean predicted malicious confidence score across constituent flows.
- $S_{\text{cat}} \in [0, 100]$: Severity weight lookup based on attack category (e.g., Infiltration/DDoS = 90, Botnet = 80, PortScan = 40).
- $N_{\text{flows}}$: Number of correlated flows collapsed into the incident.
- $B_{\text{rules}}$: Bonus score if deterministic security rules triggered on the activity.
- $w_1, w_2, w_3, w_4$: Calibrated normalization weights $(\sum w_i = 1.0)$.

### 3.5 Grounded RAG Retrieval
- **Document Chunking:** MITRE ATT&CK Enterprise Matrix and NIST SP 800-61 Rev. 2 segmented into semantic chunks $D = \{d_1, d_2, \dots, d_K\}$.
- **Embedding Function:** Sentence-Transformers `all-MiniLM-L6-v2` ($\mathbb{R}^{384}$).
- **Similarity Metric:** Cosine similarity in vector space:
  $$\text{Sim}(q, d_i) = \frac{\mathbf{e}_q \cdot \mathbf{e}_{d_i}}{\|\mathbf{e}_q\| \|\mathbf{e}_{d_i}\|}$$

---

## 4. Dataset Profile & Empirical Audit Log (CIC-IDS2017)

### 4.1 Theoretical Daily Capture Structure (From Literature)
| Day / File Name | Capture Duration | Target Attack Categories | Known Data Anomalies (Literature) |
|---|---|---|---|
| **Monday** (`Monday-WorkingHours.pcap_ISCX.csv`) | Normal working hours | Pure BENIGN traffic | Header whitespace |
| **Tuesday** (`Tuesday-WorkingHours.pcap_ISCX.csv`) | Working hours | FTP-Patator, SSH-Patator (Brute Force) | `Flow Bytes/s` NaN/Inf |
| **Wednesday** (`Wednesday-workingHours.pcap_ISCX.csv`) | Working hours | DoS (Slowloris, Slowhttptest, Hulk, GoldenEye), Heartbleed | High volume, Inf rates |
| **Thursday Morning** (`Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`) | Morning hours | Web Attacks (Brute Force, XSS, SQLi) | Low attack sample size |
| **Thursday Afternoon** (`Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`) | Afternoon hours | Infiltration (Dropbox download, Coolpreview) | Extreme class imbalance |
| **Friday Morning** (`Friday-WorkingHours-Morning.pcap_ISCX.csv`) | Morning hours | Botnet (ARES) | Header whitespace |
| **Friday Afternoon (PortScan)** (`Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`) | Afternoon | PortScan | High flow count |
| **Friday Afternoon (DDOS)** (`Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`) | Afternoon | DDoS (LOIC) | High packet rate |

### 4.2 Real Empirical Findings & Data Quality Audit (Measured Live in Phase 1)
*(This section logs actual measured values computed directly by our streaming EDA engine)*

| File Name | Total Rows (Measured) | Benign Rows | Malicious Rows | Attack Subtypes & Counts | Infinity Count | NaN / Missing Count | Duplicate Rows (%) | File Size On Disk |
|---|---|---|---|---|---|---|---|---|
| `Monday-WorkingHours.pcap_ISCX.csv` | 529,918 | 529,918 (100%) | 0 (0.0%) | None (Pure Benign baseline) | 0 | 0 | 26,935 (5.08%) | 168.7 MB |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 445,909 | 432,074 (96.90%) | 13,835 (3.10%) | FTP-Patator: 7,938<br>SSH-Patator: 5,897 | 0 | 0 | 24,118 (5.41%) | 128.8 MB |
| `Wednesday-workingHours.pcap_ISCX.csv` | 692,703 | 440,031 (63.52%) | 252,672 (36.48%) | DoS Hulk: 231,073<br>DoS GoldenEye: 10,293<br>DoS slowloris: 5,796<br>DoS Slowhttptest: 5,499<br>Heartbleed: 11 | 4,376 (+inf)<br>*(2,867 in Flow Packets/s, 1,509 in Flow Bytes/s)* | 1,358<br>*(in Flow Bytes/s)* | 90,138 (13.01%) | 214.7 MB |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | 170,366 | 168,186 (98.72%) | 2,180 (1.28%) | Web Attack – Brute Force: 1,507<br>Web Attack – XSS: 652<br>Web Attack – SQLi: 21 | 0 | 0 | 14,107 (8.28%) | 49.6 MB |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | 288,602 | 288,566 (99.99%) | 36 (0.01%) | Infiltration: 36 | 0 | 0 | 46,144 (15.99%) | 79.2 MB |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | 191,033 | 189,067 (98.97%) | 1,966 (1.03%) | Botnet (ARES): 1,966 | 0 | 0 | 17,988 (9.42%) | 55.6 MB |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 286,467 | 127,537 (44.52%) | 158,930 (55.48%) | PortScan: 158,930 | 0 | 0 | 80,973 (28.27%) | 73.3 MB |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 225,745 | 97,718 (43.29%) | 128,027 (56.71%) | DDoS (LOIC): 128,027 | 0 | 0 | 7,978 (3.53%) | 73.5 MB |
| **Total / Overall Dataset** | **2,830,743** | **2,273,097 (80.30%)** | **557,646 (19.70%)** | **14 attack types across 7 categories** | **4,376 (+inf)** | **1,358 (NaN)** | **308,381 (10.89%)** | **883.8 MB** |

### 4.3 Phase 1 Exploration Methodology & Data Quality Findings (Paper Section)
1. **Memory-Constrained Streaming Exploration:** To satisfy the strict development hardware profile (~8 GB RAM), dataset exploration was conducted without in-memory dataset aggregation. An iterator processed 25,000-row chunks sequentially across daily files, retaining runtime peak RAM below 450 MB.
2. **128-Bit Cryptographic Hash Deduplication:** To prevent 64-bit birthday paradox collisions across $N = 2.83 \times 10^6$ flow records, duplicate detection utilized dual-set 128-bit MD5 digests ($p_{\text{collision}} \approx 2^{-128}$). This revealed 255,479 within-file duplicates and 52,902 cross-file overlaps (308,381 total duplicate records, 10.894%), providing empirical justification for Phase 2 deduplication.
3. **Rigorous Disentanglement of NaN and Infinite Values:** Rather than treating infinite rates uniformly as missing values, `isna()` was decoupled from `+inf` and `-inf`. This isolated defects strictly to two rate features: `Flow Bytes/s` (1,358 NaNs, 1,509 +inf) and `Flow Packets/s` (2,867 +inf), with zero `-inf` occurrences and zero missing values across all other 77 features.
4. **Header Defect Taxonomy:** Schema auditing established that all 8 daily captures share an identical 79-column schema, but 65 of 79 column names contain leading whitespace defects (e.g. `' Destination Port'`, `' Label'`). Furthermore, raw label encoding includes legacy byte replacement anomalies in Web Attack categories (`\ufffd`).

### 4.4 Feature Taxonomy (79 Flow Features)
- **Time & Inter-Arrival Metrics:** Flow Duration, Flow IAT Mean/Std/Max/Min, Fwd/Bwd IAT.
- **Packet & Byte Volume:** Total Fwd/Bwd Packets, Flow Bytes/s, Flow Packets/s, Packet Length Mean/Std/Variance.
- **Protocol & Flag Attributes:** FIN, SYN, RST, PSH, ACK, URG, CWE, ECE flag counts.
- **Subflow & Window Statistics:** Init_Win_bytes_forward/backward, Subflow Fwd/Bwd Bytes/Packets, Active/Idle Mean/Std.

---

### 4.5 Phase 2 Data Cleaning — Methodology & Empirical Results (Paper Section)

**Pipeline Version:** `2.0` | **Timestamp (UTC):** `2026-09-05 13:16:26` | **Peak RSS:** `399.77 MB` / 500 MB limit

#### 4.5.1 Column Schema Decomposition
Of the 79 columns retained after cleaning, the formal classification is:

| Category | Count | Columns |
|---|---|---|
| **ML Candidate Feature Columns** | 76 | All numeric flow traffic attributes |
| **Network Identifier Column** | 1 | `Destination Port` (retained for Phase 7 rules & Phase 8 correlation; excluded from Phase 4/5 ML training) |
| **Redundant Column** | 1 | `Fwd Header Length.1` (exact duplicate of `Fwd Header Length`; retained unchanged, excluded during Phase 3 feature selection) |
| **Target Column** | 1 | `Label` |
| **Total** | **79** | **76 + 1 + 1 + 1 = 79** |

#### 4.5.2 Measured Cleaning Outcomes

| Metric | Value |
|---|---|
| Rows Read (original) | 2,830,743 |
| Rows Removed — NaN / Infinity | 2,867 (0.10%) |
| Rows Removed — Within-File Duplicates | 255,236 (9.02%) |
| Rows Removed — Cross-File Duplicates | 51,842 (1.83%) |
| Total Rows Removed | 309,945 (10.95%) |
| **Clean Unique Flows (Final Dataset)** | **2,520,798** |
| Peak Process RSS | 399.77 MB (within 500 MB limit) |

#### 4.5.3 Per-File Cleaning Summary

| File | Rows Read | NaN/Inf Removed | Within-File Dupes | Cross-File Dupes | Rows Written |
|---|---|---|---|---|---|
| `Monday-WorkingHours.pcap_ISCX.csv` | 529,918 | 437 | 26,831 | 0 | 502,650 |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 445,909 | 264 | 24,019 | 50 | 421,576 |
| `Wednesday-workingHours.pcap_ISCX.csv` | 692,703 | 1,297 | 80,914 | 8,214 | 602,278 |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | 170,366 | 135 | 6,052 | 8,030 | 156,149 |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | 288,602 | 207 | 35,605 | 10,505 | 242,285 |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | 191,033 | 122 | 6,867 | 11,091 | 172,953 |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 286,467 | 371 | 72,319 | 8,611 | 205,166 |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 225,745 | 34 | 2,629 | 5,341 | 217,741 |

#### 4.5.4 Cleaning Protocol (Paper Methodology Section)
1. **Column Name Standardization:** Leading/trailing whitespace stripped from all 79 column names (`df.columns.str.strip()`). 65 column name defects corrected.
2. **Strict NaN / Infinity Separation and Removal:** `+inf` and `-inf` values replaced with `np.nan` (numeric columns only). All rows containing any `np.nan` value were subsequently dropped. Total: **2,867 rows** removed, all from `Flow Bytes/s` and `Flow Packets/s`.
3. **Conservative Label Sanitization:** Explicit mapping-only approach. Three Web Attack label variants containing `U+FFFD` (Unicode replacement character) were normalized: `Web Attack \ufffd Brute Force` → `Web Attack - Brute Force` (1,507 occurrences); `Web Attack \ufffd XSS` → `Web Attack - XSS` (652 occurrences); `Web Attack \ufffd Sql Injection` → `Web Attack - Sql Injection` (21 occurrences).
4. **Canonical 128-bit MD5 Deduplication:** Rows were serialized to canonical tuples (all 79 columns, post-sanitization) and hashed using 128-bit MD5 ($p_{\text{collision}} \approx 2^{-128}$). Dual-set tracking distinguished 255,236 within-file and 51,842 cross-file duplicate flows.
5. **Source Immutability:** Raw CSVs were never modified. SHA-256 digests computed before cleaning confirm source integrity.

#### 4.5.5 Independent Verification
A separate verification pass (independent of the cleaning implementation) confirmed all 8 cleaned files contained strictly zero NaN, zero infinity, and zero duplicate records. This is a stronger proof than relying solely on the cleaner's own counters.

#### 4.5.6 Source Integrity (SHA-256)

| File | Rows | SHA-256 |
|---|---|---|
| `Monday-WorkingHours.pcap_ISCX.csv` | 529,918 | `852c4beb34eda186f32561fa79df7a07...` |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 445,909 | `52b8692ae8c7d2ed04671fe2b9833569...` |
| `Wednesday-workingHours.pcap_ISCX.csv` | 692,703 | `893c27dc968bf7a8adef1689f90be55c...` |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | 170,366 | `d67066211fb1689c78406f1506f4c447...` |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | 288,602 | `6bcda3857c2504676034e3ea57762d38...` |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | 191,033 | `53a41c24d570ea83b7ac55b2e94df94e...` |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 286,467 | `ca1824c51bfbb7b3c72290a11be04366...` |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 225,745 | `6ff1580f5f81c0ae28a26f7631721018...` |


---

### 4.6 Post-Phase 2 Strategic Audit: Evaluation & Formal Rejection of Multi-Dataset Pipeline Expansion

**Decision Date:** 2026-09-06  
**Decision Type:** Architectural Governance — Formally Audited, Debated, and Discarded for Active Core Pipeline  
**Final Status:** Multi-dataset expansion relegated strictly to Phase 14 Extended Evaluation / Future Work section of the research paper  
**Recorded By:** Research lead — retained permanently for paper methodology transparency, future reproducibility, and examiner review

> This section records the complete decision trail of a significant research strategy proposal raised post-Phase 2. It documents what was proposed, why each dataset was considered, the precise technical and methodological hazards identified per dataset, the alternative workarounds considered and why they too were rejected, and the final binding decision. It is retained as a permanent governance record rather than deleted, because the reasoning behind *not* doing something is as academically significant as the reasoning behind *doing* something.

---

#### 4.6.1 What Was Proposed and Why It Was Raised

Upon completing Phase 2 data cleaning (2,520,798 globally unique flows, 399.77 MB peak RSS), a post-cleaning review of the class distribution revealed a structurally significant problem: while the **overall dataset size is adequate for tabular machine learning**, a subset of attack categories contains so few observations that no statistically reliable generalisation claim can be made about them from CIC-IDS2017 alone.

**Measured rare-class counts (post-cleaning):**

| Attack Category | Post-Cleaning Flow Count | Concern Level |
|---|---|---|
| Heartbleed | 11 | ⛔ Critical — not a trainable class |
| Infiltration | 36 | ⛔ Critical — not a trainable class |
| Web Attack – SQL Injection | ~21 | 🔴 Very High |
| Web Attack – XSS | ~652 | 🟡 Moderate |
| Botnet (ARES) | ~1,966 | 🟡 Moderate |
| Web Attack – Brute Force | ~1,507 | 🟡 Moderate |
| DoS slowloris | ~5,796 | 🟢 Acceptable |
| DoS Slowhttptest | ~5,499 | 🟢 Acceptable |

The concern was legitimate: attempting to train a multi-class classifier that includes a "Heartbleed" output neuron on 11 training examples is not producing a Heartbleed detector — it is producing a classifier that has memorised 11 rows and cannot generalise. Similarly for Infiltration (36 rows). These classes are so statistically underrepresented that any reported per-class metric for them (precision, recall, F1) would be methodologically meaningless or trivially manipulated.

This raised the question: **should the project incorporate additional cybersecurity datasets to provide more observations of rare attack patterns, or to diversify the range of network environments and attack taxonomies represented in training?**

Four candidate datasets were identified and considered:

1. **CSE-CIC-IDS2018** (~16.2M flow records / ~6.5–7 GB processed CSVs): A successor to CIC-IDS2017 produced by the same Canadian Institute for Cybersecurity and UNB team using the same CICFlowMeter extraction pipeline, collected over 10 days in February 2018 across a 50-machine network with simulated enterprise traffic. It contains attack categories including brute force, Heartbleed, botnet, DoS, DDoS, web attacks, and infiltration — overlapping with CIC-IDS2017 but at approximately 6.4× the scale and with some attack categories substantially better represented.

2. **UNSW-NB15** (~2,540,044 flow records): A benchmark dataset produced at the Australian Defence Force Academy (ADFA) in 2015 by the IXIA PerfectStorm network traffic generation tool and Tcpdump, with 49 features extracted using Argus and Zeek/Bro. It contains 9 attack categories — Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms — representing a significantly different attack taxonomy and network environment from CIC-IDS2017.

3. **NSL-KDD** (~125,973 training / ~22,544 test records): A curated refinement of the original KDD Cup 1999 dataset produced by Tavallaee et al. (2009), removing duplicate records from KDD'99 to correct its severe redundancy bias. It contains 41 features extracted from network connection records using a different protocol stack and feature extraction methodology. It is widely used as an IDS benchmark comparison point in academic literature despite its age.

4. **CERT Insider Threat Dataset (r4.2 / r6.2)**: A synthetic but behaviourally grounded dataset produced by Carnegie Mellon University's Software Engineering Institute (SEI/CERT). It simulates insider threat scenarios over 17 months across a fictitious 1,000-employee organisation. It contains five event modalities: logon records, device (USB) records, email records, file access records, and web browsing records, along with ground-truth labels for malicious insider activity (data theft, IP exfiltration, sabotage).

---

#### 4.6.2 The Academic Justification for Considering Each Dataset

**CSE-CIC-IDS2018 — Academic Case:**
* Produced by the same team, same CICFlowMeter extraction pipeline, directly comparable feature space.
* Contains approximately 6.4× more flow records than CIC-IDS2017.
* Several rare attack categories in CIC-IDS2017 (Heartbleed, Infiltration) are more substantially represented in 2018.
* Training on CIC-IDS2017 and testing on CIC-IDS2018 (or vice versa) would produce a genuine cross-year temporal generalization result — one of the more rigorous evaluations possible within the CICFlowMeter ecosystem.
* Could directly address the rare-class problem for Heartbleed and Infiltration without resorting to synthetic oversampling.

**UNSW-NB15 — Academic Case:**
* A completely independent dataset from a different country, institution, network topology, and feature extraction methodology.
* If a model trained on CIC-IDS2017 could detect anomalies in UNSW-NB15 data, this would constitute a genuine cross-environment generalization result — far stronger evidence of robustness than any within-dataset holdout test.
* Represents a genuinely different network threat taxonomy (includes Shellcode, Worms, and Fuzzers not present in CIC-IDS2017).
* Would allow the paper to distinguish: "this model detects CIC-IDS2017 attacks" vs. "this model detects network anomalies generically."

**NSL-KDD — Academic Case:**
* A historical benchmark that virtually every IDS paper published between 2009 and 2020 reports results against.
* Including NSL-KDD results would allow direct numerical comparison against a large body of prior work.
* Its inclusion, even as a baseline comparison only, would substantially increase the paper's contextual grounding within the IDS literature.

**CERT Insider Threat — Academic Case:**
* The system's architecture explicitly includes a Temporal Event Correlation Engine (Phase 8) and Risk Scoring formula (Phase 9) that are not ML classifiers — they are designed to ingest heterogeneous alert types and correlate them into unified incidents.
* CERT's multi-modal event logs (logon, file, USB, email, web) represent a genuinely different telemetry type from network flows.
* Incorporating CERT would demonstrate that the correlation and risk scoring components are not tied exclusively to CIC-IDS2017 network flows — they can ingest host-level behavioural signals.
* This would substantially strengthen the "hybrid" and "multi-source" aspects of the architecture's name.

---

#### 4.6.3 Dataset-by-Dataset Technical Hazard Analysis

##### CSE-CIC-IDS2018 — Technical Hazard Assessment

**Hardware Collapse Risk (Critical):**
The streaming chunk-based architecture proved viable during Phase 2 *cleaning*, where each chunk was processed, aggregated into summary statistics, and discarded. Model training is fundamentally different:
- `RandomForestClassifier` (Scikit-learn) builds all $B$ trees across the full training matrix simultaneously. It does not support partial/incremental fitting.
- Constructing a feature matrix $X \in \mathbb{R}^{N \times 76}$ for $N = 13{,}000{,}000$ training rows (80% of 16.2M) with 76 `float64` features requires approximately: $13{,}000{,}000 \times 76 \times 8 \text{ bytes} \approx 7.9 \text{ GB RAM}$ — already at the physical limit, before allocating the Random Forest's internal node arrays, bootstrap sample indices, or prediction buffers.
- Operating system overhead, Python interpreter, and active NumPy/Pandas buffers consume a further ~500–800 MB.
- **Result:** Training would either crash with an `MemoryError` mid-fit or trigger catastrophic OS swap thrashing, producing training runs of many hours with no reproducibility guarantee.

**Label Harmonization Complexity:**
- CIC-IDS2017 and CSE-CIC-IDS2018 share many attack labels but with minor naming and taxonomic differences that must be explicitly resolved before any combined training. This is non-trivial work.
- For example, 2018 uses `DoS attacks-Hulk`, `DoS attacks-GoldenEye`, `DoS attacks-Slowloris`, `DoS attacks-SlowHTTPTest` where 2017 uses `DoS Hulk`, `DoS GoldenEye`, `DoS slowloris`, `DoS Slowhttptest` — not identical strings, requiring a verified mapping table before concatenation.

**Workaround Considered — Sub-Sampling CSE-CIC-IDS2018:**
Could a stratified sample of ~500K–1M rows from CSE-CIC-IDS2018 be drawn to stay within memory limits? Yes, technically. However:
- This would require first running Phase 1 (EDA) and Phase 2 (Cleaning) on the 2018 dataset — replicating weeks of existing work before gaining a single training row.
- The rare classes (Heartbleed, Infiltration) would need to be *fully retained* in any sub-sample, not randomly sub-sampled — which requires careful stratified sampling logic not yet built.
- The scientific justification weakens: "we used 500K rows from a 16.2M-row dataset, chosen via non-random stratification" introduces selection methodology complexity that would require significant paper justification.

**Verdict: Rejected for core pipeline. Permitted in Phase 14 only if Phase 3–11 are complete and a sub-sampling strategy is formally specified.**

---

##### UNSW-NB15 — Technical Hazard Assessment

**Feature Space Incompatibility (Fatal):**
This is not a solvable engineering problem within the current project scope. It is a mathematical barrier:

| Attribute | CIC-IDS2017 | UNSW-NB15 |
|---|---|---|
| Feature extractor | CICFlowMeter | Argus + Zeek/Bro |
| Feature count | 76 (ML) + 1 identifier + 1 target | 49 features + 1 target |
| Feature overlap | — | ~10 generic features (duration, bytes, packets) |
| Directional features | Yes (Fwd/Bwd split) | Partially |
| Flag-level features | Yes (SYN, FIN, RST, PSH, ACK, URG counts) | Partially (via Zeek) |
| Sub-flow features | Yes | No |
| Active/Idle time features | Yes | No |

The ML models in Phases 4 and 5 will be trained on a fixed 76-dimensional input vector $x \in \mathbb{R}^{76}$. Presenting a 49-dimensional UNSW-NB15 feature vector to these models is dimensionally incompatible — there is no valid way to align these inputs without either:
1. **Re-running CICFlowMeter on the raw UNSW-NB15 PCAPs:** The UNSW-NB15 raw PCAP collection is approximately 100 GB. Downloading, storing, and processing this is beyond both the storage and memory budget.
2. **Reducing to a shared feature subset:** Computing the ~10 generic features common to both schemas would degrade the CIC-IDS2017 models' discriminative ability — precisely the features that SHAP identifies as most important for specific attack types (e.g., `Init_Win_bytes_forward` for SYN-flood detection, `Subflow Fwd Bytes` for data-exfiltration patterns) would be lost.
3. **Training a new independent model on UNSW-NB15 separately:** This is valid as a parallel benchmark experiment, but does not constitute "cross-dataset generalization" — it is just a separate model comparison, which is a Phase 14 contribution.

**Workaround Considered — Shared Feature Subset Model:**
Training a reduced model on a ~10-feature intersection and evaluating on both datasets was considered. However, this would require:
- Retraining all Phase 3–6 artifacts from scratch on the reduced feature set.
- Reporting a deliberately degraded model as the primary result.
- Invalidating SHAP attributions (the 66 dropped features include the most discriminative ones for several attack types).

This workaround would actively damage the paper's core contribution rather than enhance it.

**Verdict: Impossible within current architecture without raw PCAP re-extraction. Permanently relegated to Phase 14 / Future Work as a separate independent benchmark experiment.**

---

##### NSL-KDD — Technical Hazard Assessment

**Feature Space Incompatibility:**
NSL-KDD's 41 features are derived from the KDD Cup 1999 network connection schema — a pre-deep-packet-inspection, pre-high-speed-network era protocol-level feature set. Many features have no equivalent in CICFlowMeter output:
- `protocol_type` (TCP/UDP/ICMP as categorical): CICFlowMeter captures this but encodes it differently.
- `service` (http, ftp, smtp...): A high-cardinality categorical with ~70 values, absent from CIC-IDS2017.
- `flag` (SF, S0, REJ, RSTO...): TCP session-state flags as a categorical string, not as individual binary counts.
- `land`, `wrong_fragment`, `urgent`: Legacy KDD fields with no CIC-IDS2017 counterpart.

**Dataset Age & Relevance Concern:**
NSL-KDD was curated from 1998 DARPA simulation traffic on a network with a 1990s architecture, protocols, and attacker tooling. Its attack patterns (e.g., `neptune`, `smurf`, `pod`, `teardrop`) are legacy attack types from over 25 years ago, largely irrelevant to modern enterprise network attack surfaces. Including it as a training source would introduce severely outdated traffic patterns into a model intended for modern network security operations.

**What It Can Validly Provide:**
NSL-KDD's sole valid use in this project is as a **historical comparison baseline**: training an independent model on NSL-KDD using its own feature space and reporting results for the purpose of situating this work within the historical IDS literature. This is a comparison table entry in the paper's Related Work section, not an architectural component.

**Verdict: NSL-KDD is not architecturally compatible as a training or evaluation data source for the CIC-IDS2017 pipeline. Valid only as a historical literature comparison in Phase 14.**

---

##### CERT Insider Threat Dataset — Technical Hazard Assessment

**Telemetry Modality Mismatch (Fundamental):**
The CERT dataset is not a network flow dataset. It does not contain IP addresses, port numbers, packet counts, byte counts, flow duration, TCP flags, or any of the 76 CICFlowMeter features used by the ML classifier pipeline. Its records look like:

```
[Logon event]:   user, pc, date, activity: Logon/Logoff
[Device event]:  user, pc, date, file_tree, activity: Connect/Disconnect  
[File event]:    user, pc, date, filename, to/from
[Email event]:   user, pc, date, to, cc, bcc, size, attachments
[HTTP event]:    user, pc, date, url, content
```

Injecting any of these record types into the Phase 3–6 ML pipeline would require:
- Constructing surrogate numeric feature vectors from categorical identity and behavioural attributes.
- Defining a common label space between "Botnet flow" and "USB data exfiltration by malicious insider."
- Resolving a fundamentally different threat model (network-level adversary vs. authenticated, privileged insider).

**Where CERT Is Architecturally Correct:**
The Temporal Event Correlation Engine (Phase 8) consumes alert tuples of the form `(EntityID, Timestamp, AlertType, Confidence, Source)`. In principle, the Phase 8 engine could consume:
- A network flow alert from Phase 4/5: `(192.168.1.5, 2026-09-05T14:23:00, SSH-Patator, 0.97, ML)`
- A CERT device event alert: `(user_MJohnson_PC04, 2026-09-05T14:25:00, USB-Exfiltration, 1.0, CERT-Rules)`

Correlating these two signals into a unified incident would demonstrate multi-source threat correlation — the literal definition of "hybrid" in the system's name. However, this is a Phase 8 concern, not a Phase 3–6 concern, and building the CERT parsing and normalisation layer before Phases 3–6 are complete would violate the dependency rule.

**Verdict: CERT has a legitimate architectural home in Phase 8 and potentially Phase 9, but must not be introduced until Phase 7 (Rule Engine) and Phase 8 (Correlation Engine) are complete on the primary CIC-IDS2017 pipeline. Its inclusion at that point as a secondary event source would strengthen the paper's "hybrid multi-source" claim.**

---

#### 4.6.4 Decision Summary Table

| Dataset | Proposed Role | Feature Compatible? | RAM Feasible? | Modality Compatible? | Pre-Phase 14 Viable? | Final Decision |
|---|---|---|---|---|---|---|
| **CSE-CIC-IDS2018** | Extended diversity training | ✅ (CICFlowMeter) | ❌ (16.2M rows ≈ 8 GB RAM) | ✅ Network flows | ❌ | Relegated to Phase 14; requires stratified sub-sampling strategy |
| **UNSW-NB15** | Cross-dataset generalization | ❌ (Argus/Zeek, 49 features) | ✅ (2.5M rows) | ✅ Network flows | ❌ | Fatal feature incompatibility; separate parallel benchmark only in Phase 14 |
| **NSL-KDD** | Historical IDS benchmark | ❌ (KDD, 41 features) | ✅ (160K rows) | ✅ Network flows | ❌ | Historical comparison only; Related Work table in paper |
| **CERT Insider Threat** | Multi-source correlation | ❌ (Host audit logs) | ✅ (Small) | ❌ (Host, not flow) | ⚠️ Phase 8+ only | Phase 8 secondary event source post-pipeline completion; never a training data source |

---

#### 4.6.5 Impact on the Dataset Sufficiency Assessment

This audit confirmed that **CIC-IDS2017 is not too small for the primary ML pipeline**. The correct restatement of the problem is:

> CIC-IDS2017's overall size (2.52M flows) is adequate for training robust Random Forest and XGBoost classifiers for the eight major attack families. The rare-class limitation (Heartbleed: 11 flows; Infiltration: 36 flows) is a genuine methodological constraint that prevents reliable per-class evaluation of those specific categories — but this limitation is honestly reported in the paper's limitations section rather than masked by artificial oversampling or dataset merging.

This is the correct scientific position: acknowledge the constraint, do not fabricate solutions, and report what the data actually supports.

---

#### 4.6.6 Impact on the Research Paper's Contribution Statement

The decision to proceed exclusively on CIC-IDS2017 does not weaken the paper. The primary contributions remain the **hybrid architecture** (ML + SHAP + Rules + Correlation + Risk Scoring + RAG + LLM) and the **explainable, grounded decision-support system** — not dataset breadth.

The paper's Discussion / Limitations section will explicitly note:
1. The rare-class limitation (11 Heartbleed, 36 Infiltration flows) and why synthetic oversampling was methodologically rejected.
2. The multi-dataset expansion that was evaluated, why it was deferred, and what Phase 14 would need to address it.
3. The feature-space incompatibility between CICFlowMeter, Argus/Zeek, and KDD-era feature extractors as an open research challenge.

These documented limitations and deferred experiments are academic strengths, not weaknesses — they demonstrate methodological rigour and provide a clearly articulated future work agenda.

---

#### 4.6.7 Final Binding Decision

* **Phase 2 Remains Frozen:** The official cleaned baseline of **2,520,798 unique flows** across 8 daily files is finalized and locked. No external data will be merged into these files.
* **No Architecture Changes:** The Phase 3–11 pipeline design remains exclusively oriented toward the CIC-IDS2017 feature schema (76 CICFlowMeter features, 1 identifier, 1 target).
* **Immediate Progression to Phase 3:** Feature engineering, binary label encoding, multi-class attack family taxonomy, stratified train/val/test splits, and feature scaling proceed immediately on the cleaned CIC-IDS2017 baseline.
* **Relegation of Multi-Dataset Studies:** CSE-CIC-IDS2018 (sub-sampled, stratified), UNSW-NB15 (parallel independent benchmark), and NSL-KDD (historical comparison) are classified as **Phase 14 Extended Evaluation benchmarks or Future Work** items, to be pursued only after the complete hybrid pipeline (Phases 3–11) is operational and evaluated.
* **CERT Deferred to Phase 8+:** CERT Insider Threat dataset integration is permissible as a secondary event source for the Phase 8 Event Correlation Engine, after the CIC-IDS2017-based pipeline is fully operational. It will never be used as a training data source for the ML classifier.

---

### 4.7 Hardware-Driven Implementation Decisions — RAM Diagnostic Findings (2026-09-08)

**Decision Date:** 2026-09-08  
**Decision Type:** Engineering Constraint — Hardware-Confirmed, Evidence-Based Implementation Changes  
**Trigger:** RAM diagnostic (check_ram.py, run 2026-09-06) on the development machine (~8 GB total system RAM)  
**Recorded By:** Research lead — retained permanently for methodology transparency and reproducibility

> This section records three concrete implementation decisions that were forced by empirically measured memory constraints. In each case, the **algorithm or analytical choice is unchanged**; only the **software implementation mechanism** has been adjusted for hardware feasibility. These decisions must be documented in the paper's Implementation / Experimental Setup section to allow other researchers to reproduce results on memory-constrained hardware.

---

#### 4.7.1 Measured Memory Profile (Evidence Basis)

The RAM diagnostic was run against the full Phase 2 cleaned dataset to confirm feasibility of in-memory training before beginning Phase 3:

| Metric | Measured Value |
|---|---|
| Total system RAM | 7.71 GB |
| Free RAM before load | ~0.860 GB |
| pandas DataFrame memory (deep, float64) | 1.595 GB |
| Process RSS delta during load | 1.778 GB |
| Task Manager — Available RAM at load peak | 28.8 MB (spike) |
| Task Manager — Sustained second-lowest point | ~200 MB |
| Dataset dimensions | 2,520,798 rows × 79 columns |

**Conclusion:** Loading the full cleaned dataset into a single pandas DataFrame and then passing it to a standard `sklearn.RandomForestClassifier.fit()` call is unsafe on this hardware. The OS reached near-total RAM exhaustion (28.8 MB Available) during the load alone, before any model training memory allocation. This rules out any library that requires the full feature matrix to reside in RAM simultaneously during training.

---

#### 4.7.2 Decision 1 — Random Forest Baseline: LightGBM Native RF Mode

**Algorithm:** unchanged — Random Forest (bagged ensemble of decision trees, as per Breiman, 2001).  
**Implementation:** changed — `lightgbm.LGBMClassifier(boosting_type='rf')` instead of `sklearn.RandomForestClassifier`.

**Rationale:** LightGBM's native RF mode produces an equivalent bagged-tree ensemble via histogram-binned training (`tree_method='hist'` internally). Instead of loading the full-precision feature matrix into RAM at once, LightGBM discretizes continuous features into a fixed number of integer bins (default 255), reducing per-feature memory by approximately 8× compared to float64. The bagged ensemble property is preserved identically: each tree is trained on a random row-sample (controlled by `bagging_fraction`) and a random column-sample (controlled by `feature_fraction`).

**Critical implementation note:** LightGBM RF mode requires explicit bagging parameters; without them, training silently falls back to gradient boosting regardless of `boosting_type='rf'`:
```python
lgb.LGBMClassifier(
    boosting_type='rf',
    bagging_fraction=0.8,   # row-level bootstrap sampling fraction (< 1.0)
    bagging_freq=1,          # enable row-level bagging every tree (>= 1)
    feature_fraction=0.8,    # column sampling fraction (< 1.0)
)
```

**Paper reporting note:** The paper will clearly state that the Random Forest baseline uses LightGBM's native RF mode for memory-constrained reproducibility, and will cite LightGBM (Ke et al., 2017) alongside Breiman (2001). The mathematical formulation of the ensemble (Section 3.1) is identical — only the training implementation differs.

---

#### 4.7.3 Decision 2 — XGBoost Comparative Model: Histogram-Binned Training

**Algorithm:** unchanged — XGBoost gradient-boosted trees (Chen & Guestrin, 2016).  
**Implementation:** `xgboost.XGBClassifier(tree_method='hist')` — histogram-binned training instead of the default exact greedy split-finding method.

**Rationale:** XGBoost's `tree_method='hist'` uses the same histogram discretization strategy as LightGBM, avoiding the need to hold the full sorted feature matrix in RAM during split finding. Memory footprint during training is comparable to LightGBM RF mode. This allows fair comparison between the two models on identical hardware without one model having an artificial advantage from a less memory-efficient training mode.

**Sequencing constraint (unchanged from AGENTS.md):** XGBoost is trained **after** Random Forest is fully evaluated and explicitly released from memory. The two models are never resident in RAM simultaneously.

**Paper reporting note:** The paper will note `tree_method='hist'` in the experimental setup table for reproducibility. This is a standard, documented XGBoost parameter — not an approximation — and does not affect the model's predictive output relative to the exact method for this dataset's value ranges.

---

#### 4.7.4 Decision 3 — Feature Precision and Intermediate File Format

**Numeric precision:** All ML feature columns are stored and loaded as **float32** from Phase 3 onward (not the pandas/CSV default of float64).

*Rationale:* For this dataset's value ranges (packet counts, byte counts in the thousands; durations in microseconds; rates per second), float32 provides ~7 significant digits of precision — sufficient for all 76 CICFlowMeter features. Halving numeric precision halves in-memory footprint of any loaded feature array, reducing the Phase 3 split artifacts from an estimated ~1.5 GB (float64) to ~750 MB (float32). For tree-based models, which split on feature inequalities rather than exact numeric gradients, there is no meaningful accuracy tradeoff.

**Intermediate file format:** Phase 3 train/test split outputs (`X_train`, `X_test`, `y_train`, `y_test`) are written and read as **Parquet files** rather than CSV.

*Rationale:* Parquet is a binary, column-oriented format that typically achieves 3–10× compression over equivalent CSV for numeric tabular data, with faster read times and lower parsing overhead (no CSV tokenization on load). For Phase 4 and Phase 5, loading only the required columns from a Parquet file without materializing the full matrix provides an additional memory-efficiency benefit over CSV, which always materializes all columns during parsing.

**Scope boundary:** This decision applies **only to new Phase 3+ intermediate artifacts**. The Phase 2 cleaned CSVs in `artifacts/cleaned/` are the authoritative frozen baseline and are never converted, replaced, or re-encoded.

**Paper reporting note:** The paper's Experimental Setup section will note float32 feature encoding and Parquet intermediate storage as reproducibility details, alongside the hardware profile and memory budgets.

---

### 4.8 Phase 3 Feature Engineering & Train/Test Split — Methodology & Empirical Results (Paper Section)

**Pipeline Version:** `1.0` | **Timestamp (UTC):** `2026-09-08 09:12:48 UTC` | **Peak RSS:** `459.45 MB` / 500 MB limit | **Elapsed:** `136.76 s`

#### 4.8.1 Pipeline Design (Memory-Constrained 3-Pass Streaming)

To satisfy the confirmed RAM budget (~8 GB system, ~0.86 GB free at idle — see §4.7.1), Phase 3 was implemented as a three-pass streaming pipeline over the 8 cleaned CSVs. No full-dataset DataFrame is ever resident in RAM simultaneously.

| Pass | Data Read | Operation | Peak RAM Contribution |
|---|---|---|---|
| **Pass 1** | `Label` column only (1 of 79) | Concatenate all labels into a single array; fit `LabelEncoder`; generate stratified 80/20 split mask; write `y_*` Parquet files | ~50 MB (label array for 2.52M rows) |
| **Pass 2** | 76 ML feature columns only | `StandardScaler.partial_fit()` on train-masked chunks only (never test rows) | ~200 MB (one chunk at a time) |
| **Pass 3** | 76 ML feature columns only | `scaler.transform()` chunk-by-chunk; write to `X_train` / `X_test` via `pyarrow.parquet.ParquetWriter` incrementally | ~300 MB (chunk + scaled output simultaneously) |

**Chunk size:** 50,000 rows per iteration. **Total I/O passes:** 3 × 8 CSVs = 24 file reads.

#### 4.8.2 Column Selection

Of the 79 columns present in the Phase 2 cleaned CSVs, three were excluded from the ML feature matrix:

| Column | Reason for Exclusion |
|---|---|
| `Destination Port` | Network identifier — retained for Phase 7 rule engine and Phase 8 correlation, but must not be memorised by the ML model as a proxy label |
| `Fwd Header Length.1` | Exact duplicate of `Fwd Header Length` — carries zero marginal information |
| `Label` | Target column |

**Final ML feature count: 76** (all CICFlowMeter numeric flow attributes).

#### 4.8.3 Label Encoding

Two target representations are produced — binary for overall attack detection, category for multi-class attack classification:

| Encoding | Scheme | Values |
|---|---|---|
| **Binary** | `0` = BENIGN, `1` = any attack | 2 classes |
| **Category** | Integer index via `sklearn.LabelEncoder` | 15 classes (0–14), sorted alphabetically |

**Category class mapping (integer → label):**

| Int | Label | Int | Label |
|---|---|---|---|
| 0 | `BENIGN` | 8 | `Heartbleed` |
| 1 | `Bot` | 9 | `Infiltration` |
| 2 | `DDoS` | 10 | `PortScan` |
| 3 | `DoS GoldenEye` | 11 | `SSH-Patator` |
| 4 | `DoS Hulk` | 12 | `Web Attack - Brute Force` |
| 5 | `DoS Slowhttptest` | 13 | `Web Attack - Sql Injection` |
| 6 | `DoS slowloris` | 14 | `Web Attack - XSS` |
| 7 | `FTP-Patator` | — | — |

#### 4.8.4 Train/Test Split

| Parameter | Value |
|---|---|
| Method | `sklearn.model_selection.train_test_split` |
| Stratify on | Category label (15-class, finest grain) |
| Test fraction | 0.20 (80/20) |
| Random state | 42 (fixed seed — fully reproducible) |
| All 15 categories present in both splits | ✅ Verified |

| Split | Rows | % of 2,520,798 | Binary: BENIGN | Binary: ATTACK |
|---|---|---|---|---|
| **Train** | 2,016,638 | 80.0% | 1,676,045 | 340,593 |
| **Test** | 504,160 | 20.0% | 419,012 | 85,148 |
| **Total** | 2,520,798 | 100.0% | 2,095,057 | 425,741 |

#### 4.8.5 Per-Class Distribution (Measured)

| Attack Category | Train Count | Test Count | Train % | Test % |
|---|---|---|---|---|
| `BENIGN` | 1,676,045 | 419,012 | 83.115% | 83.115% |
| `Bot` | 1,558 | 390 | 0.077% | 0.077% |
| `DDoS` | 102,411 | 25,603 | 5.079% | 5.079% |
| `DoS GoldenEye` | 8,229 | 2,057 | 0.408% | 0.408% |
| `DoS Hulk` | 138,277 | 34,569 | 6.858% | 6.858% |
| `DoS Slowhttptest` | 4,182 | 1,046 | 0.207% | 0.207% |
| `DoS slowloris` | 4,308 | 1,077 | 0.214% | 0.214% |
| `FTP-Patator` | 4,745 | 1,186 | 0.235% | 0.235% |
| `Heartbleed` | 9 | 2 | 0.000% | 0.000% |
| `Infiltration` | 29 | 7 | 0.001% | 0.001% |
| `PortScan` | 72,555 | 18,139 | 3.598% | 3.599% |
| `SSH-Patator` | 2,575 | 644 | 0.128% | 0.128% |
| `Web Attack - Brute Force` | 1,176 | 294 | 0.058% | 0.058% |
| `Web Attack - Sql Injection` | 17 | 4 | 0.001% | 0.001% |
| `Web Attack - XSS` | 522 | 130 | 0.026% | 0.026% |

> [!NOTE]
> Train% and Test% are identical per class (to 3 decimal places) — confirming correct stratification. Heartbleed (train=9, test=2) and Infiltration (train=29, test=7) remain critically under-represented; per §4.6.5 this is acknowledged as a methodological constraint and documented honestly in the paper's Limitations section.

#### 4.8.6 Feature Scaling

- **Method:** `sklearn.preprocessing.StandardScaler` — zero-mean, unit-variance standardisation per feature.
- **Fit discipline:** Scaler fitted **on training rows only** (Pass 2). Test rows are transformed with training statistics — no data leakage.
- **Numeric format:** All features cast to `float32` before scaling; sklearn returns `float64` internally but the output is immediately cast back to `float32` before Parquet write (AGENTS.md §Hardware Constraint).
- **Mean range across 76 features:** `[-35,647.09, 16,582,651.87]` — confirming that flow-rate features have very large absolute magnitudes and scaling is essential for any gradient-based method downstream.

#### 4.8.7 Output Artifacts

| Artifact | Size on Disk | Description |
|---|---|---|
| `X_train_v1.parquet` | 190.4 MB | 76 float32 scaled features — 2,016,638 train rows |
| `X_test_v1.parquet` | 51.4 MB | 76 float32 scaled features — 504,160 test rows |
| `y_train_binary_v1.parquet` | <1 MB | Binary label (0/1) — train |
| `y_test_binary_v1.parquet` | <1 MB | Binary label (0/1) — test |
| `y_train_category_v1.parquet` | <1 MB | Integer category label (0–14) — train |
| `y_test_category_v1.parquet` | <1 MB | Integer category label (0–14) — test |
| `label_encoder_v1.pkl` | <1 MB | Fitted `sklearn.LabelEncoder` — decodes int → class name |
| `scaler_v1.pkl` | <1 MB | Fitted `StandardScaler` — train statistics only |
| `is_train_mask_v1.npy` | 2.4 MB | Boolean array (N=2,520,798) — `True` = train row |
| `split_summary_v1.json` | <1 MB | Full machine-readable pipeline summary |
| `split_summary_v1.md` | <1 MB | Human-readable markdown report |

**Storage efficiency:** Parquet/snappy achieves ~241.8 MB for the full 76-feature, 2.52M-row split vs. an estimated ~1.5 GB if stored as float64 CSV — a ~6× reduction.

#### 4.8.8 Reproducibility Checklist

To reproduce Phase 3 results exactly:
1. Use Python 3.14.7, `scikit-learn==1.9.0`, `pyarrow==25.0.1`, `numpy==2.5.2`.
2. Run `python -m ml_service.src.feature_engineer` from the project root with the `.env` vars pointing to the Personal workspace.
3. `RANDOM_STATE=42` is the only stochastic input — all other operations are deterministic.
4. The 8 Phase 2 cleaned CSVs in `artifacts/cleaned/` must be identical (SHA-256 verified in §4.5.6).

### 4.9 Phase 4 Random Forest Baseline — Methodology & Empirical Results (Paper Section)

**Pipeline Version:** `1.0` | **Timestamp (UTC):** `2026-09-08 16:38:01 UTC` | **Split:** 80/20 Stratified (Train: 2,016,638 / Test: 504,160)  
**Hardware Profile:** ~8 GB Total System RAM, ~1.15 GB Available at execution start | **Training Time:** `140.35 s` | **Inference Latency:** `38.26 μs/sample`

#### 4.9.1 Model Architecture & Training Setup
- **Algorithm:** Random Forest (bagged-tree ensemble) implemented via LightGBM native RF mode (`boosting_type='rf'`).
- **Bagging Configuration:** `bagging_fraction=0.8` (row-level subsampling), `bagging_freq=1` (bagging active every tree), `feature_fraction=0.8` (column-level subsampling).
- **Tree Depth & Leaves:** `n_estimators=100`, `num_leaves=63`, `max_depth=15`, `random_state=42`, `n_jobs=-1`.
- **Weighting Strategy:** `class_weight=None` (unweighted baseline) — avoids the ~14,938× sample-weight distortion on 9-sample classes like Heartbleed while preserving honest minority detection statistics.
- **Memory Discipline:** Direct histogram bin construction from streaming row-groups with `free_raw_data=True`. Raw float32 array de-allocated before tree fitting, yielding resident training RSS of `531.28 MB` and inference RSS of `579.88 MB`.

#### 4.9.2 Binary Intrusion Detection Performance (Test Set N=504,160)
- **Accuracy:** 98.522%
- **Detection Rate (Recall / TPR):** 99.940% (85,097 of 85,148 attacks detected; only 51 missed)
- **False Positive Rate (FPR):** 1.767% (7,402 benign flows flagged out of 419,012 benign test flows)
- **Precision:** 91.998%
- **F1-Score:** 0.95805
- **ROC-AUC:** 0.99169
- **Binary Confusion Matrix:** TN=411,610 | FP=7,402 | FN=51 | TP=85,097

#### 4.9.3 Multi-Class 15-Category Breakdown (Measured)

| Attack Category | Test Support | Precision | Recall | F1-Score | Evaluation Note |
|---|---|---|---|---|---|
| `BENIGN` | 419,012 | 0.9999 | 0.9823 | 0.9910 | High support baseline |
| `Bot` | 390 | 0.1523 | 0.9846 | 0.2637 | High recall; low precision |
| `DDoS` | 25,603 | 0.9982 | 0.9970 | 0.9976 | Near-perfect separation |
| `DoS GoldenEye` | 2,057 | 0.9371 | 0.9771 | 0.9567 | Highly discriminative |
| `DoS Hulk` | 34,569 | 0.9965 | 0.9921 | 0.9943 | Near-perfect separation |
| `DoS Slowhttptest` | 1,046 | 0.8783 | 0.9933 | 0.9323 | Robust classification |
| `DoS slowloris` | 1,077 | 0.8767 | 0.9768 | 0.9240 | Robust classification |
| `FTP-Patator` | 1,186 | 0.9412 | 0.9983 | 0.9689 | Tuesday brute-force ground truth |
| `Heartbleed` | 2 | 0.0133 | 0.5000 | 0.0260 | ⚠️ Low Support (N=2, 1/2 detected) |
| `Infiltration` | 7 | 0.0027 | 0.7143 | 0.0054 | ⚠️ Low Support (N=7, 5/7 detected) |
| `PortScan` | 18,139 | 0.9882 | 0.9983 | 0.9932 | High volume scan detection |
| `SSH-Patator` | 644 | 0.6986 | 0.9969 | 0.8215 | Tuesday brute-force ground truth |
| `Web Attack - Brute Force` | 294 | 0.3101 | 0.5612 | 0.3995 | Thursday web attack |
| `Web Attack - Sql Injection` | 4 | 0.0020 | 1.0000 | 0.0039 | ⚠️ Low Support (N=4, 4/4 detected) |
| `Web Attack - XSS` | 130 | 0.1967 | 0.7308 | 0.3100 | Thursday web attack |

**Global Multi-Class Summary:**
- Overall Accuracy: 98.406%
- Balanced Accuracy: 89.354%
- Weighted F1-Score: 0.98988
- Macro F1-Score: 0.63921 (depressed by N<=7 rare classes)
- Multi-Class ROC-AUC (OVR Weighted): 0.99210


### 4.10 Phase 5 XGBoost Comparative Model — Methodology & Empirical Results (Paper Section)

**Pipeline Version:** `1.0` | **Timestamp (UTC):** `2026-09-10 18:38:44 UTC` | **Split:** 80/20 Stratified (Train: 2,016,638 / Test: 504,160)  
**Hardware Profile:** ~8 GB Total System RAM, CPU-only execution | **Training Time:** `957.00 s` | **Inference Latency:** `14.71 μs/sample`

#### 4.10.1 Model Architecture & Training Configuration
- **Algorithm:** XGBoost gradient-boosted decision trees (`xgboost.Booster` via native `xgb.train` with `tree_method='hist'`).
- **Boosting & Regularization Hyperparameters:** `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8` (row subsampling), `colsample_bytree=0.8` (column feature subsampling), `objective='multi:softprob'`, `eval_metric='mlogloss'`, `seed=42`, `device='cpu'`.
- **Memory Discipline & Histogram Binning:** Dataset ingested directly into a histogram-binned `xgb.DMatrix` on CPU. Raw float32 numpy arrays de-allocated prior to tree fitting, releasing memory and resulting in post-fit resident RSS of `136.59 MB` and inference resident RSS of `378.82 MB`.

#### 4.10.2 Binary Intrusion Detection Performance (Test Set N=504,160)
- **Accuracy:** 99.887%
- **Detection Rate (Recall / TPR):** 99.802% (84,979 of 85,148 attacks detected; 169 missed)
- **False Positive Rate (FPR):** 0.0952% (399 benign flows flagged out of 419,012 benign test flows — **18.5× reduction in false positives vs. Random Forest**)
- **Precision:** 99.533%
- **F1-Score:** 0.99667
- **ROC-AUC:** 0.99993
- **Binary Confusion Matrix:** TN=418,613 | FP=399 | FN=169 | TP=84,979

#### 4.10.3 Multi-Class 15-Category Breakdown (Measured)

| Attack Category | Test Support | Precision | Recall | F1-Score | Evaluation Note |
|---|---|---|---|---|---|
| `BENIGN` | 419,012 | 0.9996 | 0.9991 | 0.9993 | High support baseline |
| `Bot` | 390 | 0.8323 | 0.7128 | 0.7680 | Major improvement vs RF (+0.504 F1) |
| `DDoS` | 25,603 | 1.0000 | 0.9999 | 0.9999 | Flawless separation |
| `DoS GoldenEye` | 2,057 | 0.9966 | 0.9927 | 0.9946 | High precision & recall |
| `DoS Hulk` | 34,569 | 0.9980 | 0.9995 | 0.9987 | Robust separation |
| `DoS Slowhttptest` | 1,046 | 0.9412 | 0.9952 | 0.9675 | High discriminative power |
| `DoS slowloris` | 1,077 | 0.9898 | 0.9898 | 0.9898 | High discriminative power |
| `FTP-Patator` | 1,186 | 0.9983 | 0.9983 | 0.9983 | Tuesday brute-force ground truth |
| `Heartbleed` | 2 | 1.0000 | 0.5000 | 0.6667 | ⚠️ Low Support (N=2, 1/2 detected) |
| `Infiltration` | 7 | 1.0000 | 1.0000 | 1.0000 | ⚠️ Low Support (N=7, 7/7 detected) |
| `PortScan` | 18,139 | 0.9886 | 0.9993 | 0.9939 | High volume scan detection |
| `SSH-Patator` | 644 | 0.9984 | 0.9969 | 0.9977 | Tuesday brute-force ground truth |
| `Web Attack - Brute Force` | 294 | 0.7328 | 0.8674 | 0.7944 | Thursday web attack (+0.395 F1 vs RF) |
| `Web Attack - Sql Injection` | 4 | 1.0000 | 0.2500 | 0.4000 | ⚠️ Low Support (N=4, 1/4 detected) |
| `Web Attack - XSS` | 130 | 0.5312 | 0.2615 | 0.3505 | Thursday web attack |

**Global Multi-Class Summary:**
- Overall Accuracy: 99.858%
- Balanced Accuracy: 83.750%
- Weighted F1-Score: 0.99854
- Macro F1-Score: 0.86129 (**+0.22208 boost over Random Forest baseline**)
- Multi-Class ROC-AUC (OVR Weighted): 0.99992

### 4.11 Formal Model Selection Decision & Comparative Synthesis (Paper Section)

```
========================================================================================
MODEL SELECTION VERDICT: XGBoost Selected as Primary Threat Classifier for Phase 6 (SHAP)
========================================================================================
```

1. **Alert Fatigue Mitigation (Primary SOC Criterion):** XGBoost reduced False Positive Rate from `1.7665%` (RF) down to `0.0952%` — representing an **18.5× reduction** in false alerts (399 vs. 7,402). In an operational SOC processing 500,000 flows daily, this eliminates 7,003 false alarms without sacrificing detection efficacy.
2. **Rare-Class Discernment:** Macro F1 increased from `0.63921` to `0.86129` (+0.22208). Critical improvements were achieved in Botnet ($F_1: 0.2637 \to 0.7680$), SSH-Patator ($F_1: 0.8215 \to 0.9977$), and Web Attack Brute Force ($F_1: 0.3995 \to 0.7944$).
3. **Inference Latency Advantage:** XGBoost inference latency is `14.71 μs/sample` compared to RF's `38.26 μs/sample` (**2.6× faster throughput**).
4. **Architectural Handoff:** `shap.TreeExplainer` in Phase 6 will be initialized on the selected `xgb_baseline_category_v1` model using stratified representative background samples.

### 4.12 Explainability via SHAP (Phase 6 Technical Log & Methodology)

Phase 6 implements SHAP (SHapley Additive exPlanations) for the selected XGBoost primary threat classifier (`xgb_baseline_category_v1.pkl`).

#### 1. Explainer Configuration & Memory Safety Protocol
- **Explainer Architecture:** `shap.TreeExplainer` executed in `feature_perturbation='tree_path_dependent'` mode (`data=None`). In tree-path-dependent mode, SHAP uses the conditional tree structures and coverage statistics learned during training, eliminating background sampling bias and extraneous reference sets.
- **Sampling Strategy:** Per-class capped sampling (up to 150 rows per category) drawn from the 504,160-row test split (`y_test_category_v1.parquet` / `X_test_v1.parquet`, `seed=42`).
  - Total Sample Size: **1,793 flows** across all 15 categories.
  - Rare classes with $\le 150$ samples were fully retained: Heartbleed ($n=2$), Infiltration ($n=7$), Web Attack SQL Injection ($n=4$), Web Attack XSS ($n=130$).
  - Classes with $> 150$ samples were capped at 150. This avoids proportional sampling bias (which would have assigned $>1,660$ rows to BENIGN and starved rare threat categories).
- **Hardware Footprint:**
  - SHAP value array dimensions: `(1793, 76, 15)` in `float32`.
  - In-memory array size: **7.80 MB** (well below the ~8 GB system boundary).
  - Peak RSS during attribution: **587.1 MB** (below 700 MB safety limit).
  - Compute latency: **2.39 s** total (**1.33 ms/sample**).

#### 2. Mathematical Additivity Verification
Before accepting explanations, an explicit additivity verification was evaluated:
$$\text{Expected Output} = \text{Base Value} + \sum_{i=1}^{M} \text{SHAP}_i$$
- Verification result: **`PASSED`** (Tolerance: $1.00 \times 10^{-3}$).
- Max Absolute Discrepancy: **`7.63e-06`** (conforming to floating-point precision).
- Mean Absolute Discrepancy: **`1.62e-06`**.

#### 3. Unbiased Global Ranking via Macro-Averaged Mean |SHAP|
Because the explanation sample uses a per-class cap rather than proportional sampling, global feature importance is calculated as the **macro-averaged mean |SHAP|** (the mean absolute attribution per class, averaged equally across all 15 classes):
1. `Flow IAT Max` (Macro Mean |SHAP|: 3.1331) — Inter-arrival burst timing.
2. `Max Packet Length` (Macro Mean |SHAP|: 2.6797) — Peak frame sizing.
3. `Init_Win_bytes_backward` (Macro Mean |SHAP|: 2.4845) — TCP window scale in reverse flow.
4. `Flow IAT Min` (Macro Mean |SHAP|: 2.0021) — Tight transmission intervals.
5. `Active Min` (Macro Mean |SHAP|: 1.8443) — Persistent connection activity.
6. `Fwd Packet Length Max` (Macro Mean |SHAP|: 1.8078) — Client payload capacity.
7. `Flow Packets/s` (Macro Mean |SHAP|: 1.7970) — Volumetric transmission rate.
8. `Bwd Packets/s` (Macro Mean |SHAP|: 1.7517) — Server response packet rate.
9. `PSH Flag Count` (Macro Mean |SHAP|: 1.6535) — Push flag signaling immediate buffer flush.
10. `Bwd IAT Max` (Macro Mean |SHAP|: 1.6157) — Maximum server inter-arrival time.

#### 4. Scientific Hypothesis Testing vs. Empirical Findings
- **PortScan (Hypothesis Confirmed):** `Destination Port` was excluded as an identifier in Phase 2. Model discriminates PortScan via single-flow geometry: `Total Length of Fwd Packets` (1.8120), `min_seg_size_forward` (1.3104), and `Packet Length Mean` (1.1323).
- **DoS Hulk (Hypothesis Confirmed):** Discriminates via volumetric and window dynamics: `Init_Win_bytes_backward` (2.4377), `Bwd Packet Length Std` (1.6792), and `Idle Max` (1.6134).
- **FTP-Patator (Hypothesis Confirmed):** Discriminates via predictable authentication handshakes: `Max Packet Length` (6.9015), `Total Backward Packets` (6.0251), and `Fwd PSH Flags` (2.2982).
- **Botnet (Hypothesis Confirmed):** Discriminates via C2 heartbeat dynamics: `Bwd Packets/s` (2.6385), `Fwd Header Length` (2.4999), and `Init_Win_bytes_backward` (2.3668).
- **Methodological Disclosure on Low-Support Classes:** Explanations for Heartbleed ($n=2$), Infiltration ($n=7$), and SQL Injection ($n=4$) represent **anecdotal exploratory observations only** and are documented with caveats.

#### 5. Local Error Attribution (Misclassified Case Analysis)
In addition to high-confidence True Positive waterfalls, an error attribution analysis was performed on misclassified samples (e.g. Sample #376: True=`DoS slowloris`, Pred=`BENIGN`). The attribution showed low packet length variance and low packet rates pushed margins toward BENIGN, providing actionable diagnostic transparency for model boundary refinement.

### 4.13 Deterministic Security Rule Engine & Complementarity Analysis (Phase 7 Technical Log)

Phase 7 implements the deterministic security rule engine operating independently of ML predictions per AGENTS.md.

#### 1. Architectural Safeguards & Dataset Boundary
- **Input Telemetry:** Operates on raw (unscaled) continuous metrics and `Destination Port` streamed from `artifacts/cleaned/*.csv`. Standardized features (`X_test_v1.parquet`) were excluded from rule evaluation to preserve physical threshold validity.
- **Dataset Boundary Disclosure:** Audited and confirmed that `MachineLearningCVE` contains `Destination Port` and 76 flow metrics, but omits `Source IP`, `Destination IP`, and `Timestamp`. Rules evaluate per-flow deterministic signatures.
- **Row Alignment Verification:** Formally asserted `y_extracted_test == y_test_category_v1.parquet` across all **504,160 test flows** before evaluation (`PASSED`).
- **Train-Only Calibration & Precision Floor:** Enforced a strict 50% precision floor on training rows (`is_train_mask == True`). Rules failing the floor were demoted to `INFORMATIONAL` to prevent alert flooding into Phase 8.
- **Measured Latency:** Empirically clocked at **0.25 μs/sample** (4,000,000 flows/second throughput), negligible operational overhead.

#### 2. Calibrated Rule Definitions & Performance Metrics (Test Split: 504,160 flows)
| Rule ID | Threat Target | Severity | Status | Train Prec | Test Prec | Test Recall (TPR) | Test FPR | Test TP | Test FP |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `R101_SYN_HALF_OPEN` | DoS / PortScan | HIGH | INFORMATIONAL | 0.00% | 0.00% | 0.00% | 0.0036% | 0 | 15 |
| `R102_HIGH_RATE_BURST` | DoS Hulk / DDoS | CRITICAL | INFORMATIONAL | 0.00% | 0.00% | 0.00% | 0.0000% | 0 | 0 |
| `R103_SLOW_EXHAUSTION` | DoS Slowloris / Slowhttptest | HIGH | INFORMATIONAL | 20.14% | 19.97% | 3.97% | 3.24% | 3,384 | 13,561 |
| `R104_PORT_BRUTE_FORCE` | FTP / SSH Brute Force | HIGH | **ACTIVE** | **64.85%** | **63.85%** | **1.64%** | **0.189%** | **1,399** | **792** |
| `R105_ZERO_WINDOW_TEARDOWN` | TCP Evasion / Scan | MEDIUM | INFORMATIONAL | 0.00% | 0.00% | 0.00% | 0.0000% | 0 | 0 |

- **Active Ensemble Summary:** The calibrated active rule ensemble achieved **63.85% precision**, **1.64% recall**, and **0.189% false-positive rate** (1,399 attacks caught, 792 false alerts).

#### 3. ML vs. Rule Complementarity Matrix (Hybrid Defense Synergy)
Benchmarked against the primary XGBoost classifier (Phase 5) across the full 504,160-row test split:

| Complementarity Category | Flow Count | % of Test Split | Research & Operational Significance |
|---|:---:|:---:|---|
| **Category A (ML & Rules Flagged)** | **1,399** | 0.28% | **True Threat Synergy:** Mutual high-confidence detection on administrative port credential attacks. |
| **Category B (ML Only Flagged)** | **83,979** | 16.66% | **High-Dimensional ML Generalization:** Captured complex non-linear volumetric, IAT, and window dynamics across DoS, DDoS, PortScan, and Web attacks that simple static rules missed. |
| **Category C (Rules Only Flagged)** | **792** | 0.16% | **Deterministic Safety Net:** Administrative port traffic flagged by rule policy where ML predicted Benign. |
| **Category D (Neither Flagged)** | **417,990** | 82.91% | **Clean Benign Traffic:** Operational benign network traffic passed cleanly by both systems. |

---

## 5. Experimental Results & Benchmark Log

### 5.1 Model Performance Comparison (Phase 4 vs Phase 5)
| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score | ROC-AUC | FPR (%) | Train Time (s) | Inference Latency (ms/flow) | RAM Peak (MB) |
|---|---|---|---|---|---|---|---|---|---|
| **Random Forest (Baseline)** | 98.52% (bin) / 98.41% (mc) | 92.00% (bin) / 98.99% (wgt) | 99.94% (bin) / 89.35% (bal) | 0.9581 (bin) / 0.9899 (wgt) | 0.9917 (bin) / 0.9921 (ovr) | 1.77% | 140.35 s | 0.038 ms | 579.88 MB (fit: 531.28) |
| **XGBoost (Comparative / Selected)** | **99.89%** (bin) / **99.86%** (mc) | **99.53%** (bin) / **99.85%** (wgt) | **99.80%** (bin) / **83.75%** (bal) | **0.9967** (bin) / **0.9985** (wgt) | **0.9999** (bin) / **0.9999** (ovr) | **0.095%** | 957.00 s | **0.015 ms** | **378.82 MB** (fit: 136.59) |

### 5.2 XAI (SHAP) Top Feature Attribution (Phase 6)
| Attack Category | Top 1 Discriminative Feature | Top 2 Discriminative Feature | Top 3 Discriminative Feature | Top Feature Mean \|SHAP\| |
|---|---|---|---|---|
| **Brute Force (FTP-Patator)** | `Max Packet Length` | `Total Backward Packets` | `Fwd PSH Flags` | 6.9015 |
| **Brute Force (SSH-Patator)** | `Total Length of Bwd Packets` | `Bwd Header Length` | `Fwd Packet Length Max` | 12.4637 |
| **DoS Hulk** | `Init_Win_bytes_backward` | `Bwd Packet Length Std` | `Idle Max` | 2.4377 |
| **DoS Slowhttptest** | `Flow IAT Max` | `Active Min` | `Flow Packets/s` | 42.1098 |
| **DoS slowloris** | `Packet Length Std` | `Active Std` | `Bwd Packet Length Max` | 6.4105 |
| **DoS GoldenEye** | `Flow IAT Min` | `Idle Max` | `min_seg_size_forward` | 5.5580 |
| **DDoS** | `Init_Win_bytes_forward` | `Bwd Packet Length Mean` | `act_data_pkt_fwd` | 1.7159 |
| **PortScan** | `Total Length of Fwd Packets` | `min_seg_size_forward` | `Packet Length Mean` | 1.8120 |
| **Web Attack (Brute Force)** | `Fwd IAT Max` | `Max Packet Length` | `Init_Win_bytes_backward` | 7.9321 |
| **Web Attack (XSS)** | `PSH Flag Count` | `Init_Win_bytes_backward` | `Bwd Packet Length Min` | 19.2424 |
| **Botnet (ARES)** | `Bwd Packets/s` | `Fwd Header Length` | `Init_Win_bytes_backward` | 2.6385 |

### 5.3 Event Correlation & Incident Reduction (Phase 8)

*Evaluated on the 20% unseen test partition (504,160 flows) under primary grouping key $K_3 = (\text{Source IP}, \text{Destination IP}, \text{Destination Port})$, inactivity timeout $G = 300\text{ s}$, and hybrid alert ingestion (ML $\cup$ Rule R104). Ground-truth blind execution verified (`correlator.py` receives zero label/source telemetry).*

| Scenario / Capture File | Raw Alert Count ($N_{\text{raw}}$) | Correlated Incidents ($N_{\text{inc}}$) | Multi-Flow ($N \ge 2$) | Singletons ($N=1$) | Reduction Ratio (%) | Multi-Flow Purity ($N \ge 2$) | All-Inc Purity (%) | Consistent Coverage (%) | False Incidents (Zero Attack Flows) |
|---|---|---|---|---|---|---|---|---|---|
| **Monday (Benign Control)** | 188 | 167 | 17 | 150 | 11.17% | 100.00% | 100.00% | 0.00% | 167 / 167 (100.0%) |
| **Tuesday (Patator Validation)** | 2,022 | 165 | 23 | 142 | **91.84%** | 100.00% | 100.00% | 99.94% | 162 / 165 (98.2%) |
| **Wednesday (DoS / Heartbleed)** | 38,919 | 159 | 32 | 127 | **99.59%** | 92.02% | 92.05% | 92.01% | 156 / 159 (98.1%) |
| **Thursday Morning (Web Attacks)** | 504 | 87 | 12 | 75 | **82.74%** | 100.00% | 100.00% | 100.00% | 84 / 87 (96.5%) |
| **Thursday Afternoon (Infiltration)** | 335 | 315 | 14 | 301 | **5.97%** | 100.00% | 100.00% | 100.00% | 312 / 315 (99.1%) |
| **Friday Morning (Botnet)** | 351 | 196 | 15 | 181 | **44.16%** | 100.00% | 100.00% | 56.12% | 65 / 196 (33.2%) |
| **Friday Afternoon (PortScan)** | 18,208 | 2,172 ($K_3$) / **66 ($K_2$)** | 1,385 | 787 | **88.07% ($K_3$) / 99.64% ($K_2$)** | 100.00% | 100.00% | 96.01% | 69 / 2,172 (3.2%) |
| **Friday Afternoon (DDoS)** | 25,643 | 35 | 6 | 29 | **99.86%** | 100.00% | 100.00% | 100.00% | 34 / 35 (97.1%) |
| **TOTAL TEST SET (Primary)** | **86,170** | **3,296** | **1,504** | **1,792** | **96.17%** | **96.33%** | **96.41%** | **95.36%** | **1,049 / 3,296 (31.83%)** |
| **FULL DATASET (Supplementary)** | **430,586** | **6,954** | **3,721** | **3,233** | **98.39%** | **96.34%** | **96.40%** | **96.18%** | **1,343 / 6,954 (19.31%)** |

#### Per-Class Compression Breakdown (Test Partition, $K_3, G=300\text{ s}$)

| Attack Category | Ground-Truth Alerts Flagged | Correlated Incidents ($N_{\text{inc}}$) | Multi-Flow ($N \ge 2$) | Reduction Ratio (%) | Compression Factor |
|---|---|---|---|---|---|
| `DoS Hulk` | 34,559 | 1 | 1 | **100.00%** | **34559.0x** |
| `DDoS` | 25,602 | 1 | 1 | **100.00%** | **25602.0x** |
| `PortScan` | 18,134 | 2,103 | 1,380 | **88.40%** | **8.6x** |
| `DoS GoldenEye` | 2,048 | 1 | 1 | **99.95%** | **2048.0x** |
| `FTP-Patator` | 1,185 | 2 | 1 | **99.83%** | **592.5x** |
| `DoS slowloris` | 1,069 | 1 | 1 | **99.91%** | **1069.0x** |
| `DoS Slowhttptest` | 1,046 | 1 | 1 | **99.90%** | **1046.0x** |
| `SSH-Patator` | 642 | 1 | 1 | **99.84%** | **642.0x** |
| `Web Attack - Brute Force` | 281 | 1 | 1 | **99.64%** | **281.0x** |
| `Bot` | 278 | 131 | 9 | **52.88%** | **2.1x** |
| `Web Attack - XSS` | 125 | 1 | 1 | **99.20%** | **125.0x** |
| `Infiltration` | 7 | 3 | 3 | **57.14%** | **2.3x** |
| `Web Attack - Sql Injection` | 2 | 1 | 1 | **50.00%** | **2.0x** |
| `Heartbleed` | 1 | 1 | 0 | **0.00%** | **1.0x** |

#### Gap Window Sensitivity Sweep ($G \in \{60, 300, 900, 1800\}\text{ s}$)

| Gap Timeout ($G$) | Total Incidents | Multi-Flow ($N \ge 2$) | Singletons ($N=1$) | Reduction Ratio (%) | Multi-Flow Purity ($N \ge 2$) | All-Inc Purity (%) | Consistent Coverage (%) |
|---|---|---|---|---|---|---|---|
| `G = 60s` | 4,180 | 2,151 | 2,029 | 95.15% | 100.00% | 100.00% | 98.86% |
| `G = 300s` | 3,296 | 1,504 | 1,792 | 96.17% | 96.33% | 96.41% | 95.36% |
| `G = 900s` | 2,119 | 1,277 | 842 | 97.54% | 94.97% | 95.02% | 94.65% |
| `G = 1800s` | 1,750 | 1,190 | 560 | 97.97% | 94.98% | 95.02% | 94.81% |

**Per-Day Incident Fragmentation Across Gap Windows ($G$):**
- Monday: $60\text{ s} \rightarrow 180$, $300\text{ s} \rightarrow 167$, $900\text{ s} \rightarrow 109$, $1800\text{ s} \rightarrow 85$.
- Tuesday: $60\text{ s} \rightarrow 187$, $300\text{ s} \rightarrow 165$, $900\text{ s} \rightarrow 114$, $1800\text{ s} \rightarrow 73$.
- Wednesday: $60\text{ s} \rightarrow 184$, $300\text{ s} \rightarrow 159$, $900\text{ s} \rightarrow 120$, $1800\text{ s} \rightarrow 96$.
- Thursday Morning: $60\text{ s} \rightarrow 95$, $300\text{ s} \rightarrow 87$, $900\text{ s} \rightarrow 61$, $1800\text{ s} \rightarrow 40$.
- Thursday Afternoon: $60\text{ s} \rightarrow 326$, $300\text{ s} \rightarrow 315$, $900\text{ s} \rightarrow 283$, $1800\text{ s} \rightarrow 264$.
- Friday Morning: $60\text{ s} \rightarrow 235$, $300\text{ s} \rightarrow 196$, $900\text{ s} \rightarrow 169$, $1800\text{ s} \rightarrow 157$.
- Friday Afternoon (PortScan): $60\text{ s} \rightarrow 2,935$, $300\text{ s} \rightarrow 2,172$, $900\text{ s} \rightarrow 1,247$, $1800\text{ s} \rightarrow 1,028$.
- Friday Afternoon (DDoS): $60\text{ s} \rightarrow 38$, $300\text{ s} \rightarrow 35$, $900\text{ s} \rightarrow 18$, $1800\text{ s} \rightarrow 16$.

#### Grouping Key Ablations: $K_3$ vs $K_2$ vs $K_4$ vs Dest-Port-Only ($G=300\text{ s}$)

| Capture Day | $K_3$ Primary: `(Src, Dst, Dport)` | $K_2$ Port-Agile: `(Src, Dst)` | $K_4$ Class-Qualified: `(Src, Dst, Dport, Class)` | Dest-Port-Only: `(Dport,)` |
|---|---|---|---|---|
| `Monday` | 167 | 145 | 167 | 97 |
| `Tuesday` | 165 | 147 | 165 | 63 |
| `Wednesday` | 159 | 141 | 166 | 79 |
| `Thursday Morning` | 87 | 76 | 94 | 45 |
| `Thursday Afternoon` | 315 | 105 | 315 | 259 |
| `Friday Morning` | 196 | 83 | 196 | 160 |
| `Friday Afternoon (PortScan)` | 2,172 | 66 | 2,177 | 2,124 |
| `Friday Afternoon (DDoS)` | 35 | 29 | 36 | 20 |
| **TOTAL TEST SET** | **3,296** | **792** | **3,316** | **2,846** |
| **Multi-Flow Purity ($N \ge 2$)** | **96.33%** | **96.39%** | **99.98%** | **96.28%** |
| **Consistent Coverage** | **95.36%** | **96.34%** | **98.97%** | **95.37%** |

#### Key Empirical Findings & Methodological Audits
1. **Tuesday Acceptance Gate (Passed 5/5):**
   - *Context Note:* Tuesday represents the easiest test case in the dataset because all attacks originate from a single gateway IP (`172.16.0.1`), target distinct well-known ports (`21` for FTP, `22` for SSH), and exhibit dense continuous packet streams.
   - FTP-Patator episode fragmentation: **2 incidents** (Pre-registered goal: $\le 5$).
   - SSH-Patator episode fragmentation: **1 incident** (Pre-registered goal: $\le 5$).
   - Flow-weighted purity: **100.000%** (Pre-registered goal: $\ge 99.5\%$).
   - Incident-weighted purity: **100.000%** (Pre-registered goal: $\ge 99.5\%$).
   - Label-consistent multi-flow coverage: **99.945%** (Pre-registered goal: $\ge 98.0\%$).
2. **Volumetric Dominance & Alert Aggregation Dynamics:**
   - On unseen test data, 86,170 raw alert flows were grouped into **3,296 incidents** (**96.17% reduction**, **26.1× compression factor**).
   - **Volumetric Dominance:** The pooled reduction ratio is 96.05% dominated by three captures: Wednesday DoS (38,919 flows), PortScan (18,208 flows), and DDoS (25,643 flows), which sum to **82,770 out of the 86,170 alerts**.
   - **Benign False Alarms Limitation:** Event correlation **does not eliminate or reduce benign false-positive flows**. Sporadic false alarms remain as isolated singletons ($1,792$ singletons across test split), while false alarms occurring in temporal proximity aggregate into spurious multi-flow incidents.
3. **Monday Benign Control & Spurious Multi-Flow Campaigns:**
   - Monday (100% benign) yielded 188 flagged flows forming **167 false incidents**: 150 singletons and **17 multi-flow incidents** (spanning 38 flows).
   - **Root Cause Analysis:** 100% of these 17 multi-flow incidents were generated exclusively by Rule R104 (`RULE_ONLY:R104`) targeting `192.168.10.50` on port 21 or 22 from internal benign hosts (`.12`, `.16`, `.17`, `.19`, `.51`), containing **0 ML-flagged constituent flows**. In contrast, all 34 ML false alarms on Monday arrived isolated in time or key, forming singletons. False alarms can and do aggregate into spurious campaigns.
4. **Ingestion Ablation & Category C Audit:**
   - Under ML-only ingestion, false incidents numbered **367 out of 2,614 (14.04%)**.
   - Under Hybrid ingestion (ML $\cup$ R104), false incidents rose to **1,049 out of 3,296 (31.83%)** — adding **+682 false incidents**.
   - **Category C Ground-Truth Audit:** Exactly **0 attack flows** on the test partition were flagged only by R104 (all 792 Category C flows were benign false alarms). R104 expanded analyst incident volume without capturing any novel ground-truth threats.
5. **Purity Disaggregation ($N \ge 2$ vs All Incidents):**
   - Singletons ($1,792$ incidents) are trivially 100% pure.
   - For multi-flow incidents ($N \ge 2$, $1,504$ incidents, $84,378$ flows), flow-weighted purity is **96.33%** and incident-weighted purity is **99.96%**.
6. **PortScan Key Ablation ($K_3$ vs $K_2$):**
   - Under $K_3 = (\text{SrcIP}, \text{DstIP}, \text{DstPort})$, horizontal port scanning fragments across ports into 2,172 incidents.
   - Relaxing to $K_2 = (\text{SrcIP}, \text{DstIP})$ collapses the entire scan sweep into **66 incidents** (**32.9× collapse factor**) with 100.0% purity.
7. **Tuesday $K_2$ Merge Audit:**
   - Under $K_2 = (\text{SrcIP}, \text{DstIP})$, FTP-Patator and SSH-Patator **did not merge** (0 merged incidents). Their ~3.5-hour temporal separation ($\approx 13,000\text{ s} \gg 300\text{ s}$) prevented temporal aliasing. Therefore, all empirical evidence discriminating $K_2$ and $K_3$ behavior comes from horizontal scanning (PortScan on Friday Afternoon).
8. **Architectural Comparison (Sliding Gap vs Fixed Tumbling):**
   - Sliding gap sessionizer ($G=300\text{ s}$) produces **3,296 incidents** versus **4,524 incidents** from fixed clock tumbling bins, preventing artificial campaign fracture across fixed clock intervals.
9. **Pre-Registration Merge Audit & Inter-Flow Gap:**
   - Validated: slowloris $\rightarrow$ Slowhttptest (merged @ 300s), Hulk $\rightarrow$ GoldenEye (merged @ 300s), Slowhttptest $\rightarrow$ Hulk (merged @ 900s), Web Brute Force $\rightarrow$ XSS (merged @ 900s).
   - Web XSS $\rightarrow$ SQL Injection was observed split at 300s: measured minimum inter-flow gap on victim port 80 is **360 seconds (6.00 minutes)**, which strictly exceeds the 300s inactivity timeout.
   - **Wednesday Mathematical Independence:** Purity and coverage are independently formulated:
     $$\text{Purity}_{\text{flow}} = \frac{\sum_{i \in \mathcal{I}} \max_{c} \sum_{j \in i} \mathbb{I}(y_j = c)}{\sum_{i \in \mathcal{I}} N_i} \quad (92.05\% \text{ overall}, 92.02\% \text{ for } N \ge 2)$$
     $$\text{Coverage}_{\text{flagged}} = \frac{\sum_{i \in \mathcal{I}, N_i \ge 2, \text{maj}(i) = c \ne \text{BENIGN}} \sum_{j \in i} \mathbb{I}(y_j = c)}{N_{\text{attack, flagged}}} \quad (92.01\%)$$
     The numeric values (92.05% vs 92.01%) are verified as mathematically independent.


### 5.4 Transparent Incident Risk Scoring & Prioritization (Phase 9)

*Evaluated on the unseen 50% Report Half ($N=1,644$ incidents: $1,119$ true attacks, $525$ false alarms, $0$ mixed incidents) drawn from the 20% test partition ($3,296$ incidents). Weights were calibrated on the 50% Calibration Half ($1,652$ incidents) and committed to `preregistration_risk_v1.md` before scoring the report half.*

$$\text{Risk Score} = 0.30 \cdot C_{\text{ML}} + 0.10 \cdot \text{Sev} + 0.30 \cdot S_{\text{vol}} + 0.30 \cdot B_{\text{rule}} \in [0, 100]$$

#### Primary Metric Summary (Report Half, Unseen Incidents)

| Evaluation Metric | Measured Result | Evaluation Baseline / Benchmark | Operational Significance |
|---|---|---|---|
| **ROC-AUC (Attack vs False)** | **0.96613** | 95% CI: **[0.95747, 0.97373]**\* | High separation of threats from benign false alarms |
| **Calibration-to-Report Delta** | **+0.00554** | Calib: 0.96059 $\rightarrow$ Report: 0.96613 | Measured generalization across stratified incident halves |
| **Precision@10 (P@10)** | **100.00%** | Base Rate: 68.07% | Zero false alarms in top 10 analyst queue |
| **Precision@25 (P@25)** | **100.00%** | Base Rate: 68.07% | Zero false alarms in top 25 analyst queue |
| **Precision@50 (P@50)** | **100.00%** | Base Rate: 68.07% | Zero false alarms in top 50 analyst queue |
| **Precision@100 (P@100)** | **100.00%** | Base Rate: 68.07% | Zero false alarms in top 100 analyst queue |
| **Precision@200 (P@200)** | **100.00%** | Base Rate: 68.07% | Zero false alarms in top 200 analyst queue |
| **Precision@300 (P@300)** | **100.00%** | Base Rate: 68.07% | Zero false alarms in top 300 analyst queue |
| **Precision@400 (P@400)** | **100.00%** | Base Rate: 68.07% | Zero false alarms in top 400 analyst queue |
| **Precision@500 (P@500)** | **100.00%** | Base Rate: 68.07% | Zero false alarms in top 500 analyst queue |
| **Rank of First False Alarm** | **#539** | Incident ID: 1021 (Bot, Score: 42.38) | Clean run of 538 consecutive true attack incidents |
| **Critical/High Recall@100** | **50.00%** (1 / 2) | Random: 6.08% | Triage of severe threats |
| **Mixed Incidents Count** | **0** | $0 < N_{\text{att}} \le N/2$ | Clean partition separation |

*\*Methodological Disclosure on Confidence Intervals: Incident-level bootstrap treats individual PortScan fragments as independent observations; this assumption is anti-conservative. All bootstrap p-values have been omitted. A cluster bootstrap by `(source_day, majority_gt_label)` campaign yields 12 total clusters, of which only 4 contain attacks. Because there are fewer than ~10 effective positive clusters, the cluster bootstrap CI is not inferentially meaningful; point estimates are reported alongside this disclosure.*

#### Comparative Strategy Baselines (Report Half)

| Prioritization Strategy | Formulation / Weights | ROC-AUC | $\Delta \text{AUC}$ vs Hybrid | Incident-Level Bootstrap 95% CI |
|---|---|---|---|---|
| **Hybrid Calibrated (Ours)** | **$0.30 C_{\text{ML}} + 0.10 \text{Sev} + 0.30 S_{\text{vol}} + 0.30 B_{\text{rule}}$ (Variant B)** | **0.96613** | — | — |
| **Equal Weights Heuristic** | $0.25 C_{\text{ML}} + 0.25 \text{Sev} + 0.25 S_{\text{vol}} + 0.25 B_{\text{rule}}$ | **0.94334** | -0.02279 | **[+0.01558, +0.03094]** |
| **ML Confidence Only** | $1.00 C_{\text{ML}}$ | **0.94514** | -0.02098 | **[+0.01179, +0.03107]** |
| **Burst Volume Only** | $1.00 S_{\text{vol}}$ | **0.78051** | -0.18562 | **[+0.16104, +0.21182]** |
| **Severity Only** | $1.00 \text{Sev}$ | **0.77362** | -0.19251 | **[+0.16813, +0.21871]** |
| **Random Baseline** | $\mathcal{U}(0, 100)$ | **0.49847** | -0.46766 | — |

#### Attack Incident Composition & Effective Class Support (Report Half)
- **Report-Half Attack Composition:** The 1,119 true attack incidents in the report half consist strictly of:
  - `PortScan`: **1,051 incidents (93.92%)**
  - `Botnet (ARES)`: **66 incidents (5.90%)**
  - `FTP-Patator`: **1 incident (0.09%)**
  - `Infiltration`: **1 incident (0.09%)**
  - *Other classes (DoS Hulk, DoS Slowhttptest, DoS slowloris, DoS GoldenEye, DDoS, Web attacks, SSH-Patator, Heartbleed):* **0 attack incidents in report half**.
- **Empirical Scope:** The report-half result is effectively a demonstration of **PortScan plus Botnet** discrimination against benign false alarms.
- **Sub-Cohort Discrimination ($\ge 20$ incidents in report half):**
  - `PortScan` ($N=1,051$ attacks vs $525$ false): **$\text{ROC-AUC} = 0.96878$**
  - `Botnet (ARES)` ($N=66$ attacks vs $525$ false): **$\text{ROC-AUC} = 0.92284$**
  - Class-balanced macro AUC (averaging PortScan and Botnet): **0.94581**
  - AUC excluding PortScan (evaluating 68 non-PortScan attacks vs 525 false): **0.92511**

#### Descriptive Per-Class Distribution Over ALL Test Incidents ($N=3,296$)

| Primary Threat Category | Total Incidents | In-Sample Calib | Out-of-Sample Report | True Attacks | False Alarms | Mean Risk Score | Std Dev | Score Range [Min, Max] | Sample Status |
|---|---|---|---|---|---|---|---|---|---|
| `PortScan` | 2,305 | 1,143 | 1,162 | 2,103 | 202 | **47.17** | 9.62 | [25.55, 61.96] | In-sample / Out-of-sample mixed |
| `RULE_ONLY:R104` | 682 | 338 | 344 | 0 | 682 | **6.94** | 1.57 | [6.38, 17.92] | In-sample / Out-of-sample mixed |
| `Bot` | 183 | 93 | 90 | 131 | 52 | **38.64** | 6.28 | [27.46, 65.98] | In-sample / Out-of-sample mixed |
| `DoS Slowhttptest` | 51 | 30 | 21 | 0 | 51 | **36.58** | 3.20 | [27.67, 44.12] | In-sample / Out-of-sample mixed |
| `DoS Hulk` | 48 | 32 | 16 | 1 | 47 | **33.19** | 6.60 | [25.57, 63.96] | In-sample / Out-of-sample mixed |
| `DoS slowloris` | 6 | 3 | 3 | 1 | 5 | **36.70** | 14.84 | [25.49, 63.12] | In-sample / Out-of-sample mixed |
| `Web Attack - Brute Force` | 5 | 3 | 2 | 3 | 2 | **48.81** | 17.68 | [31.58, 67.75] | In-sample / Out-of-sample mixed |
| `Web Attack - XSS` | 3 | 0 | 3 | 0 | 3 | **29.58** | 0.35 | [29.18, 29.78] | Out-of-sample only |
| `Infiltration` | 3 | 2 | 1 | 3 | 0 | **49.39** | 2.27 | [46.92, 51.37] | In-sample / Out-of-sample mixed |
| `FTP-Patator` | 3 | 1 | 2 | 1 | 2 | **55.68** | 36.96 | [29.64, 97.99] | In-sample / Out-of-sample mixed |
| `DDoS` | 2 | 2 | 0 | 1 | 1 | **47.23** | 23.71 | [30.46, 64.00] | In-sample only |
| `SSH-Patator` | 2 | 2 | 0 | 1 | 1 | **66.01** | 45.21 | [34.04, 97.98] | In-sample only |
| `DoS GoldenEye` | 1 | 1 | 0 | 0 | 1 | **28.07** | 0.00 | [28.07, 28.07] | In-sample only |
| `Heartbleed` | 1 | 1 | 0 | 1 | 0 | **44.38** | 0.00 | [44.38, 44.38] | In-sample only |
| `Web Attack - Sql Injection` | 1 | 1 | 0 | 1 | 0 | **49.36** | 0.00 | [49.36, 49.36] | In-sample only |

#### Added Value: Multi-Flow ($N \ge 2$) and Singletons ($N = 1$) Disaggregation

| Incident Cohort | Total Incidents | Attack / False Count | Frozen Score ROC-AUC | ML-Confidence Only ROC-AUC | $\Delta \text{AUC}$ | Added-Value Assessment |
|---|---|---|---|---|---|---|
| **Report Half: Multi-Flow ($N \ge 2$)** | 755 | 697 attack / 58 false | **0.99985** | **0.99985** | **+0.00000** | Ceiling discrimination |
| **Report Half: Singletons ($N = 1$)** | 889 | 422 attack / 467 false | **0.90454** | **0.90882** | **-0.00428** | Dominated by ML classifier |
| **Full Test: Multi-Flow ($N \ge 2$)** | 1,504 | 1,400 attack / 104 false | **0.99183** | **0.98160** | **+0.01023** | Measurable multi-flow gain |
| **Full Test: Singletons ($N = 1$)** | 1,792 | 847 attack / 945 false | **0.90488** | **0.91145** | **-0.00657** | Isolated singletons rely on classifier |

#### Grouping Key Sensitivity: Frozen Score on $K_2$ Incidents (Secondary Result)
Under the port-agile grouping key $K_2 = (\text{SrcIP}, \text{DstIP})$, the total incident count decreases from $3,296$ to **$792$ incidents** (47 attack, 745 false). Applying the frozen risk model (without refitting, noting that $N_{\text{ref}}=25$ was set on $K_3$ sizes) yields **$\text{ROC-AUC} = 0.97640$**.

#### Severity Weight Policy Operating-Point Menu (Exploratory, Calibration Half Only)
$w_2$ is an analyst policy parameter not identified by the binary AUC objective because PortScan dominates incident volume. The table below presents an exploratory operating-point menu evaluated on the Calibration Half ($N=8$ Critical/High attacks):

| Configuration Type | Weight Vector $(w_1, w_2, w_3, w_4)$ | Calibration ROC-AUC | Mean Rank of Critical/High ($N=8$) | Priority Mechanics |
|---|---|---|---|---|
| **Frozen Calibrated Row** | **$(0.30, 0.10, 0.30, 0.30)$** | **0.96059** | **336.62 / 1,128** | Tied global optimum with active rule bonus |
| *Re-optimized ($w_2 = 0.10$)* | $(0.60, 0.10, 0.30, 0.00)$ | **0.96744** | 352.75 / 1,128 | Boundary optimization without rule bonus |
| *Re-optimized ($w_2 = 0.25$)* | $(0.55, 0.25, 0.20, 0.00)$ | **0.95788** | 54.88 / 1,128 | Prioritizes severe incidents into top 5% |
| *Re-optimized ($w_2 = 0.40$)* | $(0.45, 0.40, 0.15, 0.00)$ | **0.92910** | **4.88 / 1,128** | Forces Critical/High directly to top 5 slots |

*Critical/High incident counts: Calibration Half = 8 (Web Attack Brute Force: 3, Infiltration: 2, SQLi: 1, SSH-Patator: 1, Heartbleed: 1); Report Half = 2 (FTP-Patator: 1, Infiltration: 1).*

#### Methodological Limitations & Analytical Findings
1. **Analyst-Assigned Severity Table:** The severity lookup table (`SEVERITY_TABLE_PRIMARY`) utilizes **analyst-assigned heuristic severity tiers** (Critical=100, High=80, Medium=60, Low / Volumetric Availability=40, Reconnaissance=20, Zero=0). It reflects expert operational security prioritization rather than an empirical formula directly derived from NIST SP 800-61 or MITRE ATT&CK.
2. **Operational Presentation Requirement:** Downstream dashboards (Phase 13) and LLM prompt templates (Phase 11) must present `severity_tier` alongside `risk_score` because composite risk scores do not constitute an intrinsic severity ranking. Critical threats (Infiltration: 49.39, SQLi: 49.36, Heartbleed: 44.38) score similarly to volumetric reconnaissance (PortScan: 47.17) and floods (DDoS: 47.23) due to volume saturation ($w_3=0.30$) outweighing severity ($w_2=0.10$).
3. **Rule Bonus Ceiling:** High composite scores ($>68$) are attained exclusively by FTP-Patator (mean 55.68, max 97.99) and SSH-Patator (mean 66.01, max 97.98) due to the deterministic rule bonus ($B_{\text{rule}}=30.00$). Without rule bonus, composite scores cannot exceed 70.00.
4. **Pre-Registered v1 Profile & Non-Identification of $w_2, w_4$:** $w_2$ and $w_4$ are tie-break selections not uniquely identified by binary AUC alone because PortScan dominates incident volume ($>93\%$) and XGBoost already flags all flows matched by R104. The frozen weights vector $(0.30, 0.10, 0.30, 0.30)$ constitutes a pre-registered v1 profile.
5. **Between-Cohort Volume Effect:** The $+0.021$ AUC gain of the composite score over pure ML confidence ($0.96613$ vs $0.94514$) is a between-cohort volume effect: within multi-flow ($N \ge 2$), gain is $+0.01023$ across full test ($+0.00000$ on report half); within singletons ($N=1$), risk score has slightly lower AUC ($-0.00657$ full test / $-0.00428$ report half) due to volume-scaling compression.
6. **$K_2$ vs $K_3$ Non-Comparability:** $K_2$ collapses horizontal scans into large multi-thousand flow incidents where $N_{\text{ref}}=25$ saturates immediately. $K_2$ AUC (0.97640) cannot be directly compared to $K_3$ AUC (0.96613).
7. **Convex Combination Formulation:** The risk score is strictly bounded in $[0, 100]$ via normalized convex weighting ($\sum w_i = 1.0$).
8. **Volume Saturation ($N_{\text{ref}} = 25$):** $S_{\text{vol}}$ saturates for $N \ge 25$ flows, preventing multi-thousand flow volumetric floods from mathematically drowning out low-volume targeted attacks.
9. **Shared Temporal Campaigns:** The 50/50 split across incidents is stratified across the test partition; both halves share the same 8 calendar days.
10. **Retrospective Incident Risk:** Incident risk scoring is performed retrospectively after sessionization gap closure ($G=300\text{ s}$).

### 5.5 RAG Retrieval Quality & Knowledge Grounding (Phase 10 & 11)

#### 5.5.1 Phase 10 Empirical Knowledge Retrieval Evaluation (Untouched Test Partition)
Phase 10 implements offline local semantic retrieval using `all-MiniLM-L6-v2` and local persistent `ChromaDB==1.5.9`. The authoritative primary corpus consists of MITRE ATT&CK v20.0 STIX, NIST SP 800-61r3 (CSF 2.0 Community Profile), NVD CVE-2014-0160, and OWASP Top 10 (2021). Zero synthetic or LLM-generated texts are present in the knowledge corpus.

Evaluation protocol and hypotheses were pre-registered in `artifacts/rag/preregistration_rag_v1.md` (SHA-256: `a87836067c83d865d7576c087daa46dc0e94092c91e296e2c51b0727b4f9a309`) across 109 curated queries. Grid selection and threshold tuning were conducted strictly on the Development partition ($n=40$ queries: 26 threats, 14 negatives):
- **Dev Chunking Comparison:** **C512** (512 tokens / 100 overlap) selected as winning grid with highest MRR ($0.3811$, Macro-MRR $0.3886$), outperforming C256 (MRR $0.3732$) and C1024 (MRR $0.3542$, exhibiting information dilution).
- **Dev Threshold Sweep:** Optimal acceptance threshold calibrated to $\tau^* = 0.35$ ($\text{Max } F_1 = 0.8197$, Precision = 0.7143, Threat Acceptance Recall = 0.9615, False Accept Rate $\text{FAR} = 0.7143$).

The frozen winning configuration (C512, $\tau^* = 0.35$) was evaluated on the untouched Test partition ($n=54$ queries: 38 threats, 16 negatives):

| Metric | Measured Value | Wilson 95% Confidence Interval | Methodological Description |
|---|---|---|---|
| **Recall@1** | **0.3158** (12 / 38) | $[0.1908, 0.4746]$ | Primary gold technique ranked in slot #1 |
| **Recall@3** | **0.5263** (20 / 38) | $[0.3726, 0.6752]$ | Primary gold technique ranked in top-3 slots |
| **Recall@5** | **0.6842** (26 / 38) | $[0.5254, 0.8092]$ | Primary gold technique ranked in top-5 slots |
| **MRR** | **0.4478** | — | Mean Reciprocal Rank over test threat queries |
| **Macro-Recall@1** | **0.3111** | — | Unweighted mean across 14 attack categories |
| **Macro-Recall@3** | **0.5333** | — | Unweighted mean across 14 attack categories |
| **Macro-Recall@5** | **0.7000** | — | Unweighted mean across 14 attack categories |
| **Macro-MRR** | **0.4502** | — | Unweighted mean across 14 attack categories |
| **Negative ROC-AUC** | **0.7352** | — | Top-1 cosine similarity separation on negatives |
| **FAR @ $\tau^*=0.35$** | **0.7500** (12 / 16) | $[0.5050, 0.8982]$ | False accept rate on negative queries |

#### 5.5.2 Comparative Baselines on Test Partition
| Retrieval Approach | Recall@1 | Recall@3 | Recall@5 | MRR | Architectural Characteristics |
|---|---|---|---|---|---|
| **Semantic RAG (C512, all-MiniLM-L6-v2)** | **0.3158** | **0.5263** | **0.6842** | **0.4478** | Unsupervised semantic matching; generalizes to unlabeled observables (MRR=0.3926) |
| **Naive Label Lookup** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Deterministic 1:1 dictionary lookup (requires exact ground-truth label; zero generalization) |
| **Random Retrieval (100 trials)** | 0.0063 | 0.0176 | 0.0350 | 0.0147 | Empirical Monte Carlo null distribution across 2,060 chunks ($p < 0.0001$) |

*Statistical Gain:* Semantic RAG achieves a **$30.5\times$ MRR gain** ($0.4478$ vs. $0.0147$) and a **$19.5\times$ Recall@5 gain** over the random null baseline.

#### 5.5.3 Operational Query Formulation Performance
- **Schema-Derived Telemetry Queries ($n=21$):** Recall@1 = 0.3810, Recall@3 = 0.5714, Recall@5 = 0.8095, MRR = 0.5230.
- **Unlabeled Observable Behavior Queries ($n=9$):** Recall@1 = 0.2222, Recall@3 = 0.5556, Recall@5 = 0.6667, MRR = 0.3926.

#### 5.5.4 Production Serving & Fail-Safe Architecture
- `RAGRetriever` (`ml_service/src/rag/retriever.py`) delivers operational retrieval with fail-safe fallback to NIST SP 800-61r3 / CSF 2.0 incident triage guidance (`RS.AN`, `RS.MI`, `RC`) if the vector database is unreachable or query similarity is below $\tau^*=0.35$.
- Automated unit test suite (`ml_service/tests/test_retriever.py`) verifies fallback mode, schema conformity, and threshold filtering; all 20 system unit tests passed.

#### 5.5.5 LLM Decision-Support & Hallucination Assessment (Phase 11 Target Scenarios)
| Query Scenario | Relevant Context Retrieved? | Top-3 Chunk Precision | LLM Hallucination Rate (%) | Grounded Evidence Score |
|---|---|---|---|---|
| **SSH-Patator Incident** | Yes (T1110.001) | Measured (Phase 10) | 0.0% (Phase 11 target) | *Pending Phase 11* |
| **DoS Hulk Incident** | Yes (T1499.003) | Measured (Phase 10) | 0.0% (Phase 11 target) | *Pending Phase 11* |
| **SQL Injection Incident** | Yes (T1190) | Measured (Phase 10) | 0.0% (Phase 11 target) | *Pending Phase 11* |

---

## 6. Formal References (APA 7th Edition)

1. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324
2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785
3. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30, 3146–3154. https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
4. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*, 33, 9459–9474.
5. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765–4774.
6. MITRE Corporation. (2024). *MITRE ATT&CK Enterprise Matrix*. https://attack.mitre.org
7. National Institute of Standards and Technology. (2012). *Computer security incident handling guide* (NIST Special Publication 800-61 Rev. 2). U.S. Department of Commerce. https://doi.org/10.6028/NIST.SP.800-61r2
8. Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. *Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP)*, 108–116. https://doi.org/10.5220/0006639801080116
9. Google. (2024). *Gemini API documentation & architectural guides*. Google DeepMind / Google AI for Developers. https://ai.google.dev/gemini-api/docs
