# Phase 10: RAG Knowledge Base Pre-Registration Specification (v1.0)

**Document Status:** Formal Pre-Registration (Locked prior to Stage B Chunking, Embedding, or Indexing)  
**Creation Date:** 2026-11-02T14:23:13.084776+00:00  
**Governing Standard:** AGENTS.md Phase 10 Pre-Registration Mandate  
**Corpus Grounding:** MITRE ATT&CK v20.0, NIST SP 800-61r3, NVD API 2.0 (CVE-2014-0160), OWASP Top 10 (2021)

---

## 1. Methodological Foundations & Assumptions

1. **Secondary Decision-Support Mandate:** The RAG retrieval pipeline functions strictly as a decision-support and evidence-grounding layer. Retrieval queries are formulated exclusively from upstream operational telemetry (incident schema properties, extracted flow observables, and plurality predicted threat classes). Zero ground-truth test labels or test-set supervisory telemetry are ever ingested by the retrieval engine.
2. **Gold Label Criteria:** For every query q, the gold relevant corpus targets are pre-registered at the (technique_id, section_type) level:
   - A retrieved chunk is scored as **Relevant** if its primary metadata matches any pre-registered gold ATT&CK technique ID (e.g. T1595, T1110.001, T1190, T1071.001, T1498.001, T1499.003), specific CVE identifier (CVE-2014-0160), or OWASP risk category (OWASP-2021-A03, OWASP-2021-A07).
   - For negative queries, the gold target set is strictly empty. Any retrieval score exceeding similarity threshold tau is counted as a False Positive / False Accept.
3. **Strict Zero-LLM Corpus Rule:** The knowledge corpus is formed exclusively of authoritative, immutable text snapshots directly downloaded from primary standards bodies. Zero synthetic or LLM-generated documents exist in the corpus.

---

## 2. Query Suite Structure & Stratification

The pre-registered query suite contains **109 total queries** partitioned across query paradigms:

| Category | Description | Query Count | Purpose |
|---|---|---|---|
| **Schema Queries** | Natural-language translation of operational incident schema fields (port, flow count, duration, plurality label) | 28 | Tests retrieval when standard operational telemetry is populated |
| **Free-Text Observables** | Flow observables written strictly from Part A empirical telemetry without naming the attack class | 28 | Tests semantic retrieval when analyst describes raw network anomalies |
| **Wrong-Label / Low-Confidence** | Telemetry describes attack mechanics while classifier predicted benign or incorrect label | 14 | Evaluates whether RAG can surface correct grounding despite upstream ML confusion |
| **R104-Only Queries** | Incidents flagged strictly by rule R104 without ML corroboration | 4 | Tests guidance marking candidate alerts as unconfirmed/spurious |
| **Routine IT Negatives** | Benign administrative tasks, helpdesk inquiries, standard IT procedures | 25 | Evaluates rejection of non-attack queries |
| **Hard Negatives** | Benign queries containing overlapping technical keywords (SSH, SQL, backup on port 445, authorized scan) | 10 | Evaluates discriminative power against keyword matching traps |
| **Total Query Suite** | Full set of structured pre-registered queries | **109** | Full benchmark coverage |

### Partitioning & Split Protocol
- **Randomization Seed:** Fixed at `seed = 42`.
- **Partitioning Strategy:**
  - **Dev Partition (40 queries):** Used exclusively for tuning the similarity threshold tau and exploring chunking parameters.
  - **Test Partition (54 queries):** Held completely untouched until Stage B final evaluation.
  - **Blind Reserved Partition (15 queries):** Reserved for blind human analyst annotation during Phase 14 testing.

---

## 3. Pre-Registered Evaluation Metrics

### 3.1 Ranking & Retrieval Metrics (Evaluated on Threat Queries)
For each query q with pre-registered gold technique set G_q:

1. **Recall@k (k in {1, 3, 5}):**
   Proportion of queries where at least one chunk among the top k retrieved matches G_q.

2. **Mean Reciprocal Rank (MRR):**
   Average of reciprocal ranks (1 / rank) of the first relevant chunk matching G_q across all queries.

3. **Wilson Score 95% Confidence Intervals:**
   Calculated with z = 1.96 for all recall proportions to guarantee exact statistical reporting.

4. **Macro-Averaging:**
   All recall and ranking metrics will be reported both as micro-averages across all queries and macro-averaged across the 14 distinct attack classes to prevent dominant classes from biasing aggregate performance.

### 3.2 Negative Rejection & Calibration Metrics (Evaluated on Negative Queries)
1. **ROC-AUC on Top-1 Cosine Similarity Score:**
   Evaluates how cleanly the top-1 retrieval similarity score separates threat queries from negative queries across all possible decision thresholds.
2. **False Accept Rate (FAR) at Threshold tau:**
   Proportion of negative queries where top-1 similarity exceeds tau.

---

## 4. Frozen Threshold Decision Rule

- The minimum similarity acceptance threshold tau will be optimized strictly on the **Dev partition** to maximize F1(tau).
- Once determined on Dev, tau is **permanently frozen** and applied to the Test partition. Queries with top-1 similarity < tau trigger the `GENERIC_FALLBACK` triage protocol.

---

## 5. Comparative Baselines (Upfront Commitment)

The semantic RAG retrieval engine (`all-MiniLM-L6-v2` + cosine distance in ChromaDB) will be compared directly against:
1. **Naive Label-to-Technique Metadata Lookup:** Deterministic dictionary lookup mapping the plurality threat label directly to its primary ATT&CK ID without performing semantic vector search.
2. **Random Retrieval Baseline:** Uniformly random sampling of k chunks from the vector database.

---

## 6. Stage B Chunking Grid Specification

In Stage B, documents will be segmented and evaluated across the following pre-registered parameter grid:

| Grid ID | Chunk Size (tokens) | Chunk Overlap (tokens) | Primary Rationale |
|---|---|---|---|
| **C256** | 256 | 50 | High granularity; targets individual mitigation statements and detection rules |
| **C512 (Baseline)** | 512 | 100 | Balanced context window; fits full ATT&CK technique descriptions and NIST sections |
| **C1024** | 1024 | 200 | Broad context; captures multiple mitigation relationships and full vulnerability analyses |

*Note: The chunking grid is pre-registered here. No chunking or embedding execution will occur until Stage B.*

---

## 7. Blind Human Annotation Protocol

15 queries (15 queries) are segregated into the `blind_reserved` split. During evaluation, these queries will be presented in randomized, unlabelled order for human analyst validation to evaluate inter-annotator agreement against the pre-registered gold labels.
