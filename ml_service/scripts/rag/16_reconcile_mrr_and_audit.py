"""
16_reconcile_mrr_and_audit.py
──────────────────────────────
Reconciles C512 DEV MRR (0.3811 vs 0.3635) and performs comprehensive Oracle Audit
on DEV and TEST partitions with zero ground-truth label contamination.
"""
import json
from pathlib import Path
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

_ROOT = Path(__file__).resolve().parents[3]
rag_dir = _ROOT / "artifacts" / "rag"
chroma_path = rag_dir / "chroma_db"
model_dir = _ROOT / "models" / "embeddings" / "all-MiniLM-L6-v2"

def main():
    queries_file = rag_dir / "rag_evaluation_queries_v1.json"
    with open(queries_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    all_queries = data["queries"]
    dev_threats = [q for q in all_queries if q["split"] == "dev" and q["label"] != "BENIGN"]
    test_threats = [q for q in all_queries if q["split"] == "test" and q["label"] != "BENIGN"]
    
    client = chromadb.PersistentClient(path=str(chroma_path), settings=Settings(anonymized_telemetry=False))
    coll = client.get_collection("rag_knowledge_c512")
    model = SentenceTransformer(str(model_dir), device="cpu")
    model.max_seq_length = 256
    
    # ---------------------------------------------------------
    # PART 1: RECONCILE C512 DEV MRR (0.3811 vs 0.3635)
    # ---------------------------------------------------------
    print("=== PART 1: RECONCILING C512 DEV MRR ===")
    rr_top5 = []
    rr_top10 = []
    discrepant_queries = []
    
    for q in dev_threats:
        gold_set = set(q["gold"])
        v = model.encode([q["text"]], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
        res10 = coll.query(query_embeddings=[v], n_results=10)
        metas10 = res10["metadatas"][0]
        
        rank10 = None
        for idx, m in enumerate(metas10):
            tid = m["technique_id"]
            if (tid in gold_set) or any(tid.startswith(g) or g.startswith(tid) for g in gold_set):
                rank10 = idx + 1
                break
                
        rank5 = rank10 if (rank10 is not None and rank10 <= 5) else None
        
        rr10 = (1.0 / rank10) if rank10 else 0.0
        rr5 = (1.0 / rank5) if rank5 else 0.0
        
        rr_top10.append(rr10)
        rr_top5.append(rr5)
        
        if rank10 != rank5:
            discrepant_queries.append({
                "id": q["id"],
                "type": q["type"],
                "text": q["text"],
                "gold": q["gold"],
                "rank_top10": rank10,
                "rank_top5": rank5,
                "rr_top10": round(rr10, 4),
                "rr_top5": round(rr5, 4),
                "diff": round(rr10 - rr5, 4)
            })
            
    mrr_top10 = float(np.mean(rr_top10))
    mrr_top5 = float(np.mean(rr_top5))
    print(f"Mean MRR with n_results=10 (Stage B / 07_tune_and_evaluate.py): {mrr_top10:.6f} -> rounded: {mrr_top10:.4f}")
    print(f"Mean MRR with n_results=5  (Closure Pass 2 / 14_eval_token.py):    {mrr_top5:.6f} -> rounded: {mrr_top5:.4f}")
    print(f"Delta: {mrr_top10 - mrr_top5:.6f}")
    print(f"Discrepant queries (ranks 6-10): {len(discrepant_queries)}")
    for dq in discrepant_queries:
        print(f"  {dq['id']} ({dq['type']}): rank in top-10={dq['rank_top10']}, rank in top-5={dq['rank_top5']} | text: '{dq['text'][:60]}...'")
        
    # ---------------------------------------------------------
    # PART 2: ORACLE AUDIT ACROSS DEV AND TEST THREAT QUERIES
    # ---------------------------------------------------------
    print("\n=== PART 2: ORACLE AUDIT ===")
    # Load label maps
    with open(rag_dir / "label_technique_map_v1.json", "r", encoding="utf-8") as f:
        lmap_v1 = json.load(f)["mappings"]
    with open(rag_dir / "label_technique_map_v2.json", "r", encoding="utf-8") as f:
        lmap_v2 = json.load(f)["mappings"]

    # Check query JSON schema:
    # Does any query have a separate 'predicted_label' field?
    has_pred_field = any("predicted_label" in q for q in all_queries)
    print(f"Does any query in rag_evaluation_queries_v1.json have 'predicted_label' field? {has_pred_field}")
    
    # In rag_evaluation_queries_v1.json, the fields are:
    # id, label, type, text, gold, split
    # For every threat query, q['label'] is the TRUE attack category!
    # In wrong_label queries, the text describes the predicted label, but q['label'] holds the true class.
    # In observable queries, the text describes raw telemetry, but q['label'] holds the true class.
    # Therefore, ANY script that used q['label'] for retrieval or lookup on wrong_label or observable queries
    # was accessing an ORACLE ground-truth label!
    
    # Define exact predicted labels from query text for wrong_label queries:
    # Dev wrong_label (n=6):
    dev_wrong_pred = {
        "Q_BOT_WRG1": "BENIGN",
        "Q_HULK_WRG1": "BENIGN",
        "Q_SLOW_WRG1": "BENIGN",
        "Q_SHTT_WRG1": "BENIGN",
        "Q_DDOS_WRG1": "PortScan",
        "Q_WXSS_WRG1": "BENIGN"
    }
    # Test wrong_label (n=6):
    test_wrong_pred = {
        "Q_PORTSCAN_WRG1": "BENIGN",
        "Q_INFIL_WRG1": "BENIGN",
        "Q_HBLD_WRG1": "BENIGN",
        "Q_FTP_WRG1": "BENIGN",
        "Q_SSH_WRG1": "BENIGN",
        "Q_WSQL_WRG1": "BENIGN"
    }
    
    # Define Evaluation Functions without ANY Oracle Labels:
    # 1. Stated-Label Queries (schema, wrong_label, r104_only):
    #    - schema: predicted label is explicit in query header
    #    - wrong_label: predicted label is the error stated in query text
    #    - r104_only: predicted label is "RULE_ONLY:R104"
    #    Policy on RULE_ONLY:R104: Falls back to generic NIST incident response block (is_generic=True, 0 ATT&CK technique hits).
    #    Policy on unmapped / BENIGN: Falls back to generic NIST block.
    #    Evaluated on Mapped Path (coll.get metadata) and Label Lookup.
    #
    # 2. Observable Queries (observable):
    #    - Zero label stated in prompt.
    #    - Mapped path cannot run without a label (or returns generic NIST block).
    #    - Evaluated in Unfiltered Semantic Search over query text!
    
    def evaluate_clean_pipeline(split_name: str, threat_list: list[dict], wrong_dict: dict):
        schema_queries = [q for q in threat_list if q["type"] == "schema"]
        wrong_queries = [q for q in threat_list if q["type"] == "wrong_label"]
        r104_queries = [q for q in threat_list if q["type"] == "r104_only"]
        obs_queries = [q for q in threat_list if q["type"] == "observable"]
        
        stated_queries = schema_queries + wrong_queries + r104_queries
        
        # A. Mapped Metadata Lookup on Stated-Label Queries
        mapped_r1 = []
        mapped_r5 = []
        mapped_rr = []
        
        for q in stated_queries:
            gold_set = set(q["gold"])
            q_type = q["type"]
            if q_type == "schema":
                pred_lbl = q["label"]
            elif q_type == "wrong_label":
                pred_lbl = wrong_dict[q["id"]]
            elif q_type == "r104_only":
                pred_lbl = "RULE_ONLY:R104"
                
            # Operational Policy:
            if pred_lbl in lmap_v1 and pred_lbl != "RULE_ONLY:R104" and pred_lbl != "BENIGN":
                candidate_tids = [lmap_v1[pred_lbl]["primary_technique_id"]] + lmap_v1[pred_lbl].get("candidate_technique_ids", [])
                first_rank = None
                for idx, tid in enumerate(candidate_tids):
                    if (tid in gold_set) or any(tid.startswith(g) or g.startswith(tid) for g in gold_set):
                        first_rank = idx + 1
                        break
                r1 = (first_rank == 1)
                r5 = (first_rank is not None and first_rank <= 5)
                rr = 1.0 / first_rank if first_rank else 0.0
            else:
                # RULE_ONLY:R104 or BENIGN -> Generic NIST block (no ATT&CK match)
                first_rank = None
                r1 = False
                r5 = False
                rr = 0.0
                
            mapped_r1.append(r1)
            mapped_r5.append(r5)
            mapped_rr.append(rr)
            
        # B. Unfiltered Semantic Search on Observable Queries
        obs_r1 = []
        obs_r5 = []
        obs_rr = []
        
        for q in obs_queries:
            gold_set = set(q["gold"])
            v = model.encode([q["text"]], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
            res = coll.query(query_embeddings=[v], n_results=5)
            first_rank = None
            for idx, m in enumerate(res["metadatas"][0]):
                tid = m["technique_id"]
                if (tid in gold_set) or any(tid.startswith(g) or g.startswith(tid) for g in gold_set):
                    first_rank = idx + 1
                    break
            r1 = (first_rank == 1)
            r5 = (first_rank is not None and first_rank <= 5)
            rr = 1.0 / first_rank if first_rank else 0.0
            obs_r1.append(r1)
            obs_r5.append(r5)
            obs_rr.append(rr)
            
        # C. End-to-End Operational Pipeline across ALL Threat Queries:
        #    - Stated-label queries: Mapped path
        #    - Observable queries: Unfiltered semantic search
        all_r1 = mapped_r1 + obs_r1
        all_r5 = mapped_r5 + obs_r5
        all_rr = mapped_rr + obs_rr
        
        # D. Realistic Label Lookup Baseline on ALL Threat Queries:
        #    - Stated-label queries: lookup on stated predicted label
        #    - Observable queries: unmapped (0.0) because no label is stated!
        lookup_r1 = list(mapped_r1) + [False] * len(obs_queries)
        lookup_r5 = list(mapped_r5) + [False] * len(obs_queries)
        lookup_rr = list(mapped_rr) + [0.0] * len(obs_queries)
        
        return {
            "split": split_name,
            "counts": {
                "schema": len(schema_queries),
                "wrong_label": len(wrong_queries),
                "r104_only": len(r104_queries),
                "observable": len(obs_queries),
                "total_threats": len(threat_list)
            },
            "mapped_on_stated_queries": {
                "n": len(stated_queries),
                "recall_at_1": round(float(np.mean(mapped_r1)), 4),
                "hits_r1": sum(mapped_r1),
                "recall_at_5": round(float(np.mean(mapped_r5)), 4),
                "hits_r5": sum(mapped_r5),
                "mrr": round(float(np.mean(mapped_rr)), 4)
            },
            "semantic_on_observable_queries": {
                "n": len(obs_queries),
                "recall_at_1": round(float(np.mean(obs_r1)), 4),
                "hits_r1": sum(obs_r1),
                "recall_at_5": round(float(np.mean(obs_r5)), 4),
                "hits_r5": sum(obs_r5),
                "mrr": round(float(np.mean(obs_rr)), 4)
            },
            "end_to_end_pipeline_all_threats": {
                "n": len(threat_list),
                "recall_at_1": round(float(np.mean(all_r1)), 4),
                "hits_r1": sum(all_r1),
                "recall_at_5": round(float(np.mean(all_r5)), 4),
                "hits_r5": sum(all_r5),
                "mrr": round(float(np.mean(all_rr)), 4)
            },
            "realistic_label_lookup_baseline": {
                "n": len(threat_list),
                "recall_at_1": round(float(np.mean(lookup_r1)), 4),
                "hits_r1": sum(lookup_r1),
                "recall_at_5": round(float(np.mean(lookup_r5)), 4),
                "hits_r5": sum(lookup_r5),
                "mrr": round(float(np.mean(lookup_rr)), 4)
            }
        }
        
    dev_clean = evaluate_clean_pipeline("dev", dev_threats, dev_wrong_pred)
    test_clean = evaluate_clean_pipeline("test", test_threats, test_wrong_pred)
    
    print("\n--- DEV CLEAN EVALUATION (EXPLORATORY, n=26) ---")
    print(json.dumps(dev_clean, indent=2))
    
    print("\n--- TEST CLEAN EVALUATION (DESCRIPTIVE BENCHMARK, n=38) ---")
    print(json.dumps(test_clean, indent=2))
    
    out_file = rag_dir / "clean_reconciled_evaluation_v1.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "part1_mrr_reconciliation": {
                "dev_mrr_top10": round(mrr_top10, 4),
                "dev_mrr_top5": round(mrr_top5, 4),
                "delta": round(mrr_top10 - mrr_top5, 4),
                "explanation": "07_tune_and_evaluate.py queried top_k=10 chunks, awarding partial reciprocal rank to 2 queries whose first hit was at rank 6 (1/6 = 0.1667) and rank 8 (1/8 = 0.1250); 14_eval_token_capped_grids.py queried n_results=5, treating rank > 5 as 0.0. Recall@1 and Recall@5 are mathematically identical in both runs.",
                "discrepant_queries": discrepant_queries
            },
            "part2_oracle_audit_summary": {
                "contaminated_metrics": {
                    "mapped_dev_pass1_mrr": 0.9808,
                    "mode_a_dev_pass2_mrr": 0.7308,
                    "mode_b_dev_pass2_mrr": 0.7885,
                    "realistic_label_lookup_test_mrr": 0.8421,
                    "cause": "q['label'] in rag_evaluation_queries_v1.json held the ground-truth attack class for observable and wrong_label queries. Using q['label'] acted as an oracle."
                },
                "dev_clean": dev_clean,
                "test_clean": test_clean
            }
        }, f, indent=2)
    print(f"\nSaved reconciled evaluation to {out_file.name}")

if __name__ == "__main__":
    main()
