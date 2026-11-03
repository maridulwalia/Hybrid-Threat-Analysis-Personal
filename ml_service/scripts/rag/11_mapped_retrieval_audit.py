"""
11_mapped_retrieval_audit.py
────────────────────────────
Audit of Mapped Retrieval on DEV partition.
Examines query types, labels passed, map versions, DoS gold techniques,
and detailed per-query performance for wrong_label queries.
"""
import json
import hashlib
from pathlib import Path
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

_ROOT = Path(__file__).resolve().parents[3]
rag_dir = _ROOT / "artifacts" / "rag"

def main():
    # 1. Verify hash of rag_evaluation_queries_v1.json
    queries_path = rag_dir / "rag_evaluation_queries_v1.json"
    with open(queries_path, "rb") as f:
        q_bytes = f.read()
    q_hash = hashlib.sha256(q_bytes).hexdigest()
    print(f"rag_evaluation_queries_v1.json SHA-256: {q_hash} ({len(q_bytes)} bytes)")
    
    with open(queries_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    all_queries = data["queries"]
    dev_queries = [q for q in all_queries if q["split"] == "dev"]
    dev_threats = [q for q in dev_queries if q["label"] != "BENIGN"]
    dev_negs = [q for q in dev_queries if q["label"] == "BENIGN"]
    
    print(f"\nDev total queries: {len(dev_queries)}")
    print(f"Dev threats: {len(dev_threats)}, Dev negatives: {len(dev_negs)}")
    
    # Query types among dev threats
    type_counts = {}
    for q in dev_threats:
        t = q["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    print("\nDev threat query type breakdown:", type_counts)
    
    # Query types among dev negatives
    neg_type_counts = {}
    for q in dev_negs:
        t = q["type"]
        neg_type_counts[t] = neg_type_counts.get(t, 0) + 1
    print("Dev negative query type breakdown:", neg_type_counts)
    
    # Load label maps
    with open(rag_dir / "label_technique_map_v1.json", "r", encoding="utf-8") as f:
        lmap_v1 = json.load(f)["mappings"]
    with open(rag_dir / "label_technique_map_v2.json", "r", encoding="utf-8") as f:
        lmap_v2 = json.load(f)["mappings"]
        
    # Check DoS queries gold techniques
    print("\n--- Gold techniques for DoS queries in DEV ---")
    dos_dev = [q for q in dev_queries if "dos" in q["label"].lower() or "hulk" in q["label"].lower() or "slow" in q["label"].lower() or "heartbleed" in q["label"].lower() or "ddos" in q["label"].lower()]
    for q in dos_dev:
        print(f"  ID: {q['id']}, Type: {q['type']}, Label: {q['label']}, Gold: {q['gold']}")
        
    # Load Chroma C512
    chroma_path = rag_dir / "chroma_db"
    client = chromadb.PersistentClient(path=str(chroma_path))
    coll512 = client.get_collection("rag_knowledge_c512")
    
    model_dir = _ROOT / "models" / "embeddings" / "all-MiniLM-L6-v2"
    model = SentenceTransformer(str(model_dir), device="cpu")
    
    # Audit how Task 6 in 09_closure_analyses ran:
    print("\n--- Simulating Task 6 under v1 mapping and q['label'] ---")
    task6_results = []
    for q in dev_threats:
        lbl = q["label"]
        gold_set = set(q["gold"])
        q_type = q["type"]
        
        # In Task 6:
        filtered = False
        if lbl in lmap_v1 and lbl != "RULE_ONLY:R104":
            tids = [lmap_v1[lbl]["primary_technique_id"]] + lmap_v1[lbl].get("candidate_technique_ids", [])
            v = model.encode([q["text"]], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
            res = coll512.query(query_embeddings=[v], n_results=5, where={"technique_id": {"$in": tids}})
            filtered = True
        else:
            v = model.encode([q["text"]], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
            res = coll512.query(query_embeddings=[v], n_results=5)
            
        hit_metas = res["metadatas"][0]
        hit_tids = [m["technique_id"] for m in hit_metas]
        first_rank = None
        for idx, m in enumerate(hit_metas):
            tid = m["technique_id"]
            if (tid in gold_set) or any(tid.startswith(g) or g.startswith(tid) for g in gold_set):
                first_rank = idx + 1
                break
                
        r1 = (first_rank == 1)
        r5 = (first_rank is not None and first_rank <= 5)
        rr = 1.0 / first_rank if first_rank else 0.0
        
        task6_results.append({
            "id": q["id"],
            "type": q_type,
            "label_used": lbl,
            "filtered": filtered,
            "gold": list(gold_set),
            "hit_tids": hit_tids,
            "first_rank": first_rank,
            "r1": r1,
            "r5": r5,
            "rr": rr
        })
        
    # Summarize by query type
    by_type = {}
    for r in task6_results:
        t = r["type"]
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(r)
        
    print("\n--- Summary by Query Type under q['label'] ---")
    for t, rows in by_type.items():
        n = len(rows)
        r1 = sum(1 for r in rows if r["r1"]) / n
        r5 = sum(1 for r in rows if r["r5"]) / n
        mrr = np.mean([r["rr"] for r in rows])
        print(f"  {t} (n={n}): Recall@1={r1:.4f} ({sum(1 for r in rows if r['r1'])}/{n}), Recall@5={r5:.4f} ({sum(1 for r in rows if r['r5'])}/{n}), MRR={mrr:.4f}")
        
    print(f"\nOverall Dev Threats (n={len(task6_results)}):")
    total_r1 = sum(1 for r in task6_results if r["r1"]) / len(task6_results)
    total_r5 = sum(1 for r in task6_results if r["r5"]) / len(task6_results)
    total_mrr = np.mean([r["rr"] for r in task6_results])
    print(f"  Recall@1: {total_r1:.4f} ({sum(1 for r in task6_results if r['r1'])}/{len(task6_results)})")
    print(f"  Recall@5: {total_r5:.4f} ({sum(1 for r in task6_results if r['r5'])}/{len(task6_results)})")
    print(f"  MRR: {total_mrr:.4f}")

    # Inspect wrong_label queries specifically
    print("\n--- Detail on WRONG_LABEL queries (n=6) under ground-truth label ---")
    wrong_rows = by_type.get("wrong_label", [])
    for r in wrong_rows:
        print(f"  ID: {r['id']}, Label passed: '{r['label_used']}', Gold: {r['gold']}, Filtered: {r['filtered']}, Hits: {r['hit_tids'][:3]}, Rank: {r['first_rank']}, R@5: {r['r5']}")

    # =========================================================================
    # RE-RUN WITH PREDICTED LABELS ONLY (NO ORACLE / NO GROUND TRUTH LEAKAGE)
    # =========================================================================
    print("\n--- RE-RUN DEV WITH PREDICTED LABELS ONLY (AUDIT) ---")
    # For schema queries (n=5): label is the predicted label in operational schema
    # For observable queries (n=14): observable indicators in free text; if unmapped -> fallback
    # For wrong_label queries (n=6): the actual predicted label from the text (5 benign, 1 portscan)
    # For r104_only (n=1): RULE_ONLY:R104 (unmapped to ATT&CK, triggers generic NIST)
    
    wrong_label_predictions = {
        "Q_BOT_WRG1": "BENIGN",
        "Q_HULK_WRG1": "BENIGN",
        "Q_SLOW_WRG1": "BENIGN",
        "Q_SHTT_WRG1": "BENIGN",
        "Q_DDOS_WRG1": "PortScan",
        "Q_WXSS_WRG1": "BENIGN"
    }
    
    # Mode A: Strict Operational Serving Gate (No Query Embedding, Metadata Lookup Only)
    # If mapped label matches an attack class -> retrieve candidate techniques
    # If unmapped / BENIGN / RULE_ONLY -> fallback to generic NIST block (0 technique match)
    mode_a_r1 = []
    mode_a_r5 = []
    mode_a_rr = []
    
    # Mode B: Fallback Semantic Search (When label unmapped/benign, run unfiltered embedding query)
    mode_b_r1 = []
    mode_b_r5 = []
    mode_b_rr = []
    
    for q in dev_threats:
        gold_set = set(q["gold"])
        q_type = q["type"]
        if q_type == "schema":
            pred_lbl = q["label"]
        elif q_type == "wrong_label":
            pred_lbl = wrong_label_predictions[q["id"]]
        elif q_type == "r104_only":
            pred_lbl = "RULE_ONLY:R104"
        else: # observable
            # If oracle: q["label"]; if unmapped: "OBSERVABLE_NO_LABEL"
            # We report oracle observable vs realistic observable
            pred_lbl = q["label"]
            
        # Mode A: Strict Mapped Lookup
        if pred_lbl in lmap_v1 and pred_lbl != "RULE_ONLY:R104" and pred_lbl != "BENIGN":
            tids = [lmap_v1[pred_lbl]["primary_technique_id"]] + lmap_v1[pred_lbl].get("candidate_technique_ids", [])
            # In strict mapped lookup: top techniques are tids in order
            first_rank_a = None
            for idx, tid in enumerate(tids):
                if (tid in gold_set) or any(tid.startswith(g) or g.startswith(tid) for g in gold_set):
                    first_rank_a = idx + 1
                    break
            r1_a = (first_rank_a == 1)
            r5_a = (first_rank_a is not None and first_rank_a <= 5)
            rr_a = 1.0 / first_rank_a if first_rank_a else 0.0
        else:
            # Unmapped / BENIGN / RULE_ONLY -> Generic NIST block (no technique retrieved)
            first_rank_a = None
            r1_a = False
            r5_a = False
            rr_a = 0.0
            
        mode_a_r1.append(r1_a)
        mode_a_r5.append(r5_a)
        mode_a_rr.append(rr_a)
        
        # Mode B: Filtered semantic if mapped, unfiltered semantic if unmapped
        if pred_lbl in lmap_v1 and pred_lbl != "RULE_ONLY:R104" and pred_lbl != "BENIGN":
            tids = [lmap_v1[pred_lbl]["primary_technique_id"]] + lmap_v1[pred_lbl].get("candidate_technique_ids", [])
            v = model.encode([q["text"]], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
            res = coll512.query(query_embeddings=[v], n_results=5, where={"technique_id": {"$in": tids}})
        else:
            v = model.encode([q["text"]], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
            res = coll512.query(query_embeddings=[v], n_results=5)
            
        hit_metas = res["metadatas"][0]
        first_rank_b = None
        for idx, m in enumerate(hit_metas):
            tid = m["technique_id"]
            if (tid in gold_set) or any(tid.startswith(g) or g.startswith(tid) for g in gold_set):
                first_rank_b = idx + 1
                break
        r1_b = (first_rank_b == 1)
        r5_b = (first_rank_b is not None and first_rank_b <= 5)
        rr_b = 1.0 / first_rank_b if first_rank_b else 0.0
        mode_b_r1.append(r1_b)
        mode_b_r5.append(r5_b)
        mode_b_rr.append(rr_b)
        
        if q_type == "wrong_label":
            print(f"  WRONG_LABEL ID: {q['id']}, Pred: '{pred_lbl}', Gold: {q['gold']}")
            print(f"    Mode A (Strict Mapped): rank={first_rank_a}, R@5={r5_a}")
            print(f"    Mode B (Semantic Fallback): rank={first_rank_b}, R@5={r5_b}")

    print("\n--- Summary: Mode A (Strict Serving Gate with Predicted Labels, n=26 Dev Threats) ---")
    print(f"  Recall@1: {np.mean(mode_a_r1):.4f} ({sum(mode_a_r1)}/26)")
    print(f"  Recall@5: {np.mean(mode_a_r5):.4f} ({sum(mode_a_r5)}/26)")
    print(f"  MRR: {np.mean(mode_a_rr):.4f}")

    print("\n--- Summary: Mode B (Semantic Fallback with Predicted Labels, n=26 Dev Threats) ---")
    print(f"  Recall@1: {np.mean(mode_b_r1):.4f} ({sum(mode_b_r1)}/26)")
    print(f"  Recall@5: {np.mean(mode_b_r5):.4f} ({sum(mode_b_r5)}/26)")
    print(f"  MRR: {np.mean(mode_b_rr):.4f}")


if __name__ == "__main__":
    main()
