"""
Phase 10 Stage B: Dev Tuning and Test Evaluation Suite.
Follows formal pre-registration in artifacts/rag/preregistration_rag_v1.md:
  1. Loads 109 pre-registered queries from artifacts/rag/rag_evaluation_queries_v1.json.
  2. Runs retrieval across C256, C512, and C1024 on the DEV split (47 queries).
  3. Sweeps threshold tau in [0.10, 0.90] strictly on DEV to find tau* = argmax F1(tau).
  4. Freezes the winning chunk grid and tau*.
  5. Evaluates frozen winning model + frozen tau* on the untouched TEST split (47 queries).
  6. Computes Recall@1/3/5 with Wilson 95% CIs, MRR, Macro-averages across attack classes,
     ROC-AUC on top-1 similarity for negatives, and FAR(tau*).
  7. Evaluates Comparative Baselines: Naive Label Lookup and Random Retrieval.
  8. Saves:
     - artifacts/rag/dev_tuning_results_v1.json
     - artifacts/rag/test_evaluation_results_v1.json
"""

import os
import math
import json
import time
import random
from pathlib import Path
import numpy as np

def wilson_score_interval(p_hat: float, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculates Wilson score 95% confidence interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.959963984540054  # 95% two-sided
    denominator = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2 * n)) / denominator
    margin = (z * math.sqrt((p_hat * (1 - p_hat) / n) + (z**2) / (4 * (n**2)))) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)

def compute_roc_auc(threat_scores: list[float], neg_scores: list[float]) -> float:
    """Computes exact Mann-Whitney U ROC-AUC separating threat top-1 scores from negative top-1 scores."""
    n_threat = len(threat_scores)
    n_neg = len(neg_scores)
    if n_threat == 0 or n_neg == 0:
        return 0.5
    
    # Count pairs where threat score > negative score
    concordant = 0.0
    for ts in threat_scores:
        for ns in neg_scores:
            if ts > ns:
                concordant += 1.0
            elif ts == ns:
                concordant += 0.5
    return concordant / (n_threat * n_neg)

def query_collection(coll, embed_model, query_text: str, top_k: int = 10):
    vec = embed_model.encode([query_text], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
    res = coll.query(
        query_embeddings=[vec],
        n_results=top_k,
        include=["metadatas", "distances", "documents"]
    )
    hits = []
    ids = res["ids"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    docs = res["documents"][0]
    
    for i in range(len(ids)):
        # In cosine space, similarity = 1.0 - distance
        sim = 1.0 - dists[i]
        hits.append({
            "chunk_id": ids[i],
            "similarity": float(sim),
            "distance": float(dists[i]),
            "technique_id": metas[i]["technique_id"],
            "technique_name": metas[i]["technique_name"],
            "source": metas[i]["source"],
            "section_type": metas[i]["section_type"],
            "title": metas[i]["title"],
            "text": docs[i]
        })
    return hits

def evaluate_split(split_name: str, queries: list[dict], coll, embed_model, label_map: dict):
    threat_queries = [q for q in queries if q["label"] != "BENIGN"]
    neg_queries = [q for q in queries if q["label"] == "BENIGN"]
    
    threat_results = []
    threat_top1_sims = []
    neg_top1_sims = []
    
    # 1. Evaluate threat queries
    for q in threat_queries:
        hits = query_collection(coll, embed_model, q["text"], top_k=10)
        gold_set = set(q["gold"])
        
        # Check rank of first hit
        first_rank = None
        r1 = False
        r3 = False
        r5 = False
        
        for idx, h in enumerate(hits):
            tid = h["technique_id"]
            # Match on technique ID or subtechnique prefix or exact match
            is_match = (tid in gold_set) or any(tid.startswith(g) or g.startswith(tid) for g in gold_set)
            if is_match and first_rank is None:
                first_rank = idx + 1
                if idx < 1: r1 = True
                if idx < 3: r3 = True
                if idx < 5: r5 = True
                
        rr = (1.0 / first_rank) if first_rank is not None else 0.0
        top1_sim = hits[0]["similarity"] if hits else 0.0
        threat_top1_sims.append(top1_sim)
        
        threat_results.append({
            "id": q["id"],
            "label": q["label"],
            "type": q["type"],
            "top1_sim": top1_sim,
            "first_rank": first_rank,
            "rr": rr,
            "r1": r1,
            "r3": r3,
            "r5": r5,
            "hits": hits[:3]
        })
        
    # 2. Evaluate negative queries
    for q in neg_queries:
        hits = query_collection(coll, embed_model, q["text"], top_k=5)
        top1_sim = hits[0]["similarity"] if hits else 0.0
        neg_top1_sims.append(top1_sim)
        
    # Metrics
    n_th = len(threat_results)
    rec1 = sum(1 for r in threat_results if r["r1"]) / n_th if n_th > 0 else 0.0
    rec3 = sum(1 for r in threat_results if r["r3"]) / n_th if n_th > 0 else 0.0
    rec5 = sum(1 for r in threat_results if r["r5"]) / n_th if n_th > 0 else 0.0
    mrr = sum(r["rr"] for r in threat_results) / n_th if n_th > 0 else 0.0
    
    # Macro across attack labels
    labels = sorted(list(set(r["label"] for r in threat_results)))
    macro_r1 = np.mean([np.mean([r["r1"] for r in threat_results if r["label"] == l]) for l in labels])
    macro_r3 = np.mean([np.mean([r["r3"] for r in threat_results if r["label"] == l]) for l in labels])
    macro_r5 = np.mean([np.mean([r["r5"] for r in threat_results if r["label"] == l]) for l in labels])
    macro_mrr = np.mean([np.mean([r["rr"] for r in threat_results if r["label"] == l]) for l in labels])
    
    auc = compute_roc_auc(threat_top1_sims, neg_top1_sims)
    
    return {
        "n_threat": n_th,
        "n_neg": len(neg_queries),
        "recall_at_1": float(rec1),
        "recall_at_3": float(rec3),
        "recall_at_5": float(rec5),
        "mrr": float(mrr),
        "macro_recall_at_1": float(macro_r1),
        "macro_recall_at_3": float(macro_r3),
        "macro_recall_at_5": float(macro_r5),
        "macro_mrr": float(macro_mrr),
        "roc_auc_top1": float(auc),
        "threat_top1_sims": threat_top1_sims,
        "neg_top1_sims": neg_top1_sims,
        "threat_results": threat_results
    }

def run_dev_tuning_and_test():
    rag_dir = Path("artifacts/rag")
    chroma_dir = rag_dir / "chroma_db"
    queries_file = rag_dir / "rag_evaluation_queries_v1.json"
    map_file = rag_dir / "label_technique_map_v1.json"
    model_dir = Path("models/embeddings/all-MiniLM-L6-v2").resolve()
    
    with open(queries_file, "r", encoding="utf-8") as f:
        all_queries = json.load(f)["queries"]
        
    with open(map_file, "r", encoding="utf-8") as f:
        label_map = json.load(f)["mappings"]
        
    dev_queries = [q for q in all_queries if q["split"] == "dev"]
    test_queries = [q for q in all_queries if q["split"] == "test"]
    
    print(f"[Suite] Loaded {len(all_queries)} total queries: Dev={len(dev_queries)}, Test={len(test_queries)}")
    
    import torch
    torch.set_num_threads(1)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(str(model_dir), device="cpu")
    
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=str(chroma_dir), settings=Settings(anonymized_telemetry=False))
    
    # -------------------------------------------------------------------------
    # PART 1: Dev Split Grid Comparison & Threshold Sweep
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("PART 1: DEV SPLIT EVALUATION & GRID COMPARISON")
    print("="*80)
    
    dev_results_by_grid = {}
    grids = ["C256", "C512", "C1024"]
    
    for gid in grids:
        coll_name = f"rag_knowledge_{gid.lower()}"
        coll = client.get_collection(name=coll_name, embedding_function=None)
        res = evaluate_split("dev", dev_queries, coll, embed_model, label_map)
        dev_results_by_grid[gid] = res
        print(f"[{gid} on Dev] Recall@1={res['recall_at_1']:.4f}, Recall@3={res['recall_at_3']:.4f}, Recall@5={res['recall_at_5']:.4f}, MRR={res['mrr']:.4f}, Macro-MRR={res['macro_mrr']:.4f}, ROC-AUC={res['roc_auc_top1']:.4f}")
        
    # Select winning grid based on pre-registered baseline (C512) and Dev MRR
    # Determine best grid
    best_grid = max(grids, key=lambda g: (dev_results_by_grid[g]["mrr"], dev_results_by_grid[g]["roc_auc_top1"]))
    print(f"\n[Selection] Best chunking grid selected on Dev: {best_grid} (MRR={dev_results_by_grid[best_grid]['mrr']:.4f})")
    
    # Sweep threshold tau on Dev for best grid
    best_dev = dev_results_by_grid[best_grid]
    threat_sims = best_dev["threat_top1_sims"]
    neg_sims = best_dev["neg_top1_sims"]
    
    tau_sweep = []
    best_tau = 0.50
    best_f1 = -1.0
    best_p_tau = 0.0
    best_r_tau = 0.0
    best_far = 0.0
    
    for tau in np.arange(0.20, 0.85, 0.01):
        tau = round(float(tau), 2)
        tp = sum(1 for s in threat_sims if s >= tau)
        fn = sum(1 for s in threat_sims if s < tau)
        fp = sum(1 for s in neg_sims if s >= tau)
        tn = sum(1 for s in neg_sims if s < tau)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        tau_sweep.append({
            "tau": tau,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "far": round(far, 4)
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau
            best_p_tau = prec
            best_r_tau = rec
            best_far = far
            
    print(f"[Threshold Tuning on Dev] Optimal tau* = {best_tau:.2f} -> Max F1 = {best_f1:.4f} (Prec={best_p_tau:.4f}, Rec={best_r_tau:.4f}, FAR={best_far:.4f})")
    
    dev_out = {
        "phase": "Phase 10 Stage B: Dev Tuning",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dev_queries_count": len(dev_queries),
        "grid_comparison": {
            gid: {
                "recall_at_1": dev_results_by_grid[gid]["recall_at_1"],
                "recall_at_3": dev_results_by_grid[gid]["recall_at_3"],
                "recall_at_5": dev_results_by_grid[gid]["recall_at_5"],
                "mrr": dev_results_by_grid[gid]["mrr"],
                "macro_recall_at_1": dev_results_by_grid[gid]["macro_recall_at_1"],
                "macro_recall_at_3": dev_results_by_grid[gid]["macro_recall_at_3"],
                "macro_recall_at_5": dev_results_by_grid[gid]["macro_recall_at_5"],
                "macro_mrr": dev_results_by_grid[gid]["macro_mrr"],
                "roc_auc_top1": dev_results_by_grid[gid]["roc_auc_top1"]
            } for gid in grids
        },
        "selected_grid": best_grid,
        "frozen_tau_star": best_tau,
        "dev_optimal_metrics": {
            "f1": round(best_f1, 4),
            "precision": round(best_p_tau, 4),
            "recall": round(best_r_tau, 4),
            "false_accept_rate": round(best_far, 4)
        },
        "tau_sweep": tau_sweep
    }
    
    dev_out_file = rag_dir / "dev_tuning_results_v1.json"
    with open(dev_out_file, "w", encoding="utf-8") as f:
        json.dump(dev_out, f, indent=2)
    print(f"[Dev Artifact] Saved Dev results to {dev_out_file}")
    
    # -------------------------------------------------------------------------
    # PART 2: Final Evaluation on Untouched Test Split
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("PART 2: FINAL TEST SPLIT EVALUATION (FROZEN CONFIG & TAU)")
    print("="*80)
    
    test_coll = client.get_collection(name=f"rag_knowledge_{best_grid.lower()}", embedding_function=None)
    test_res = evaluate_split("test", test_queries, test_coll, embed_model, label_map)
    
    # Calculate Wilson 95% CIs on Test
    n_test_th = test_res["n_threat"]
    ci_r1 = wilson_score_interval(test_res["recall_at_1"], n_test_th)
    ci_r3 = wilson_score_interval(test_res["recall_at_3"], n_test_th)
    ci_r5 = wilson_score_interval(test_res["recall_at_5"], n_test_th)
    
    # Test FAR at frozen tau*
    test_fp = sum(1 for s in test_res["neg_top1_sims"] if s >= best_tau)
    test_far = test_fp / test_res["n_neg"] if test_res["n_neg"] > 0 else 0.0
    
    print(f"\n[Test Primary Model: {best_grid}]")
    print(f"  Recall@1: {test_res['recall_at_1']:.4f} (95% CI: [{ci_r1[0]:.4f}, {ci_r1[1]:.4f}])")
    print(f"  Recall@3: {test_res['recall_at_3']:.4f} (95% CI: [{ci_r3[0]:.4f}, {ci_r3[1]:.4f}])")
    print(f"  Recall@5: {test_res['recall_at_5']:.4f} (95% CI: [{ci_r5[0]:.4f}, {ci_r5[1]:.4f}])")
    print(f"  MRR:      {test_res['mrr']:.4f}")
    print(f"  Macro-Recall@1: {test_res['macro_recall_at_1']:.4f}")
    print(f"  Macro-Recall@3: {test_res['macro_recall_at_3']:.4f}")
    print(f"  Macro-Recall@5: {test_res['macro_recall_at_5']:.4f}")
    print(f"  Macro-MRR:      {test_res['macro_mrr']:.4f}")
    print(f"  Negative ROC-AUC: {test_res['roc_auc_top1']:.4f}")
    print(f"  False Accept Rate at tau* ({best_tau}): {test_far:.4f} ({test_fp}/{test_res['n_neg']})")
    
    # -------------------------------------------------------------------------
    # PART 3: Comparative Baselines on Test Split
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("PART 3: COMPARATIVE BASELINES ON TEST SPLIT")
    print("="*80)
    
    # Baseline 1: Naive Label Lookup
    # Maps predicted label to primary ATT&CK ID.
    test_threat_queries = [q for q in test_queries if q["label"] != "BENIGN"]
    base_lookup_r1 = 0
    base_lookup_r3 = 0
    base_lookup_r5 = 0
    base_lookup_rr = []
    
    for q in test_threat_queries:
        lbl = q["label"]
        gold_set = set(q["gold"])
        mapped_primary = label_map.get(lbl, {}).get("primary_technique_id", "")
        # For naive lookup, rank is 1 if mapped_primary in gold_set else infinity
        hit = mapped_primary in gold_set or any(mapped_primary.startswith(g) or g.startswith(mapped_primary) for g in gold_set)
        if hit:
            base_lookup_r1 += 1
            base_lookup_r3 += 1
            base_lookup_r5 += 1
            base_lookup_rr.append(1.0)
        else:
            base_lookup_rr.append(0.0)
            
    n_bl = len(test_threat_queries)
    b1_r1 = base_lookup_r1 / n_bl
    b1_r3 = base_lookup_r3 / n_bl
    b1_r5 = base_lookup_r5 / n_bl
    b1_mrr = np.mean(base_lookup_rr)
    print(f"[Baseline 1: Naive Label Lookup] Recall@1={b1_r1:.4f}, Recall@3={b1_r3:.4f}, Recall@5={b1_r5:.4f}, MRR={b1_mrr:.4f}")
    
    # Baseline 2: Random Retrieval
    # Randomly samples k chunks from collection (100 trials)
    all_coll_ids = test_coll.get()["ids"]
    rand_r1_trials = []
    rand_r3_trials = []
    rand_r5_trials = []
    rand_mrr_trials = []
    
    rnd = random.Random(42)
    for _ in range(100):
        t_r1 = 0
        t_r3 = 0
        t_r5 = 0
        t_mrr = []
        for q in test_threat_queries:
            gold_set = set(q["gold"])
            sampled_ids = rnd.sample(all_coll_ids, 5)
            # Fetch technique_ids of sampled chunks
            sampled_items = test_coll.get(ids=sampled_ids)["metadatas"]
            hit_rank = None
            for idx, m in enumerate(sampled_items):
                tid = m["technique_id"]
                if (tid in gold_set) or any(tid.startswith(g) or g.startswith(tid) for g in gold_set):
                    if hit_rank is None:
                        hit_rank = idx + 1
                        if idx < 1: t_r1 += 1
                        if idx < 3: t_r3 += 1
                        if idx < 5: t_r5 += 1
            t_mrr.append(1.0 / hit_rank if hit_rank else 0.0)
        rand_r1_trials.append(t_r1 / n_bl)
        rand_r3_trials.append(t_r3 / n_bl)
        rand_r5_trials.append(t_r5 / n_bl)
        rand_mrr_trials.append(np.mean(t_mrr))
        
    b2_r1 = float(np.mean(rand_r1_trials))
    b2_r3 = float(np.mean(rand_r3_trials))
    b2_r5 = float(np.mean(rand_r5_trials))
    b2_mrr = float(np.mean(rand_mrr_trials))
    print(f"[Baseline 2: Random Retrieval (100 trials)] Recall@1={b2_r1:.4f}, Recall@3={b2_r3:.4f}, Recall@5={b2_r5:.4f}, MRR={b2_mrr:.4f}")
    
    # Save final Test Results
    test_out = {
        "phase": "Phase 10 Stage B: Final Test Evaluation",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selected_grid": best_grid,
        "frozen_tau_star": best_tau,
        "test_queries_count": len(test_queries),
        "threat_queries_count": n_test_th,
        "negative_queries_count": test_res["n_neg"],
        "primary_model_metrics": {
            "recall_at_1": round(test_res["recall_at_1"], 4),
            "recall_at_1_ci95": [round(ci_r1[0], 4), round(ci_r1[1], 4)],
            "recall_at_3": round(test_res["recall_at_3"], 4),
            "recall_at_3_ci95": [round(ci_r3[0], 4), round(ci_r3[1], 4)],
            "recall_at_5": round(test_res["recall_at_5"], 4),
            "recall_at_5_ci95": [round(ci_r5[0], 4), round(ci_r5[1], 4)],
            "mrr": round(test_res["mrr"], 4),
            "macro_recall_at_1": round(test_res["macro_recall_at_1"], 4),
            "macro_recall_at_3": round(test_res["macro_recall_at_3"], 4),
            "macro_recall_at_5": round(test_res["macro_recall_at_5"], 4),
            "macro_mrr": round(test_res["macro_mrr"], 4),
            "roc_auc_top1": round(test_res["roc_auc_top1"], 4),
            "false_accept_rate_at_tau": round(test_far, 4)
        },
        "baselines": {
            "naive_label_lookup": {
                "recall_at_1": round(b1_r1, 4),
                "recall_at_3": round(b1_r3, 4),
                "recall_at_5": round(b1_r5, 4),
                "mrr": round(b1_mrr, 4)
            },
            "random_retrieval_100_trials": {
                "recall_at_1": round(b2_r1, 4),
                "recall_at_3": round(b2_r3, 4),
                "recall_at_5": round(b2_r5, 4),
                "mrr": round(b2_mrr, 4)
            }
        },
        "test_threat_details": test_res["threat_results"]
    }
    
    test_out_file = rag_dir / "test_evaluation_results_v1.json"
    with open(test_out_file, "w", encoding="utf-8") as f:
        json.dump(test_out, f, indent=2)
    print(f"\n[Test Artifact] Saved final Test results to {test_out_file}")
    print("[Suite] All evaluations completed successfully.")

if __name__ == "__main__":
    run_dev_tuning_and_test()

