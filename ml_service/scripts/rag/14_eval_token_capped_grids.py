"""
14_eval_token_capped_grids.py
─────────────────────────────
Evaluates DEV queries (n=26 threats) across:
  - rag_knowledge_tok128 (token cap 128)
  - rag_knowledge_tok256 (token cap 256)
  - rag_knowledge_c512   (frozen baseline, word cap 512)
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
        
    dev_threats = [q for q in data["queries"] if q["split"] == "dev" and q["label"] != "BENIGN"]
    print(f"Total DEV threat queries: {len(dev_threats)}")
    
    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False)
    )
    
    model = SentenceTransformer(str(model_dir), device="cpu")
    model.max_seq_length = 256
    
    grids = [
        ("tok128", "rag_knowledge_tok128"),
        ("tok256", "rag_knowledge_tok256"),
        ("c512",   "rag_knowledge_c512"),
    ]
    
    results = {}
    for grid_name, coll_name in grids:
        coll = client.get_collection(coll_name)
        r1_list = []
        r5_list = []
        rr_list = []
        
        for q in dev_threats:
            gold_set = set(q["gold"])
            v = model.encode([q["text"]], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
            res = coll.query(query_embeddings=[v], n_results=5)
            
            metas = res["metadatas"][0]
            first_rank = None
            for idx, m in enumerate(metas):
                tid = m["technique_id"]
                if (tid in gold_set) or any(tid.startswith(g) or g.startswith(tid) for g in gold_set):
                    first_rank = idx + 1
                    break
                    
            r1 = (first_rank == 1)
            r5 = (first_rank is not None and first_rank <= 5)
            rr = 1.0 / first_rank if first_rank else 0.0
            
            r1_list.append(r1)
            r5_list.append(r5)
            rr_list.append(rr)
            
        n = len(dev_threats)
        results[grid_name] = {
            "collection": coll_name,
            "chunk_count": coll.count(),
            "n_threat_queries": n,
            "recall_at_1": round(float(np.mean(r1_list)), 4),
            "hits_r1": sum(r1_list),
            "recall_at_5": round(float(np.mean(r5_list)), 4),
            "hits_r5": sum(r5_list),
            "mrr": round(float(np.mean(rr_list)), 4)
        }
        
    print("\n=== DEV Retrieval Performance Across Chunking Grids ===")
    print(json.dumps(results, indent=2))
    
    # Save to json
    out_path = rag_dir / "dev_token_grid_comparison_v1.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved grid comparison to {out_path.name}")

if __name__ == "__main__":
    main()

