"""
13_embed_token_grid.py
──────────────────────
Embeds token-capped chunks into a dedicated ChromaDB collection in a fresh process.
Usage: python 13_embed_token_grid.py --grid tok128
       python 13_embed_token_grid.py --grid tok256
"""
import sys
import os
import json
import argparse
import time
import psutil
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

_ROOT = Path(__file__).resolve().parents[3]

def get_peak_rss_mb() -> float:
    p = psutil.Process()
    # On Windows, peak working set is in peak_wset
    info = p.memory_info()
    if hasattr(info, "peak_wset"):
        return info.peak_wset / (1024 * 1024)
    return info.rss / (1024 * 1024)

def embed_grid(grid_id: str, batch_size: int = 128):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    rag_dir = _ROOT / "artifacts" / "rag"
    chunk_file = rag_dir / f"chunks_{grid_id}_v1.json"
    chroma_path = rag_dir / "chroma_db"
    model_dir = _ROOT / "models" / "embeddings" / "all-MiniLM-L6-v2"
    
    print(f"\n[{grid_id}] Loading chunks from {chunk_file.name}...")
    with open(chunk_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data["chunks"]
    total = len(chunks)
    print(f"[{grid_id}] Total chunks to embed: {total}")
    
    print(f"[{grid_id}] Initializing ChromaDB client at {chroma_path}...")
    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False)
    )
    
    coll_name = f"rag_knowledge_{grid_id}"
    # Delete existing collection if present (for clean re-indexing)
    try:
        client.delete_collection(coll_name)
        print(f"[{grid_id}] Replaced prior collection '{coll_name}'.")
    except Exception:
        pass
        
    coll = client.create_collection(
        name=coll_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    print(f"[{grid_id}] Loading SentenceTransformer from {model_dir} (CPU)...")
    model = SentenceTransformer(str(model_dir), device="cpu")
    model.max_seq_length = 256
    
    t0 = time.time()
    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [{
            "grid_id": c["grid_id"],
            "source": c["source"],
            "technique_id": c["technique_id"],
            "technique_name": c["technique_name"],
            "tactics": c["tactics"],
            "section_type": c["section_type"],
            "title": c["title"],
            "token_count": c["token_count"]
        } for c in batch]
        
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        ).tolist()
        
        coll.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts
        )
        if (i // batch_size) % 5 == 0 or (i + batch_size) >= total:
            print(f"[{grid_id}] Indexed {min(i + batch_size, total)}/{total} chunks...")
            
    elapsed = time.time() - t0
    peak_rss = get_peak_rss_mb()
    print(f"[{grid_id}] Done! Indexed {coll.count()} chunks in {elapsed:.2f} s.")
    print(f"[{grid_id}] Peak RSS: {peak_rss:.2f} MB")
    return {"grid_id": grid_id, "count": coll.count(), "elapsed_s": round(elapsed, 2), "peak_rss_mb": round(peak_rss, 2)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True, choices=["tok128", "tok256"])
    args = parser.parse_args()
    embed_grid(args.grid)

