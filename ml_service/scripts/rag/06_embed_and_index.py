"""
Phase 10 Stage B: Embedding and Vector Indexing.
Encodes pre-registered chunk grids using all-MiniLM-L6-v2 (CPU-only, offline)
and populates persistent ChromaDB collections in artifacts/rag/chroma_db.

Hardware Constraint Management:
- Sets torch.set_num_threads(1) to avoid thread pool bloat.
- Sets HF_HUB_OFFLINE=1.
- Decouples embedding generation from ChromaDB indexing:
    Step 1: Batch-encode chunks -> write embeddings_{grid}_v1.npz -> release model & gc.collect().
    Step 2: Load vectors from disk -> bulk-insert into ChromaDB PersistentClient.
- Monitors and records exact peak RSS to ensure compliance with < 500 MB constraint.
"""

import os
import gc
import json
import time
from pathlib import Path
import numpy as np
import psutil

proc = psutil.Process()

def get_peak_rss_mb():
    mem = proc.memory_info()
    peak = getattr(mem, 'peak_wset', mem.rss)
    return peak / (1024 * 1024)

def encode_grid(grid_id: str, chunks_file: Path, model_dir: Path, out_npz: Path, batch_size: int = 32):
    print(f"\n[Embedder] Loading chunks from {chunks_file}...")
    with open(chunks_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data["chunks"]
    print(f"[Embedder] Total chunks to encode for {grid_id}: {len(chunks)}")
    
    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]
    
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    import torch
    torch.set_num_threads(1)
    
    from sentence_transformers import SentenceTransformer
    print(f"[Embedder] Loading SentenceTransformer from {model_dir} (CPU-only)...")
    t0 = time.time()
    model = SentenceTransformer(str(model_dir), device="cpu")
    t_load = time.time() - t0
    print(f"[Embedder] Model loaded in {t_load:.2f}s. Peak RSS: {get_peak_rss_mb():.2f} MB")
    
    print(f"[Embedder] Encoding {len(texts)} chunks (batch_size={batch_size})...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    t_enc = time.time() - t0
    peak_enc = get_peak_rss_mb()
    print(f"[Embedder] Encoded {len(texts)} chunks in {t_enc:.2f}s ({len(texts)/t_enc:.1f} chunks/s). Peak RSS: {peak_enc:.2f} MB")
    
    np.savez_compressed(
        out_npz,
        chunk_ids=np.array(chunk_ids),
        embeddings=embeddings.astype(np.float32)
    )
    print(f"[Embedder] Saved embeddings to {out_npz} ({out_npz.stat().st_size:,} bytes)")
    
    # Release model and reclaim memory
    del model
    del texts
    del embeddings
    gc.collect()
    print(f"[Embedder] Released model. Current RSS: {proc.memory_info().rss / (1024*1024):.2f} MB")
    return t_enc, peak_enc

def index_into_chroma(grid_id: str, chunks_file: Path, npz_file: Path, chroma_dir: Path, bulk_batch: int = 500):
    print(f"\n[Indexer] Loading chunks and vectors for {grid_id}...")
    with open(chunks_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data["chunks"]
    
    npz_data = np.load(npz_file)
    chunk_ids = npz_data["chunk_ids"].tolist()
    embeddings = npz_data["embeddings"].tolist()
    
    assert len(chunks) == len(chunk_ids) == len(embeddings), "Mismatch between chunks and embedding counts!"
    
    import chromadb
    from chromadb.config import Settings
    
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False)
    )
    
    coll_name = f"rag_knowledge_{grid_id.lower()}"
    try:
        client.delete_collection(name=coll_name)
    except Exception:
        pass
        
    coll = client.create_collection(
        name=coll_name,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"}
    )
    
    print(f"[Indexer] Inserting {len(chunks)} items into collection '{coll_name}'...")
    t0 = time.time()
    for i in range(0, len(chunks), bulk_batch):
        batch_chunks = chunks[i:i + bulk_batch]
        batch_ids = chunk_ids[i:i + bulk_batch]
        batch_vecs = embeddings[i:i + bulk_batch]
        
        batch_docs = [c["text"] for c in batch_chunks]
        batch_metas = [{
            "source": c["source"],
            "technique_id": c["technique_id"],
            "technique_name": c["technique_name"],
            "tactics": c["tactics"],
            "section_type": c["section_type"],
            "title": c["title"]
        } for c in batch_chunks]
        
        coll.add(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_vecs,
            metadatas=batch_metas
        )
        print(f"  - Inserted {min(i + bulk_batch, len(chunks))}/{len(chunks)} items...")
        
    t_idx = time.time() - t0
    assert coll.count() == len(chunks), f"Expected {len(chunks)}, got {coll.count()}"
    print(f"[Indexer] Collection '{coll_name}' indexed in {t_idx:.2f}s. Total items verified: {coll.count()}")
    
    # Test query
    q_res = coll.query(
        query_embeddings=[embeddings[0]],
        n_results=2
    )
    assert q_res["ids"][0][0] == chunk_ids[0], f"Integrity error: expected {chunk_ids[0]}, got {q_res['ids'][0][0]}"
    print(f"[Indexer] Top-1 verification query passed (returned {q_res['ids'][0][0]})")
    
    del coll
    del client
    gc.collect()
    return t_idx

def main():
    rag_dir = Path("artifacts/rag")
    chroma_dir = rag_dir / "chroma_db"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    
    model_dir = Path("models/embeddings/all-MiniLM-L6-v2").resolve()
    assert model_dir.exists(), f"Embedding model not found at {model_dir}"
    
    grids = ["C256", "C512", "C1024"]
    stats = {}
    
    for gid in grids:
        c_file = rag_dir / f"chunks_{gid.lower()}_v1.json"
        npz_file = rag_dir / f"embeddings_{gid.lower()}_v1.npz"
        
        t_enc, peak_enc = encode_grid(gid, c_file, model_dir, npz_file, batch_size=32)
        t_idx = index_into_chroma(gid, c_file, npz_file, chroma_dir, bulk_batch=500)
        
        stats[gid] = {
            "encode_time_s": round(t_enc, 2),
            "peak_rss_mb": round(peak_enc, 2),
            "index_time_s": round(t_idx, 2)
        }
        
    stats_path = rag_dir / "indexing_metrics_v1.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        
    print(f"\n[Index Summary] All 3 grids embedded and indexed into ChromaDB at {chroma_dir}")
    print(f"[Index Summary] Metrics saved to {stats_path}")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()

