"""
12_hard_capped_token_grids.py
─────────────────────────────
Generates token-capped chunking grids using all-MiniLM-L6-v2's WordPiece tokenizer:
  - tok128: Hard cap of 128 tokens (overlap 24)
  - tok256: Hard cap of 256 tokens (overlap 48)

Guarantees 100.0% of chunks have total tokens <= 256 (and <= 128 for tok128),
strictly eliminating embedding-window truncation.
"""
import sys
import json
import re
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

import importlib.util

_ROOT = Path(__file__).resolve().parents[3]
chunker_path = _ROOT / "ml_service" / "scripts" / "rag" / "05_parse_and_chunk.py"
spec = importlib.util.spec_from_file_location("parse_and_chunk", str(chunker_path))
parse_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parse_mod)

parse_mitre_stix = parse_mod.parse_mitre_stix
parse_nist_pdf = parse_mod.parse_nist_pdf
parse_nvd_json = parse_mod.parse_nvd_json
parse_owasp_docs = parse_mod.parse_owasp_docs

def build_token_capped_corpus(
    grid_id: str,
    max_tokens: int,
    overlap_tokens: int,
    parsed_sections: list[dict],
    tokenizer: AutoTokenizer
) -> list[dict]:
    chunks = []
    max_content_tokens = max_tokens - 2 # reserve 2 tokens for [CLS] and [SEP]
    step = max_content_tokens - overlap_tokens
    if step <= 0:
        step = max_content_tokens // 2
        
    for sec in parsed_sections:
        raw_text = sec["text"]
        input_ids = tokenizer.encode(raw_text, add_special_tokens=False)
        
        if not input_ids:
            continue
            
        if len(input_ids) <= max_content_tokens:
            chunk_spans = [input_ids]
        else:
            chunk_spans = []
            for i in range(0, len(input_ids), step):
                span = input_ids[i : i + max_content_tokens]
                if not span:
                    break
                chunk_spans.append(span)
                if i + max_content_tokens >= len(input_ids):
                    break
                    
        for idx, span in enumerate(chunk_spans):
            chunk_text = tokenizer.decode(span, clean_up_tokenization_spaces=True, skip_special_tokens=True).strip()
            # Verify strict token cap with special tokens added
            re_encoded = tokenizer.encode(chunk_text, add_special_tokens=True)
            # Iteratively trim if decode introduced expansion
            while len(re_encoded) > max_tokens and len(span) > 1:
                span = span[:-1]
                chunk_text = tokenizer.decode(span, clean_up_tokenization_spaces=True, skip_special_tokens=True).strip()
                re_encoded = tokenizer.encode(chunk_text, add_special_tokens=True)
                
            actual_tok_count = len(re_encoded)
            cid = f"{grid_id}_{sec['technique_id']}_{sec['section_type']}_{idx}"
            cid = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', cid)
            
            chunks.append({
                "chunk_id": cid,
                "grid_id": grid_id,
                "source": sec["source"],
                "technique_id": sec["technique_id"],
                "technique_name": sec["technique_name"],
                "tactics": sec["tactics"],
                "section_type": sec["section_type"],
                "title": sec["title"],
                "chunk_index": idx,
                "total_chunks_in_section": len(chunk_spans),
                "word_count": len(chunk_text.split()),
                "token_count": actual_tok_count,
                "text": chunk_text
            })
            
    return chunks

def main():
    raw_dir = _ROOT / "artifacts" / "rag" / "raw"
    out_dir = _ROOT / "artifacts" / "rag"
    model_dir = _ROOT / "models" / "embeddings" / "all-MiniLM-L6-v2"
    
    print(f"[Tokenizer] Loading tokenizer from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    
    print("[Corpus] Parsing raw snapshots...")
    mitre_sections = parse_mitre_stix(raw_dir / "enterprise-attack-20.0.json")
    nist_sections = parse_nist_pdf(raw_dir / "NIST.SP.800-61r3.pdf")
    nvd_sections = parse_nvd_json(raw_dir / "nvd_cve_2014_0160.json")
    owasp_sections = parse_owasp_docs(raw_dir / "owasp_top10_2021")
    
    all_sections = mitre_sections + nist_sections + nvd_sections + owasp_sections
    print(f"[Corpus] Total parsed document sections: {len(all_sections)}")
    
    grids = [
        ("tok128", 128, 24),
        ("tok256", 256, 48),
    ]
    
    stats = {}
    for grid_id, max_tokens, overlap in grids:
        print(f"\n[Chunker] Building grid {grid_id} (max_tokens={max_tokens}, overlap={overlap})...")
        chunks = build_token_capped_corpus(grid_id, max_tokens, overlap, all_sections, tokenizer)
        
        tok_lens = [c["token_count"] for c in chunks]
        over_256 = sum(1 for t in tok_lens if t > 256)
        pct_over_256 = (over_256 / len(tok_lens)) * 100.0
        
        stats[grid_id] = {
            "chunk_count": len(chunks),
            "max_tokens_cap": max_tokens,
            "min_tokens": int(np.min(tok_lens)),
            "median_tokens": float(np.median(tok_lens)),
            "mean_tokens": round(float(np.mean(tok_lens)), 2),
            "max_tokens_observed": int(np.max(tok_lens)),
            "count_over_256": over_256,
            "pct_over_256": pct_over_256
        }
        
        out_file = out_dir / f"chunks_{grid_id}_v1.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "grid_id": grid_id,
                "token_cap": max_tokens,
                "overlap_tokens": overlap,
                "chunk_count": len(chunks),
                "stats": stats[grid_id],
                "chunks": chunks
            }, f, indent=2)
            
        print(f"  -> Saved {len(chunks)} chunks to {out_file.name}")
        print(f"  -> Min: {stats[grid_id]['min_tokens']}, Median: {stats[grid_id]['median_tokens']:.1f}, Max: {stats[grid_id]['max_tokens_observed']}")
        print(f"  -> Count > 256: {over_256} ({pct_over_256:.2f}%)")
        assert over_256 == 0, f"Violation: {over_256} chunks exceeded 256 tokens in grid {grid_id}!"
        assert stats[grid_id]["max_tokens_observed"] <= max_tokens, f"Violation: max token cap exceeded in {grid_id}!"
        
    print("\n=== Token Capping Summary ===")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
