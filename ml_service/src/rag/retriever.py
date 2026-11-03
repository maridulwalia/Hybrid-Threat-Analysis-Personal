"""
Operational RAG Retrieval Service for Incident Decision Support (Phase 10).
Retrieves grounded cybersecurity intelligence from authoritative bodies
(MITRE ATT&CK, NIST SP 800-61r3, NVD, OWASP) using ChromaDB.

Serving Architecture:
1. Operational Incident Serving Mode (Production Path):
   - Fast, zero-embedding metadata lookup: primary_threat -> candidate technique IDs
     -> Chroma coll.get(where={"technique_id": {"$in": ids}})
   - Orders chunks by candidate technique priority and section type: [description, mitigation, detection].
   - Operates with ZERO PyTorch / SentenceTransformer overhead (low RSS < 150 MB).
   - If primary_threat is unmapped, BENIGN, or RULE_ONLY: falls back directly to generic NIST block.

2. Freeform Semantic Search Mode (Analyst Query Path):
   - Dense vector search over freeform natural language queries via all-MiniLM-L6-v2.
   - Retains cosine similarity purely as an informational field; no similarity threshold gating.

Fail-safe Requirement:
- If RAG database or embedding models are unavailable, returns structured generic guidance.
- Core ML/XAI/Rule/Risk pipeline operates independently of this layer.
"""

import os
import json
import logging
from typing import Optional, Any
from pathlib import Path

logger = logging.getLogger("rag_retriever")

GENERIC_FALLBACK_GUIDANCE = {
    "chunk_id": "generic_nist_csf_rs",
    "technique_id": "generic-nist-csf-rs",
    "technique_name": "NIST SP 800-61r3: Incident Response & Triage Framework",
    "title": "NIST SP 800-61r3: Incident Response Recommendations (CSF 2.0 Community Profile)",
    "section_type": "generic",
    "source": "generic",
    "is_generic": True,
    "similarity": 0.0,
    "text": (
        "Standard Incident Response Protocol (NIST SP 800-61 Revision 3 / CSF 2.0):\n"
        "1. Analysis (RS.AN): Acquire and preserve full flow and host telemetry for the affected endpoint.\n"
        "2. Mitigation (RS.MI): Isolate the suspect host or apply perimeter filtering if suspicious traffic persists.\n"
        "3. Eradication & Recovery (RC): Identify root cause, verify system integrity, and document lessons learned."
    )
}

SECTION_ORDER = {"description": 0, "mitigation": 1, "detection": 2, "web_vulnerability": 3, "incident_handling": 4}

class RAGRetriever:
    def __init__(
        self,
        chroma_path: str = "artifacts/rag/chroma_db",
        model_path: str = "models/embeddings/all-MiniLM-L6-v2",
        map_path: str = "artifacts/rag/label_technique_map_v2.json",
        collection_name: str = "rag_knowledge_c512"
    ):
        self.chroma_path = Path(chroma_path)
        self.model_path = Path(model_path)
        self.map_path = Path(map_path)
        self.collection_name = collection_name
        
        self._client = None
        self._coll = None
        self._embed_model = None
        self._mapping = None
        self._is_ready = False
        
        self._initialize()

    def _initialize(self):
        try:
            if not self.chroma_path.exists():
                logger.warning(f"ChromaDB path {self.chroma_path} does not exist. Operating in fallback mode.")
                return
                
            # Load label-to-technique mapping for operational fast path
            if self.map_path.exists():
                with open(self.map_path, "r", encoding="utf-8") as f:
                    map_data = json.load(f)
                    self._mapping = map_data.get("mappings", {})
            else:
                # Fallback to v1 map if v2 not found
                alt_map = self.map_path.parent / "label_technique_map_v1.json"
                if alt_map.exists():
                    with open(alt_map, "r", encoding="utf-8") as f:
                        map_data = json.load(f)
                        self._mapping = map_data.get("mappings", {})
                else:
                    self._mapping = {}
                    
            import chromadb
            from chromadb.config import Settings
            
            self._client = chromadb.PersistentClient(
                path=str(self.chroma_path),
                settings=Settings(anonymized_telemetry=False)
            )
            self._coll = self._client.get_collection(
                name=self.collection_name,
                embedding_function=None
            )
            self._is_ready = True
            logger.info(f"RAGRetriever initialized successfully on collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to initialize RAGRetriever: {e}. Operating in fallback mode.")
            self._is_ready = False

    def _get_embed_model(self):
        """Lazy load SentenceTransformer only when semantic embedding retrieval is explicitly called."""
        if self._embed_model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Embedding model directory {self.model_path} not found.")
                
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            
            import torch
            torch.set_num_threads(1)
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer(str(self.model_path), device="cpu")
        return self._embed_model

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def retrieve_mapped_metadata(self, primary_threat: str, token_budget: int = 1500) -> dict:
        """
        Operational Zero-Embedding Serving Path:
        Maps primary_threat -> candidate ATT&CK IDs -> ChromaDB metadata get.
        Never imports or executes PyTorch / SentenceTransformer models.
        """
        if not self._is_ready:
            return {
                "primary_threat": primary_threat,
                "is_generic": True,
                "candidate_technique_ids": [],
                "chunks": [GENERIC_FALLBACK_GUIDANCE]
            }

        if not primary_threat or primary_threat == "BENIGN" or primary_threat.startswith("RULE_ONLY") or (self._mapping and primary_threat not in self._mapping):
            return {
                "primary_threat": primary_threat,
                "is_generic": True,
                "candidate_technique_ids": [],
                "chunks": [GENERIC_FALLBACK_GUIDANCE]
            }

        entry = self._mapping.get(primary_threat, {})
        primary_tid = entry.get("primary_technique_id")
        candidate_tids = [primary_tid] if primary_tid else []
        for tid in entry.get("candidate_technique_ids", []):
            if tid not in candidate_tids:
                candidate_tids.append(tid)

        if not candidate_tids:
            return {
                "primary_threat": primary_threat,
                "is_generic": True,
                "candidate_technique_ids": [],
                "chunks": [GENERIC_FALLBACK_GUIDANCE]
            }

        try:
            res = self._coll.get(where={"technique_id": {"$in": candidate_tids}})
            ids = res.get("ids", [])
            metas = res.get("metadatas", [])
            docs = res.get("documents", [])
            
            if not ids:
                return {
                    "primary_threat": primary_threat,
                    "is_generic": True,
                    "candidate_technique_ids": candidate_tids,
                    "chunks": [GENERIC_FALLBACK_GUIDANCE]
                }
                
            raw_chunks = []
            for i in range(len(ids)):
                tid = metas[i].get("technique_id", "")
                stype = metas[i].get("section_type", "description")
                tech_prio = candidate_tids.index(tid) if tid in candidate_tids else 99
                sec_prio = SECTION_ORDER.get(stype, 99)
                
                raw_chunks.append({
                    "chunk_id": ids[i],
                    "technique_id": tid,
                    "technique_name": metas[i].get("technique_name", ""),
                    "title": metas[i].get("title", ""),
                    "source": metas[i].get("source", ""),
                    "section_type": stype,
                    "is_generic": False,
                    "similarity": None, # pure metadata lookup, no embedding similarity
                    "text": docs[i],
                    "_prio": (tech_prio, sec_prio)
                })
                
            # Order by candidate priority, then section order
            raw_chunks.sort(key=lambda x: x["_prio"])
            
            # Apply token budget
            selected_chunks = []
            accumulated_tokens = 0
            for c in raw_chunks:
                # Approximate token count by word length * 1.3
                approx_tokens = int(len(c["text"].split()) * 1.3)
                if selected_chunks and (accumulated_tokens + approx_tokens > token_budget):
                    continue
                accumulated_tokens += approx_tokens
                clean_chunk = {k: v for k, v in c.items() if k != "_prio"}
                selected_chunks.append(clean_chunk)
                
            if not selected_chunks:
                selected_chunks = [GENERIC_FALLBACK_GUIDANCE]
                
            return {
                "primary_threat": primary_threat,
                "is_generic": False,
                "candidate_technique_ids": candidate_tids,
                "chunks": selected_chunks
            }
        except Exception as e:
            logger.error(f"Error in retrieve_mapped_metadata: {e}")
            return {
                "primary_threat": primary_threat,
                "is_generic": True,
                "candidate_technique_ids": candidate_tids,
                "chunks": [GENERIC_FALLBACK_GUIDANCE]
            }

    def retrieve_for_incident(self, incident: dict, top_k: int = 3) -> dict:
        """
        Retrieves knowledge for an operational incident.
        Uses fast zero-embedding metadata retrieval on primary_threat.
        """
        primary_threat = incident.get("primary_threat", "Unknown Threat")
        return self.retrieve_mapped_metadata(primary_threat=primary_threat)

    def retrieve_for_text(self, query_text: str, top_k: int = 3) -> dict:
        """
        Retrieves top-k relevant knowledge chunks for freeform text query via semantic vector search.
        Similarity threshold gating is removed; cosine similarity is returned as an informational field.
        """
        if not self._is_ready:
            return {
                "query": query_text,
                "is_generic": True,
                "reason": "RAGRetriever service unavailable (fail-safe fallback).",
                "top_1_similarity": 0.0,
                "chunks": [GENERIC_FALLBACK_GUIDANCE]
            }
            
        try:
            embed_model = self._get_embed_model()
            vec = embed_model.encode([query_text], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
            res = self._coll.query(
                query_embeddings=[vec],
                n_results=top_k,
                include=["metadatas", "distances", "documents"]
            )
            
            ids = res["ids"][0]
            metas = res["metadatas"][0]
            dists = res["distances"][0]
            docs = res["documents"][0]
            
            chunks = []
            for i in range(len(ids)):
                sim = float(1.0 - dists[i])
                chunks.append({
                    "chunk_id": ids[i],
                    "similarity": round(sim, 4),
                    "technique_id": metas[i]["technique_id"],
                    "technique_name": metas[i]["technique_name"],
                    "title": metas[i]["title"],
                    "source": metas[i]["source"],
                    "section_type": metas[i]["section_type"],
                    "is_generic": False,
                    "text": docs[i]
                })
                
            top1_sim = chunks[0]["similarity"] if chunks else 0.0
            return {
                "query": query_text,
                "is_generic": False,
                "top_1_similarity": top1_sim,
                "chunks": chunks
            }
        except Exception as e:
            logger.error(f"Error during retrieval query: {e}")
            return {
                "query": query_text,
                "is_generic": True,
                "reason": f"Retrieval error: {e}",
                "top_1_similarity": 0.0,
                "chunks": [GENERIC_FALLBACK_GUIDANCE]
            }
