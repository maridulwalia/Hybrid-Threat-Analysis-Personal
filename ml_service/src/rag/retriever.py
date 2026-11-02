"""
Operational RAG Retrieval Service for Incident Decision Support (Phase 10).
Retrieves grounded cybersecurity intelligence from authoritative bodies
(MITRE ATT&CK, NIST SP 800-61r3, NVD, OWASP) using semantic vector search in ChromaDB.

Fail-safe Requirement:
- If RAG database or embedding models are unavailable, returns structured fallback guidance.
- Core ML/XAI/Rule/Risk pipeline operates independently of this layer.
"""

import os
import logging
from typing import Optional, Any
from pathlib import Path

logger = logging.getLogger("rag_retriever")

GENERIC_FALLBACK_GUIDANCE = {
    "technique_id": "NIST-CSF-RS",
    "technique_name": "NIST SP 800-61r3: Incident Response & Triage Framework",
    "title": "NIST SP 800-61r3: Incident Response Recommendations (CSF 2.0 Community Profile)",
    "section_type": "incident_handling",
    "source": "NIST SP 800-61r3",
    "similarity": 0.0,
    "text": (
        "Standard Incident Response Protocol (NIST SP 800-61 Revision 3 / CSF 2.0):\n"
        "1. Analysis (RS.AN): Acquire and preserve full flow and host telemetry for the affected endpoint.\n"
        "2. Mitigation (RS.MI): Isolate the suspect host or apply perimeter filtering if suspicious traffic persists.\n"
        "3. Eradication & Recovery (RC): Identify root cause, verify system integrity, and document lessons learned."
    )
}

class RAGRetriever:
    def __init__(
        self,
        chroma_path: str = "artifacts/rag/chroma_db",
        model_path: str = "models/embeddings/all-MiniLM-L6-v2",
        collection_name: str = "rag_knowledge_c512",
        tau: float = 0.45
    ):
        self.chroma_path = Path(chroma_path)
        self.model_path = Path(model_path)
        self.collection_name = collection_name
        self.tau = tau
        self._client = None
        self._coll = None
        self._embed_model = None
        self._is_ready = False
        
        self._initialize()
        
    def _initialize(self):
        try:
            if not self.chroma_path.exists():
                logger.warning(f"ChromaDB path {self.chroma_path} does not exist. Operating in fallback mode.")
                return
            if not self.model_path.exists():
                logger.warning(f"Embedding model path {self.model_path} does not exist. Operating in fallback mode.")
                return
                
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            
            import torch
            torch.set_num_threads(1)
            
            from sentence_transformers import SentenceTransformer
            import chromadb
            from chromadb.config import Settings
            
            self._embed_model = SentenceTransformer(str(self.model_path), device="cpu")
            self._client = chromadb.PersistentClient(
                path=str(self.chroma_path),
                settings=Settings(anonymized_telemetry=False)
            )
            self._coll = self._client.get_collection(
                name=self.collection_name,
                embedding_function=None
            )
            self._is_ready = True
            logger.info(f"RAGRetriever initialized successfully on collection '{self.collection_name}' (tau={self.tau}).")
        except Exception as e:
            logger.error(f"Failed to initialize RAGRetriever: {e}. Falling back to default guidance.")
            self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def retrieve_for_incident(self, incident: dict, top_k: int = 3) -> dict:
        """
        Retrieves knowledge chunks given an operational incident dictionary conforming to Phase 9 schema.
        """
        primary_threat = incident.get("primary_threat", "Unknown Threat")
        dport = incident.get("destination_port", 0)
        flow_count = incident.get("flow_count", 1)
        
        # Build natural-language query from operational telemetry
        query_text = f"Incident targeting port {dport}: {primary_threat} attack campaign with {flow_count} flows."
        return self.retrieve_for_text(query_text, top_k=top_k)

    def retrieve_for_text(self, query_text: str, top_k: int = 3) -> dict:
        """
        Retrieves top-k relevant knowledge chunks for arbitrary analyst text query.
        """
        if not self._is_ready:
            return {
                "query": query_text,
                "is_fallback": True,
                "reason": "RAGRetriever service unavailable (fail-safe fallback).",
                "top_1_similarity": 0.0,
                "chunks": [GENERIC_FALLBACK_GUIDANCE]
            }
            
        try:
            vec = self._embed_model.encode([query_text], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
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
                    "text": docs[i]
                })
                
            top1_sim = chunks[0]["similarity"] if chunks else 0.0
            is_fallback = top1_sim < self.tau
            
            if is_fallback:
                return {
                    "query": query_text,
                    "is_fallback": True,
                    "reason": f"Top similarity score ({top1_sim:.4f}) below acceptance threshold tau ({self.tau:.2f}).",
                    "top_1_similarity": top1_sim,
                    "chunks": [GENERIC_FALLBACK_GUIDANCE] + chunks
                }
                
            return {
                "query": query_text,
                "is_fallback": False,
                "top_1_similarity": top1_sim,
                "chunks": chunks
            }
        except Exception as e:
            logger.error(f"Error during retrieval query: {e}")
            return {
                "query": query_text,
                "is_fallback": True,
                "reason": f"Retrieval error: {e}",
                "top_1_similarity": 0.0,
                "chunks": [GENERIC_FALLBACK_GUIDANCE]
            }

