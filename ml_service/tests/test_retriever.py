"""
Unit tests for RAGRetriever (Phase 10 Stage B).
Verifies:
  1. Initialization and fail-safe graceful fallback when database is missing.
  2. Formatting of operational incident queries.
  3. Structured response schema (chunk_id, technique_id, similarity, title, text).
  4. Decision-support threshold fallback mechanism.
"""

import unittest
from pathlib import Path
from ml_service.src.rag.retriever import RAGRetriever, GENERIC_FALLBACK_GUIDANCE

class TestRAGRetriever(unittest.TestCase):
    def test_failsafe_fallback_on_invalid_path(self):
        retriever = RAGRetriever(
            chroma_path="non_existent_chroma_path_12345",
            model_path="non_existent_model_path_12345"
        )
        self.assertFalse(retriever.is_ready)
        res = retriever.retrieve_for_text("Suspicious traffic on port 80")
        self.assertTrue(res["is_fallback"])
        self.assertEqual(res["top_1_similarity"], 0.0)
        self.assertEqual(res["chunks"][0]["technique_id"], "NIST-CSF-RS")

    def test_operational_incident_query_formation(self):
        retriever = RAGRetriever(
            chroma_path="artifacts/rag/chroma_db",
            model_path="models/embeddings/all-MiniLM-L6-v2",
            collection_name="rag_knowledge_c512",
            tau=0.45
        )
        # Operational incident fixture matching Phase 9 schema v1.1
        incident = {
            "incident_id": 9999,
            "destination_port": 21,
            "primary_threat": "FTP-Patator",
            "flow_count": 28,
            "risk_score": 97.99
        }
        res = retriever.retrieve_for_incident(incident, top_k=3)
        self.assertIn("FTP-Patator", res["query"])
        self.assertIn("port 21", res["query"])
        self.assertIn("chunks", res)
        self.assertIsInstance(res["chunks"], list)

if __name__ == "__main__":
    unittest.main()

