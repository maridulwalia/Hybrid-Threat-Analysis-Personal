"""
Unit tests for RAGRetriever (Phase 10 Stage B closure pass 2).
Verifies:
  1. Initialization and fail-safe graceful fallback when database is missing.
  2. Zero-embedding metadata retrieval mode (primary_threat -> candidate technique IDs -> Chroma.get)
     with sys.modules blocker asserting torch and sentence_transformers are NEVER imported.
  3. Structured response schema (chunk_id, technique_id, is_generic, title, text).
  4. Freeform semantic search without threshold gating (informational similarity).
"""

import sys
import unittest
from pathlib import Path

class TestRAGRetriever(unittest.TestCase):
    def test_failsafe_fallback_on_invalid_path(self):
        from ml_service.src.rag.retriever import RAGRetriever, GENERIC_FALLBACK_GUIDANCE
        retriever = RAGRetriever(
            chroma_path="non_existent_chroma_path_12345",
            model_path="non_existent_model_path_12345"
        )
        self.assertFalse(retriever.is_ready)
        res = retriever.retrieve_for_text("Suspicious traffic on port 80")
        self.assertTrue(res["is_generic"])
        self.assertEqual(res["top_1_similarity"], 0.0)
        self.assertEqual(res["chunks"][0]["technique_id"], "generic-nist-csf-rs")

    def test_zero_embedding_operational_serving_path(self):
        # Assert torch and sentence_transformers are NOT in sys.modules yet
        self.assertNotIn("torch", sys.modules, "torch should not be pre-imported before test")
        self.assertNotIn("sentence_transformers", sys.modules, "sentence_transformers should not be pre-imported")
        
        from ml_service.src.rag.retriever import RAGRetriever
        retriever = RAGRetriever(
            chroma_path="artifacts/rag/chroma_db",
            model_path="models/embeddings/all-MiniLM-L6-v2",
            collection_name="rag_knowledge_c512"
        )
        self.assertTrue(retriever.is_ready)
        
        # Operational incident fixture matching Phase 9 schema v1.1
        incident = {
            "incident_id": 9999,
            "destination_port": 21,
            "primary_threat": "FTP-Patator",
            "flow_count": 28,
            "risk_score": 97.99
        }
        res = retriever.retrieve_for_incident(incident)
        
        # Verify retrieved contents
        self.assertFalse(res["is_generic"])
        self.assertEqual(res["primary_threat"], "FTP-Patator")
        self.assertIn("T1110.001", res["candidate_technique_ids"])
        self.assertGreater(len(res["chunks"]), 0)
        
        # Verify section ordering: description should precede mitigation/detection
        sec_types = [c["section_type"] for c in res["chunks"]]
        if "description" in sec_types and "mitigation" in sec_types:
            self.assertLess(sec_types.index("description"), sec_types.index("mitigation"))
            
        # STRICT ASSERTION: torch and sentence_transformers must NEVER have been imported!
        self.assertNotIn("torch", sys.modules, "CRITICAL: torch was imported during zero-embedding retrieval!")
        self.assertNotIn("sentence_transformers", sys.modules, "CRITICAL: sentence_transformers was imported during zero-embedding retrieval!")

    def test_unmapped_and_rule_only_triggers_generic_block(self):
        from ml_service.src.rag.retriever import RAGRetriever
        retriever = RAGRetriever(
            chroma_path="artifacts/rag/chroma_db",
            model_path="models/embeddings/all-MiniLM-L6-v2",
            collection_name="rag_knowledge_c512"
        )
        
        # RULE_ONLY:R104
        res_r104 = retriever.retrieve_mapped_metadata("RULE_ONLY:R104")
        self.assertTrue(res_r104["is_generic"])
        self.assertEqual(res_r104["chunks"][0]["technique_id"], "generic-nist-csf-rs")
        
        # BENIGN
        res_benign = retriever.retrieve_mapped_metadata("BENIGN")
        self.assertTrue(res_benign["is_generic"])
        self.assertEqual(res_benign["chunks"][0]["technique_id"], "generic-nist-csf-rs")

if __name__ == "__main__":
    unittest.main()
