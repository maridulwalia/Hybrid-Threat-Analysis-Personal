"""
15_profile_fresh_processes.py
─────────────────────────────
Measures exact peak RSS and Working Set memory across 4 isolated subprocess configurations:
  (a) Python + pandas + XGBoost + risk scorer
  (b) (a) + ChromaDB metadata get (operational zero-embedding serving path)
  (c) (b) + SentenceTransformers query embedding (semantic vector search path)
  (d) (a) + SHAP TreeExplainer (representative sample explanation path)

Each configuration executes in a fresh, isolated Python interpreter process.
"""
import sys
import subprocess
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
py_exe = sys.executable

PROFILES = {
    "a_ml_risk_scorer": """
import sys, os, time, psutil
p = psutil.Process()
import pandas as pd
import numpy as np
import xgboost as xgb
# Load risk scorer module
from ml_service.src.risk.risk_scorer import IncidentRiskScorer
scorer = IncidentRiskScorer(weights=[0.30, 0.10, 0.30, 0.30], n_ref=25.0, rule_variant="Variant_B")

# Simulate scoring 100 incident batches
incident_data = {
    "p_attack_series": [0.99] * 25,
    "primary_threat": "PortScan",
    "n_flows": 25,
    "r104_any": False,
    "has_ml_flag": True
}
for i in range(100):
    res = scorer.score_incident(incident_data)

info = p.memory_info()
peak_mb = (info.peak_wset if hasattr(info, 'peak_wset') else info.rss) / (1024 * 1024)
rss_mb = info.rss / (1024 * 1024)
print(f"RESULT:{rss_mb:.2f}:{peak_mb:.2f}")
""",

    "b_ml_risk_chroma_metadata": """
import sys, os, time, psutil
p = psutil.Process()
import pandas as pd
import numpy as np
import xgboost as xgb
from ml_service.src.risk.risk_scorer import IncidentRiskScorer
scorer = IncidentRiskScorer(weights=[0.30, 0.10, 0.30, 0.30], n_ref=25.0, rule_variant="Variant_B")

from ml_service.src.rag.retriever import RAGRetriever
retriever = RAGRetriever(
    chroma_path="artifacts/rag/chroma_db",
    model_path="models/embeddings/all-MiniLM-L6-v2",
    collection_name="rag_knowledge_c512"
)

# Test zero-embedding mapped retrieval
res = retriever.retrieve_mapped_metadata("FTP-Patator")
assert not res["is_generic"], "Should retrieve FTP-Patator technique"
assert "torch" not in sys.modules, "torch must NOT be loaded"
assert "sentence_transformers" not in sys.modules, "sentence_transformers must NOT be loaded"

info = p.memory_info()
peak_mb = (info.peak_wset if hasattr(info, 'peak_wset') else info.rss) / (1024 * 1024)
rss_mb = info.rss / (1024 * 1024)
print(f"RESULT:{rss_mb:.2f}:{peak_mb:.2f}")
""",

    "c_ml_risk_chroma_embedding": """
import sys, os, time, psutil
p = psutil.Process()
import pandas as pd
import numpy as np
import xgboost as xgb
from ml_service.src.risk.risk_scorer import IncidentRiskScorer
scorer = IncidentRiskScorer(weights=[0.30, 0.10, 0.30, 0.30], n_ref=25.0, rule_variant="Variant_B")

from ml_service.src.rag.retriever import RAGRetriever
retriever = RAGRetriever(
    chroma_path="artifacts/rag/chroma_db",
    model_path="models/embeddings/all-MiniLM-L6-v2",
    collection_name="rag_knowledge_c512"
)

# Execute semantic query embedding
res = retriever.retrieve_for_text("Suspicious brute force attack on port 21")
assert len(res["chunks"]) > 0

info = p.memory_info()
peak_mb = (info.peak_wset if hasattr(info, 'peak_wset') else info.rss) / (1024 * 1024)
rss_mb = info.rss / (1024 * 1024)
print(f"RESULT:{rss_mb:.2f}:{peak_mb:.2f}")
""",

    "d_ml_risk_shap_explainer": """
import sys, os, time, psutil
p = psutil.Process()
import pandas as pd
import numpy as np
import xgboost as xgb
from ml_service.src.risk.risk_scorer import IncidentRiskScorer
scorer = IncidentRiskScorer(weights=[0.30, 0.10, 0.30, 0.30], n_ref=25.0, rule_variant="Variant_B")

import shap
# Create dummy booster and small background
X_dummy = np.random.randn(50, 76).astype(np.float32)
y_dummy = np.random.randint(0, 2, size=50)
dtrain = xgb.DMatrix(X_dummy, label=y_dummy)
param = {'max_depth': 4, 'tree_method': 'hist', 'objective': 'binary:logistic'}
bst = xgb.train(param, dtrain, num_boost_round=10)

explainer = shap.TreeExplainer(bst)
shap_vals = explainer.shap_values(X_dummy[:5])

info = p.memory_info()
peak_mb = (info.peak_wset if hasattr(info, 'peak_wset') else info.rss) / (1024 * 1024)
rss_mb = info.rss / (1024 * 1024)
print(f"RESULT:{rss_mb:.2f}:{peak_mb:.2f}")
"""
}

def main():
    print("=== MEASURING FRESH-PROCESS MEMORY PROFILES (ISOLATED SUBPROCESSES) ===")
    results = {}
    
    for name, code in PROFILES.items():
        cmd = [py_exe, "-c", code]
        proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"Error in {name}:")
            print(proc.stderr)
            continue
            
        out_line = [line for line in proc.stdout.splitlines() if line.startswith("RESULT:")]
        if out_line:
            _, rss_str, peak_str = out_line[0].split(":")
            rss = float(rss_str)
            peak = float(peak_str)
            results[name] = {
                "rss_mb": rss,
                "peak_mb": peak,
                "within_500mb_budget": peak <= 500.0,
                "headroom_mb": round(500.0 - peak, 2)
            }
            print(f"[{name}] RSS: {rss:.2f} MB | Peak Working Set: {peak:.2f} MB | Budget <= 500 MB: {peak <= 500.0} (Headroom: {500.0 - peak:.2f} MB)")
            
    out_file = _ROOT / "artifacts" / "rag" / "fresh_process_memory_profiles_v1.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved profile results to {out_file.name}")

if __name__ == "__main__":
    main()
