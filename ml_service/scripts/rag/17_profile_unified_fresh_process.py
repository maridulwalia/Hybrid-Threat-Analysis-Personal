"""
17_profile_unified_fresh_process.py
───────────────────────────────────
Profiles ONE fresh, unified Python process executing the full decision-support pipeline:
  1. Imports FastAPI app (ml_service.main.app)
  2. Loads trained XGBoost model (models/xgb_baseline_category_v1.json)
  3. Initializes IncidentRiskScorer
  4. Initializes RAGRetriever (ChromaDB zero-embedding operational serving mode)
  5. Initializes SHAP TreeExplainer
  6. Measures RSS before request
  7. Executes ONE full unified request:
     - XGBoost inference on 1 flow (76 features)
     - RiskScorer composite scoring
     - ChromaDB metadata get for grounded ATT&CK technique chunks
     - SHAP TreeExplainer local feature attribution on the flow
  8. Measures RSS after request & Peak Working Set
  9. Evaluates exact headroom against Render 512 MB Free/Starter and 2048 MB Standard tiers.
"""

import sys
import os
import json
import time
import psutil
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

def get_mem_mb(p: psutil.Process) -> tuple[float, float]:
    info = p.memory_info()
    rss = info.rss / (1024 * 1024)
    peak = (info.peak_wset if hasattr(info, "peak_wset") else info.rss) / (1024 * 1024)
    return rss, peak

def main():
    p = psutil.Process()
    print("=== STARTING UNIFIED PROCESS PROFILING ===")
    
    # 1. Import FastAPI
    from ml_service.main import app
    print(f"[Step 1] Imported FastAPI app. Uptime: {app.title}")
    
    # 2. Load XGBoost Model
    import xgboost as xgb
    import numpy as np
    model_path = _ROOT / "models" / "xgb_baseline_category_v1.json"
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    print(f"[Step 2] Loaded XGBoost model from {model_path.name}")
    
    # 3. Initialize Risk Scorer
    from ml_service.src.risk.risk_scorer import IncidentRiskScorer
    scorer = IncidentRiskScorer(weights=[0.30, 0.10, 0.30, 0.30], n_ref=25.0, rule_variant="Variant_B")
    print("[Step 3] Initialized IncidentRiskScorer (Frozen weights v1)")
    
    # 4. Initialize RAG Retriever
    from ml_service.src.rag.retriever import RAGRetriever
    retriever = RAGRetriever(
        chroma_path="artifacts/rag/chroma_db",
        model_path="models/embeddings/all-MiniLM-L6-v2",
        map_path="artifacts/rag/label_technique_map_v2.json",
        collection_name="rag_knowledge_c512"
    )
    assert retriever.is_ready, "Retriever must be ready"
    print("[Step 4] Initialized RAGRetriever (ChromaDB collection rag_knowledge_c512)")
    
    # 5. Initialize SHAP TreeExplainer
    import shap
    explainer = shap.TreeExplainer(booster)
    print("[Step 5] Initialized SHAP TreeExplainer on booster")
    
    # Measure Before Request
    rss_before, peak_before = get_mem_mb(p)
    print(f"\n>>> RSS BEFORE REQUEST: {rss_before:.2f} MB (Peak WS: {peak_before:.2f} MB)")
    
    # 6. Execute One Unified Request
    t0 = time.time()
    # Dummy flow telemetry: 76 float32 features
    flow_features = np.random.randn(1, 76).astype(np.float32)
    dmatrix = xgb.DMatrix(flow_features)
    preds = booster.predict(dmatrix) # multiclass probabilities (15 classes)
    pred_class_idx = int(np.argmax(preds[0]))
    p_attack = float(1.0 - preds[0][0]) # binary attack prob (class 0 = BENIGN)
    
    # Incident scoring
    incident_data = {
        "p_attack_series": [p_attack] * 10,
        "primary_threat": "FTP-Patator",
        "n_flows": 10,
        "r104_any": True,
        "has_ml_flag": True
    }
    score_res = scorer.score_incident(incident_data)
    
    # Chroma metadata get (zero-embedding serving path)
    rag_res = retriever.retrieve_mapped_metadata(primary_threat="FTP-Patator")
    assert not rag_res["is_generic"], "Should retrieve FTP-Patator technique"
    assert "torch" not in sys.modules, "torch must NOT be loaded in zero-embedding serving path"
    assert "sentence_transformers" not in sys.modules, "sentence_transformers must NOT be loaded"
    
    # SHAP local explanation for the flow
    shap_vals = explainer.shap_values(dmatrix)
    
    elapsed = time.time() - t0
    rss_after, peak_after = get_mem_mb(p)
    print(f">>> RSS AFTER REQUEST:  {rss_after:.2f} MB (Peak WS: {peak_after:.2f} MB)")
    print(f">>> Request Latency:    {elapsed*1000.0:.2f} ms")
    print(f">>> Risk Score Output:  {score_res['risk_score']} (FTP-Patator)")
    print(f">>> Grounded RAG Hits:  {len(rag_res['chunks'])} chunks ({rag_res['candidate_technique_ids']})")
    
    # 7. Render Headroom Evaluation
    render_caps = {
        "Internal_Safety_Budget": 500.0,
        "Render_Free_Tier": 512.0,
        "Render_Starter_Tier": 512.0,
        "Render_Standard_Tier": 2048.0
    }
    
    summary = {
        "service": "ml_service_fastapi_unified",
        "python_version": sys.version,
        "rss_before_request_mb": round(rss_before, 2),
        "rss_after_request_mb": round(rss_after, 2),
        "delta_rss_mb": round(rss_after - rss_before, 2),
        "peak_working_set_mb": round(peak_after, 2),
        "request_latency_ms": round(elapsed * 1000.0, 2),
        "limits": {}
    }
    
    print("\n=== HEADROOM EVALUATION ACROSS RENDER TIERS ===")
    for plan, cap in render_caps.items():
        headroom = round(cap - peak_after, 2)
        status = "PASS" if peak_after <= cap else "FAIL_OOM"
        summary["limits"][plan] = {
            "cap_mb": cap,
            "peak_mb": round(peak_after, 2),
            "headroom_mb": headroom,
            "status": status
        }
        print(f"  {plan:<25} Cap: {cap:>7.1f} MB | Peak: {peak_after:>7.2f} MB | Headroom: {headroom:>7.2f} MB | Status: {status}")
        
    out_path = _ROOT / "artifacts" / "rag" / "unified_serving_memory_profile_v1.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved unified profile to {out_path.name}")

if __name__ == "__main__":
    main()
