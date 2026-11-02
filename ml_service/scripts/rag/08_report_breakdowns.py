import json
from pathlib import Path
import numpy as np

def analyze():
    test_file = Path("artifacts/rag/test_evaluation_results_v1.json")
    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    threats = data["test_threat_details"]
    print("=== BREAKDOWN BY QUERY TYPE ===")
    for qtype in ["schema", "observable", "freeform"]:
        sub = [t for t in threats if t["type"] == qtype]
        if sub:
            r1 = np.mean([t["r1"] for t in sub])
            r3 = np.mean([t["r3"] for t in sub])
            r5 = np.mean([t["r5"] for t in sub])
            mrr = np.mean([t["rr"] for t in sub])
            print(f"Type '{qtype}' (n={len(sub)}): R@1={r1:.4f}, R@3={r3:.4f}, R@5={r5:.4f}, MRR={mrr:.4f}")
            
    print("\n=== BREAKDOWN BY ATTACK CLASS ===")
    classes = sorted(list(set(t["label"] for t in threats)))
    for cls in classes:
        sub = [t for t in threats if t["label"] == cls]
        r1 = np.mean([t["r1"] for t in sub])
        r5 = np.mean([t["r5"] for t in sub])
        mrr = np.mean([t["rr"] for t in sub])
        print(f"Class '{cls}' (n={len(sub)}): R@1={r1:.4f}, R@5={r5:.4f}, MRR={mrr:.4f}")

if __name__ == "__main__":
    analyze()

