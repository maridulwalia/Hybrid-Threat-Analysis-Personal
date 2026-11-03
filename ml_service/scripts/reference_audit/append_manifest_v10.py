import hashlib
from pathlib import Path

_ROOT = Path(r"d:\Hybrid Threat Analysis and Cyber Defence Support System - Personal")

new_files = [
    "artifacts/correlation/ml_flags_v3.parquet",
    "artifacts/llm/preregistration_v1.md",
    "artifacts/llm/llm_evaluation_summary_v1.json",
    "artifacts/llm/blind_review_v1.json",
    "artifacts/llm/llm_evaluation_report_v1.md",
]

entries = []
for rel in new_files:
    p = _ROOT / rel
    data = p.read_bytes()
    h = hashlib.sha256(data).hexdigest()
    sz = len(data)
    entry = f"{h}  {sz} bytes  {rel.replace('/', chr(92))}"
    entries.append(entry)
    print(entry)

manifest_v10 = _ROOT / "artifacts" / "MANIFEST_sha256_v10.txt"
with open(manifest_v10, "a", encoding="utf-8") as f:
    f.write("\n# --- Phase 11 Decision Support Integration Artifacts ---\n")
    for e in entries:
        f.write(e + "\n")

print("Appended 5 entries to MANIFEST_sha256_v10.txt")

