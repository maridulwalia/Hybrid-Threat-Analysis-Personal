"""
preflight_check.py
──────────────────
Verify all SHA-256 hashes in artifacts/MANIFEST_sha256_v1.txt against current filesystem.
"""
import sys
import hashlib
from pathlib import Path

_ROOT = Path(r"d:\Hybrid Threat Analysis and Cyber Defence Support System - Personal")

manifest_name = sys.argv[1] if len(sys.argv) > 1 else "MANIFEST_sha256_v2.txt"
manifest_path = _ROOT / "artifacts" / manifest_name
if not manifest_path.exists():
    print("Manifest file not found!")
    sys.exit(1)

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()

all_ok = True
checked_count = 0

with open(manifest_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 4 and parts[2] == "bytes":
            exp_hash = parts[0]
            exp_bytes = int(parts[1])
            rel_path = parts[3]
            full_path = _ROOT / rel_path
            
            if not full_path.exists():
                print(f"MISSING: {rel_path}")
                all_ok = False
                continue
            
            obs_bytes = full_path.stat().st_size
            obs_hash = sha256_file(full_path)
            
            if obs_bytes != exp_bytes or obs_hash != exp_hash:
                print(f"MISMATCH in {rel_path}:")
                print(f"  Exp: {exp_hash} ({exp_bytes} bytes)")
                print(f"  Obs: {obs_hash} ({obs_bytes} bytes)")
                all_ok = False
            else:
                checked_count += 1

print(f"Pre-flight verification: {checked_count} files checked.")
print(f"Integrity check passed: {all_ok}")
if not all_ok:
    sys.exit(1)

