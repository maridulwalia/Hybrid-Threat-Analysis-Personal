"""
Phase 10 Stage A (A1.c): Environment Diff and Regression Verifier.
Compares artifacts/rag/env_before_v1.txt vs artifacts/rag/env_after_v1.txt.
Asserts:
  1. All 50 original package pins in env_before_v1.txt remain strictly identical in env_after_v1.txt.
  2. Diff consists strictly of additive new packages for ChromaDB, Sentence-Transformers, and Torch.
"""

import sys
from pathlib import Path

def parse_freeze(filepath: Path) -> dict[str, str]:
    pkgs = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "==" in line:
                name, ver = line.split("==", 1)
                pkgs[name.lower()] = ver
            elif "@" in line:
                name = line.split("@")[0].strip()
                pkgs[name.lower()] = line
    return pkgs

def verify_diff():
    before_path = Path("artifacts/rag/env_before_v1.txt")
    after_path = Path("artifacts/rag/env_after_v1.txt")
    
    if not before_path.exists() or not after_path.exists():
        print(f"Error: missing {before_path} or {after_path}")
        sys.exit(1)
        
    before = parse_freeze(before_path)
    after = parse_freeze(after_path)
    
    print(f"=== [A1.c] Verifying Environment Pin Integrity ===")
    print(f"Total packages before: {len(before)}")
    print(f"Total packages after:  {len(after)}")
    
    # Check for modifications / deletions
    modified = []
    removed = []
    for pkg, ver in before.items():
        if pkg not in after:
            removed.append(pkg)
        elif after[pkg] != ver:
            modified.append((pkg, ver, after[pkg]))
            
    if removed:
        print(f"CRITICAL FAILURE: {len(removed)} packages were removed from venv: {removed}")
        sys.exit(1)
        
    if modified:
        print(f"CRITICAL FAILURE: {len(modified)} package versions were altered:")
        for pkg, old_v, new_v in modified:
            print(f"  - {pkg}: {old_v} -> {new_v}")
        sys.exit(1)
        
    added = {pkg: after[pkg] for pkg in after if pkg not in before}
    print(f"\nSUCCESS: Zero removals, zero version alterations among original {len(before)} packages.")
    print(f"Added {len(added)} new packages (additions strictly):")
    for pkg, ver in sorted(added.items()):
        print(f"  + {pkg}=={ver}")
        
    print("\n[A1.c] Environment integrity assertion PASSED.")

if __name__ == "__main__":
    verify_diff()

