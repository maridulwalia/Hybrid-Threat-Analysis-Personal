"""
check_ram.py — Phase 2 diagnostic
Measures:
  1. Free RAM before loading the cleaned dataset
  2. RSS delta while loading (actual RAM consumed by the DataFrames)
  3. Free RAM after loading

Usage:
    python ml_service/scripts/check_ram.py
"""

import os
import sys
import glob
import psutil
import pandas as pd

# ── path to cleaned CSVs (produced by Phase 2 pipeline) ────────────────────
CLEANED_DIR = os.path.join(
    os.path.dirname(__file__),          # ml_service/scripts/
    "..", "..",                         # project root
    "artifacts", "cleaned"
)
CLEANED_DIR = os.path.normpath(CLEANED_DIR)


def bytes_to_gb(b: int) -> float:
    return b / (1024 ** 3)


def ram_snapshot(label: str) -> dict:
    proc  = psutil.Process(os.getpid())
    vm    = psutil.virtual_memory()
    rss   = proc.memory_info().rss
    free  = vm.available
    total = vm.total
    print(f"\n[{label}]")
    print(f"  Process RSS : {bytes_to_gb(rss):.3f} GB")
    print(f"  Free RAM    : {bytes_to_gb(free):.3f} GB  ({free / total * 100:.1f}% of {bytes_to_gb(total):.2f} GB total)")
    return {"rss": rss, "free": free}


def main() -> None:
    csv_files = sorted(glob.glob(os.path.join(CLEANED_DIR, "*.csv")))
    if not csv_files:
        print(f"[ERROR] No CSV files found in: {CLEANED_DIR}")
        print("        Run the Phase 2 cleaning pipeline first.")
        sys.exit(1)

    print(f"Found {len(csv_files)} cleaned CSV file(s) in:\n  {CLEANED_DIR}")
    for f in csv_files:
        size_mb = os.path.getsize(f) / (1024 ** 2)
        print(f"  {os.path.basename(f):60s}  {size_mb:7.1f} MB on disk")

    before = ram_snapshot("BEFORE LOAD")

    print("\nLoading all cleaned CSVs into a single DataFrame ...")
    chunks = []
    for f in csv_files:
        df_chunk = pd.read_csv(f, low_memory=False)
        chunks.append(df_chunk)

    df = pd.concat(chunks, ignore_index=True)
    del chunks  # release list references

    after = ram_snapshot("AFTER LOAD")

    # ── in-memory size reported by pandas ──────────────────────────────────
    pandas_mem = df.memory_usage(deep=True).sum()

    # ── derived metrics ─────────────────────────────────────────────────────
    rss_delta = after["rss"] - before["rss"]

    print("\n" + "=" * 58)
    print("  SUMMARY")
    print("=" * 58)
    print(f"  Rows x Columns          : {df.shape[0]:,} x {df.shape[1]}")
    print(f"  DataFrame RAM (pandas)  : {bytes_to_gb(pandas_mem):.3f} GB")
    print(f"  Process RSS delta       : {bytes_to_gb(rss_delta):.3f} GB")
    print(f"  Free RAM while loaded   : {bytes_to_gb(after['free']):.3f} GB")
    print(f"  Free RAM before load    : {bytes_to_gb(before['free']):.3f} GB")
    print(f"  RAM consumed by load    : {bytes_to_gb(before['free'] - after['free']):.3f} GB")
    print("=" * 58)

    # ── safety check against project's 8 GB hard cap ───────────────────────
    SAFETY_LIMIT_GB = 0.5   # project caps ML pipeline at 500 MB peak RSS delta
    if bytes_to_gb(rss_delta) > SAFETY_LIMIT_GB:
        print(f"\n[WARNING] RSS delta ({bytes_to_gb(rss_delta):.3f} GB) "
              f"exceeds the {SAFETY_LIMIT_GB} GB single-load safety limit.")
        print("          Consider using chunked reads in the ML pipeline (Phase 3+).")
    else:
        print(f"\n[OK] RSS delta is within the {SAFETY_LIMIT_GB} GB single-load safety limit.")


if __name__ == "__main__":
    main()
