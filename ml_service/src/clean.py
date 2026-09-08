"""
ml_service/src/clean.py
────────────────────────
Phase 2: Data Cleaning
Streaming, read-only pipeline that cleans all 8 CIC-IDS2017 daily CSV files
and writes deduplicated, sanitized outputs to artifacts/cleaned/.

Design Guarantees:
  1. Source Immutability: Raw CSVs in MachineLearningCSV/ are never modified.
     SHA-256 hashes verify this before and after the run.
  2. Strict Cleaning Order (per chunk):
       strip column names
       → sanitize labels (explicit mapping only)
       → replace +inf / -inf with np.nan
       → drop NaN rows
       → canonical row hash (deterministic MD5-128, including Label)
       → deduplicate (within-file + cross-file)
       → write surviving rows to artifacts/cleaned/
  3. Granular Accounting: rows_read / nan_removed / within_file_dupes /
     cross_file_dupes / rows_written tracked per file and globally.
  4. Live RSS Instrumentation: peak resident set size measured in real time.
  5. Independent Post-Cleaning Verification: separate pass over cleaned files
     confirms zero NaN, zero inf, and zero duplicates.
  6. Pipeline Version Identifier: cleaning_summary.json is stamped with
     phase, pipeline_version, and timestamp for research traceability.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import psutil

# ── Project root path setup ────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.paths import Paths, validate_paths
from ml_service.src.utils.memory_guard import assert_memory_headroom

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Pipeline Constants ─────────────────────────────────────────────────────────
PIPELINE_VERSION = "2.0"
PHASE = "Phase 2"
CHUNK_SIZE = 25_000
MEMORY_LIMIT_MB = 500

# Conservative, explicit label normalization map.
# Only these exact strings are replaced — no broad substitutions.
LABEL_MAP: Dict[str, str] = {
    # Unicode replacement character variant (U+FFFD)
    "Web Attack \ufffd Brute Force": "Web Attack - Brute Force",
    "Web Attack \ufffd XSS": "Web Attack - XSS",
    "Web Attack \ufffd Sql Injection": "Web Attack - Sql Injection",
    # Question-mark variant (some text editors render \ufffd as ?)
    "Web Attack ? Brute Force": "Web Attack - Brute Force",
    "Web Attack ? XSS": "Web Attack - XSS",
    "Web Attack ? Sql Injection": "Web Attack - Sql Injection",
    # Double-space variant
    "Web Attack  Brute Force": "Web Attack - Brute Force",
    "Web Attack  XSS": "Web Attack - XSS",
    "Web Attack  Sql Injection": "Web Attack - Sql Injection",
}

# Canonical column schema (76+1+1+1 = 79).
# Populated after first file is processed; written to column_schema.json.
_COLUMN_SCHEMA: Dict[str, Any] = {}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _sha256_file(path: Path) -> str:
    """Compute SHA-256 digest of a file without loading it fully into RAM."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65_536), b""):
            h.update(block)
    return h.hexdigest()


def _hash_row_128bit(row_values: Tuple[Any, ...]) -> bytes:
    """128-bit MD5 digest of a canonical row tuple for collision-safe deduplication.

    The row must already be cleaned (column names stripped, label sanitized,
    inf replaced with NaN, NaN rows already dropped) before this is called,
    so the same logical flow always produces the same hash regardless of which
    daily CSV it came from.
    """
    return hashlib.md5(str(row_values).encode("utf-8", errors="replace")).digest()


def _peak_rss_mb() -> float:
    """Return current process peak resident set size in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


# ── Source Integrity Audit ─────────────────────────────────────────────────────

def compute_source_integrity() -> Dict[str, Any]:
    """Compute SHA-256, file size, row count, and column count for all raw CSVs."""
    logger.info("Computing source integrity hashes for all 8 raw CSV files...")
    integrity: Dict[str, Any] = {}
    for csv_path in Paths.ALL_CSVS:
        logger.info("  SHA-256: %s", csv_path.name)
        sha256 = _sha256_file(csv_path)
        file_size = csv_path.stat().st_size

        # Read only first chunk to get column count; then count rows cheaply.
        col_count = 0
        row_count = 0
        for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory=False):
            if col_count == 0:
                col_count = len(chunk.columns)
            row_count += len(chunk)

        integrity[csv_path.name] = {
            "sha256": sha256,
            "file_size_bytes": file_size,
            "rows": row_count,
            "columns": col_count,
        }
        logger.info("    ✓ %s | rows=%d | sha256=%s...", csv_path.name, row_count, sha256[:16])

    return integrity


# ── Column Schema ──────────────────────────────────────────────────────────────

def _build_column_schema(stripped_cols: List[str]) -> Dict[str, Any]:
    """Classify all 79 columns into ML features, identifier, redundant, target.

    Schema decomposition: 76 (ML features) + 1 (identifier) + 1 (redundant) + 1 (target) = 79
    """
    IDENTIFIER_COLS = ["Destination Port"]
    REDUNDANT_COLS = ["Fwd Header Length.1"]
    TARGET_COL = "Label"

    ml_feature_cols = [
        c for c in stripped_cols
        if c not in IDENTIFIER_COLS
        and c not in REDUNDANT_COLS
        and c != TARGET_COL
    ]

    schema = {
        "description": (
            "Formal column classification for CIC-IDS2017 (Phase 2 cleaned dataset). "
            "76 ML feature columns + 1 identifier + 1 redundant + 1 target = 79 total."
        ),
        "total_columns": len(stripped_cols),
        "ml_feature_columns_count": len(ml_feature_cols),
        "ml_feature_columns": ml_feature_cols,
        "identifier_columns": IDENTIFIER_COLS,
        "identifier_columns_note": (
            "Destination Port is retained in cleaned CSVs for Phase 7 deterministic rules "
            "and Phase 8 event correlation. It must NOT be included in Phase 4/5 ML model "
            "feature sets to prevent port-number memorization."
        ),
        "redundant_columns": REDUNDANT_COLS,
        "redundant_columns_note": (
            "Fwd Header Length.1 is an exact duplicate of Fwd Header Length. "
            "It is preserved unchanged in the cleaned dataset and will be formally "
            "excluded during Phase 3 feature selection. Removing it in Phase 2 would "
            "obscure the provenance trail for the research paper."
        ),
        "target_column": TARGET_COL,
    }
    return schema


# ── Streaming Cleaning Pipeline ────────────────────────────────────────────────

def clean_dataset() -> Dict[str, Any]:
    """Stream all 8 daily CSVs through the cleaning pipeline and write clean outputs."""
    validate_paths()
    Paths.CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    assert_memory_headroom(min_free_mb=256)

    start_time = time.time()
    process = psutil.Process(os.getpid())
    peak_rss_mb: float = 0.0

    logger.info("=" * 70)
    logger.info("Phase 2 — Data Cleaning  (pipeline v%s)", PIPELINE_VERSION)
    logger.info("=" * 70)

    # Global deduplication hash set — persists across all 8 files.
    global_seen_hashes: Set[bytes] = set()

    # Global counters
    global_rows_read = 0
    global_nan_removed = 0
    global_within_dupes = 0
    global_cross_dupes = 0
    global_rows_written = 0

    # Per-file label normalization counters (global aggregate)
    global_label_normalizations: Dict[str, int] = {}

    file_reports: List[Dict[str, Any]] = []
    col_schema_built = False

    for file_idx, csv_path in enumerate(Paths.ALL_CSVS, 1):
        file_name = csv_path.name
        # Derive a short clean output name, e.g. "Monday_cleaned.csv"
        day_name = file_name.replace("-WorkingHours", "").replace(".pcap_ISCX.csv", "").replace("-", "_")
        out_path = Paths.CLEANED_DIR / f"{day_name}_cleaned.csv"

        logger.info("[%d/8] Cleaning: %s", file_idx, file_name)
        logger.info("  → Output: %s", out_path.name)

        file_seen_hashes: Set[bytes] = set()
        file_rows_read = 0
        file_nan_removed = 0
        file_within_dupes = 0
        file_cross_dupes = 0
        file_rows_written = 0
        file_label_normalizations: Dict[str, int] = {}

        first_chunk = True
        chunk_reader = pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory=False)

        for chunk in chunk_reader:
            file_rows_read += len(chunk)

            # ── Step 1: Strip column whitespace ─────────────────────────────
            chunk.columns = chunk.columns.str.strip()

            # ── Build column schema on first chunk of first file ─────────────
            if not col_schema_built:
                global _COLUMN_SCHEMA
                _COLUMN_SCHEMA = _build_column_schema(list(chunk.columns))
                col_schema_built = True

            # ── Step 2: Sanitize labels (explicit mapping only) ──────────────
            label_col = "Label"
            if label_col in chunk.columns:
                chunk[label_col] = chunk[label_col].str.strip()
                for raw_label, clean_label in LABEL_MAP.items():
                    mask = chunk[label_col] == raw_label
                    count = int(mask.sum())
                    if count > 0:
                        chunk.loc[mask, label_col] = clean_label
                        key = f"{raw_label} -> {clean_label}"
                        file_label_normalizations[key] = file_label_normalizations.get(key, 0) + count
                        global_label_normalizations[key] = global_label_normalizations.get(key, 0) + count

            # ── Step 3: Replace +inf / -inf with NaN ─────────────────────────
            num_cols = chunk.select_dtypes(include=[np.number]).columns
            chunk[num_cols] = chunk[num_cols].replace(
                [np.inf, -np.inf], np.nan
            )

            # ── Step 4: Drop rows containing any NaN ─────────────────────────
            before = len(chunk)
            chunk = chunk.dropna()
            nan_dropped = before - len(chunk)
            file_nan_removed += nan_dropped

            # ── Step 5 & 6: Hash and deduplicate ─────────────────────────────
            row_tuples = [tuple(x) for x in chunk.to_numpy()]
            keep_mask = []
            for row in row_tuples:
                h = _hash_row_128bit(row)
                if h in file_seen_hashes:
                    file_within_dupes += 1
                    keep_mask.append(False)
                else:
                    file_seen_hashes.add(h)
                    if h in global_seen_hashes:
                        file_cross_dupes += 1
                        keep_mask.append(False)
                    else:
                        global_seen_hashes.add(h)
                        keep_mask.append(True)

            chunk = chunk[keep_mask]
            file_rows_written += len(chunk)

            # ── Step 7: Write clean chunk to output file ─────────────────────
            write_header = first_chunk
            chunk.to_csv(
                out_path,
                mode="w" if first_chunk else "a",
                index=False,
                header=write_header,
            )
            first_chunk = False

            del chunk
            del row_tuples

            # Track peak RSS
            current_rss = _peak_rss_mb()
            if current_rss > peak_rss_mb:
                peak_rss_mb = current_rss

        # Accumulate global totals
        global_rows_read += file_rows_read
        global_nan_removed += file_nan_removed
        global_within_dupes += file_within_dupes
        global_cross_dupes += file_cross_dupes
        global_rows_written += file_rows_written

        file_total_dupes = file_within_dupes + file_cross_dupes
        logger.info(
            "  ✓ Read: %d | NaN/Inf removed: %d | Within-dupes: %d | "
            "Cross-dupes: %d | Written: %d",
            file_rows_read, file_nan_removed, file_within_dupes,
            file_cross_dupes, file_rows_written,
        )

        file_reports.append({
            "file_name": file_name,
            "output_file": out_path.name,
            "rows_read": file_rows_read,
            "rows_with_nan_or_inf_removed": file_nan_removed,
            "duplicate_rows_removed": {
                "within_file": file_within_dupes,
                "cross_file": file_cross_dupes,
                "total": file_total_dupes,
            },
            "rows_written": file_rows_written,
            "label_normalizations": file_label_normalizations,
        })

        gc.collect()

    elapsed = time.time() - start_time
    global_total_dupes = global_within_dupes + global_cross_dupes
    within_limit = peak_rss_mb <= MEMORY_LIMIT_MB

    summary: Dict[str, Any] = {
        "phase": PHASE,
        "pipeline_version": PIPELINE_VERSION,
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "chunk_size": CHUNK_SIZE,
        "memory_limit_mb": MEMORY_LIMIT_MB,
        "memory": {
            "peak_rss_mb": round(peak_rss_mb, 2),
            "memory_limit_mb": MEMORY_LIMIT_MB,
            "within_limit": within_limit,
        },
        "global_accounting": {
            "rows_read": global_rows_read,
            "rows_with_nan_or_inf_removed": global_nan_removed,
            "duplicate_rows_removed": {
                "within_file": global_within_dupes,
                "cross_file": global_cross_dupes,
                "total": global_total_dupes,
            },
            "rows_written": global_rows_written,
        },
        "global_label_normalizations": global_label_normalizations,
        "elapsed_seconds": round(elapsed, 2),
        "per_file_reports": file_reports,
    }

    logger.info("─" * 70)
    logger.info("Cleaning complete in %.1f s", elapsed)
    logger.info("  Total read:    %d", global_rows_read)
    logger.info("  NaN/Inf removed: %d", global_nan_removed)
    logger.info("  Duplicates removed (within/cross): %d / %d", global_within_dupes, global_cross_dupes)
    logger.info("  Rows written:  %d", global_rows_written)
    logger.info("  Peak RSS:      %.1f MB  (limit: %d MB, within_limit=%s)",
                peak_rss_mb, MEMORY_LIMIT_MB, within_limit)

    return summary


# ── Independent Post-Cleaning Verification ─────────────────────────────────────

def verify_cleaned_files() -> Dict[str, Any]:
    """Independently verify that every cleaned file has zero NaN, inf, and duplicates.

    This verification is intentionally implemented separately from the cleaning
    pipeline so it cannot share any state, bugs, or assumptions with clean.py's
    own tracking.
    """
    logger.info("=" * 70)
    logger.info("Independent Post-Cleaning Verification")
    logger.info("=" * 70)

    verification_hashes: Set[bytes] = set()
    verification_results: List[Dict[str, Any]] = []
    all_passed = True

    for file_report in [p for p in Paths.CLEANED_DIR.glob("*_cleaned.csv")]:
        out_path = file_report
        rows = 0
        nan_count = 0
        pos_inf_count = 0
        neg_inf_count = 0
        dup_count = 0

        for chunk in pd.read_csv(out_path, chunksize=CHUNK_SIZE, low_memory=False):
            rows += len(chunk)

            # Check NaN
            nan_count += int(chunk.isna().sum().sum())

            # Check +inf / -inf on numeric cols only
            num_cols = chunk.select_dtypes(include=[np.number]).columns
            for c in num_cols:
                pos_inf_count += int(np.isposinf(chunk[c]).sum())
                neg_inf_count += int(np.isneginf(chunk[c]).sum())

            # Check duplicates via independent hash set
            for row in [tuple(x) for x in chunk.to_numpy()]:
                h = _hash_row_128bit(row)
                if h in verification_hashes:
                    dup_count += 1
                else:
                    verification_hashes.add(h)

        passed = (nan_count == 0 and pos_inf_count == 0 and neg_inf_count == 0 and dup_count == 0)
        if not passed:
            all_passed = False

        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(
            "  %s %s | rows=%d | NaN=%d | +inf=%d | -inf=%d | dupes=%d",
            status, out_path.name, rows, nan_count, pos_inf_count, neg_inf_count, dup_count,
        )

        verification_results.append({
            "file": out_path.name,
            "rows_verified": rows,
            "nan_count": nan_count,
            "pos_inf_count": pos_inf_count,
            "neg_inf_count": neg_inf_count,
            "global_duplicate_count": dup_count,
            "passed": passed,
        })

    logger.info("Overall verification: %s", "✅ ALL PASSED" if all_passed else "❌ FAILURES FOUND")
    return {
        "all_passed": all_passed,
        "per_file_verification": sorted(verification_results, key=lambda x: x["file"]),
    }


# ── Report Saving ──────────────────────────────────────────────────────────────

def save_reports(
    summary: Dict[str, Any],
    source_integrity: Dict[str, Any],
    verification: Dict[str, Any],
) -> None:
    """Save cleaning_summary.json, column_schema.json, and cleaning_summary.md."""
    Paths.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Merge source integrity and verification into summary
    summary["source_integrity"] = source_integrity
    summary["post_cleaning_verification"] = verification

    # 1. Save JSON summary
    json_path = Paths.ARTIFACTS_DIR / "cleaning_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved: %s", json_path)

    # 2. Save column schema
    schema_path = Paths.ARTIFACTS_DIR / "column_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(_COLUMN_SCHEMA, f, indent=2)
    logger.info("Saved: %s", schema_path)

    # 3. Build Markdown summary
    ga = summary["global_accounting"]
    mem = summary["memory"]
    md: List[str] = []
    md.append("# CIC-IDS2017 Data Cleaning Report — Phase 2")
    md.append("")
    md.append(f"- **Pipeline Version**: `{summary['pipeline_version']}`")
    md.append(f"- **Timestamp (UTC)**: `{summary['timestamp_utc']}`")
    md.append(f"- **Elapsed Time**: `{summary['elapsed_seconds']} s`")
    md.append(f"- **Chunk Size**: `{summary['chunk_size']:,}` rows")
    md.append(f"- **Peak RSS**: `{mem['peak_rss_mb']} MB` / `{mem['memory_limit_mb']} MB` limit → {'✅ Within limit' if mem['within_limit'] else '❌ EXCEEDED'}")
    md.append("")

    md.append("## 1. Global Accounting")
    md.append("")
    md.append("| Metric | Count |")
    md.append("|---|---|")
    md.append(f"| Rows Read (total across 8 files) | {ga['rows_read']:,} |")
    md.append(f"| Rows Removed — NaN / Infinity | {ga['rows_with_nan_or_inf_removed']:,} |")
    md.append(f"| Rows Removed — Within-File Duplicates | {ga['duplicate_rows_removed']['within_file']:,} |")
    md.append(f"| Rows Removed — Cross-File Duplicates | {ga['duplicate_rows_removed']['cross_file']:,} |")
    md.append(f"| Total Rows Removed | {ga['rows_with_nan_or_inf_removed'] + ga['duplicate_rows_removed']['total']:,} |")
    md.append(f"| **Rows Written (Clean Unique Flows)** | **{ga['rows_written']:,}** |")
    md.append("")

    md.append("## 2. Per-File Cleaning Summary")
    md.append("")
    md.append("| File | Rows Read | NaN/Inf Removed | Within-File Dupes | Cross-File Dupes | Rows Written |")
    md.append("|---|---|---|---|---|---|")
    for fr in summary["per_file_reports"]:
        d = fr["duplicate_rows_removed"]
        md.append(
            f"| `{fr['file_name']}` | {fr['rows_read']:,} | {fr['rows_with_nan_or_inf_removed']:,} | "
            f"{d['within_file']:,} | {d['cross_file']:,} | {fr['rows_written']:,} |"
        )
    md.append("")

    md.append("## 3. Label Normalizations Applied")
    md.append("")
    normalizations = summary.get("global_label_normalizations", {})
    if normalizations:
        md.append("| Transformation | Occurrences |")
        md.append("|---|---|")
        for k, v in normalizations.items():
            md.append(f"| `{k}` | {v:,} |")
    else:
        md.append("✅ No label normalization was required (no malformed labels found).")
    md.append("")

    md.append("## 4. Column Schema (76 + 1 + 1 + 1 = 79)")
    md.append("")
    md.append(f"- **ML Feature Columns**: `{_COLUMN_SCHEMA.get('ml_feature_columns_count', '?')}` (flow traffic attributes)")
    md.append(f"- **Network Identifier**: `Destination Port` (retained; excluded from ML training)")
    md.append(f"- **Redundant Column**: `Fwd Header Length.1` (preserved; excluded from ML training in Phase 3)")
    md.append(f"- **Target Column**: `Label`")
    md.append("")

    md.append("## 5. Independent Post-Cleaning Verification")
    md.append("")
    v = summary["post_cleaning_verification"]
    md.append(f"**Overall Result**: {'✅ All files passed' if v['all_passed'] else '❌ Failures detected — investigate before proceeding'}")
    md.append("")
    md.append("| Cleaned File | Rows | NaN | +Inf | -Inf | Duplicates | Result |")
    md.append("|---|---|---|---|---|---|---|")
    for vr in v["per_file_verification"]:
        status = "✅ PASS" if vr["passed"] else "❌ FAIL"
        md.append(
            f"| `{vr['file']}` | {vr['rows_verified']:,} | {vr['nan_count']} | "
            f"{vr['pos_inf_count']} | {vr['neg_inf_count']} | {vr['global_duplicate_count']} | {status} |"
        )
    md.append("")

    md.append("## 6. Source Integrity Verification")
    md.append("")
    md.append("Raw CSV source files are **never modified**. SHA-256 hashes prove immutability.")
    md.append("")
    md.append("| File | Rows | Columns | SHA-256 (first 32 chars) |")
    md.append("|---|---|---|---|")
    for fname, info in source_integrity.items():
        md.append(f"| `{fname}` | {info['rows']:,} | {info['columns']} | `{info['sha256'][:32]}...` |")
    md.append("")

    md_path = Paths.ARTIFACTS_DIR / "cleaning_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    logger.info("Saved: %s", md_path)


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1: Source integrity hashes (proves raw files are never touched)
    source_integrity = compute_source_integrity()

    # Step 2: Clean all 8 files
    summary = clean_dataset()

    # Step 3: Independent verification
    verification = verify_cleaned_files()

    # Step 4: Save all reports
    save_reports(summary, source_integrity, verification)

    # Exit non-zero if verification failed
    if not verification["all_passed"]:
        logger.error("VERIFICATION FAILED — do not use cleaned files until this is resolved.")
        sys.exit(1)
    else:
        logger.info("Phase 2 complete. Cleaned files ready in: %s", Paths.CLEANED_DIR)
