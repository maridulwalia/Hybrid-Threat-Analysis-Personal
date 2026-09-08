"""
ml_service/src/feature_engineer.py
────────────────────────────────────
Phase 3: Feature Engineering & Train/Test Split

Memory-safe, 3-pass streaming pipeline that reads the 8 Phase 2 cleaned CSVs
and produces versioned Parquet split artifacts in artifacts/splits/.

Design Guarantees
─────────────────
1. No full DataFrame in RAM at any point — each pass streams 50,000-row chunks.
2. Pass 1 (labels): only the Label column is read. The full 2.5 M label array
   (~50 MB) is the largest object held in memory during this pass.
3. Pass 2 (scaler fit): only the 76 ML feature columns are read, and only
   train-row chunks are passed to StandardScaler.partial_fit(). Peak RSS ~200 MB.
4. Pass 3 (transform & write): features read in chunks, split train/test, scaled,
   written incrementally via pyarrow.parquet.ParquetWriter. Peak RSS ~300 MB.
5. All feature columns cast to float32 (AGENTS.md Phase 3 constraint).
6. Excluded from ML feature set: Destination Port (identifier), Fwd Header
   Length.1 (redundant duplicate). Both present in cleaned CSVs but never in
   Parquet outputs.
7. Stratified split on attack-category label (finest grain), 80/20, seed=42.
8. All outputs versioned (v1 suffix) — never overwrite silently.

Outputs (artifacts/splits/)
────────────────────────────
  X_train_v1.parquet          76 float32 scaled features, train rows
  X_test_v1.parquet           76 float32 scaled features, test rows
  y_train_binary_v1.parquet   binary label (0=BENIGN, 1=ATTACK), train
  y_test_binary_v1.parquet    binary label, test
  y_train_category_v1.parquet integer attack-category label, train
  y_test_category_v1.parquet  integer attack-category label, test
  label_encoder_v1.pkl        fitted sklearn LabelEncoder (int -> class name)
  scaler_v1.pkl               fitted sklearn StandardScaler (train only)
  is_train_mask_v1.npy        bool array shape (2520798,) -- row assignments
  split_summary_v1.json       counts, class distribution, memory, elapsed time
  split_summary_v1.md         human-readable markdown report

Run
───
  python -m ml_service.src.feature_engineer
"""

from __future__ import annotations

import gc
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Project root on sys.path ───────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.paths import Paths
from ml_service.src.utils.memory_guard import assert_memory_headroom

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
PIPELINE_VERSION = "1.0"
PHASE = "Phase 3"
CHUNK_SIZE = 50_000
MEMORY_LIMIT_MB = 500
TEST_FRACTION = 0.20
RANDOM_STATE = 42
VERSION_TAG = "v1"

# Columns excluded from ML feature set (per column_schema.json):
#   Destination Port  -- identifier; must not be memorised by the model
#   Fwd Header Length.1 -- exact duplicate of Fwd Header Length
#   Label             -- target column
EXCLUDE_COLS = {"Destination Port", "Fwd Header Length.1", "Label"}

BENIGN_LABEL = "BENIGN"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _peak_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def _load_feature_cols() -> list[str]:
    """Load the 76 ML feature column names from column_schema.json.

    Falls back to deriving them from the first cleaned CSV header if the
    schema file is missing (should not happen post-Phase 2).
    """
    schema_path = Paths.ARTIFACTS_DIR / "column_schema.json"
    if schema_path.exists():
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)
        cols: list[str] = schema["ml_feature_columns"]
        logger.info("Loaded %d ML feature columns from column_schema.json", len(cols))
        return cols

    logger.warning("column_schema.json not found -- deriving feature cols from CSV header")
    first_chunk = next(pd.read_csv(Paths.CLEANED_CSVS[0], nrows=0, chunksize=1,
                                   skipinitialspace=True))
    all_cols = list(first_chunk.columns.str.strip())
    feature_cols = [c for c in all_cols if c not in EXCLUDE_COLS]
    logger.info("Derived %d ML feature columns from CSV header", len(feature_cols))
    return feature_cols


# ── Pass 1: Label Collection & Split Assignment ────────────────────────────────

def collect_labels() -> dict[str, Any]:
    """Stream all 8 cleaned CSVs, read only the Label column.

    This pass reads ONLY the Label column (1 of 79) to minimise RAM usage.
    Feature columns are NOT needed here — they are read in Passes 2 and 3.

    Returns a dict with keys:
        all_labels_raw   np.ndarray[str]   shape (N,)
        binary_labels    np.ndarray[int8]  shape (N,)  0=BENIGN 1=ATTACK
        category_labels  np.ndarray[int16] shape (N,)  integer-encoded category
        label_encoder    fitted LabelEncoder
        is_train_mask    np.ndarray[bool]  shape (N,)
        train_indices    np.ndarray[int]
        test_indices     np.ndarray[int]
        n_total, n_train, n_test  int
        classes          list[str]
    """
    logger.info("=" * 70)
    logger.info("Phase 3 Pass 1 -- Label Collection & Split Assignment")
    logger.info("=" * 70)
    assert_memory_headroom(min_free_mb=200)

    label_chunks: list[np.ndarray] = []
    expected_rows: list[int] = []          # per-file row counts for post-loop assertion

    for csv_path in Paths.CLEANED_CSVS:
        file_rows = 0
        _label_dtype: Any = {"Label": str}
        for chunk in pd.read_csv(
            csv_path,
            usecols=["Label"],
            chunksize=CHUNK_SIZE,
            skipinitialspace=True,
            dtype=_label_dtype,
        ):
            chunk.columns = chunk.columns.str.strip()
            # to_numpy(dtype=object) gives Pyright a concrete DTypeLike; str.strip() already
            # guarantees Python str values, so the object array holds plain str elements.
            labels: np.ndarray = chunk["Label"].str.strip().to_numpy(dtype=object)
            label_chunks.append(labels)
            file_rows += len(labels)

        expected_rows.append(file_rows)
        logger.info("  %s -> %d rows", csv_path.name, file_rows)

    all_labels_raw: np.ndarray = np.concatenate(label_chunks)
    del label_chunks
    gc.collect()

    # Verify total matches sum of per-file counts (catches mid-stream truncation)
    assert len(all_labels_raw) == sum(expected_rows), (
        f"Label count mismatch: concatenated {len(all_labels_raw)} "
        f"but per-file sum is {sum(expected_rows)}"
    )
    logger.info("Total labels collected: %d (matches per-file sum OK)", len(all_labels_raw))

    # Binary encoding: 0 = BENIGN, 1 = ATTACK
    # np.where returns a typed ndarray so Pyright does not lose track through a bool comparison.
    binary_labels: np.ndarray = np.where(
        all_labels_raw == BENIGN_LABEL, np.int8(0), np.int8(1)
    ).astype(np.int8)
    logger.info(
        "Binary -- BENIGN: %d | ATTACK: %d",
        int((binary_labels == 0).sum()),
        int((binary_labels == 1).sum()),
    )

    # Category encoding via LabelEncoder
    le = LabelEncoder()
    # Annotate explicitly: fit_transform stubs return ArrayLike; we know it's ndarray[int].
    category_labels: np.ndarray = np.asarray(
        le.fit_transform(all_labels_raw), dtype=np.int16
    )
    # Cast le.classes_ to plain Python str immediately — numpy strings (np.str_)
    # are not reliably JSON-serializable and produce ugly repr in logs.
    assert le.classes_ is not None, "LabelEncoder.classes_ is None after fit"
    classes: list[str] = [str(c) for c in le.classes_]
    logger.info("Attack categories (%d): %s", len(classes), classes)

    # Stratified 80/20 split on category labels
    indices = np.arange(len(all_labels_raw))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_FRACTION,
        random_state=RANDOM_STATE,
        stratify=category_labels,
    )
    logger.info(
        "Split: train=%d (%.1f%%) | test=%d (%.1f%%)",
        len(train_idx), 100 * len(train_idx) / len(indices),
        len(test_idx),  100 * len(test_idx)  / len(indices),
    )

    # Boolean mask (True = train row)
    is_train_mask = np.zeros(len(all_labels_raw), dtype=bool)
    is_train_mask[train_idx] = True

    # Sanity checks
    assert is_train_mask.sum() + (~is_train_mask).sum() == len(all_labels_raw)
    train_cats = set(category_labels[is_train_mask])
    test_cats  = set(category_labels[~is_train_mask])
    missing_in_test = train_cats - test_cats
    if missing_in_test:
        missing_names = [classes[i] for i in missing_in_test]
        logger.warning(
            "Stratification warning -- categories absent from test set: %s",
            missing_names,
        )
    else:
        logger.info("Stratification sanity check: all categories present in both splits OK")

    return {
        "all_labels_raw":  all_labels_raw,
        "binary_labels":   binary_labels,
        "category_labels": category_labels,
        "label_encoder":   le,
        "is_train_mask":   is_train_mask,
        "train_indices":   train_idx,
        "test_indices":    test_idx,
        "n_total":         len(all_labels_raw),
        "n_train":         len(train_idx),
        "n_test":          len(test_idx),
        "classes":         classes,
    }


def save_label_artifacts(label_data: dict[str, Any]) -> None:
    """Write y Parquet files, label encoder, and is_train_mask to splits dir."""
    Paths.SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    is_train = label_data["is_train_mask"]
    binary   = label_data["binary_labels"]
    category = label_data["category_labels"]
    le       = label_data["label_encoder"]

    def _write_y(arr: np.ndarray, col: str, path: Path) -> None:
        pq.write_table(pa.table({col: arr}), path, compression="snappy")
        logger.info("Saved: %s (%d rows)", path.name, len(arr))

    _write_y(binary[is_train],    "binary_label",   Paths.SPLITS_DIR / f"y_train_binary_{VERSION_TAG}.parquet")
    _write_y(binary[~is_train],   "binary_label",   Paths.SPLITS_DIR / f"y_test_binary_{VERSION_TAG}.parquet")
    _write_y(category[is_train],  "category_label", Paths.SPLITS_DIR / f"y_train_category_{VERSION_TAG}.parquet")
    _write_y(category[~is_train], "category_label", Paths.SPLITS_DIR / f"y_test_category_{VERSION_TAG}.parquet")

    le_path = Paths.SPLITS_DIR / f"label_encoder_{VERSION_TAG}.pkl"
    with open(le_path, "wb") as f:
        pickle.dump(le, f, protocol=5)
    logger.info("Saved: %s (classes: %s)", le_path.name, list(le.classes_))

    mask_path = Paths.SPLITS_DIR / f"is_train_mask_{VERSION_TAG}.npy"
    np.save(mask_path, is_train)
    logger.info("Saved: %s (%d rows, %d train)", mask_path.name, len(is_train), int(is_train.sum()))


# ── Pass 2: Scaler Fit on Train Rows ──────────────────────────────────────────

def fit_scaler(
    feature_cols: list[str],
    is_train_mask: np.ndarray,
) -> StandardScaler:
    """Stream all 8 cleaned CSVs and partial_fit a StandardScaler on train rows.

    Only feature columns are read. Only rows where is_train_mask is True are
    passed to partial_fit. Peak RSS ~200 MB.
    """
    logger.info("=" * 70)
    logger.info("Phase 3 Pass 2 -- Scaler Fit (train rows only)")
    logger.info("=" * 70)
    assert_memory_headroom(min_free_mb=200)

    scaler = StandardScaler()
    global_row_offset = 0
    chunks_fit = 0

    for csv_path in Paths.CLEANED_CSVS:
        _feat_dtype: Any = {c: "float32" for c in feature_cols}
        for chunk in pd.read_csv(
            csv_path,
            usecols=feature_cols,
            chunksize=CHUNK_SIZE,
            skipinitialspace=True,
            dtype=_feat_dtype,
        ):
            chunk.columns = chunk.columns.str.strip()
            chunk_len = len(chunk)
            chunk_mask = is_train_mask[global_row_offset: global_row_offset + chunk_len]
            chunk_arr: np.ndarray = chunk.to_numpy(dtype=np.float32)
            train_chunk = chunk_arr[chunk_mask]

            if len(train_chunk) > 0:
                scaler.partial_fit(train_chunk)
                chunks_fit += 1

            global_row_offset += chunk_len
            del chunk, train_chunk

        gc.collect()
        logger.info("  Fit progress: %s done (global offset: %d)", csv_path.name, global_row_offset)

    assert global_row_offset == is_train_mask.shape[0], (
        f"Row count mismatch: streamed {global_row_offset} "
        f"but mask has {is_train_mask.shape[0]}"
    )
    # Verify scaler was actually fitted; mean_ is None before partial_fit is called.
    scaler_mean: np.ndarray | None = getattr(scaler, "mean_", None)
    assert scaler_mean is not None and scaler_mean.shape == (len(feature_cols),), \
        "Scaler fit failed -- mean_ is None or shape mismatch"
    logger.info(
        "Scaler fit complete: %d chunks, %d features, mean range [%.4f, %.4f]",
        chunks_fit, len(feature_cols),
        float(scaler_mean.min()), float(scaler_mean.max()),
    )

    scaler_path = Paths.SPLITS_DIR / f"scaler_{VERSION_TAG}.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f, protocol=5)
    logger.info("Saved: %s", scaler_path.name)
    return scaler


# ── Pass 3: Transform & Write Parquet ─────────────────────────────────────────

def transform_and_write(
    feature_cols: list[str],
    is_train_mask: np.ndarray,
    scaler: StandardScaler,
) -> dict[str, int]:
    """Stream all 8 cleaned CSVs, scale features, write X Parquet files.

    Uses pyarrow.parquet.ParquetWriter for incremental streaming writes --
    never holds the full train or test matrix in RAM simultaneously.

    Returns dict: x_train_rows, x_test_rows, nan_in_scaled.
    """
    logger.info("=" * 70)
    logger.info("Phase 3 Pass 3 -- Transform & Write Parquet")
    logger.info("=" * 70)
    assert_memory_headroom(min_free_mb=200)

    x_train_path = Paths.SPLITS_DIR / f"X_train_{VERSION_TAG}.parquet"
    x_test_path  = Paths.SPLITS_DIR / f"X_test_{VERSION_TAG}.parquet"
    pa_schema    = pa.schema([(col, pa.float32()) for col in feature_cols])

    x_train_rows = 0
    x_test_rows  = 0
    nan_count    = 0
    global_row_offset = 0

    with (
        pq.ParquetWriter(x_train_path, schema=pa_schema, compression="snappy") as train_writer,
        pq.ParquetWriter(x_test_path,  schema=pa_schema, compression="snappy") as test_writer,
    ):
        for csv_path in Paths.CLEANED_CSVS:
            _feat_dtype: Any = {c: "float32" for c in feature_cols}
            for chunk in pd.read_csv(
                csv_path,
                usecols=feature_cols,
                chunksize=CHUNK_SIZE,
                skipinitialspace=True,
                dtype=_feat_dtype,
            ):
                chunk.columns = chunk.columns.str.strip()
                chunk_len  = len(chunk)
                chunk_mask = is_train_mask[global_row_offset: global_row_offset + chunk_len]

                # Scale; sklearn returns float64 -- cast back to float32.
                # Break into two steps so Pyright tracks ndarray through the chain.
                scaled_f64: np.ndarray = np.asarray(
                    scaler.transform(chunk.to_numpy(dtype=np.float32))
                )
                scaled: np.ndarray = scaled_f64.astype(np.float32)
                nan_count += int(np.isnan(scaled).sum())

                train_arr = scaled[chunk_mask]
                test_arr  = scaled[~chunk_mask]

                if len(train_arr) > 0:
                    train_writer.write_table(
                        pa.table(
                            {col: train_arr[:, i] for i, col in enumerate(feature_cols)},
                            schema=pa_schema,
                        )
                    )
                    x_train_rows += len(train_arr)

                if len(test_arr) > 0:
                    test_writer.write_table(
                        pa.table(
                            {col: test_arr[:, i] for i, col in enumerate(feature_cols)},
                            schema=pa_schema,
                        )
                    )
                    x_test_rows += len(test_arr)

                global_row_offset += chunk_len
                del chunk, scaled, train_arr, test_arr

            gc.collect()
            logger.info(
                "  Written: %s | train=%d | test=%d",
                csv_path.name, x_train_rows, x_test_rows,
            )

    assert global_row_offset == is_train_mask.shape[0], (
        f"Row count mismatch in Pass 3: {global_row_offset} vs {is_train_mask.shape[0]}"
    )
    assert nan_count == 0, f"NaN detected in scaled output: {nan_count}"
    logger.info(
        "Pass 3 complete: X_train=%d | X_test=%d | NaN=0 OK",
        x_train_rows, x_test_rows,
    )
    logger.info("  X_train: %s (%.1f MB)", x_train_path.name, x_train_path.stat().st_size / 1e6)
    logger.info("  X_test:  %s (%.1f MB)", x_test_path.name,  x_test_path.stat().st_size / 1e6)
    return {"x_train_rows": x_train_rows, "x_test_rows": x_test_rows, "nan_in_scaled": nan_count}


# ── Summary Report ─────────────────────────────────────────────────────────────

def save_split_summary(
    label_data: dict[str, Any],
    write_result: dict[str, int],
    elapsed_seconds: float,
    peak_rss_mb: float,
) -> None:
    """Write split_summary_v1.json and split_summary_v1.md."""
    Paths.SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    le: LabelEncoder = label_data["label_encoder"]
    binary   = label_data["binary_labels"]
    category = label_data["category_labels"]
    is_train = label_data["is_train_mask"]
    # Guarantee plain Python str — label_data["classes"] is already cast in collect_labels,
    # but re-cast here defensively in case this function is ever called standalone.
    classes: list[str] = [str(c) for c in label_data["classes"]]

    def _class_dist(cat_arr: np.ndarray) -> dict[str, int]:
        # `classes` is list[str], so `name` is already str — no cast needed.
        return {name: int((cat_arr == i).sum()) for i, name in enumerate(classes)}

    train_dist = _class_dist(category[is_train])
    test_dist  = _class_dist(category[~is_train])
    rc = {
        "total": label_data["n_total"],
        "train": label_data["n_train"],
        "test":  label_data["n_test"],
        "x_train_written": write_result["x_train_rows"],
        "x_test_written":  write_result["x_test_rows"],
    }

    artifact_files = {
        f"X_train_{VERSION_TAG}.parquet":          "76 float32 scaled features -- train rows",
        f"X_test_{VERSION_TAG}.parquet":           "76 float32 scaled features -- test rows",
        f"y_train_binary_{VERSION_TAG}.parquet":   "binary label (0=BENIGN, 1=ATTACK) -- train",
        f"y_test_binary_{VERSION_TAG}.parquet":    "binary label -- test",
        f"y_train_category_{VERSION_TAG}.parquet": "integer attack-category label -- train",
        f"y_test_category_{VERSION_TAG}.parquet":  "integer attack-category label -- test",
        f"label_encoder_{VERSION_TAG}.pkl":        "fitted sklearn LabelEncoder",
        f"scaler_{VERSION_TAG}.pkl":               "fitted sklearn StandardScaler (train only)",
        f"is_train_mask_{VERSION_TAG}.npy":        "bool array (N,) -- True = train row",
    }

    summary: dict[str, Any] = {
        "phase": PHASE,
        "pipeline_version": PIPELINE_VERSION,
        "version_tag": VERSION_TAG,
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "parameters": {
            "test_fraction":       TEST_FRACTION,
            "random_state":        RANDOM_STATE,
            "chunk_size":          CHUNK_SIZE,
            "float_precision":     "float32",
            "parquet_compression": "snappy",
            "stratify_on":         "category_label",
        },
        "row_counts": rc,
        "feature_columns_count": 76,
        "label_encoding": {
            "binary":            {"BENIGN": 0, "ATTACK": 1},
            "category_classes":  classes,          # plain list[str] — JSON-safe
            "n_classes":         len(classes),
            "benign_total":      int((binary == 0).sum()),
            "attack_total":      int((binary == 1).sum()),
        },
        "class_distribution": {"train": train_dist, "test": test_dist},
        "memory": {
            "peak_rss_mb":    round(peak_rss_mb, 2),
            "memory_limit_mb": MEMORY_LIMIT_MB,
            "within_limit":   peak_rss_mb <= MEMORY_LIMIT_MB,
        },
        "elapsed_seconds": round(elapsed_seconds, 2),
        "artifacts": artifact_files,
    }

    json_path = Paths.SPLITS_DIR / f"split_summary_{VERSION_TAG}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved: %s", json_path.name)

    # Markdown report
    md: list[str] = []
    md += [
        f"# CIC-IDS2017 Feature Engineering & Split Report -- {PHASE}", "",
        f"- **Pipeline Version**: `{PIPELINE_VERSION}`",
        f"- **Version Tag**: `{VERSION_TAG}`",
        f"- **Timestamp (UTC)**: `{summary['timestamp_utc']}`",
        f"- **Elapsed Time**: `{summary['elapsed_seconds']} s`",
        f"- **Peak RSS**: `{summary['memory']['peak_rss_mb']} MB` / "
        f"`{MEMORY_LIMIT_MB} MB` limit -> "
        f"{'OK Within limit' if summary['memory']['within_limit'] else 'EXCEEDED'}",
        "",
        "## 1. Split Parameters", "",
        "| Parameter | Value |", "|---|---|",
        f"| Test fraction | `{TEST_FRACTION}` (80/20) |",
        f"| Random state | `{RANDOM_STATE}` |",
        "| Stratify on | attack-category label (finest grain) |",
        "| Float precision | `float32` |",
        "| Parquet compression | `snappy` |",
        f"| Chunk size | `{CHUNK_SIZE:,}` rows |",
        "",
        "## 2. Row Counts", "",
        "| Split | Rows | % of Total |", "|---|---|---|",
        f"| Total | {rc['total']:,} | 100.0% |",
        f"| Train | {rc['train']:,} | {100*rc['train']/rc['total']:.1f}% |",
        f"| Test  | {rc['test']:,}  | {100*rc['test']/rc['total']:.1f}% |",
        "",
        "## 3. Label Encoding", "",
        f"- **Binary**: BENIGN -> `0` ({summary['label_encoding']['benign_total']:,} rows), "
        f"ATTACK -> `1` ({summary['label_encoding']['attack_total']:,} rows)",
        f"- **Category classes** ({len(classes)}): "
        + ", ".join(f"`{c}`" for c in classes),
        "",
        "## 4. Class Distribution (Train vs Test)", "",
        "| Class | Train Count | Test Count | Train % | Test % |",
        "|---|---|---|---|---|",
    ]
    for cls in classes:
        tr = train_dist.get(cls, 0)
        te = test_dist.get(cls, 0)
        md.append(
            f"| `{cls}` | {tr:,} | {te:,} | "
            f"{100*tr/rc['train']:.3f}% | {100*te/rc['test']:.3f}% |"
        )
    md += [
        "",
        "## 5. Output Artifacts", "",
        "| File | Size | Description |", "|---|---|---|",
    ]
    for fname, desc in artifact_files.items():
        fpath = Paths.SPLITS_DIR / fname
        size_str = f"{fpath.stat().st_size / 1e6:.1f} MB" if fpath.exists() else "—"
        md.append(f"| `{fname}` | {size_str} | {desc} |")

    md_path = Paths.SPLITS_DIR / f"split_summary_{VERSION_TAG}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    logger.info("Saved: %s", md_path.name)


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start_time = time.time()
    Paths.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    peak_rss: float = _peak_rss_mb()

    logger.info("=" * 70)
    logger.info(
        "Phase 3 -- Feature Engineering & Train/Test Split  (pipeline v%s)",
        PIPELINE_VERSION,
    )
    logger.info("=" * 70)

    feature_cols = _load_feature_cols()
    assert len(feature_cols) == 76, f"Expected 76 feature cols, got {len(feature_cols)}"

    # Pass 1 -- Labels & split mask
    label_data = collect_labels()
    save_label_artifacts(label_data)
    peak_rss = max(peak_rss, _peak_rss_mb())

    # Pass 2 -- Scaler fit
    scaler = fit_scaler(feature_cols, label_data["is_train_mask"])
    peak_rss = max(peak_rss, _peak_rss_mb())

    # Retain only the mask before Pass 3 (free large arrays)
    is_train_mask = label_data["is_train_mask"]
    gc.collect()

    # Pass 3 -- Transform & write
    write_result = transform_and_write(feature_cols, is_train_mask, scaler)
    peak_rss = max(peak_rss, _peak_rss_mb())

    elapsed = time.time() - start_time

    # Summary
    save_split_summary(label_data, write_result, elapsed, peak_rss)

    logger.info("-" * 70)
    logger.info("Phase 3 complete in %.1f s", elapsed)
    logger.info("  Train rows : %d", label_data["n_train"])
    logger.info("  Test rows  : %d", label_data["n_test"])
    logger.info(
        "  Peak RSS   : %.1f MB  (limit: %d MB, within_limit=%s)",
        peak_rss, MEMORY_LIMIT_MB, peak_rss <= MEMORY_LIMIT_MB,
    )
    logger.info("  Artifacts  : %s", Paths.SPLITS_DIR)

    if peak_rss > MEMORY_LIMIT_MB:
        logger.warning(
            "Peak RSS %.1f MB exceeded soft limit %d MB. "
            "Review chunk size or pipeline design before Phase 4.",
            peak_rss, MEMORY_LIMIT_MB,
        )
        sys.exit(1)

    logger.info("Phase 3 OK -- split artifacts ready for Phase 4 (LightGBM RF).")
