"""
ml_service/src/correlation/generate_flags.py
─────────────────────────────────────────────
Phase 8: Precompute ML and Security Rule Ingestion Flags.

Generates unified per-flow detection flags across all 2,520,798 cleaned flows:
- ML predictions: XGBoost baseline (models/xgb_baseline_category_v1.pkl).
  * Evaluated on test partition (504,160 rows) with strict parity assertion against Phase 5.
  * Evaluated on train partition (2,016,638 rows) for supplementary full-day evaluation.
- Security Rule: Rule R104_PORT_BRUTE_FORCE evaluated on raw CSV metrics.
  * Strict parity assertion against Phase 7 calibration and test results (2,191 test hits).
- Output: artifacts/correlation/ml_flags_v1.parquet
  * Schema: [row_id: int64, predicted_class: str, confidence: float32, r104_flag: bool, is_flagged: bool]
  * is_flagged = (predicted_class != 'BENIGN') | r104_flag.

Hardware constraint: peak RSS strictly < 500 MB.
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
from typing import Any, List

import joblib
import numpy as np
import pandas as pd
import psutil  # type: ignore[import-untyped]
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.paths import Paths
from ml_service.src.utils.memory_guard import assert_memory_headroom

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_flags")

CHUNK_SIZE = 50_000
MEMORY_LIMIT_MB = 500
VERSION_TAG = "v1"


def get_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def main() -> None:
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("Phase 8 Step 2 -- ML & Rule Flag Generation (Streaming)")
    logger.info("=" * 70)
    assert_memory_headroom(min_free_mb=200)

    # 1. Load LabelEncoder & Train Mask
    logger.info("Loading label encoder and train mask...")
    le_path = Paths.SPLITS_DIR / f"label_encoder_{VERSION_TAG}.pkl"
    with open(le_path, "rb") as f:
        le = pickle.load(f)
    classes: List[str] = [str(c) for c in le.classes_]
    benign_idx = classes.index("BENIGN")
    logger.info("Categories (%d): %s (BENIGN index: %d)", len(classes), classes, benign_idx)

    mask_path = Paths.SPLITS_DIR / f"is_train_mask_{VERSION_TAG}.npy"
    is_train_mask: np.ndarray = np.load(mask_path)
    n_total = len(is_train_mask)
    assert n_total == 2_520_798, f"Unexpected mask length: {n_total}"
    train_indices = np.where(is_train_mask)[0]
    test_indices = np.where(~is_train_mask)[0]
    n_train = len(train_indices)
    n_test = len(test_indices)
    logger.info("Mask loaded: %d total rows (train: %d, test: %d)", n_total, n_train, n_test)

    # 2. Load XGBoost Model
    xgb_model_path = Paths.MODELS_DIR / f"xgb_baseline_category_{VERSION_TAG}.pkl"
    logger.info("Loading XGBoost booster: %s", xgb_model_path.name)
    booster: xgb.Booster = joblib.load(xgb_model_path)
    logger.info("Booster loaded successfully. RSS: %.2f MB", get_rss_mb())

    # Preallocate global prediction arrays
    pred_class_idx = np.zeros(n_total, dtype=np.int8)
    confidence = np.zeros(n_total, dtype=np.float32)

    # 3. XGBoost Inference on Test Set (504,160 rows)
    logger.info("Running XGBoost inference on TEST split (X_test_%s.parquet)...", VERSION_TAG)
    x_test_path = Paths.SPLITS_DIR / f"X_test_{VERSION_TAG}.parquet"
    test_pq = pq.ParquetFile(x_test_path)
    
    test_preds_list: List[np.ndarray] = []
    test_confs_list: List[np.ndarray] = []
    
    for batch in test_pq.iter_batches(batch_size=CHUNK_SIZE):
        chunk_arr = batch.to_pandas().to_numpy(dtype=np.float32)
        dtest = xgb.DMatrix(chunk_arr)
        probs = booster.predict(dtest)
        b_preds = np.argmax(probs, axis=1).astype(np.int8)
        b_confs = np.max(probs, axis=1).astype(np.float32)
        test_preds_list.append(b_preds)
        test_confs_list.append(b_confs)
        del dtest, probs, chunk_arr, batch
        gc.collect()

    test_preds = np.concatenate(test_preds_list)
    test_confs = np.concatenate(test_confs_list)
    del test_preds_list, test_confs_list
    assert len(test_preds) == n_test, f"Test count mismatch: {len(test_preds)} vs {n_test}"

    # Verify test accuracy against y_test_category
    y_test_path = Paths.SPLITS_DIR / f"y_test_category_{VERSION_TAG}.parquet"
    y_test = pq.read_table(y_test_path).to_pandas().iloc[:, 0].to_numpy()
    test_acc = float(np.mean(test_preds == y_test))
    logger.info("XGBoost Test Accuracy: %.5f (Expected ~0.99858 from Phase 5)", test_acc)
    assert abs(test_acc - 0.99858) < 1e-4, f"Test accuracy discrepancy: {test_acc}"

    # Verify Phase 5 confusion numbers
    test_is_attack_true = y_test != benign_idx
    test_is_attack_pred = test_preds != benign_idx
    tp = int(np.sum(test_is_attack_true & test_is_attack_pred))
    fp = int(np.sum((~test_is_attack_true) & test_is_attack_pred))
    fn = int(np.sum(test_is_attack_true & (~test_is_attack_pred)))
    tn = int(np.sum((~test_is_attack_true) & (~test_is_attack_pred)))
    logger.info("Test Confusion Matrix Parity: TP=%d, FP=%d, FN=%d, TN=%d", tp, fp, fn, tn)
    assert tp == 84_979, f"TP mismatch: {tp} vs 84979"
    assert fp == 399, f"FP mismatch: {fp} vs 399"
    assert fn == 169, f"FN mismatch: {fn} vs 169"
    assert tn == 418_613, f"TN mismatch: {tn} vs 418613"
    logger.info("Phase 5 Test Set Parity Assertion PASSED!")

    # Assign test predictions
    pred_class_idx[test_indices] = test_preds
    confidence[test_indices] = test_confs
    del test_preds, test_confs, y_test
    gc.collect()

    # 4. XGBoost Inference on Train Set (2,016,638 rows)
    logger.info("Running XGBoost inference on TRAIN split (X_train_%s.parquet)...", VERSION_TAG)
    x_train_path = Paths.SPLITS_DIR / f"X_train_{VERSION_TAG}.parquet"
    train_pq = pq.ParquetFile(x_train_path)

    train_preds_list: List[np.ndarray] = []
    train_confs_list: List[np.ndarray] = []

    for batch in train_pq.iter_batches(batch_size=CHUNK_SIZE):
        chunk_arr = batch.to_pandas().to_numpy(dtype=np.float32)
        dtrain = xgb.DMatrix(chunk_arr)
        probs = booster.predict(dtrain)
        b_preds = np.argmax(probs, axis=1).astype(np.int8)
        b_confs = np.max(probs, axis=1).astype(np.float32)
        train_preds_list.append(b_preds)
        train_confs_list.append(b_confs)
        del dtrain, probs, chunk_arr, batch
        gc.collect()

    train_preds = np.concatenate(train_preds_list)
    train_confs = np.concatenate(train_confs_list)
    del train_preds_list, train_confs_list
    assert len(train_preds) == n_train, f"Train count mismatch: {len(train_preds)} vs {n_train}"

    # Assign train predictions
    pred_class_idx[train_indices] = train_preds
    confidence[train_indices] = train_confs
    del train_preds, train_confs, booster
    gc.collect()
    logger.info("ML inference complete for all %d rows. RSS: %.2f MB", n_total, get_rss_mb())

    # 5. Evaluate Rule R104 on Cleaned CSVs
    logger.info("Evaluating Rule R104_PORT_BRUTE_FORCE across cleaned CSVs...")
    r104_chunks: List[np.ndarray] = []
    rule_req_cols = ["Destination Port", "min_seg_size_forward", "Total Backward Packets"]

    for csv_path in Paths.CLEANED_CSVS:
        for chunk in pd.read_csv(csv_path, usecols=rule_req_cols, chunksize=CHUNK_SIZE):
            chunk.columns = chunk.columns.str.strip()
            ports = chunk["Destination Port"].to_numpy()
            min_seg = chunk["min_seg_size_forward"].to_numpy()
            bwd_pkts = chunk["Total Backward Packets"].to_numpy()

            port_match = np.isin(ports, [21, 22])
            hit = port_match & (min_seg == 32) & (bwd_pkts >= 5)
            r104_chunks.append(hit)
            del chunk, ports, min_seg, bwd_pkts, hit

    r104_flags = np.concatenate(r104_chunks).astype(bool)
    del r104_chunks
    gc.collect()
    assert len(r104_flags) == n_total, f"R104 flags count mismatch: {len(r104_flags)} vs {n_total}"

    # Parity check on test split with Phase 7
    r104_test_hits = int(np.sum(r104_flags[test_indices]))
    logger.info("R104 hits on TEST split: %d (Expected: 2191 [1399 TP + 792 FP])", r104_test_hits)
    assert r104_test_hits == 2_191, f"R104 test hits mismatch: {r104_test_hits} vs 2191"
    logger.info("Phase 7 Rule Engine Parity Assertion PASSED!")

    # 6. Build Final Parquet Table
    logger.info("Constructing output parquet table...")
    row_ids = np.arange(n_total, dtype=np.int64)
    predicted_class_names = [classes[i] for i in pred_class_idx]
    
    is_ml_attack = pred_class_idx != benign_idx
    is_flagged = is_ml_attack | r104_flags

    # Verification on test split
    test_flagged_count = int(np.sum(is_flagged[test_indices]))
    logger.info("Total test flows flagged (ML | R104): %d (Expected: 86170)", test_flagged_count)
    assert test_flagged_count == 86_170, f"Flagged test mismatch: {test_flagged_count} vs 86170"

    total_flagged_count = int(np.sum(is_flagged))
    logger.info("Total flows flagged across all 2,520,798: %d (%.2f%%)", total_flagged_count, 100 * total_flagged_count / n_total)

    # Convert to Arrow Table with explicit types
    pa_schema = pa.schema([
        ("row_id", pa.int64()),
        ("predicted_class", pa.string()),
        ("confidence", pa.float32()),
        ("r104_flag", pa.bool_()),
        ("is_flagged", pa.bool_()),
    ])

    arrow_table = pa.Table.from_arrays(
        [
            pa.array(row_ids, type=pa.int64()),
            pa.array(predicted_class_names, type=pa.string()),
            pa.array(confidence, type=pa.float32()),
            pa.array(r104_flags, type=pa.bool_()),
            pa.array(is_flagged, type=pa.bool_()),
        ],
        schema=pa_schema,
    )

    out_dir = Paths.ARTIFACTS_DIR / "correlation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ml_flags_{VERSION_TAG}.parquet"

    logger.info("Writing output parquet: %s...", out_path.name)
    pq.write_table(arrow_table, out_path, compression="snappy")
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info("Wrote %s (%.2f MB, %d rows)", out_path.name, file_size_mb, len(arrow_table))

    peak_rss = get_rss_mb()
    elapsed = time.time() - t0
    logger.info("ML & Rule flags generation finished in %.2f s | Peak RSS: %.2f MB", elapsed, peak_rss)
    assert peak_rss < MEMORY_LIMIT_MB, f"Peak RSS {peak_rss:.2f} MB exceeded budget {MEMORY_LIMIT_MB} MB"


if __name__ == "__main__":
    main()

