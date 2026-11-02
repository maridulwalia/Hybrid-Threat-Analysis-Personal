"""
ml_service/src/risk/generate_flags_v2.py
────────────────────────────────────────
Phase 9 Data Preparation: Generate ml_flags_v2.parquet.

Extends ml_flags_v1.parquet by adding:
- p_attack: float32 = 1.0 - P(BENIGN) from the calibrated XGBoost multiclass model.

Memory Safety Protocol:
- Computes only p_attack (9.6 MB array) across test and train partitions.
- Streams existing columns (row_id, predicted_class, confidence, r104_flag, is_flagged)
  directly from ml_flags_v1.parquet in 100k chunks, eliminating redundant array duplication.
- Peak RSS target: < 350 MB (strictly below 500 MB limit).
"""

from __future__ import annotations

import gc
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import List

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
logger = logging.getLogger("generate_flags_v2")

CHUNK_SIZE = 50_000
MEMORY_LIMIT_MB = 500


def get_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def main() -> None:
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("Phase 9 Step 2a -- Memory-Safe ml_flags_v2.parquet Generation")
    logger.info("=" * 70)
    assert_memory_headroom(min_free_mb=200)

    # 1. Load LabelEncoder & Train Mask
    le_path = Paths.SPLITS_DIR / "label_encoder_v1.pkl"
    with open(le_path, "rb") as f:
        le = pickle.load(f)
    classes: List[str] = [str(c) for c in le.classes_]
    benign_idx = classes.index("BENIGN")
    logger.info("Categories (%d): BENIGN index is %d", len(classes), benign_idx)

    mask_path = Paths.SPLITS_DIR / "is_train_mask_v1.npy"
    is_train_mask: np.ndarray = np.load(mask_path)
    n_total = len(is_train_mask)
    assert n_total == 2_520_798, f"Unexpected mask length: {n_total}"
    test_indices = np.where(~is_train_mask)[0]
    train_indices = np.where(is_train_mask)[0]
    n_test = len(test_indices)
    n_train = len(train_indices)
    del is_train_mask
    gc.collect()

    # Preallocate ONLY the new p_attack array (9.6 MB)
    p_attack = np.zeros(n_total, dtype=np.float32)

    # 2. Load Booster
    xgb_model_path = Paths.MODELS_DIR / "xgb_baseline_category_v1.pkl"
    logger.info("Loading XGBoost booster: %s", xgb_model_path.name)
    booster: xgb.Booster = joblib.load(xgb_model_path)
    logger.info("Booster loaded. RSS: %.2f MB", get_rss_mb())

    # 3. Inference on Test Set
    logger.info("Streaming inference on TEST set (%d rows)...", n_test)
    x_test_path = Paths.SPLITS_DIR / "X_test_v1.parquet"
    test_pq = pq.ParquetFile(x_test_path)

    cur_idx = 0
    test_pred_attack_count = 0
    for batch in test_pq.iter_batches(batch_size=CHUNK_SIZE):
        chunk_arr = batch.to_pandas().to_numpy(dtype=np.float32)
        batch_len = len(chunk_arr)
        dtest = xgb.DMatrix(chunk_arr)
        probs = booster.predict(dtest)
        b_p_att = (1.0 - probs[:, benign_idx]).astype(np.float32)
        idx_slice = test_indices[cur_idx : cur_idx + batch_len]
        p_attack[idx_slice] = b_p_att
        test_pred_attack_count += int(np.sum(np.argmax(probs, axis=1) != benign_idx))
        cur_idx += batch_len
        del dtest, probs, chunk_arr, batch, b_p_att
        gc.collect()

    assert cur_idx == n_test, f"Test count mismatch: {cur_idx} vs {n_test}"
    logger.info("Test set complete. Predicted attacks count: %d (Expected: 85,378)", test_pred_attack_count)
    assert test_pred_attack_count == 84_979 + 399, f"Parity mismatch on predicted attacks: {test_pred_attack_count}"
    del test_pq, test_indices
    gc.collect()

    # 4. Inference on Train Set
    logger.info("Streaming inference on TRAIN set (%d rows)...", n_train)
    x_train_path = Paths.SPLITS_DIR / "X_train_v1.parquet"
    train_pq = pq.ParquetFile(x_train_path)

    cur_idx = 0
    for batch in train_pq.iter_batches(batch_size=CHUNK_SIZE):
        chunk_arr = batch.to_pandas().to_numpy(dtype=np.float32)
        batch_len = len(chunk_arr)
        dtrain = xgb.DMatrix(chunk_arr)
        probs = booster.predict(dtrain)
        b_p_att = (1.0 - probs[:, benign_idx]).astype(np.float32)
        idx_slice = train_indices[cur_idx : cur_idx + batch_len]
        p_attack[idx_slice] = b_p_att
        cur_idx += batch_len
        del dtrain, probs, chunk_arr, batch, b_p_att
        gc.collect()

    assert cur_idx == n_train, f"Train count mismatch: {cur_idx} vs {n_train}"
    logger.info("Train set complete. Releasing booster...")
    del train_pq, train_indices, booster
    gc.collect()
    logger.info("Booster released. In-memory p_attack ready. Current RSS: %.2f MB", get_rss_mb())

    # 5. Stream from ml_flags_v1.parquet and write ml_flags_v2.parquet
    v1_path = Paths.ARTIFACTS_DIR / "correlation" / "ml_flags_v1.parquet"
    v2_path = Paths.ARTIFACTS_DIR / "correlation" / "ml_flags_v2.parquet"
    logger.info("Streaming from %s and writing %s...", v1_path.name, v2_path.name)

    v1_pq = pq.ParquetFile(v1_path)
    schema_v2 = pa.schema([
        ("row_id", pa.int64()),
        ("predicted_class", pa.string()),
        ("confidence", pa.float32()),
        ("p_attack", pa.float32()),
        ("r104_flag", pa.bool_()),
        ("is_flagged", pa.bool_()),
    ])

    writer = pq.ParquetWriter(v2_path, schema_v2, compression="snappy")
    stream_chunk = 100_000
    rows_written = 0

    for batch in v1_pq.iter_batches(batch_size=stream_chunk):
        b_df = batch.to_pandas()
        b_len = len(b_df)
        b_p_att = p_attack[rows_written : rows_written + b_len]
        
        table_chunk = pa.Table.from_arrays(
            [
                pa.array(b_df["row_id"].to_numpy(dtype=np.int64), type=pa.int64()),
                pa.array(b_df["predicted_class"].tolist(), type=pa.string()),
                pa.array(b_df["confidence"].to_numpy(dtype=np.float32), type=pa.float32()),
                pa.array(b_p_att, type=pa.float32()),
                pa.array(b_df["r104_flag"].to_numpy(dtype=bool), type=pa.bool_()),
                pa.array(b_df["is_flagged"].to_numpy(dtype=bool), type=pa.bool_()),
            ],
            schema=schema_v2,
        )
        writer.write_table(table_chunk)
        rows_written += b_len
        del b_df, b_p_att, table_chunk, batch
        gc.collect()

    writer.close()
    assert rows_written == n_total, f"Total rows written mismatch: {rows_written} vs {n_total}"

    elapsed = time.time() - t0
    peak_rss = get_rss_mb()
    logger.info("Successfully generated %s (%d rows) in %.2f s | Peak RSS: %.2f MB", v2_path.name, n_total, elapsed, peak_rss)
    assert peak_rss < MEMORY_LIMIT_MB, f"Peak RSS {peak_rss:.2f} MB exceeded limit {MEMORY_LIMIT_MB} MB"


if __name__ == "__main__":
    main()
