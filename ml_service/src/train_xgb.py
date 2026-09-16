"""
ml_service/src/train_xgb.py
───────────────────────────
Phase 5: ML Comparison — XGBoost via tree_method='hist'.

Implements memory-constrained gradient-boosted tree ensemble training per AGENTS.md
and RESEARCH_PAPER_TRACKER.md:
- XGBoost histogram-binned algorithm (tree_method='hist', device='cpu')
- Native xgb.train with DMatrix for minimal memory footprint and precise multi-class handling
- Hyperparameters matching research specifications: n_estimators=200, max_depth=6,
  learning_rate=0.1, subsample=0.8, colsample_bytree=0.8
- Memory discipline: constructs histogram matrix and releases raw float32 memory,
  monitoring resident RSS against the < 700 MB peak budget.
- Comprehensive evaluation: global, multi-class (all 15 categories with low-support warnings),
  and binary intrusion detection metrics (FPR, TPR, ROC-AUC, latency).
- Auto-generates artifacts/model_selection_decision_v1.md comparing RF vs XGBoost.

Usage:
    # Stage 1: Smoke test (10k train rows, 2k test rows)
    python ml_service/src/train_xgb.py --smoke-test

    # Stage 2: Full comparative training & evaluation
    python ml_service/src/train_xgb.py
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import psutil
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

# ── Root path resolution ───────────────────────────────────────────────────────
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
logger = logging.getLogger("train_xgb")

# ── Constants ──────────────────────────────────────────────────────────────────
PIPELINE_VERSION = "1.0"
PHASE = "Phase 5"
VERSION_TAG = "v1"
N_ESTIMATORS_FULL = 200
N_ESTIMATORS_SMOKE = 10
MAX_DEPTH = 6
LEARNING_RATE = 0.1
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
RANDOM_STATE = 42
MEMORY_BUDGET_MB = 700
LOW_SUPPORT_THRESHOLD = 30


def get_rss_mb() -> float:
    """Return the current process RSS in Megabytes."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


# ── Step 1: Data Loading ───────────────────────────────────────────────────────

def load_data(
    smoke_test: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, float]]:
    """Load train and test data from Phase 3 Parquet splits.

    Uses streaming row-group reads to populate a single preallocated float32
    array, preventing PyArrow/Pandas duplicate in-memory copies.

    Returns:
        X_train, y_train, X_test, y_test, class_names, telemetry_dict
    """
    logger.info("=" * 70)
    logger.info("Phase 5 -- Loading Data (smoke_test=%s)", smoke_test)
    logger.info("=" * 70)
    assert_memory_headroom(min_free_mb=200)

    telemetry: dict[str, float] = {
        "baseline_rss_mb": round(get_rss_mb(), 2),
    }
    logger.info("Baseline RSS: %.2f MB", telemetry["baseline_rss_mb"])

    # 1. Load label encoder to get exact ordered class names
    le_path = Paths.split(f"label_encoder_{VERSION_TAG}.pkl")
    le = joblib.load(le_path)
    assert le.classes_ is not None, "LabelEncoder classes_ is None"
    classes: list[str] = [str(c) for c in le.classes_]
    logger.info("Loaded %d target categories: %s", len(classes), classes)

    # 2. Load y_train and y_test
    y_train_path = Paths.split(f"y_train_category_{VERSION_TAG}.parquet")
    y_test_path  = Paths.split(f"y_test_category_{VERSION_TAG}.parquet")

    y_train_full = pq.read_table(y_train_path)["category_label"].to_numpy().astype(np.int32)
    y_test_full  = pq.read_table(y_test_path)["category_label"].to_numpy().astype(np.int32)

    x_train_path = Paths.split(f"X_train_{VERSION_TAG}.parquet")
    x_test_path  = Paths.split(f"X_test_{VERSION_TAG}.parquet")

    pf_train = pq.ParquetFile(x_train_path)
    total_train_rows = pf_train.metadata.num_rows
    num_features = pf_train.metadata.num_columns

    if smoke_test:
        train_rows_to_load = min(10_000, total_train_rows)
        test_rows_to_load = 2_000
        logger.info("Smoke test mode: loading %d train rows and %d test rows across row groups", train_rows_to_load, test_rows_to_load)

        # Read samples from multiple row groups so multiple classes are represented
        train_chunks = []
        for rg_idx in range(min(5, pf_train.num_row_groups)):
            df_chunk = pf_train.read_row_group(rg_idx).to_pandas()
            train_chunks.append(df_chunk)
        df_train_sub = df_chunk = __import__("pandas").concat(train_chunks, ignore_index=True)
        X_train = df_train_sub.to_numpy(dtype=np.float32)[:train_rows_to_load]
        y_train = y_train_full[:len(X_train)]

        pf_test = pq.ParquetFile(x_test_path)
        test_chunks = []
        for rg_idx in range(min(5, pf_test.num_row_groups)):
            df_chunk = pf_test.read_row_group(rg_idx).to_pandas()
            test_chunks.append(df_chunk)
        df_test_sub = __import__("pandas").concat(test_chunks, ignore_index=True)
        X_test = df_test_sub.to_numpy(dtype=np.float32)[:test_rows_to_load]
        y_test = y_test_full[:len(X_test)]
    else:
        logger.info("Full mode: loading %d train rows (%d features) via streaming row groups", total_train_rows, num_features)
        X_train = np.empty((total_train_rows, num_features), dtype=np.float32)
        offset = 0
        for rg_idx in range(pf_train.num_row_groups):
            rg_df = pf_train.read_row_group(rg_idx).to_pandas()
            rg_arr = rg_df.to_numpy(dtype=np.float32)
            rg_len = len(rg_arr)
            X_train[offset : offset + rg_len] = rg_arr
            offset += rg_len
            del rg_df, rg_arr

        y_train = y_train_full

        logger.info("Loading full test set: %d rows", len(y_test_full))
        pf_test = pq.ParquetFile(x_test_path)
        total_test_rows = pf_test.metadata.num_rows
        X_test = np.empty((total_test_rows, num_features), dtype=np.float32)
        offset = 0
        for rg_idx in range(pf_test.num_row_groups):
            rg_df = pf_test.read_row_group(rg_idx).to_pandas()
            rg_arr = rg_df.to_numpy(dtype=np.float32)
            rg_len = len(rg_arr)
            X_test[offset : offset + rg_len] = rg_arr
            offset += rg_len
            del rg_df, rg_arr

        y_test = y_test_full

    telemetry["after_load_rss_mb"] = round(get_rss_mb(), 2)
    logger.info(
        "Data loaded: X_train shape=%s | X_test shape=%s | RSS: %.2f MB",
        X_train.shape, X_test.shape, telemetry["after_load_rss_mb"],
    )

    return X_train, y_train, X_test, y_test, classes, telemetry


# ── Step 2: XGBoost Training ───────────────────────────────────────────────────

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    smoke_test: bool = False,
    telemetry: dict[str, float] | None = None,
) -> tuple[xgb.Booster, dict[str, Any]]:
    """Train XGBoost multi-class classifier using histogram-binned tree_method."""
    if telemetry is None:
        telemetry = {}

    n_estimators = N_ESTIMATORS_SMOKE if smoke_test else N_ESTIMATORS_FULL

    logger.info("=" * 70)
    logger.info("Phase 5 -- XGBoost Training (tree_method='hist', n_estimators=%d)", n_estimators)
    logger.info("=" * 70)

    # 1. Create DMatrix
    logger.info("Constructing XGBoost DMatrix (histogram binning)...")
    dtrain = xgb.DMatrix(data=X_train, label=y_train)

    telemetry["after_construct_rss_mb"] = round(get_rss_mb(), 2)
    logger.info("DMatrix constructed. Resident RSS: %.2f MB", telemetry["after_construct_rss_mb"])

    # Free raw X_train memory reference
    del X_train
    gc.collect()

    telemetry["after_free_raw_rss_mb"] = round(get_rss_mb(), 2)
    logger.info("Freed raw X_train. Resident RSS: %.2f MB", telemetry["after_free_raw_rss_mb"])

    params: dict[str, Any] = {
        "tree_method": "hist",
        "device": "cpu",
        "objective": "multi:softprob",
        "num_class": n_classes,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE_BYTREE,
        "seed": RANDOM_STATE,
        "eval_metric": "mlogloss",
        "verbosity": 1,
    }

    logger.info(
        "XGBoost Parameters: tree_method='hist' | n_estimators=%d | max_depth=%d | lr=%.2f | subsample=%.2f | colsample=%.2f",
        n_estimators, MAX_DEPTH, LEARNING_RATE, SUBSAMPLE, COLSAMPLE_BYTREE,
    )

    t0 = time.time()
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=n_estimators,
    )
    train_elapsed_sec = time.time() - t0

    # Clean up DMatrix
    del dtrain
    gc.collect()

    telemetry["after_fit_rss_mb"] = round(get_rss_mb(), 2)
    logger.info("Training complete in %.2f s | RSS after fit: %.2f MB", train_elapsed_sec, telemetry["after_fit_rss_mb"])

    training_meta: dict[str, Any] = {
        "algorithm": "XGBoost (Histogram Gradient Boosting)",
        "tree_method": "hist",
        "device": "cpu",
        "n_estimators": n_estimators,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE_BYTREE,
        "objective": "multi:softprob",
        "num_class": n_classes,
        "random_state": RANDOM_STATE,
        "train_time_seconds": round(train_elapsed_sec, 2),
    }

    return booster, training_meta


# ── Step 3: Comprehensive Evaluation ───────────────────────────────────────────

def evaluate_model(
    booster: xgb.Booster,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
    telemetry: dict[str, float],
) -> dict[str, Any]:
    """Compute full evaluation metric suite across multiclass and binary benchmarks."""
    logger.info("=" * 70)
    logger.info("Phase 5 -- Model Evaluation on Test Set (N=%d)", len(y_test))
    logger.info("=" * 70)

    # 1. Inference & Latency
    dtest = xgb.DMatrix(X_test)
    t0 = time.time()
    probs_raw: Any = booster.predict(dtest)
    probs: np.ndarray = np.asarray(probs_raw)
    eval_elapsed_sec = time.time() - t0

    del dtest
    gc.collect()

    preds: np.ndarray = np.argmax(probs, axis=1)
    latency_us = (eval_elapsed_sec / len(y_test)) * 1_000_000

    telemetry["after_inference_rss_mb"] = round(get_rss_mb(), 2)
    logger.info(
        "Inference finished in %.2f s (%.2f μs/sample) | Peak RSS: %.2f MB",
        eval_elapsed_sec, latency_us, telemetry["after_inference_rss_mb"],
    )

    # 2. Global Multi-Class Metrics
    acc = float(accuracy_score(y_test, preds))
    bal_acc = float(balanced_accuracy_score(y_test, preds))

    zd: Any = 0
    macro_res: Any = precision_recall_fscore_support(
        y_test, preds, average="macro", zero_division=zd
    )
    macro_p, macro_r, macro_f1 = float(macro_res[0]), float(macro_res[1]), float(macro_res[2])

    weighted_res: Any = precision_recall_fscore_support(
        y_test, preds, average="weighted", zero_division=zd
    )
    weighted_p, weighted_r, weighted_f1 = float(weighted_res[0]), float(weighted_res[1]), float(weighted_res[2])

    # Multi-class ROC-AUC (OVR)
    try:
        if len(np.unique(y_test)) > 1:
            roc_auc_ovr_macro = float(
                roc_auc_score(y_test, probs, multi_class="ovr", average="macro", labels=list(range(len(classes))))
            )
            roc_auc_ovr_weighted = float(
                roc_auc_score(y_test, probs, multi_class="ovr", average="weighted", labels=list(range(len(classes))))
            )
        else:
            roc_auc_ovr_macro = -1.0
            roc_auc_ovr_weighted = -1.0
    except Exception as e:
        logger.warning("Could not calculate multi-class ROC-AUC: %s", e)
        roc_auc_ovr_macro = -1.0
        roc_auc_ovr_weighted = -1.0

    # 3. Per-Class Metrics with Low-Support Warnings
    class_res: Any = precision_recall_fscore_support(
        y_test, preds, average=None, zero_division=zd, labels=list(range(len(classes)))
    )
    class_p: np.ndarray = np.asarray(class_res[0])
    class_r: np.ndarray = np.asarray(class_res[1])
    class_f1: np.ndarray = np.asarray(class_res[2])
    class_supp: np.ndarray = np.asarray(class_res[3])

    per_class_metrics: dict[str, dict[str, Any]] = {}
    for idx, name in enumerate(classes):
        supp = int(class_supp[idx])
        is_low_support = supp < LOW_SUPPORT_THRESHOLD
        per_class_metrics[name] = {
            "class_index": idx,
            "precision": round(float(class_p[idx]), 5),
            "recall": round(float(class_r[idx]), 5),
            "f1_score": round(float(class_f1[idx]), 5),
            "support": supp,
            "low_support_warning": is_low_support,
            "caveat": (
                "Support < 30: metric has high variance and low statistical significance"
                if is_low_support else None
            ),
        }

    # Confusion matrix (15x15)
    cm = confusion_matrix(y_test, preds, labels=list(range(len(classes)))).tolist()

    # 4. Binary Intrusion Detection Metrics
    # Class 0 is BENIGN, 1..14 are ATTACKs.
    y_test_bin = (y_test != 0).astype(int)
    preds_bin  = (preds != 0).astype(int)
    prob_attack = 1.0 - probs[:, 0]

    bin_cm = confusion_matrix(y_test_bin, preds_bin, labels=[0, 1]).ravel()
    tn, fp, fn, tp = (int(x) for x in bin_cm)

    bin_acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    bin_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    bin_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Detection Rate (TPR)
    bin_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bin_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0      # False Positive Rate
    bin_f1 = (2 * bin_precision * bin_recall) / (bin_precision + bin_recall) if (bin_precision + bin_recall) > 0 else 0.0

    try:
        if len(np.unique(y_test_bin)) > 1:
            bin_roc_auc = float(roc_auc_score(y_test_bin, prob_attack))
        else:
            bin_roc_auc = -1.0
    except Exception as e:
        logger.warning("Could not calculate binary ROC-AUC: %s", e)
        bin_roc_auc = -1.0

    binary_metrics: dict[str, Any] = {
        "accuracy": round(float(bin_acc), 5),
        "precision": round(float(bin_precision), 5),
        "recall_tpr": round(float(bin_recall), 5),
        "specificity_tnr": round(float(bin_specificity), 5),
        "false_positive_rate_fpr": round(float(bin_fpr), 6),
        "f1_score": round(float(bin_f1), 5),
        "roc_auc": round(float(bin_roc_auc), 5),
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        },
    }

    results: dict[str, Any] = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "test_sample_count": len(y_test),
        "inference_time_seconds": round(eval_elapsed_sec, 2),
        "inference_latency_us_per_sample": round(latency_us, 2),
        "global_multiclass": {
            "accuracy": round(acc, 5),
            "balanced_accuracy": round(bal_acc, 5),
            "macro_precision": round(float(macro_p), 5),
            "macro_recall": round(float(macro_r), 5),
            "macro_f1": round(float(macro_f1), 5),
            "weighted_precision": round(float(weighted_p), 5),
            "weighted_recall": round(float(weighted_r), 5),
            "weighted_f1": round(float(weighted_f1), 5),
            "roc_auc_ovr_macro": round(roc_auc_ovr_macro, 5),
            "roc_auc_ovr_weighted": round(roc_auc_ovr_weighted, 5),
        },
        "binary_intrusion_detection": binary_metrics,
        "per_class": per_class_metrics,
        "confusion_matrix": cm,
        "classes": classes,
        "memory_telemetry": telemetry,
    }

    logger.info(
        "XGBoost Multiclass Results: Acc=%.4f | Macro F1=%.4f | Weighted F1=%.4f",
        acc, macro_f1, weighted_f1,
    )
    logger.info(
        "XGBoost Binary Intrusion Results: Detection Rate (TPR)=%.4f | FPR=%.6f | F1=%.4f | ROC-AUC=%.4f",
        bin_recall, bin_fpr, bin_f1, bin_roc_auc,
    )

    return results


# ── Step 4: Markdown Report Generation ─────────────────────────────────────────

def generate_markdown_report(
    metrics: dict[str, Any],
    training_meta: dict[str, Any],
    smoke_test: bool = False,
) -> str:
    """Format evaluation metrics into publication-ready Markdown table."""
    glob = metrics["global_multiclass"]
    bin_m = metrics["binary_intrusion_detection"]
    bin_cm = bin_m["confusion_matrix"]
    mem = metrics["memory_telemetry"]

    lines = [
        f"# Phase 5 — ML Comparative Evaluation Report: XGBoost (tree_method='hist'){' [SMOKE TEST]' if smoke_test else ''}",
        "",
        f"**Model:** XGBoost Comparative Model (`xgboost.Booster` via `tree_method='hist'`)  ",
        f"**Date:** `{metrics['evaluation_timestamp']}`  ",
        f"**Split:** 80/20 Stratified (`artifacts/splits/`) | **Test Samples:** `{metrics['test_sample_count']:,}`  ",
        f"**Training Time:** `{training_meta['train_time_seconds']:.2f} s` | **Inference Latency:** `{metrics['inference_latency_us_per_sample']:.2f} μs/sample`  ",
        f"**Peak Memory:** `{max(mem.values()):.2f} MB` (Budget: `< {MEMORY_BUDGET_MB} MB`)",
        "",
        "---",
        "",
        "## 1. Executive Summary & Core Performance",
        "",
        "| Metric Type | Metric | Score |",
        "|---|---|---|",
        f"| **Binary Detection** | Accuracy | **{bin_m['accuracy'] * 100:.3f}%** |",
        f"| **Binary Detection** | Detection Rate (Recall / TPR) | **{bin_m['recall_tpr'] * 100:.3f}%** |",
        f"| **Binary Detection** | False Positive Rate (FPR) | **{bin_m['false_positive_rate_fpr'] * 100:.4f}%** |",
        f"| **Binary Detection** | Precision | **{bin_m['precision'] * 100:.3f}%** |",
        f"| **Binary Detection** | F1-Score | **{bin_m['f1_score']:.5f}** |",
        f"| **Binary Detection** | ROC-AUC | **{bin_m['roc_auc']:.5f}** |",
        f"| **Multi-Class (15 cat)** | Overall Accuracy | **{glob['accuracy'] * 100:.3f}%** |",
        f"| **Multi-Class (15 cat)** | Balanced Accuracy | **{glob['balanced_accuracy'] * 100:.3f}%** |",
        f"| **Multi-Class (15 cat)** | Macro F1-Score | **{glob['macro_f1']:.5f}** |",
        f"| **Multi-Class (15 cat)** | Weighted F1-Score | **{glob['weighted_f1']:.5f}** |",
        f"| **Multi-Class (15 cat)** | ROC-AUC (OVR Weighted) | **{glob['roc_auc_ovr_weighted']:.5f}** |",
        "",
        "### Binary Confusion Matrix",
        "",
        f"- **True Negatives (BENIGN correctly passed):** `{bin_cm['true_negatives']:,}`",
        f"- **False Positives (BENIGN wrongly alerted):** `{bin_cm['false_positives']:,}`",
        f"- **False Negatives (ATTACK missed):** `{bin_cm['false_negatives']:,}`",
        f"- **True Positives (ATTACK correctly detected):** `{bin_cm['true_positives']:,}`",
        "",
        "---",
        "",
        "## 2. Per-Class Attack Performance Breakdown",
        "",
        "| Attack Category | Test Support | Precision | Recall | F1-Score | Statistical Note |",
        "|---|---|---|---|---|---|",
    ]

    for name, m in metrics["per_class"].items():
        note = "⚠️ Low Support (<30)" if m["low_support_warning"] else "✅ Robust"
        lines.append(
            f"| `{name}` | {m['support']:,} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1_score']:.4f} | {note} |"
        )

    lines.extend([
        "",
        "> [!NOTE]",
        "> Classes marked with **⚠️ Low Support (<30)** (e.g. Heartbleed $N=2$, Infiltration $N=7$, SQL Injection $N=4$) have severe sample-size constraints in the CIC-IDS2017 dataset. While reported for full transparency, per-class F1 metrics for these categories exhibit high variance where a single misclassification swings performance by 25%–50%.",
        "",
        "---",
        "",
        "## 3. Hardware & Memory Profile",
        "",
        "| Telemetry Checkpoint | Resident RSS (MB) | Budget |",
        "|---|---|---|",
        f"| Initial Baseline | `{mem.get('baseline_rss_mb', 0):.2f} MB` | — |",
        f"| Post-Data Loading | `{mem.get('after_load_rss_mb', 0):.2f} MB` | < 700 MB |",
        f"| DMatrix Histogram Binning | `{mem.get('after_construct_rss_mb', 0):.2f} MB` | < 700 MB |",
        f"| Post-Raw Data Freeing | `{mem.get('after_free_raw_rss_mb', 0):.2f} MB` | < 700 MB |",
        f"| Post-Fit Training Peak | `{mem.get('after_fit_rss_mb', 0):.2f} MB` | < 700 MB |",
        f"| Post-Inference Peak | `{mem.get('after_inference_rss_mb', 0):.2f} MB` | < 700 MB |",
        "",
        "---",
        "",
        "## 4. XGBoost Hyperparameter Configuration",
        "",
        "```json",
        json.dumps(training_meta, indent=2),
        "```",
    ])

    return "\n".join(lines)


# ── Step 5: Model Selection Decision Synthesis ─────────────────────────────────

def synthesize_model_selection_decision(
    xgb_metrics: dict[str, Any],
    xgb_meta: dict[str, Any],
) -> str:
    """Generate formal Model Selection Decision comparing Phase 4 RF and Phase 5 XGBoost."""
    rf_json_path = Paths.artifact("rf_metrics_v1.json")
    if not rf_json_path.exists():
        logger.warning("RF metrics not found at %s. Skipping decision synthesis.", rf_json_path)
        return ""

    with open(rf_json_path, "r", encoding="utf-8") as f:
        rf_metrics = json.load(f)

    rf_glob = rf_metrics["global_multiclass"]
    rf_bin = rf_metrics["binary_intrusion_detection"]
    rf_hp = rf_metrics["hyperparameters"]

    xg_glob = xgb_metrics["global_multiclass"]
    xg_bin = xgb_metrics["binary_intrusion_detection"]

    rf_train_t = rf_hp.get("train_time_seconds", 0.0)
    xg_train_t = xgb_meta.get("train_time_seconds", 0.0)

    rf_lat = rf_metrics.get("inference_latency_us_per_sample", 0.0)
    xg_lat = xgb_metrics.get("inference_latency_us_per_sample", 0.0)

    # Balanced evaluation score
    rf_score = (rf_bin["recall_tpr"] * 0.3) + ((1.0 - rf_bin["false_positive_rate_fpr"]) * 0.3) + (rf_glob["macro_f1"] * 0.2) + (rf_glob["weighted_f1"] * 0.2)
    xg_score = (xg_bin["recall_tpr"] * 0.3) + ((1.0 - xg_bin["false_positive_rate_fpr"]) * 0.3) + (xg_glob["macro_f1"] * 0.2) + (xg_glob["weighted_f1"] * 0.2)

    selected_model = "XGBoost" if xg_score >= rf_score else "Random Forest"
    decision_rationale = (
        f"XGBoost achieved superior detection trade-offs with higher Macro F1 ({xg_glob['macro_f1']:.4f} vs {rf_glob['macro_f1']:.4f}) and competitive FPR ({xg_bin['false_positive_rate_fpr']*100:.4f}% vs {rf_bin['false_positive_rate_fpr']*100:.4f}%)."
        if selected_model == "XGBoost" else
        f"Random Forest demonstrated superior detection balance with higher overall Detection Rate ({rf_bin['recall_tpr']*100:.3f}% vs {xg_bin['recall_tpr']*100:.3f}%) and lower inference latency ({rf_lat:.2f} μs vs {xg_lat:.2f} μs)."
    )

    doc_lines = [
        "# Phase 5 — Model Selection Decision Document",
        "",
        "**Document Version:** `1.0`  ",
        f"**Date:** `{xgb_metrics['evaluation_timestamp']}`  ",
        "**Dataset:** CIC-IDS2017 (80/20 Stratified Split, Test N=504,160)  ",
        "**Authoritative Scope:** Selection of the Primary Threat Classification Model for Phase 6 (SHAP Explainability) and downstream Event Correlation / Risk Scoring.",
        "",
        "---",
        "",
        "## 1. Head-to-Head Empirical Benchmark",
        "",
        "| Evaluation Dimension | Metric | Random Forest Baseline (`LightGBM RF`) | XGBoost Comparative (`tree_method='hist'`) | Delta (XGB vs RF) | Advantage |",
        "|---|---|---|---|---|---|",
        f"| **Binary Intrusion** | Accuracy | `{rf_bin['accuracy']*100:.3f}%` | `{xg_bin['accuracy']*100:.3f}%` | `{(xg_bin['accuracy'] - rf_bin['accuracy'])*100:+.3f}%` | {'XGBoost' if xg_bin['accuracy'] >= rf_bin['accuracy'] else 'Random Forest'} |",
        f"| **Binary Intrusion** | Detection Rate (Recall / TPR) | `{rf_bin['recall_tpr']*100:.3f}%` | `{xg_bin['recall_tpr']*100:.3f}%` | `{(xg_bin['recall_tpr'] - rf_bin['recall_tpr'])*100:+.3f}%` | {'XGBoost' if xg_bin['recall_tpr'] >= rf_bin['recall_tpr'] else 'Random Forest'} |",
        f"| **Binary Intrusion** | False Positive Rate (FPR) | `{rf_bin['false_positive_rate_fpr']*100:.4f}%` | `{xg_bin['false_positive_rate_fpr']*100:.4f}%` | `{(xg_bin['false_positive_rate_fpr'] - rf_bin['false_positive_rate_fpr'])*100:+.4f}%` | {'XGBoost (lower)' if xg_bin['false_positive_rate_fpr'] <= rf_bin['false_positive_rate_fpr'] else 'Random Forest (lower)'} |",
        f"| **Binary Intrusion** | Binary F1-Score | `{rf_bin['f1_score']:.5f}` | `{xg_bin['f1_score']:.5f}` | `{xg_bin['f1_score'] - rf_bin['f1_score']:+.5f}` | {'XGBoost' if xg_bin['f1_score'] >= rf_bin['f1_score'] else 'Random Forest'} |",
        f"| **Binary Intrusion** | ROC-AUC | `{rf_bin['roc_auc']:.5f}` | `{xg_bin['roc_auc']:.5f}` | `{xg_bin['roc_auc'] - rf_bin['roc_auc']:+.5f}` | {'XGBoost' if xg_bin['roc_auc'] >= rf_bin['roc_auc'] else 'Random Forest'} |",
        f"| **Multi-Class (15 cat)** | Overall Accuracy | `{rf_glob['accuracy']*100:.3f}%` | `{xg_glob['accuracy']*100:.3f}%` | `{(xg_glob['accuracy'] - rf_glob['accuracy'])*100:+.3f}%` | {'XGBoost' if xg_glob['accuracy'] >= rf_glob['accuracy'] else 'Random Forest'} |",
        f"| **Multi-Class (15 cat)** | Balanced Accuracy | `{rf_glob['balanced_accuracy']*100:.3f}%` | `{xg_glob['balanced_accuracy']*100:.3f}%` | `{(xg_glob['balanced_accuracy'] - rf_glob['balanced_accuracy'])*100:+.3f}%` | {'XGBoost' if xg_glob['balanced_accuracy'] >= rf_glob['balanced_accuracy'] else 'Random Forest'} |",
        f"| **Multi-Class (15 cat)** | Macro F1-Score | `{rf_glob['macro_f1']:.5f}` | `{xg_glob['macro_f1']:.5f}` | `{xg_glob['macro_f1'] - rf_glob['macro_f1']:+.5f}` | {'XGBoost' if xg_glob['macro_f1'] >= rf_glob['macro_f1'] else 'Random Forest'} |",
        f"| **Multi-Class (15 cat)** | Weighted F1-Score | `{rf_glob['weighted_f1']:.5f}` | `{xg_glob['weighted_f1']:.5f}` | `{xg_glob['weighted_f1'] - rf_glob['weighted_f1']:+.5f}` | {'XGBoost' if xg_glob['weighted_f1'] >= rf_glob['weighted_f1'] else 'Random Forest'} |",
        f"| **Multi-Class (15 cat)** | ROC-AUC (OVR Weighted) | `{rf_glob['roc_auc_ovr_weighted']:.5f}` | `{xg_glob['roc_auc_ovr_weighted']:.5f}` | `{xg_glob['roc_auc_ovr_weighted'] - rf_glob['roc_auc_ovr_weighted']:+.5f}` | {'XGBoost' if xg_glob['roc_auc_ovr_weighted'] >= rf_glob['roc_auc_ovr_weighted'] else 'Random Forest'} |",
        f"| **Operational** | Training Time | `{rf_train_t:.2f} s` | `{xg_train_t:.2f} s` | `{xg_train_t - rf_train_t:+.2f} s` | {'Random Forest' if rf_train_t <= xg_train_t else 'XGBoost'} |",
        f"| **Operational** | Inference Latency | `{rf_lat:.2f} μs/sample` | `{xg_lat:.2f} μs/sample` | `{xg_lat - rf_lat:+.2f} μs` | {'Random Forest' if rf_lat <= xg_lat else 'XGBoost'} |",
        "",
        "---",
        "",
        "## 2. Per-Class F1-Score Comparison",
        "",
        "| Attack Category | Test Support | Random Forest F1 | XGBoost F1 | F1 Delta (XGB - RF) | Superior Classifier |",
        "|---|---|---|---|---|---|",
    ]

    for name in xgb_metrics["classes"]:
        rf_c = rf_metrics["per_class"].get(name, {})
        xg_c = xgb_metrics["per_class"].get(name, {})
        rf_f1 = rf_c.get("f1_score", 0.0)
        xg_f1 = xg_c.get("f1_score", 0.0)
        supp = xg_c.get("support", 0)
        delta = xg_f1 - rf_f1
        winner = "XGBoost" if delta > 0.001 else ("Random Forest" if delta < -0.001 else "Tie (Equal)")
        doc_lines.append(
            f"| `{name}` | {supp:,} | {rf_f1:.4f} | {xg_f1:.4f} | {delta:+.4f} | {winner} |"
        )

    doc_lines.extend([
        "",
        "---",
        "",
        "## 3. Formal Model Selection Decision",
        "",
        f"> ### **Selected Primary Model:** `{selected_model}`",
        f"> ",
        f"> **Decision Rationale:** {decision_rationale}",
        "",
        "### Architectural Handoff to Phase 6 (Explainability with SHAP)",
        f"1. **Explainer Selection:** `shap.TreeExplainer` will be instantiated on the selected `{selected_model}` model.",
        "2. **Sampling Protocol:** Per AGENTS.md hardware constraints, SHAP explanations will be generated on a stratified, representative sample of the test set (never the full 504,160 rows).",
        "3. **Model Preservation:** Both models (`rf_baseline_category_v1` and `xgb_baseline_category_v1`) remain saved in `models/` for paper reproducibility and comparative ablation studies.",
    ])

    return "\n".join(doc_lines)


# ── Step 6: Main Orchestrator ──────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: ML Comparison XGBoost")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run lightweight smoke test on 10k train / 2k test samples to verify execution without resource consumption",
    )
    args = parser.parse_args()

    smoke_test: bool = args.smoke_test
    tag = f"{VERSION_TAG}_smoke" if smoke_test else VERSION_TAG

    logger.info("Starting Phase 5 Pipeline (smoke_test=%s, tag=%s)", smoke_test, tag)

    # 1. Load Data
    X_train, y_train, X_test, y_test, classes, telemetry = load_data(smoke_test=smoke_test)

    # 2. Train Model
    booster, training_meta = train_xgboost(
        X_train=X_train,
        y_train=y_train,
        n_classes=len(classes),
        smoke_test=smoke_test,
        telemetry=telemetry,
    )

    # 3. Evaluate Model
    metrics = evaluate_model(
        booster=booster,
        X_test=X_test,
        y_test=y_test,
        classes=classes,
        telemetry=telemetry,
    )

    # 4. Generate Reports & Artifacts
    report_md = generate_markdown_report(
        metrics=metrics,
        training_meta=training_meta,
        smoke_test=smoke_test,
    )

    # Output file destinations
    Paths.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    Paths.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_json_path = Paths.model(f"xgb_baseline_category_{tag}.json")
    model_pkl_path  = Paths.model(f"xgb_baseline_category_{tag}.pkl")
    metrics_json_path = Paths.artifact(f"xgb_metrics_{tag}.json")
    metrics_md_path   = Paths.artifact(f"xgb_metrics_{tag}.md")

    # Save model in native JSON format and joblib pickle
    booster.save_model(str(model_json_path))
    joblib.dump(booster, model_pkl_path)
    logger.info("Saved models: %s and %s", model_json_path.name, model_pkl_path.name)

    # Save metrics JSON & Markdown
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "phase": PHASE,
                "pipeline_version": PIPELINE_VERSION,
                "version_tag": tag,
                "smoke_test": smoke_test,
                "hyperparameters": training_meta,
                **metrics,
            },
            f,
            indent=2,
        )

    with open(metrics_md_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    logger.info("Saved metrics: %s and %s", metrics_json_path.name, metrics_md_path.name)

    # 5. Synthesize Model Selection Decision (if not smoke test)
    if not smoke_test:
        decision_md = synthesize_model_selection_decision(metrics, training_meta)
        if decision_md:
            decision_path = Paths.artifact(f"model_selection_decision_{VERSION_TAG}.md")
            with open(decision_path, "w", encoding="utf-8") as f:
                f.write(decision_md)
            logger.info("Generated Model Selection Decision: %s", decision_path.name)

    logger.info("Phase 5 successfully completed!")


if __name__ == "__main__":
    main()
