"""
ml_service/src/train_rf.py
──────────────────────────
Phase 4: ML Baseline — Random Forest via LightGBM RF Mode.

Implements memory-constrained bagged-tree ensemble training per AGENTS.md
and RESEARCH_PAPER_TRACKER.md:
- LightGBM native RF mode (boosting_type='rf')
- Mandatory explicit bagging parameters: bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8
- Memory discipline: constructs histogram binned lgb.Dataset with free_raw_data=True,
  releasing raw float32 matrices immediately to stay within the < 700 MB peak RSS budget.
- Comprehensive evaluation: global, multi-class (all 15 categories with low-support warnings),
  and binary intrusion detection metrics (FPR, TPR, ROC-AUC).

Usage:
    # Stage 1: Smoke test (10k train rows, 2k test rows)
    python ml_service/src/train_rf.py --smoke-test

    # Stage 2: Full baseline training & evaluation
    python ml_service/src/train_rf.py
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
import lightgbm as lgb
import numpy as np
import psutil
import pyarrow.parquet as pq
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
logger = logging.getLogger("train_rf")

# ── Constants ──────────────────────────────────────────────────────────────────
PIPELINE_VERSION = "1.0"
PHASE = "Phase 4"
VERSION_TAG = "v1"
N_ESTIMATORS_FULL = 100
N_ESTIMATORS_SMOKE = 10
NUM_LEAVES = 63
MAX_DEPTH = 15
BAGGING_FRACTION = 0.8
BAGGING_FREQ = 1
FEATURE_FRACTION = 0.8
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
    logger.info("Phase 4 -- Loading Data (smoke_test=%s)", smoke_test)
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
        logger.info("Smoke test mode: loading first %d train rows and %d test rows", train_rows_to_load, test_rows_to_load)

        # Read first row group slice
        rg0 = pf_train.read_row_group(0)
        X_train: np.ndarray = rg0.to_pandas().to_numpy(dtype=np.float32)[:train_rows_to_load]
        y_train: np.ndarray = y_train_full[:train_rows_to_load]

        pf_test = pq.ParquetFile(x_test_path)
        rg_test0 = pf_test.read_row_group(0)
        X_test: np.ndarray = rg_test0.to_pandas().to_numpy(dtype=np.float32)[:test_rows_to_load]
        y_test: np.ndarray = y_test_full[:test_rows_to_load]
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


# ── Step 2: LightGBM RF Training ───────────────────────────────────────────────

def train_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    smoke_test: bool = False,
    telemetry: dict[str, float] | None = None,
) -> tuple[lgb.Booster, dict[str, Any]]:
    """Train Random Forest model using LightGBM native RF mode.

    Constructs histogram bins directly and releases raw float32 array before
    tree construction to prevent memory ballooning.
    """
    if telemetry is None:
        telemetry = {}

    logger.info("=" * 70)
    logger.info("Phase 4 -- LightGBM RF Training (trees=%d)", N_ESTIMATORS_SMOKE if smoke_test else N_ESTIMATORS_FULL)
    logger.info("=" * 70)

    n_estimators = N_ESTIMATORS_SMOKE if smoke_test else N_ESTIMATORS_FULL

    # 1. Create LightGBM Dataset with free_raw_data=True
    logger.info("Constructing LightGBM Dataset (histogram binning)...")
    train_data = lgb.Dataset(
        data=X_train,
        label=y_train,
        free_raw_data=True,
    )
    train_data.construct()

    telemetry["after_construct_rss_mb"] = round(get_rss_mb(), 2)
    logger.info("Dataset constructed. RSS before freeing raw X: %.2f MB", telemetry["after_construct_rss_mb"])

    # Explicitly release raw array reference and collect garbage
    del X_train
    gc.collect()

    telemetry["after_free_raw_rss_mb"] = round(get_rss_mb(), 2)
    logger.info("Freed raw X_train. Resident RSS: %.2f MB", telemetry["after_free_raw_rss_mb"])

    # 2. Hyperparameters strictly matching AGENTS.md rules
    params: dict[str, Any] = {
        "boosting_type": "rf",
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "num_leaves": NUM_LEAVES,
        "max_depth": MAX_DEPTH,
        "bagging_fraction": BAGGING_FRACTION,   # Mandatory bootstrap row subsampling < 1.0
        "bagging_freq": BAGGING_FREQ,           # Mandatory row bagging every 1 tree
        "feature_fraction": FEATURE_FRACTION,   # Mandatory column subsampling < 1.0
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }

    logger.info(
        "LightGBM RF Parameters: boosting_type='rf' | bagging_fraction=%.2f | bagging_freq=%d | feature_fraction=%.2f | num_leaves=%d | max_depth=%d",
        params["bagging_fraction"], params["bagging_freq"], params["feature_fraction"],
        params["num_leaves"], params["max_depth"],
    )

    t0 = time.time()
    booster = lgb.train(
        params=params,
        train_set=train_data,
        num_boost_round=n_estimators,
    )
    train_elapsed_sec = time.time() - t0

    telemetry["after_fit_rss_mb"] = round(get_rss_mb(), 2)
    logger.info("Training complete in %.2f s | RSS after fit: %.2f MB", train_elapsed_sec, telemetry["after_fit_rss_mb"])

    training_meta: dict[str, Any] = {
        "n_estimators": n_estimators,
        "num_leaves": NUM_LEAVES,
        "max_depth": MAX_DEPTH,
        "bagging_fraction": BAGGING_FRACTION,
        "bagging_freq": BAGGING_FREQ,
        "feature_fraction": FEATURE_FRACTION,
        "class_weight": "None (unweighted baseline)",
        "random_state": RANDOM_STATE,
        "train_time_seconds": round(train_elapsed_sec, 2),
    }

    return booster, training_meta


# ── Step 3: Comprehensive Evaluation ───────────────────────────────────────────

def evaluate_model(
    booster: lgb.Booster,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
    telemetry: dict[str, float],
) -> dict[str, Any]:
    """Compute full evaluation metric suite across multiclass and binary benchmarks."""
    logger.info("=" * 70)
    logger.info("Phase 4 -- Model Evaluation on Test Set (N=%d)", len(y_test))
    logger.info("=" * 70)

    # 1. Inference & Latency
    t0 = time.time()
    # LightGBM predict on multiclass returns shape (N, n_classes) of probabilities
    probs_raw: Any = booster.predict(X_test)
    probs: np.ndarray = np.asarray(probs_raw)
    eval_elapsed_sec = time.time() - t0

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
    # In CIC-IDS2017 encoding, class 0 is BENIGN, 1..14 are ATTACKs.
    y_test_bin = (y_test != 0).astype(int)
    preds_bin  = (preds != 0).astype(int)
    # Binary attack probability is (1.0 - P(BENIGN))
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
        "Multiclass Results: Acc=%.4f | Macro F1=%.4f | Weighted F1=%.4f",
        acc, macro_f1, weighted_f1,
    )
    logger.info(
        "Binary Intrusion Results: Detection Rate (TPR)=%.4f | FPR=%.6f | F1=%.4f | ROC-AUC=%.4f",
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
        f"# Phase 4 — ML Baseline Evaluation Report: Random Forest (LightGBM RF Mode){' [SMOKE TEST]' if smoke_test else ''}",
        "",
        f"**Model:** Random Forest Baseline (`lightgbm.LGBMClassifier` via `boosting_type='rf'`)  ",
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
        f"| LightGBM Histogram Binning | `{mem.get('after_construct_rss_mb', 0):.2f} MB` | < 700 MB |",
        f"| Post-Raw Data Freeing | `{mem.get('after_free_raw_rss_mb', 0):.2f} MB` | < 700 MB |",
        f"| Post-Fit Training Peak | `{mem.get('after_fit_rss_mb', 0):.2f} MB` | < 700 MB |",
        f"| Post-Inference Peak | `{mem.get('after_inference_rss_mb', 0):.2f} MB` | < 700 MB |",
        "",
        "---",
        "",
        "## 4. LightGBM Hyperparameter Configuration",
        "",
        "```json",
        json.dumps(training_meta, indent=2),
        "```",
    ])

    return "\n".join(lines)


# ── Step 5: Main Orchestrator ──────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: ML Baseline Random Forest via LightGBM")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run lightweight smoke test on 10k train / 2k test samples to verify execution without resource consumption",
    )
    args = parser.parse_args()

    smoke_test: bool = args.smoke_test
    tag = f"{VERSION_TAG}_smoke" if smoke_test else VERSION_TAG

    logger.info("Starting Phase 4 Pipeline (smoke_test=%s, tag=%s)", smoke_test, tag)

    # 1. Load Data
    X_train, y_train, X_test, y_test, classes, telemetry = load_data(smoke_test=smoke_test)

    # 2. Train Model
    booster, training_meta = train_rf(
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

    model_txt_path = Paths.model(f"rf_baseline_category_{tag}.txt")
    model_pkl_path = Paths.model(f"rf_baseline_category_{tag}.pkl")
    metrics_json_path = Paths.artifact(f"rf_metrics_{tag}.json")
    metrics_md_path = Paths.artifact(f"rf_metrics_{tag}.md")

    # Save model
    booster.save_model(str(model_txt_path))
    joblib.dump(booster, model_pkl_path)
    logger.info("Saved models: %s and %s", model_txt_path.name, model_pkl_path.name)

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
    logger.info("Phase 4 successfully completed!")


if __name__ == "__main__":
    main()
