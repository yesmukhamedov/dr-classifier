"""Shared inference helpers for Experiments 5 and 6 (no-retrain evaluation).

Private module — not part of the public package API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.data.augmentation import FundusAugmentation
from src.data.datasets import EyePACSDataset
from src.data.splits import PatientLevelKFold
from src.models.factory import create_model
from src.preprocessing.pipeline_v5 import PreprocessingPipelineV5
from src.preprocessing.config import PreprocessingV5Config
from src.training.checkpoint import CheckpointManager
from src.training.trainer import Trainer


# ── Pipeline builder ──────────────────────────────────────────────────────────

def build_full_pipeline(v5_cfg: dict, is_training: bool = False) -> PreprocessingPipelineV5:
    """Build the full V5 8-stage preprocessing pipeline from config.

    Args:
        v5_cfg: The ``preprocessing_v5`` section of the merged config dict.
        is_training: Whether to enable stochastic augmentation.

    Returns:
        Configured V5 pipeline instance.
    """
    pp_config = PreprocessingV5Config.from_dict(v5_cfg)
    return PreprocessingPipelineV5(pp_config, is_training=is_training)


# ── Model loading / training ──────────────────────────────────────────────────

def load_or_train_model(
    config: dict[str, Any],
    output_dir: Path,
    subset_size: int | None = None,
    resume: bool = False,
) -> tuple[nn.Module, float]:
    """Return a trained EfficientNet-B3 (full preprocessing) and its EyePACS F1.

    Tries the following in order:
      1. Load `outputs/exp1/checkpoints/D_fold0/best_model.pt` (exp1 config D).
      2. Load `outputs/exp1/checkpoints/B_fold0/best_model.pt` (exp1 config B).
      3. Train a fresh model on EyePACS fold 0 with full preprocessing.

    Args:
        config: Merged experiment config.
        output_dir: Output directory for this experiment (used when training fresh).
        subset_size: Limit EyePACS rows — used for smoke tests.
        resume: Resume from checkpoint when training fresh.

    Returns:
        Tuple of (model on eval mode, eyepacs_val_f1).  eyepacs_val_f1 is the
        validation weighted-F1 from the checkpoint metrics, or NaN if unknown.
    """
    model_name = "efficientnet_b3"
    model_cfg  = config["models"][model_name]
    exp1_root  = Path(config["paths"]["output_dir"]) / "exp1" / "checkpoints"

    for candidate_tag in ("D_fold0", "B_fold0"):
        ckpt_path = exp1_root / candidate_tag / "best_model.pt"
        if ckpt_path.exists():
            print(f"  Loading exp1 checkpoint: {ckpt_path}")
            model = create_model(model_name, model_cfg)
            ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            f1 = ckpt.get("metrics", {}).get("val_weighted_f1", float("nan"))
            model.eval()
            return model, f1

    # No exp1 checkpoint — train fresh
    print("  No exp1 checkpoint found — training fresh EfficientNet-B3 on EyePACS …")
    return _train_fresh(config, model_name, model_cfg, output_dir, subset_size, resume)


def _train_fresh(
    config: dict[str, Any],
    model_name: str,
    model_cfg: dict,
    output_dir: Path,
    subset_size: int | None,
    resume: bool,
) -> tuple[nn.Module, float]:
    """Train EfficientNet-B3 with full preprocessing on EyePACS fold 0."""
    eyepacs_root = config["paths"]["eyepacs"]
    labels_csv   = str(Path(eyepacs_root) / "trainLabels.csv")
    images_root  = str(Path(eyepacs_root) / "train")

    df = pd.read_csv(labels_csv)
    if subset_size is not None:
        df = df.iloc[:subset_size].reset_index(drop=True)

    all_paths, all_labels, all_pids = [], [], []
    for _, row in df.iterrows():
        name = str(row["image"])
        p = Path(images_root) / f"{name}.jpeg"
        if not p.exists():
            continue
        all_paths.append(str(p))
        all_labels.append(int(row["level"]))
        all_pids.append(name.split("_")[0])

    cv_cfg   = config["cross_validation"]
    seed     = config.get("seed", 42)
    splitter = PatientLevelKFold(
        n_folds=cv_cfg["n_folds"], seed=seed,
        stratified=cv_cfg.get("stratified", True),
    )
    splits = splitter.split(all_paths, all_labels, all_pids)
    train_idx, val_idx = splits[0]

    v5_cfg   = config["preprocessing_v5"]
    pipeline = build_full_pipeline(v5_cfg, is_training=True)
    aug         = FundusAugmentation(config["augmentation"])
    trainer     = Trainer(config, device="auto")

    train_ds = EyePACSDataset(
        image_paths=[all_paths[i] for i in train_idx],
        labels=[all_labels[i] for i in train_idx],
        patient_ids=[all_pids[i] for i in train_idx],
        preprocessing=pipeline, augmentation=aug,
    )
    val_ds = EyePACSDataset(
        image_paths=[all_paths[i] for i in val_idx],
        labels=[all_labels[i] for i in val_idx],
        patient_ids=[all_pids[i] for i in val_idx],
        preprocessing=pipeline, augmentation=None,
    )
    train_loader = DataLoader(
        train_ds, batch_size=trainer.batch_size, shuffle=True,
        num_workers=trainer.num_workers,
        pin_memory=(trainer.device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=trainer.batch_size, shuffle=False,
        num_workers=trainer.num_workers,
        pin_memory=(trainer.device.type == "cuda"),
    )

    ckpt_dir = output_dir / "checkpoints" / "eyepacs_full_fold0"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_mgr = CheckpointManager(ckpt_dir, max_keep=5)
    model    = create_model(model_name, model_cfg)

    best = trainer.train_fold(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        fold=0,
        config_name="eyepacs_full",
        checkpoint_mgr=ckpt_mgr,
        metrics_csv_path=output_dir / "metrics_eyepacs.csv",
        resume=resume,
    )
    ckpt_mgr.load_best(model)
    model.eval()
    return model, best.get("val_weighted_f1", float("nan"))


# ── Dataset evaluation ────────────────────────────────────────────────────────

def evaluate_dataset(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    """Run inference on a dataset and return the full metrics dict.

    Args:
        model: Trained model in eval mode.
        dataset: Any Dataset that returns (image_tensor, label).
        config: Full config (reads batch_size, num_workers, mixed_precision).
        device: Device to run inference on.

    Returns:
        Dict with weighted_f1, roc_auc, cohen_kappa_quadratic, accuracy,
        sensitivity, specificity, ppv, npv, binary_roc_auc.
    """
    tc = config.get("training", {})
    batch_size  = tc.get("batch_size", 16)
    num_workers = tc.get("num_workers", 4)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    trainer  = Trainer(config, device=str(device))
    criterion = nn.CrossEntropyLoss()

    metrics, all_preds, all_probs, all_labels = trainer.evaluate(
        model, loader, criterion
    )

    # Strip the "val_" prefix for cleaner output keys
    result: dict[str, float] = {
        k.replace("val_", ""): v
        for k, v in metrics.items()
    }

    # Binary clinical ROC-AUC (referable = grade >= 2)
    y_bin = (np.asarray(all_labels) >= 2).astype(int)
    ref_prob = all_probs[:, 2:].sum(axis=1)
    try:
        result["binary_roc_auc"] = float(roc_auc_score(y_bin, ref_prob))
    except ValueError:
        result["binary_roc_auc"] = float("nan")

    return result


def evaluate_dataset_binary(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    config: dict[str, Any],
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Run inference and compute binary DR metrics for binary-labeled datasets.

    Intended for datasets such as RFMiD that provide only binary DR labels
    (0 = No DR, 1 = DR present).  The 5-class model's "any DR" probability
    is used as the score: ``any_dr_prob = sum(softmax[:, 1:])``.

    Args:
        model: Trained 5-class model in eval mode.
        dataset: Dataset returning (image_tensor, label) where label ∈ {0, 1}.
        config: Full config (reads batch_size, num_workers, mixed_precision).
        device: Device to run inference on.
        threshold: Probability threshold for the positive ("any DR") class.

    Returns:
        Dict with keys:
          sensitivity, specificity, binary_f1, binary_accuracy,
          binary_roc_auc, n_positive, n_negative,
          weighted_f1 (= binary_f1, for H-6 g_ratio compatibility),
          roc_auc     (= binary_roc_auc, for H-6 g_ratio compatibility),
          evaluation_mode, label_scheme.
    """
    tc = config.get("training", {})
    batch_size  = tc.get("batch_size", 16)
    num_workers = tc.get("num_workers", 4)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    trainer   = Trainer(config, device=str(device))
    criterion = nn.CrossEntropyLoss()

    _, all_preds, all_probs, all_labels = trainer.evaluate(
        model, loader, criterion
    )

    y_true = np.asarray(all_labels, dtype=int)

    # "Any DR" score: sum of softmax probabilities for classes 1–4
    any_dr_prob  = all_probs[:, 1:].sum(axis=1)
    y_pred_bin   = (any_dr_prob >= threshold).astype(int)

    try:
        binary_roc_auc = float(roc_auc_score(y_true, any_dr_prob))
    except ValueError:
        binary_roc_auc = float("nan")

    sensitivity = float(recall_score(y_true, y_pred_bin, pos_label=1, zero_division=0))
    specificity = float(recall_score(y_true, y_pred_bin, pos_label=0, zero_division=0))
    binary_f1   = float(f1_score(y_true, y_pred_bin, zero_division=0))
    binary_acc  = float(accuracy_score(y_true, y_pred_bin))

    n_positive = int(y_true.sum())
    n_negative = int(len(y_true) - n_positive)

    return {
        "sensitivity":      sensitivity,
        "specificity":      specificity,
        "binary_f1":        binary_f1,
        "binary_accuracy":  binary_acc,
        "binary_roc_auc":   binary_roc_auc,
        "n_positive":       n_positive,
        "n_negative":       n_negative,
        # Aliases for H-6 g_ratio + variance computation
        "weighted_f1":      binary_f1,
        "roc_auc":          binary_roc_auc,
        "evaluation_mode":  "binary",
        "label_scheme":     "0=No DR, 1=DR present (any severity)",
    }
