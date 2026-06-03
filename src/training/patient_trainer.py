"""
Patient-level training loop for per-patient blending models.

Handles dict-based batches from :class:`~src.data.datasets.EyePACSPatientPairDataset`
and implements a two-stage training protocol:

Stage 1 — Backbone pre-training as a single-image classifier.
    The backbone (with Identity head) is temporarily wrapped in a thin linear
    head and trained using the standard :class:`~src.training.trainer.Trainer`.
    After Stage 1 the linear head is discarded; the backbone retains its
    fine-tuned weights.

Stage 2 — Patient-pair blending.
    (a) Backbone is frozen.  :class:`~src.models.patient_model.PatientHead`
        is trained for ``head_warmup_epochs`` on patient-pair batches.
    (b) Backbone is unfrozen with a lower LR (1 × 10⁻⁵) while the head
        continues with a higher LR (1 × 10⁻⁴).  Training continues until
        early stopping.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import (
    compute_primary_metrics,
    compute_secondary_metrics,
    compute_clinical_metrics,
)
from src.models.patient_model import DRPatientModel
from src.training.checkpoint import CheckpointManager
from src.training.losses import compute_class_weights, create_loss
from src.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Temporary single-image wrapper (Stage 1 helper)
# ---------------------------------------------------------------------------

class _SingleImageWrapper(nn.Module):
    """Wraps a :class:`~src.models.patient_model.Backbone` + linear head
    so Stage 1 can reuse the standard single-image :class:`Trainer`.

    Args:
        backbone: Backbone with Identity classifier (as produced by
            :func:`~src.models.factory.create_patient_model`).
        num_classes: Number of output classes.
    """

    def __init__(self, backbone: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    # Expose labels attribute passthrough for Trainer class-weight extraction
    @property
    def labels(self) -> list[int]:
        return []


# ---------------------------------------------------------------------------
# PatientTrainer
# ---------------------------------------------------------------------------

class PatientTrainer:
    """Training loop for per-patient blending models.

    Parses the same config keys as :class:`~src.training.trainer.Trainer`
    and adds patient-pair epoch routines and the two-stage protocol.

    Args:
        config: Full config dict (from ``default.yaml``).  Reads the same
            ``training.*`` sub-keys as :class:`Trainer`.
        device: ``"auto"`` selects CUDA when available, otherwise CPU.
    """

    # Stage 2 hyper-parameters (can be overridden via config)
    HEAD_WARMUP_EPOCHS: int = 5
    BACKBONE_LR_STAGE2: float = 1e-5
    HEAD_LR_STAGE2: float = 1e-4

    def __init__(self, config: dict, device: str = "auto") -> None:
        self.config = config
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        tc = config.get("training", {})
        self.lr: float = tc.get("learning_rate", 1e-4)
        self.weight_decay: float = tc.get("weight_decay", 1e-4)
        self.batch_size: int = tc.get("batch_size", 16)
        self.max_epochs: int = tc.get("max_epochs", 100)
        self.mixed_precision: bool = tc.get("mixed_precision", True)
        self.num_workers: int = tc.get("num_workers", 4)
        self.use_class_weights: bool = tc.get("class_weights") == "inverse_frequency"
        self.loss_type: str = tc.get("loss_type", "focal")
        self.focal_gamma: float = tc.get("focal_gamma", 2.0)

        es = tc.get("early_stopping", {})
        self.es_patience: int = es.get("patience", 10)

        sched = tc.get("scheduler", {})
        self.sched_factor: float = sched.get("factor", 0.5)
        self.sched_patience: int = sched.get("patience", 5)
        self.sched_min_lr: float = sched.get("min_lr", 1e-6)

        self.output_dir = Path(config.get("paths", {}).get("output_dir", "outputs/"))
        self.num_classes: int = 5

    # ------------------------------------------------------------------
    # Core epoch routines — patient-pair batches
    # ------------------------------------------------------------------

    def train_one_epoch_patient(
        self,
        model: DRPatientModel,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
    ) -> dict:
        """Run one training epoch on patient-pair dict batches.

        Args:
            model: :class:`~src.models.patient_model.DRPatientModel`.
            dataloader: DataLoader yielding dicts with keys
                ``"left"``, ``"right"``, ``"label"``.
            criterion: Loss function.
            optimizer: Optimiser.
            scaler: GradScaler (no-op when mixed_precision=False).

        Returns:
            Dict with ``train_loss`` (float) and ``train_accuracy`` (float).
        """
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            img_L = batch["left"].to(self.device, non_blocking=True)
            img_R = batch["right"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.mixed_precision):
                logits = model(img_L, img_R)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        n = max(total, 1)
        return {"train_loss": total_loss / n, "train_accuracy": correct / n}

    def evaluate_patient(
        self,
        model: DRPatientModel,
        dataloader: DataLoader,
        criterion: nn.Module,
    ) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate a patient model on dict-based batches.

        Args:
            model: :class:`~src.models.patient_model.DRPatientModel`.
            dataloader: DataLoader yielding dicts with keys
                ``"left"``, ``"right"``, ``"label"``.
            criterion: Loss function (for ``val_loss`` tracking).

        Returns:
            Tuple of:
              - metrics_dict: primary + secondary + clinical metrics
              - all_preds: shape ``(N,)`` int array
              - all_probs: shape ``(N, num_classes)`` float array
              - all_labels: shape ``(N,)`` int array
        """
        model.eval()
        total_loss = 0.0
        all_preds: list[np.ndarray] = []
        all_probs: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        with torch.no_grad():
            for batch in dataloader:
                img_L = batch["left"].to(self.device, non_blocking=True)
                img_R = batch["right"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)

                logits = model(img_L, img_R)
                loss = criterion(logits.float(), labels)

                total_loss += loss.item() * labels.size(0)
                probs = torch.softmax(logits.float(), dim=1)
                preds = logits.argmax(dim=1)

                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().float().numpy())
                all_labels.append(labels.cpu().numpy())

        preds_arr = np.concatenate(all_preds)
        probs_arr = np.concatenate(all_probs)
        labels_arr = np.concatenate(all_labels)

        primary = compute_primary_metrics(labels_arr, preds_arr, probs_arr, self.num_classes)
        secondary = compute_secondary_metrics(labels_arr, preds_arr, self.num_classes)
        clinical = compute_clinical_metrics(labels_arr, preds_arr)

        metrics = {
            "val_loss": total_loss / max(len(labels_arr), 1),
            **{f"val_{k}": v for k, v in primary.items()},
            **{f"val_{k}": v for k, v in secondary.items()},
            **{f"val_{k}": v for k, v in clinical.items()},
        }
        return metrics, preds_arr, probs_arr, labels_arr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append_metrics_csv(
        self,
        path: Path,
        epoch: int,
        fold: int,
        config_name: str,
        train_loss: float,
        val_loss: float,
        val_metrics: dict,
    ) -> None:
        """Append one row to the experiment metrics CSV."""
        row = {
            "epoch": epoch,
            "fold": fold,
            "config": config_name,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "weighted_f1": f"{val_metrics.get('val_weighted_f1', float('nan')):.6f}",
            "roc_auc": f"{val_metrics.get('val_roc_auc', float('nan')):.6f}",
            "kappa": f"{val_metrics.get('val_cohen_kappa_quadratic', float('nan')):.6f}",
            "accuracy": f"{val_metrics.get('val_accuracy', float('nan')):.6f}",
        }
        write_header = not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["epoch", "fold", "config", "train_loss", "val_loss",
                            "weighted_f1", "roc_auc", "kappa", "accuracy"],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(row)
