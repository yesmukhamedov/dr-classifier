"""Experiment 2: Preprocessing Component Ablation + Parameter Sweeps (H-2).

Part A — Component ablation:
  7 V5 pipeline levels trained on EyePACS with k-fold CV using EfficientNet-B3.
  Image quality metrics (CNR, Entropy, SSIM) measured on sample images.

Part B — CLAHE threshold sensitivity sweep.
Part C — Flat-field sigma factor sweep.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.datasets import EyePACSDataset, IDRiDDataset
from src.data.splits import PatientLevelKFold
from src.models.factory import create_model
from src.preprocessing.pipeline_v5 import PreprocessingPipelineV5
from src.preprocessing.config import PreprocessingV5Config
from src.training.checkpoint import CheckpointManager
from src.training.trainer import Trainer
from src.utils.image_quality import compute_all_quality_metrics
from src.utils.seed import set_seed


class _V5PipelineAdapter:
    """Wraps PreprocessingPipelineV5 to return numpy HWC uint8 instead of torch.Tensor.

    datasets.py.__getitem__ expects preprocessing to return a numpy array
    (it handles float conversion and HWC→CHW transposition itself).
    PreprocessingPipelineV5 returns a normalised torch.Tensor (CHW float32).
    This adapter converts the tensor back to numpy HWC uint8 so datasets.py
    can process it correctly.
    """

    def __init__(self, pipeline: "PreprocessingPipelineV5") -> None:
        self._pipeline = pipeline

    def __call__(self, image: np.ndarray) -> np.ndarray:
        tensor = self._pipeline(image)          # (C, H, W) float32 tensor, ImageNet-normalised
        # Undo ImageNet normalisation → [0,1], then → uint8 HWC
        # datasets.py will re-normalise to [0,1] float32 and transpose to CHW
        arr = tensor.permute(1, 2, 0).cpu().numpy()   # (H, W, C) float32
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr  = arr * std + mean                        # undo normalisation → [0,1]
        arr  = (arr * 255).clip(0, 255).astype(np.uint8)
        return arr

# ── Ablation levels ───────────────────────────────────────────────────────────
_ABLATION_LEVELS: list[dict] = [
    {
        "name": "baseline",
        "description": "Stages 1+4 only: FOV crop+resize + ImageNet normalize",
        "flags": dict(use_canonical_flip=False, use_od_fovea_rotation=False,
                      use_flat_field=False, use_clahe=False, use_pca_color=False,
                      use_brightness_contrast=False, use_shear=False, use_stretch=False),
    },
    {
        "name": "baseline_canonical_flip",
        "description": "Stages 0+1+4",
        "flags": dict(use_canonical_flip=True, use_od_fovea_rotation=False,
                      use_flat_field=False, use_clahe=False, use_pca_color=False,
                      use_brightness_contrast=False, use_shear=False, use_stretch=False),
    },
    {
        "name": "baseline_flat_field",
        "description": "Stages 1+2+4",
        "flags": dict(use_canonical_flip=False, use_od_fovea_rotation=False,
                      use_flat_field=True, use_clahe=False, use_pca_color=False,
                      use_brightness_contrast=False, use_shear=False, use_stretch=False),
    },
    {
        "name": "baseline_clahe",
        "description": "Stages 1+3+4",
        "flags": dict(use_canonical_flip=False, use_od_fovea_rotation=False,
                      use_flat_field=False, use_clahe=True, use_pca_color=False,
                      use_brightness_contrast=False, use_shear=False, use_stretch=False),
    },
    {
        "name": "baseline_augmentation",
        "description": "Stages 1+4+5 (train-time aug only)",
        "flags": dict(use_canonical_flip=False, use_od_fovea_rotation=False,
                      use_flat_field=False, use_clahe=False, use_pca_color=True,
                      use_brightness_contrast=True, use_shear=True, use_stretch=True),
    },
    {
        "name": "full_v5",
        "description": "All V5 stages 0+1+2+3+4+5+6+7",
        "flags": dict(use_canonical_flip=True, use_od_fovea_rotation=True,
                      use_flat_field=True, use_clahe=True, use_pca_color=True,
                      use_brightness_contrast=True, use_shear=True, use_stretch=True),
    },
]

_CLAHE_SWEEP_VALUES: list[float] = [
    0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0, 10.0
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_eyepacs_index(
    root: str,
    labels_csv: str,
    subset_size: int | None = None,
) -> tuple[list[str], list[int], list[str]]:
    root = Path(root)
    df = pd.read_csv(labels_csv)
    if subset_size is not None:
        df = df.iloc[:subset_size].reset_index(drop=True)
    paths, labels, pids = [], [], []
    for _, row in df.iterrows():
        name = str(row["image"])
        p = root / f"{name}.jpeg"
        if not p.exists():
            continue
        paths.append(str(p))
        labels.append(int(row["level"]))
        pids.append(name.split("_")[0])
    return paths, labels, pids


def _build_v5_pipeline(
    v5_cfg: dict[str, Any],
    level_flags: dict[str, bool],
    is_training: bool,
) -> PreprocessingPipelineV5:
    """Instantiate a PreprocessingPipelineV5 for a given ablation level.

    Args:
        v5_cfg: The ``preprocessing_v5`` section of the merged config dict.
        level_flags: Toggle overrides for this ablation level (from ``_ABLATION_LEVELS``).
        is_training: Whether to enable stochastic CLAHE and augmentation.

    Returns:
        Configured :class:`PreprocessingPipelineV5` instance.
    """
    preproc_config = PreprocessingV5Config(
        target_size=v5_cfg.get("target_size", 512),
        flat_field_sigma=v5_cfg.get("flat_field_sigma", 45.0),
        flat_field_mode=v5_cfg.get("flat_field_mode", "adaptive"),
        flat_field_sigma_factor=v5_cfg.get("flat_field_sigma_factor", 0.07),
        flat_field_mask_only=v5_cfg.get("flat_field_mask_only", True),
        clahe_clip_factor=v5_cfg.get("clahe_clip_factor", 2.0),
        clahe_global_threshold=v5_cfg.get("clahe_global_threshold", 0.01),
        clahe_tile_grid_size=tuple(v5_cfg.get("clahe_tile_grid_size", [8, 8])),
        clahe_train_prob=v5_cfg.get("clahe_train_prob", 0.8),
        rotation_sigma=v5_cfg.get("rotation_sigma", 13.0),
        rotation_clip=v5_cfg.get("rotation_clip", 40.0),
        zoom_range=tuple(v5_cfg.get("zoom_range", [0.9, 1.1])),
        shear_range=tuple(v5_cfg.get("shear_range", [-2.0, 2.0])),
        shear_prob=v5_cfg.get("shear_prob", 0.3),
        pca_color_sigma=v5_cfg.get("pca_color_sigma", 0.1),
        pca_color_prob=v5_cfg.get("pca_color_prob", 0.5),
        brightness_alpha_range=tuple(v5_cfg.get("brightness_alpha_range", [0.9, 1.1])),
        brightness_beta_range=tuple(v5_cfg.get("brightness_beta_range", [-10.0, 10.0])),
        bc_prob=v5_cfg.get("bc_prob", 0.5),
        **level_flags,
    )
    return PreprocessingPipelineV5(preproc_config, is_training=is_training)


def _measure_quality_on_sample(
    image_paths: list[str],
    pipeline: PreprocessingPipelineV5,
    n_samples: int = 100,
    seed: int = 42,
) -> dict[str, float]:
    """Compute mean CNR, Entropy, SSIM over n_samples images.

    Args:
        image_paths: Pool of image paths to sample from.
        pipeline: Preprocessing pipeline to apply.
        n_samples: Number of images to sample.
        seed: RNG seed for reproducible sampling.

    Returns:
        Dict with mean_cnr, mean_entropy, mean_ssim.
    """
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(image_paths))
    indices = rng.choice(len(image_paths), size=n, replace=False)

    cnrs, entropies, ssims = [], [], []
    for idx in indices:
        raw = cv2.imread(image_paths[idx])
        if raw is None:
            continue
        processed = pipeline(raw)
        # adapter returns numpy HWC uint8 — pass directly to quality metrics
        if isinstance(processed, torch.Tensor):
            processed = (processed.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        q = compute_all_quality_metrics(processed, original=raw)
        cnrs.append(q["cnr"])
        entropies.append(q["entropy"])
        if q["ssim"] is not None:
            ssims.append(q["ssim"])

    result: dict[str, float] = {}
    if cnrs:
        result["mean_cnr"] = float(np.mean(cnrs))
    if entropies:
        result["mean_entropy"] = float(np.mean(entropies))
    if ssims:
        result["mean_ssim"] = float(np.mean(ssims))
    return result


def _compute_summary(per_fold: list[dict]) -> dict[str, str]:
    keys = [
        "val_weighted_f1", "val_roc_auc",
        "val_cohen_kappa_quadratic", "val_accuracy",
    ]
    summary: dict[str, str] = {}
    for k in keys:
        vals = [
            m[k] for m in per_fold
            if k in m and not np.isnan(float(m[k]))
        ]
        if vals:
            summary[k.replace("val_", "")] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
    return summary


# ── Part A ────────────────────────────────────────────────────────────────────

def _run_ablation(
    config: dict[str, Any],
    all_paths: list[str],
    all_labels: list[int],
    all_pids: list[str],
    splits: list[tuple[list[int], list[int]]],
    output_dir: Path,
    metrics_csv: Path,
    trainer: Trainer,
    fold_range: list[int],
    levels_to_run: list[str] | None,
    v5_cfg: dict,
    model_name: str,
    model_cfg: dict,
    quality_n_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Run Part A: component ablation training and quality measurement.

    Returns dict mapping level_name -> {"per_fold": [...], "quality": {...}}.
    """
    results: dict[str, Any] = {}

    for level_spec in _ABLATION_LEVELS:
        level_name = level_spec["name"]
        if levels_to_run is not None and level_name not in levels_to_run:
            continue

        level_flags = level_spec["flags"]
        print(f"\n{'='*65}")
        print(f"  Ablation Level: {level_name}")
        print(f"  Description: {level_spec['description']}")
        print(f"{'='*65}")

        pipeline_train = _V5PipelineAdapter(_build_v5_pipeline(v5_cfg, level_flags, is_training=True))
        pipeline_val   = _V5PipelineAdapter(_build_v5_pipeline(v5_cfg, level_flags, is_training=False))

        # Image quality on sample images (train pool only — fold 0 train split)
        train_idx_0 = splits[0][0]
        sample_paths = [all_paths[i] for i in train_idx_0]
        quality = _measure_quality_on_sample(
            sample_paths, pipeline_val, n_samples=quality_n_samples, seed=seed
        )
        print(f"  Quality — CNR={quality.get('mean_cnr', float('nan')):.3f}  "
              f"Entropy={quality.get('mean_entropy', float('nan')):.3f}  "
              f"SSIM={quality.get('mean_ssim', float('nan')):.3f}")

        per_fold: list[dict] = []
        n_folds = len(splits)

        for fold_idx in fold_range:
            train_idx, val_idx = splits[fold_idx]
            print(f"\n  [Level {level_name} | Fold {fold_idx+1}/{n_folds}]")

            train_ds = EyePACSDataset(
                image_paths=[all_paths[i] for i in train_idx],
                labels=[all_labels[i] for i in train_idx],
                patient_ids=[all_pids[i] for i in train_idx],
                preprocessing=pipeline_train,
                augmentation=None,
            )
            val_ds = EyePACSDataset(
                image_paths=[all_paths[i] for i in val_idx],
                labels=[all_labels[i] for i in val_idx],
                patient_ids=[all_pids[i] for i in val_idx],
                preprocessing=pipeline_val,
                augmentation=None,
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

            ckpt_dir = output_dir / "checkpoints" / f"{level_name}_fold{fold_idx}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_mgr = CheckpointManager(ckpt_dir, max_keep=5)
            model = create_model(model_name, model_cfg)

            best = trainer.train_fold(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                fold=fold_idx,
                config_name=level_name,
                checkpoint_mgr=ckpt_mgr,
                metrics_csv_path=metrics_csv,
                resume=False,
            )
            per_fold.append(best)
            f1 = best.get("val_weighted_f1", float("nan"))
            print(f"  Best F1={f1:.4f}")

        results[level_name] = {"per_fold": per_fold, "quality": quality}

    return results


# ── Part B ────────────────────────────────────────────────────────────────────

def _run_clahe_sweep(
    config: dict[str, Any],
    output_dir: Path,
    metrics_csv: Path,
    trainer: Trainer,
    clip_values: list[float],
    seed: int,
    subset_size: int | None,
    max_epochs: int | None,
) -> dict[str, Any]:
    """Run Part B: CLAHE clip_factor sweep on IDRiD.

    Returns dict mapping str(clip_factor) -> {"per_class_f1": [...], "dr1_f1", "dr2_f1"}.
    """
    v5_cfg = config.get("preprocessing_v5", config.get("preprocessing_v4", {}))

    # Load IDRiD
    idrid_ds_full = IDRiDDataset.from_directory(
        root="/mnt/d/datasets/IDRiD/B. Disease Grading/1. Original Images/a. Training Set",
        labels_csv="/mnt/d/datasets/IDRiD/B. Disease Grading/2. Groundtruths/"
                   "a. IDRiD_Disease Grading_Training Labels.csv",
        subset_indices=list(range(subset_size)) if subset_size else None,
    )
    print(f"  IDRiD: {len(idrid_ds_full)} images for CLAHE sweep")

    # Simple 80/20 train/val split (IDRiD is too small for patient-level 5-fold)
    n = len(idrid_ds_full)
    n_train = int(n * 0.8)
    idx_all = list(range(n))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx_all)
    train_idx = idx_all[:n_train]
    val_idx   = idx_all[n_train:]

    model_name = "efficientnet_b3"
    model_cfg  = dict(config["models"][model_name])
    if max_epochs is not None:
        config = dict(config)
        config["training"] = dict(config["training"])
        config["training"]["max_epochs"] = max_epochs

    sweep_results: dict[str, Any] = {}

    for clip_factor in clip_values:
        key = str(clip_factor)
        print(f"\n  CLAHE sweep — clahe_clip_factor={clip_factor}")

        # Build a V5 config with clahe enabled and the swept clip_factor
        sweep_flags = dict(
            use_canonical_flip=False, use_flat_field=False,
            use_clahe=True, use_pca_color=False,
            use_brightness_contrast=False, use_shear=False, use_stretch=False,
        )
        sweep_v5_cfg = dict(v5_cfg)
        sweep_v5_cfg["clahe_clip_factor"] = clip_factor

        pipeline_train = _V5PipelineAdapter(_build_v5_pipeline(sweep_v5_cfg, sweep_flags, is_training=True))
        pipeline_val   = _V5PipelineAdapter(_build_v5_pipeline(sweep_v5_cfg, sweep_flags, is_training=False))

        train_ds = IDRiDDataset(
            image_paths=[idrid_ds_full.image_paths[i] for i in train_idx],
            labels=[idrid_ds_full.labels[i] for i in train_idx],
            patient_ids=[idrid_ds_full.patient_ids[i] for i in train_idx],
            image_stems=[idrid_ds_full.image_stems[i] for i in train_idx],
            masks_root=None,
            preprocessing=pipeline_train,
            augmentation=None,
        )
        val_ds = IDRiDDataset(
            image_paths=[idrid_ds_full.image_paths[i] for i in val_idx],
            labels=[idrid_ds_full.labels[i] for i in val_idx],
            patient_ids=[idrid_ds_full.patient_ids[i] for i in val_idx],
            image_stems=[idrid_ds_full.image_stems[i] for i in val_idx],
            masks_root=None,
            preprocessing=pipeline_val,
            augmentation=None,
        )

        train_loader = DataLoader(
            train_ds, batch_size=trainer.batch_size, shuffle=True,
            num_workers=trainer.num_workers,
            pin_memory=(trainer.device.type == "cuda"), drop_last=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=trainer.batch_size, shuffle=False,
            num_workers=trainer.num_workers,
            pin_memory=(trainer.device.type == "cuda"),
        )

        ckpt_dir = output_dir / "checkpoints" / f"clahe_{key.replace('.','p')}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_mgr  = CheckpointManager(ckpt_dir, max_keep=2)
        sweep_csv = output_dir / "metrics_clahe_sweep.csv"
        model     = create_model(model_name, model_cfg)

        # Build a local trainer with potentially overridden max_epochs
        local_trainer = Trainer(config, device="auto")
        best = local_trainer.train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            fold=0,
            config_name=f"clahe_{key}",
            checkpoint_mgr=ckpt_mgr,
            metrics_csv_path=sweep_csv,
            resume=False,
        )

        per_class_f1 = best.get("val_per_class_f1", [])
        dr1_f1 = per_class_f1[1] if len(per_class_f1) > 1 else float("nan")
        dr2_f1 = per_class_f1[2] if len(per_class_f1) > 2 else float("nan")
        print(f"  clip_factor={clip_factor}: weighted_F1={best.get('val_weighted_f1', float('nan')):.4f}  "
              f"DR1_F1={dr1_f1:.4f}  DR2_F1={dr2_f1:.4f}")

        sweep_results[key] = {
            "clahe_clip_factor": clip_factor,
            "weighted_f1":       best.get("val_weighted_f1", float("nan")),
            "per_class_f1":      per_class_f1,
            "dr1_f1":            dr1_f1,
            "dr2_f1":            dr2_f1,
        }

    return sweep_results


# ── Entry point ───────────────────────────────────────────────────────────────

def run(
    config: dict[str, Any],
    fold: int | None = None,
    resume: bool = False,
    _subset_size: int | None = None,
    _levels_to_run: list[str] | None = None,
    _clahe_values: list[float] | None = None,
    _quality_n_samples: int = 100,
    _configs_to_run: list[str] | None = None,  # ignored — accepted for CLI compat
) -> None:
    """Run Experiment 2: V4 component ablation + CLAHE clip_factor sweep.

    Args:
        config: Full merged config dict.
        fold: If set, run only this fold index for Part A.
        resume: Resume from checkpoint (Part A only).
        _subset_size: Limit EyePACS CSV rows (smoke test).
        _levels_to_run: Run only these level names (smoke test).
        _clahe_values: Override CLAHE sweep clip_factor values (smoke test).
        _quality_n_samples: Number of images for quality metric sampling.
        _configs_to_run: Ignored — accepted for run_experiment.py CLI compatibility.
    """
    set_seed(config.get("seed", 42))
    seed = config.get("seed", 42)

    output_dir = Path(config["paths"]["output_dir"]) / "exp2"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    metrics_csv = output_dir / "metrics.csv"

    cv_cfg  = config["cross_validation"]
    v5_cfg  = config.get("preprocessing_v5", config.get("preprocessing_v4", {}))
    n_folds = cv_cfg["n_folds"]
    fold_range = [fold] if fold is not None else list(range(n_folds))

    model_name = "efficientnet_b3"
    model_cfg  = config["models"][model_name]

    # Architecture: EfficientNet-B3 (Exp 1 winner per RESEARCH_ARCHITECTURE §5.2)
    # Mixed precision: read from per-model config (D-6 design decision)
    use_amp = config.get("models", {}).get(model_name, {}).get("mixed_precision", False)

    trainer = Trainer(config, device="auto")
    trainer.mixed_precision = use_amp

    # ── Load EyePACS index ────────────────────────────────────────────────────
    eyepacs_root = config["paths"]["eyepacs"]
    labels_csv   = str(Path(eyepacs_root) / "trainLabels.csv")
    images_root  = str(Path(eyepacs_root) / "train")

    print(f"Loading EyePACS index …")
    all_paths, all_labels, all_pids = _load_eyepacs_index(
        images_root, labels_csv, subset_size=_subset_size
    )
    print(f"  {len(all_paths)} images | {len(set(all_pids))} patients")

    splitter = PatientLevelKFold(n_folds=n_folds, seed=seed,
                                 stratified=cv_cfg.get("stratified", True))
    splits = splitter.split(all_paths, all_labels, all_pids)
    ok = splitter.verify_no_leakage(splits, all_pids)
    if not ok:
        raise RuntimeError("Patient leakage in CV splits — aborting.")
    print(f"  {n_folds}-fold splits verified (no leakage)")

    # ════════════════════════════════════════════════════════════
    # PART A — Component Ablation
    # ════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("PART A — V4 Component Ablation")
    print(f"{'='*65}")

    ablation_results = _run_ablation(
        config=config,
        all_paths=all_paths,
        all_labels=all_labels,
        all_pids=all_pids,
        splits=splits,
        output_dir=output_dir,
        metrics_csv=metrics_csv,
        trainer=trainer,
        fold_range=fold_range,
        levels_to_run=_levels_to_run,
        v5_cfg=v5_cfg,
        model_name=model_name,
        model_cfg=model_cfg,
        quality_n_samples=_quality_n_samples,
        seed=seed,
    )

    # Ablation summary
    ablation_summary: dict[str, Any] = {}
    if fold is None:
        for level_name, res in ablation_results.items():
            ablation_summary[level_name] = {
                "metrics": _compute_summary(res["per_fold"]),
                "quality": res["quality"],
            }
        summary_path = output_dir / "ablation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(ablation_summary, f, indent=2)
        print(f"\n  Ablation summary saved to {summary_path}")

    # ════════════════════════════════════════════════════════════
    # PART B — CLAHE Sweep
    # ════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("PART B — CLAHE Clip Factor Sweep (H-2)")
    print(f"{'='*65}")

    clahe_values = _clahe_values if _clahe_values is not None else _CLAHE_SWEEP_VALUES

    # For the sweep on IDRiD, use max_epochs from config (or overridden)
    sweep_results = _run_clahe_sweep(
        config=config,
        output_dir=output_dir,
        metrics_csv=metrics_csv,
        trainer=trainer,
        clip_values=clahe_values,
        seed=seed,
        subset_size=_subset_size,
        max_epochs=None,
    )

    clahe_path = output_dir / "clahe_sweep.json"
    with open(clahe_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\n  CLAHE sweep results saved to {clahe_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    if fold is None and ablation_summary:
        print(f"\n{'='*65}")
        print("Experiment 2 — Ablation Summary")
        print(f"{'='*65}")
        for level_name, entry in ablation_summary.items():
            f1_str = entry["metrics"].get("weighted_f1", "N/A")
            cnr    = entry["quality"].get("mean_cnr", float("nan"))
            ent    = entry["quality"].get("mean_entropy", float("nan"))
            print(f"  {level_name:<28s}: F1={f1_str}  CNR={cnr:.3f}  Entropy={ent:.3f}")
