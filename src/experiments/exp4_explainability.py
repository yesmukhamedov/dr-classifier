"""Experiment 4: Grad-CAM Explainability Analysis (H-5, PC-7).

Protocol:
  1. Train two EfficientNet-B4 models on EyePACS:
       a. Baseline  — resize-only preprocessing
       b. Full      — all 5-component preprocessing pipeline
  2. Sample up to 10 IDRiD images per DR class (50 total).
  3. Per image: generate Grad-CAM for both models, load lesion masks,
     compute ALO (primary) and IoU (secondary) per lesion type.
  4. Save Grad-CAM comparison figures.
  5. Test H-5: IoU_preproc > IoU_baseline for ≥ 3 of 4 lesion types.

Output: <output_dir>/exp4/iou_results.json
        <output_dir>/exp4/gradcam/  (PNG per IDRiD image)

NC-14 (INVARIANTS): Grad-CAM activation does NOT constitute clinical
localization of pathology — it is an interpretability tool only.
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

from src.data.augmentation import FundusAugmentation
from src.data.datasets import EyePACSDataset, IDRiDDataset
from src.data.splits import PatientLevelKFold
from src.explainability.gradcam import GradCAMGenerator
from src.explainability.iou import (
    compute_alo_per_lesion_type,
    compute_attention_overlap,
    compute_iou_per_lesion_type,
)
from src.explainability.visualization import create_comparison_figure
from src.models.efficientnet import create_efficientnet, get_gradcam_target_layer
from src.preprocessing.pipeline_v5 import PreprocessingPipelineV5
from src.preprocessing.config import PreprocessingV5Config
from src.training.checkpoint import CheckpointManager
from src.training.trainer import Trainer
from src.utils.seed import set_seed

_LESION_TYPES = ["microaneurysms", "haemorrhages", "hard_exudates", "soft_exudates"]
_SAMPLES_PER_CLASS = 10
_NUM_CLASSES = 5


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


def _build_pipeline(config: dict, full: bool, is_training: bool = False) -> PreprocessingPipelineV5:
    """Build baseline (resize-only) or full V5 preprocessing pipeline."""
    if full:
        v5_cfg = config.get("preprocessing_v5", {})
        pp_config = PreprocessingV5Config.from_dict(v5_cfg)
        return PreprocessingPipelineV5(pp_config, is_training=is_training)
    target_size = config.get("preprocessing_v5", {}).get("target_size", 512)
    return PreprocessingPipelineV5.create_baseline_v5(
        target_size=target_size, is_training=is_training,
    )


def _train_model(
    config: dict,
    pipeline: PreprocessingPipelineV5,
    all_paths: list[str],
    all_labels: list[int],
    all_pids: list[str],
    splits: list[tuple],
    output_dir: Path,
    config_name: str,
    resume: bool,
) -> torch.nn.Module:
    """Train EfficientNet-B4 on EyePACS fold 0 with the given pipeline.

    Args:
        config: Merged experiment config.
        pipeline: Preprocessing pipeline to use.
        all_paths / all_labels / all_pids: Full EyePACS index.
        splits: 5-fold split list.
        output_dir: Experiment output root (checkpoints saved here).
        config_name: Tag string (e.g. "baseline" or "full_pipeline").
        resume: Resume from checkpoint if available.

    Returns:
        Trained nn.Module in eval mode on the trainer device.
    """
    aug_cfg   = config["augmentation"]
    model_cfg = dict(config["models"]["efficientnet_b4"])

    trainer = Trainer(config, device="auto")

    train_idx, val_idx = splits[0]
    train_ds = EyePACSDataset(
        image_paths=[all_paths[i] for i in train_idx],
        labels=[all_labels[i] for i in train_idx],
        patient_ids=[all_pids[i] for i in train_idx],
        preprocessing=pipeline,
        augmentation=FundusAugmentation(aug_cfg),
    )
    val_ds = EyePACSDataset(
        image_paths=[all_paths[i] for i in val_idx],
        labels=[all_labels[i] for i in val_idx],
        patient_ids=[all_pids[i] for i in val_idx],
        preprocessing=pipeline,
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

    ckpt_dir = output_dir / "checkpoints" / config_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_mgr = CheckpointManager(ckpt_dir, max_keep=5)
    model    = create_efficientnet(variant="b4", **model_cfg)

    trainer.train_fold(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        fold=0,
        config_name=config_name,
        checkpoint_mgr=ckpt_mgr,
        metrics_csv_path=output_dir / f"metrics_{config_name}.csv",
        resume=resume,
    )

    ckpt_mgr.load_best(model)
    model.eval()
    return model


def _sample_idrid_indices(
    dataset: IDRiDDataset,
    n_per_class: int,
    seed: int,
) -> list[int]:
    """Sample up to n_per_class indices per DR grade (0–4).

    Images with at least one lesion mask are prioritised.

    Args:
        dataset: IDRiD dataset.
        n_per_class: Maximum number of images to sample per DR grade.
        seed: RNG seed.

    Returns:
        Sorted list of dataset indices.
    """
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = {c: [] for c in range(_NUM_CLASSES)}
    for i, label in enumerate(dataset.labels):
        by_class[label].append(i)

    selected: list[int] = []
    for cls in range(_NUM_CLASSES):
        indices = by_class[cls]
        if not indices:
            continue
        # Prefer indices that have masks
        has_masks = [i for i in indices if dataset.get_lesion_masks(i) is not None]
        no_masks  = [i for i in indices if dataset.get_lesion_masks(i) is None]
        pool = has_masks + no_masks  # prioritise mask-having images
        n = min(n_per_class, len(pool))
        chosen = rng.choice(len(pool), size=n, replace=False)
        selected.extend([pool[j] for j in chosen])

    return sorted(selected)


def _image_to_tensor(
    image: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Convert a preprocessed HWC float32 [0,1] or uint8 image to (1,C,H,W) tensor.

    Args:
        image: BGR image, uint8 or float32.
        device: Target device.

    Returns:
        Float32 tensor of shape (1, 3, H, W) in [0, 1].
    """
    if image.dtype == np.uint8:
        img_f = image.astype(np.float32) / 255.0
    else:
        img_f = image.astype(np.float32)
    tensor = torch.from_numpy(
        np.ascontiguousarray(img_f.transpose(2, 0, 1))
    ).unsqueeze(0).to(device)
    return tensor


# ── Entry point ───────────────────────────────────────────────────────────────

def run(
    config: dict[str, Any],
    fold: int | None = None,
    resume: bool = False,
    _subset_size: int | None = None,
) -> None:
    """Run Experiment 4: Grad-CAM explainability analysis.

    Args:
        config: Full merged config dict.
        fold: Unused (included for CLI interface consistency).
        resume: Resume training from checkpoint if available.
        _subset_size: Limit EyePACS rows for smoke test.
    """
    set_seed(config.get("seed", 42))
    seed = config.get("seed", 42)

    output_dir = Path(config["paths"]["output_dir"]) / "exp4"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    gradcam_dir = output_dir / "gradcam"
    gradcam_dir.mkdir(exist_ok=True)

    cv_cfg      = config["cross_validation"]
    # V5 pipeline config
    n_folds     = cv_cfg["n_folds"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load EyePACS index ────────────────────────────────────────────────────
    eyepacs_root = config["paths"]["eyepacs"]
    labels_csv   = str(Path(eyepacs_root) / "trainLabels.csv")
    images_root  = str(Path(eyepacs_root) / "train")

    print("Loading EyePACS index …")
    all_paths, all_labels, all_pids = _load_eyepacs_index(
        images_root, labels_csv, subset_size=_subset_size
    )
    print(f"  {len(all_paths)} images | {len(set(all_pids))} patients")

    splitter = PatientLevelKFold(
        n_folds=n_folds, seed=seed,
        stratified=cv_cfg.get("stratified", True),
    )
    splits = splitter.split(all_paths, all_labels, all_pids)
    if not splitter.verify_no_leakage(splits, all_pids):
        raise RuntimeError("Patient leakage in CV splits — aborting.")

    # ── Build preprocessing pipelines ────────────────────────────────────────
    pipeline_baseline = _build_pipeline(config, full=False, is_training=False)
    pipeline_full     = _build_pipeline(config, full=True, is_training=False)

    # ── Train or load both models ─────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("Training Baseline model (resize only) …")
    print(f"{'='*65}")
    model_baseline = _train_model(
        config, pipeline_baseline, all_paths, all_labels, all_pids,
        splits, output_dir, "baseline", resume,
    )

    print(f"\n{'='*65}")
    print("Training Full-Pipeline model …")
    print(f"{'='*65}")
    model_full = _train_model(
        config, pipeline_full, all_paths, all_labels, all_pids,
        splits, output_dir, "full_pipeline", resume,
    )

    model_baseline = model_baseline.to(device).eval()
    model_full     = model_full.to(device).eval()

    # ── Grad-CAM generators ───────────────────────────────────────────────────
    tl_baseline = get_gradcam_target_layer(model_baseline, variant="b4")
    tl_full     = get_gradcam_target_layer(model_full,     variant="b4")
    cam_baseline = GradCAMGenerator(model_baseline, tl_baseline, device=str(device))
    cam_full     = GradCAMGenerator(model_full,     tl_full,     device=str(device))

    # ── Load IDRiD dataset ────────────────────────────────────────────────────
    idrid_root  = config["paths"]["idrid"]
    idrid_imgs  = str(Path(idrid_root) / "B. Disease Grading" /
                      "1. Original Images" / "a. Training Set")
    idrid_csv   = str(Path(idrid_root) / "B. Disease Grading" / "2. Groundtruths" /
                      "a. IDRiD_Disease Grading_Training Labels.csv")
    idrid_masks = str(Path(idrid_root) / "A. Segmentation" /
                      "2. All Segmentation Groundtruths" / "a. Training Set")

    idrid_ds = IDRiDDataset.from_directory(
        root=idrid_imgs,
        labels_csv=idrid_csv,
        lesion_mask_dir=idrid_masks,
    )
    print(f"\nIDRiD: {len(idrid_ds)} images | "
          f"{idrid_ds.count_images_with_masks()} with lesion masks")

    # ── Sample images per DR class ────────────────────────────────────────────
    sample_indices = _sample_idrid_indices(idrid_ds, _SAMPLES_PER_CLASS, seed)
    print(f"Sampled {len(sample_indices)} IDRiD images for Grad-CAM analysis")

    # ── Per-image Grad-CAM + IoU/ALO computation ──────────────────────────────
    per_image_results: list[dict[str, Any]] = []

    for idx in sample_indices:
        stem     = idrid_ds.image_stems[idx]
        dr_grade = idrid_ds.labels[idx]
        raw      = cv2.imread(str(idrid_ds.image_paths[idx]))
        if raw is None:
            print(f"  WARNING: could not load {idrid_ds.image_paths[idx]}")
            continue

        # Apply respective pipelines
        img_base  = pipeline_baseline(raw)
        img_full  = pipeline_full(raw)

        # Tensors
        t_base  = _image_to_tensor(img_base, device)
        t_full  = _image_to_tensor(img_full, device)

        # Grad-CAM heatmaps (H, W) float32 [0,1]
        hm_baseline = cam_baseline.generate(t_base,  target_class=None)
        hm_full     = cam_full.generate(t_full, target_class=None)

        # Lesion masks
        masks = idrid_ds.get_lesion_masks(idx) or {}

        # IoU and ALO per lesion type
        iou_base  = compute_iou_per_lesion_type(hm_baseline, masks)
        iou_full  = compute_iou_per_lesion_type(hm_full,     masks)
        alo_base  = compute_alo_per_lesion_type(hm_baseline, masks)
        alo_full  = compute_alo_per_lesion_type(hm_full,     masks)

        # Attention overlap between the two models
        attn_overlap = compute_attention_overlap(hm_full, hm_baseline)

        result_entry: dict[str, Any] = {
            "image":                 stem,
            "dr_grade":              dr_grade,
            "has_lesion_masks":      bool(masks),
            "baseline_iou":          iou_base,
            "preprocessed_iou":      iou_full,
            "baseline_alo":          alo_base,
            "preprocessed_alo":      alo_full,
            "attention_overlap":     attn_overlap,
        }
        per_image_results.append(result_entry)

        # Save comparison figure
        fig_path = gradcam_dir / f"{stem}_comparison.png"
        create_comparison_figure(
            image=img_base,
            heatmap_baseline=hm_baseline,
            heatmap_preproc=hm_full,
            lesion_masks=masks if masks else None,
            save_path=fig_path,
        )
        mask_info = f"{len(masks)} lesion types" if masks else "no masks"
        print(f"  [{stem}] grade={dr_grade} | {mask_info} | "
              f"attn_overlap={attn_overlap:.3f}")

    # ── Aggregate across images that have masks ───────────────────────────────
    def _mean_iou_over_images(results: list[dict], key: str) -> dict[str, float]:
        """Average per-lesion-type metrics across images that have that type."""
        accumulator: dict[str, list[float]] = {lt: [] for lt in _LESION_TYPES}
        for r in results:
            for lt, val in r.get(key, {}).items():
                if lt in accumulator:
                    accumulator[lt].append(val)
        return {
            lt: float(np.mean(vals)) if vals else float("nan")
            for lt, vals in accumulator.items()
        }

    mean_iou_baseline  = _mean_iou_over_images(per_image_results, "baseline_iou")
    mean_iou_full      = _mean_iou_over_images(per_image_results, "preprocessed_iou")
    mean_alo_baseline  = _mean_iou_over_images(per_image_results, "baseline_alo")
    mean_alo_full      = _mean_iou_over_images(per_image_results, "preprocessed_alo")

    # H-5 test: IoU_preproc > IoU_baseline for ≥ 3 of 4 lesion types
    improved_lesion_types = [
        lt for lt in _LESION_TYPES
        if (not np.isnan(mean_iou_full.get(lt, float("nan")))
            and not np.isnan(mean_iou_baseline.get(lt, float("nan")))
            and mean_iou_full[lt] > mean_iou_baseline[lt])
    ]
    h5_supported = len(improved_lesion_types) >= 3

    # ALO H-5 direction (primary metric per INVARIANTS)
    improved_alo_types = [
        lt for lt in _LESION_TYPES
        if (not np.isnan(mean_alo_full.get(lt, float("nan")))
            and not np.isnan(mean_alo_baseline.get(lt, float("nan")))
            and mean_alo_full[lt] > mean_alo_baseline[lt])
    ]
    h5_alo_supported = len(improved_alo_types) >= 3

    results_out: dict[str, Any] = {
        "per_image": per_image_results,
        "summary": {
            "n_images_analysed":      len(per_image_results),
            "n_images_with_masks":    sum(1 for r in per_image_results if r["has_lesion_masks"]),
            "mean_iou_baseline":      mean_iou_baseline,
            "mean_iou_preprocessed":  mean_iou_full,
            "mean_alo_baseline":      mean_alo_baseline,
            "mean_alo_preprocessed":  mean_alo_full,
            "h5_supported":           h5_supported,
            "h5_alo_supported":       h5_alo_supported,
            "lesion_types_improved_iou": improved_lesion_types,
            "lesion_types_improved_alo": improved_alo_types,
            "nc14_note": (
                "Grad-CAM activation does NOT constitute clinical localization "
                "of pathology (INVARIANTS NC-14).  These metrics are "
                "interpretability evidence only."
            ),
        },
    }

    out_path = output_dir / "iou_results.json"
    with open(out_path, "w") as f:
        json.dump(results_out, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nResults saved → {out_path}")

    # ── Summary printout ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("Experiment 4 — Grad-CAM Summary")
    print(f"{'='*65}")
    print(f"  Images analysed: {len(per_image_results)} "
          f"({results_out['summary']['n_images_with_masks']} with lesion masks)")
    print(f"\n  IoU per lesion type (baseline → preprocessed):")
    for lt in _LESION_TYPES:
        b = mean_iou_baseline.get(lt, float("nan"))
        p = mean_iou_full.get(lt, float("nan"))
        arrow = "↑" if lt in improved_lesion_types else ("↓" if not np.isnan(b) else "–")
        print(f"    {lt:<20}: {b:.4f} → {p:.4f}  {arrow}")
    print(f"\n  ALO per lesion type (baseline → preprocessed):")
    for lt in _LESION_TYPES:
        b = mean_alo_baseline.get(lt, float("nan"))
        p = mean_alo_full.get(lt, float("nan"))
        arrow = "↑" if lt in improved_alo_types else ("↓" if not np.isnan(b) else "–")
        print(f"    {lt:<20}: {b:.4f} → {p:.4f}  {arrow}")
    print(f"\n  H-5 (IoU) supported: {h5_supported}  "
          f"({len(improved_lesion_types)}/4 lesion types improved)")
    print(f"  H-5 (ALO) supported: {h5_alo_supported}  "
          f"({len(improved_alo_types)}/4 lesion types improved)")
