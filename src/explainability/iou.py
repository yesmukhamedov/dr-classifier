"""IoU and ALO metrics for comparing Grad-CAM heatmaps with lesion masks.

Two metrics are computed (per INVARIANTS EH-2 / H-5):
  ALO  — Attention-Lesion Overlap (PRIMARY metric per H-5):
          |activation ∩ lesion| / |lesion|
          Fraction of the lesion area covered by high-activation regions.
  IoU  — Intersection-over-Union (SECONDARY metric):
          |activation ∩ lesion| / |activation ∪ lesion|

H-5 success criterion: IoU_preproc > IoU_baseline for at least 3 of 4
lesion types (microaneurysms, haemorrhages, hard_exudates, soft_exudates).

NC-14 note: These metrics quantify heatmap-mask overlap as interpretability
evidence; they do NOT constitute clinical lesion localization.
"""

from __future__ import annotations

import cv2
import numpy as np


def compute_iou(
    heatmap: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute Intersection-over-Union between a Grad-CAM heatmap and a mask.

    Both inputs are binarized at their respective thresholds.

    Args:
        heatmap: Float32 array (H, W) in [0, 1].  Values >= threshold are
                 considered activated.
        mask: Uint8 binary array.  Values > 0 are considered lesion pixels.
              May differ in spatial resolution from heatmap — it is resized
              to match heatmap before comparison.
        threshold: Binarization threshold for the heatmap.  Default: 0.5.

    Returns:
        IoU scalar in [0, 1].  Returns 0.0 if the union is empty.
    """
    activation = (heatmap >= threshold).astype(np.uint8)
    lesion = _resize_mask(mask, heatmap.shape)

    intersection = int(np.logical_and(activation, lesion).sum())
    union        = int(np.logical_or(activation, lesion).sum())
    if union == 0:
        return 0.0
    return intersection / union


def compute_alo(
    heatmap: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute Attention-Lesion Overlap (ALO) — primary H-5 metric.

    ALO = |activation ∩ lesion| / |lesion|

    Measures the fraction of the lesion area that the model attends to.
    Unlike IoU, this is independent of the activation region size, making it
    more sensitive to whether lesion pixels are covered by attention.

    Args:
        heatmap: Float32 array (H, W) in [0, 1].
        mask: Uint8 binary mask (any resolution; resized internally).
        threshold: Heatmap binarization threshold.  Default: 0.5.

    Returns:
        ALO in [0, 1].  Returns 0.0 if the lesion mask is empty.
    """
    activation = (heatmap >= threshold).astype(np.uint8)
    lesion = _resize_mask(mask, heatmap.shape)

    lesion_count = int(lesion.sum())
    if lesion_count == 0:
        return 0.0
    intersection = int(np.logical_and(activation, lesion).sum())
    return intersection / lesion_count


def compute_iou_per_lesion_type(
    heatmap: np.ndarray,
    lesion_masks: dict[str, np.ndarray | None],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute IoU between a heatmap and each available lesion mask.

    Args:
        heatmap: Float32 array (H, W) in [0, 1].
        lesion_masks: Dict mapping lesion type key to mask array or None.
            Expected keys: "microaneurysms", "haemorrhages",
            "hard_exudates", "soft_exudates".
        threshold: Heatmap binarization threshold.  Default: 0.5.

    Returns:
        Dict mapping lesion type key to IoU float.  Types where the mask is
        None are omitted from the result.
    """
    result: dict[str, float] = {}
    for lesion_type, mask in lesion_masks.items():
        if mask is None:
            continue
        result[lesion_type] = compute_iou(heatmap, mask, threshold)
    return result


def compute_alo_per_lesion_type(
    heatmap: np.ndarray,
    lesion_masks: dict[str, np.ndarray | None],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute ALO between a heatmap and each available lesion mask.

    Args:
        heatmap: Float32 array (H, W) in [0, 1].
        lesion_masks: Dict mapping lesion type key to mask array or None.
        threshold: Heatmap binarization threshold.  Default: 0.5.

    Returns:
        Dict mapping lesion type key to ALO float.  Types where the mask is
        None are omitted from the result.
    """
    result: dict[str, float] = {}
    for lesion_type, mask in lesion_masks.items():
        if mask is None:
            continue
        result[lesion_type] = compute_alo(heatmap, mask, threshold)
    return result


def compute_attention_overlap(
    heatmap_preproc: np.ndarray,
    heatmap_baseline: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute attention consistency between two heatmaps (IoU of binarized maps).

    Measures whether preprocessing shifts the regions of high activation,
    irrespective of lesion masks.

    Args:
        heatmap_preproc: Heatmap from the preprocessed model, (H, W) float32.
        heatmap_baseline: Heatmap from the baseline model, (H, W) float32.
            Resized to match heatmap_preproc if shapes differ.
        threshold: Binarization threshold.  Default: 0.5.

    Returns:
        IoU scalar in [0, 1].
    """
    act_preproc  = (heatmap_preproc >= threshold).astype(np.uint8)
    act_baseline = _resize_mask(
        (heatmap_baseline >= threshold).astype(np.uint8),
        heatmap_preproc.shape,
    )
    intersection = int(np.logical_and(act_preproc, act_baseline).sum())
    union        = int(np.logical_or(act_preproc, act_baseline).sum())
    if union == 0:
        return 0.0
    return intersection / union


# ── Internal helpers ──────────────────────────────────────────────────────────

def _resize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize mask to target_shape (H, W) using nearest-neighbour interpolation.

    Binarizes after resize so that partial-pixel values from linear
    interpolation do not create artefacts.

    Args:
        mask: Uint8 or bool binary mask of any spatial resolution.
        target_shape: Desired (H, W) tuple.

    Returns:
        Uint8 binary mask of shape target_shape.
    """
    h, w = target_shape
    if mask.shape[:2] == (h, w):
        return (mask > 0).astype(np.uint8)
    resized = cv2.resize(
        mask.astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )
    return (resized > 0).astype(np.uint8)
