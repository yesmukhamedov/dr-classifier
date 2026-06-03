"""Visualization helpers for Grad-CAM overlays and comparison figures (Exp 4).

NC-14 (INVARIANTS): Saved figures are interpretability evidence only.
They do NOT constitute clinical lesion localization.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def overlay_gradcam(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend a Grad-CAM heatmap onto an image using JET colormap.

    The heatmap is colourized with cv2.COLORMAP_JET (red = high activation,
    blue = low) and blended over the original image:
        result = alpha * heatmap_coloured + (1 - alpha) * image

    Args:
        image: BGR uint8 image, any spatial resolution.
        heatmap: Float32 array (H, W) in [0, 1], same spatial resolution as
                 `image`.  Resized to match if shapes differ.
        alpha: Heatmap blend weight in [0, 1].  Default: 0.4.

    Returns:
        BGR uint8 blended image of the same shape as `image`.
    """
    # Ensure image is uint8 BGR
    if image.dtype != np.uint8:
        img_u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        img_u8 = image.copy()

    # Ensure image is 3-channel
    if img_u8.ndim == 2:
        img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

    h, w = img_u8.shape[:2]

    # Resize heatmap to image size if necessary
    if heatmap.shape != (h, w):
        hm = cv2.resize(heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        hm = heatmap.astype(np.float32)

    hm_u8 = (np.clip(hm, 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)

    blended = cv2.addWeighted(colored, alpha, img_u8, 1.0 - alpha, 0)
    return blended


def create_comparison_figure(
    image: np.ndarray,
    heatmap_baseline: np.ndarray,
    heatmap_preproc: np.ndarray,
    lesion_masks: dict[str, np.ndarray] | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Save a side-by-side Grad-CAM comparison figure.

    Four-panel layout:
        [Original] [Baseline Grad-CAM] [Preprocessed Grad-CAM] [Lesion mask]

    The lesion mask panel shows the union of all available masks overlaid on
    the original image (green channel).  If no masks are available it shows
    a blank (zeroed) panel.

    Args:
        image: BGR uint8 (or float32) image as displayed to the user.
        heatmap_baseline: Grad-CAM heatmap from the baseline model (H, W).
        heatmap_preproc: Grad-CAM heatmap from the preprocessed model (H, W).
        lesion_masks: Dict mapping lesion type key to mask array, or None.
        save_path: File path to save the figure.  Saved only if not None.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if image.dtype != np.uint8:
        img_u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        img_u8 = image.copy()
    img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    overlay_base  = cv2.cvtColor(overlay_gradcam(img_u8, heatmap_baseline), cv2.COLOR_BGR2RGB)
    overlay_prepr = cv2.cvtColor(overlay_gradcam(img_u8, heatmap_preproc),  cv2.COLOR_BGR2RGB)

    # Build union lesion mask panel
    if lesion_masks:
        union_mask = np.zeros((h, w), dtype=np.uint8)
        for mask in lesion_masks.values():
            if mask is not None:
                resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                union_mask = np.logical_or(union_mask, resized > 0).astype(np.uint8)
        # Green overlay: lesion pixels highlighted green on original
        mask_panel = img_rgb.copy()
        mask_panel[union_mask > 0] = [0, 220, 0]
    else:
        mask_panel = np.zeros_like(img_rgb)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    panels = [
        (img_rgb,       "Original"),
        (overlay_base,  "Baseline Grad-CAM\n(resize only)"),
        (overlay_prepr, "Preprocessed Grad-CAM\n(full pipeline)"),
        (mask_panel,    "Lesion Masks\n(green = lesion)"),
    ]
    for ax, (panel, title) in zip(axes, panels):
        ax.imshow(panel)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
