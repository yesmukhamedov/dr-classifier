"""Grad-CAM explainability tools for DR classification (Experiment 4).

NC-14 (INVARIANTS): Grad-CAM activation does NOT constitute clinical
localization of pathology — it is an interpretability tool, not a diagnostic
output.
"""

from src.explainability.gradcam import GradCAMGenerator
from src.explainability.iou import (
    compute_alo,
    compute_alo_per_lesion_type,
    compute_attention_overlap,
    compute_iou,
    compute_iou_per_lesion_type,
)
from src.explainability.visualization import create_comparison_figure, overlay_gradcam

__all__ = [
    "GradCAMGenerator",
    "compute_iou",
    "compute_alo",
    "compute_iou_per_lesion_type",
    "compute_alo_per_lesion_type",
    "compute_attention_overlap",
    "overlay_gradcam",
    "create_comparison_figure",
]
