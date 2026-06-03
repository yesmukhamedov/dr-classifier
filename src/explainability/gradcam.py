"""Grad-CAM activation map generation for EfficientNet models (Experiment 4).

Wraps the pytorch_grad_cam library to provide a clean interface for generating
heatmaps from fundus images.  The heatmaps are used as interpretability
evidence for H-5 (preprocessing directs attention toward lesion regions).

NC-14 (INVARIANTS): Grad-CAM activation does NOT constitute clinical
localization of pathology — it is an interpretability tool, not a diagnostic
output.  Grad-CAM overlays indicate regions of high gradient-weighted
activation in the final convolutional layer and do not represent pixel-level
diagnostic delineation of lesion boundaries.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class GradCAMGenerator:
    """Generate Grad-CAM heatmaps from a CNN model.

    Uses the pytorch_grad_cam library with a specified target layer
    (typically the last convolutional layer before global pooling).

    Args:
        model: Trained nn.Module.  Should be in eval mode and on `device`.
        target_layer: The convolutional layer to hook.  For timm EfficientNet
            models, use ``model.conv_head`` (see
            ``src.models.efficientnet.get_gradcam_target_layer``).
        device: Device string — "cuda" or "cpu".
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.device = device
        self._cam = GradCAM(model=model, target_layers=[target_layer])

    # ------------------------------------------------------------------

    def generate(
        self,
        image_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Generate a single Grad-CAM heatmap.

        Args:
            image_tensor: Input tensor of shape (1, 3, H, W), float32 [0, 1].
                          Should already be on the correct device.
            target_class: Class index to explain.  If None, the predicted
                          class (argmax of logits) is used.

        Returns:
            Heatmap as a float32 np.ndarray of shape (H, W), values in [0, 1].
            Larger values indicate higher activation for the target class.
        """
        targets = (
            [ClassifierOutputTarget(target_class)]
            if target_class is not None
            else None
        )
        # pytorch_grad_cam expects CPU or CUDA tensor — pass as-is
        grayscale_cam = self._cam(
            input_tensor=image_tensor,
            targets=targets,
        )  # shape: (1, H, W), float32
        return grayscale_cam[0]

    # ------------------------------------------------------------------

    def generate_batch(
        self,
        images: list[torch.Tensor],
        target_classes: list[int] | None = None,
    ) -> list[np.ndarray]:
        """Generate Grad-CAM heatmaps for multiple images.

        Each image is processed independently to support different target
        classes per image.  For large batches consider calling generate()
        directly with a batched tensor if all target classes are identical.

        Args:
            images: List of tensors, each of shape (1, 3, H, W).
            target_classes: Optional list of target class indices, one per
                            image.  If None, the predicted class is used for
                            every image.

        Returns:
            List of float32 np.ndarray heatmaps, each of shape (H, W) in
            [0, 1], in the same order as `images`.
        """
        heatmaps: list[np.ndarray] = []
        for i, img in enumerate(images):
            tc = target_classes[i] if target_classes is not None else None
            heatmaps.append(self.generate(img, target_class=tc))
        return heatmaps
