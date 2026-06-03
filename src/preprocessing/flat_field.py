"""
Stage 4 (V5): Flat-Field Correction.

Reduces uneven illumination by subtracting a heavily blurred version of the
image and re-centering at 128:

    corrected = image − GaussianBlur(image, σ) + 128

A large σ captures only the low-frequency illumination gradient, so the
subtraction removes broad brightness variation while preserving local vessel
and lesion detail.

In V5, σ is computed adaptively as σ = 0.07 × FOV_diameter.  Correction
is applied only inside the FOV mask (padding pixels are left at zero).

Input/output images are RGB uint8 NumPy arrays.
"""

from __future__ import annotations

import cv2
import numpy as np


def apply_flat_field(
    image: np.ndarray,
    sigma: float = 45.0,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply flat-field correction to reduce uneven illumination.

    Algorithm::

        blur      = GaussianBlur(image, σ)
        corrected = image − blur + 128

    When *mask* is provided, correction is applied only inside the mask
    (``mask > 0``). Padding areas (``mask == 0``) are left at zero.

    Kernel size is derived automatically from *sigma* (passed as ``(0, 0)``
    to :func:`cv2.GaussianBlur`).

    Args:
        image: RGB uint8 NumPy array of shape ``(H, W, 3)``.
        sigma: Gaussian blur σ controlling the spatial scale of the
            illumination estimate.
        mask: Optional binary mask of shape ``(H, W)`` (float32 or uint8).
            When provided, only pixels where ``mask > 0`` are corrected;
            padding regions remain zero.

    Returns:
        Corrected RGB uint8 NumPy array of shape ``(H, W, 3)``.
    """
    blur = cv2.GaussianBlur(image, (0, 0), sigma)
    corrected = image.astype(np.float32) - blur.astype(np.float32) + 128.0
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    if mask is not None:
        mask_3ch = np.expand_dims(mask > 0, axis=-1).astype(np.uint8)
        corrected = corrected * mask_3ch  # zero out padding
    return corrected
