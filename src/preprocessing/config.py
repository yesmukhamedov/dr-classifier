"""
V5 preprocessing configuration dataclass and pipeline presets.

All V5 pipeline stages are parameterised through :class:`PreprocessingV5Config`.
Use :meth:`PreprocessingV5Config.from_dict` to load from a YAML-parsed dict and
:meth:`PreprocessingV5Config.from_preset` to start from a named model preset.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import cv2


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingV5Config:
    """
    Unified configuration for the V5 preprocessing + augmentation pipeline.

    Fields are grouped by pipeline stage.  Boolean toggles allow individual
    stages/sub-stages to be disabled (e.g. for ablation experiments).

    Args:
        mode: Pipeline mode. ``"full"`` runs all V5 stages. ``"baseline"``
            does simple stretch-resize + ImageNet normalize (3 channels, no mask).
        use_canonical_flip: Enable Stage 0 horizontal flip for left-eye images.
        use_od_fovea_rotation: Enable Stage 1 OD–fovea axis rotation normalization.
        use_flat_field: Enable Stage 4 Gaussian flat-field correction.
        use_clahe: Enable Stage 5 upgraded CLAHE L-channel enhancement.
        use_pca_color: Enable Stage 6 PCA colour jitter augmentation.
        use_brightness_contrast: Enable Stage 6 brightness/contrast augmentation.
        use_shear: Enable Stage 6 shear component of the affine transform.
        use_stretch: Enable Stage 6 anisotropic stretch component.
        target_size: Output spatial resolution in pixels (square).
        flat_field_mode: ``"adaptive"`` (σ=factor·D) or ``"fixed"`` (σ=flat_field_sigma).
        flat_field_sigma_factor: σ = factor × FOV diameter D (used when mode=adaptive).
        flat_field_sigma: Gaussian σ for flat-field blur subtraction (fixed mode or fallback).
        flat_field_mask_only: Apply correction only inside FOV mask when ``True``.
        clahe_tile_grid_size: CLAHE tile grid as (rows, cols).
        clahe_clip_factor: Clip-limit scale factor (× tile_area / 256).
        clahe_global_threshold: Additional global clip limit (× tile_area).
        clahe_train_prob: Probability of applying CLAHE at train time.
        rotation_sigma: σ of the truncated Gaussian rotation distribution (°).
        rotation_clip: Hard clip on sampled rotation angle (°).
        zoom_range: Log-uniform zoom range [min, max].
        stretch_range: Log-uniform anisotropic stretch range [min, max].
        shear_range: Uniform shear range [min_deg, max_deg].
        shear_prob: Probability of applying shear augmentation.
        interp_weights: Sampling weights for (LINEAR, CUBIC, NEAREST).
        border_mode: OpenCV border mode for warpAffine (default BORDER_REFLECT).
        pca_color_sigma: σ of Normal distribution for PCA colour noise.
        pca_color_prob: Probability of applying PCA colour jitter.
        brightness_alpha_range: Contrast multiplier range [min, max].
        brightness_beta_range: Brightness additive range [min, max] (uint8 scale).
        bc_prob: Probability of applying brightness/contrast augmentation.
        normalize_mode: ``"dataset_specific"`` or ``"imagenet"``.
        dataset_mean: Per-channel mean from EyePACS training set (mask=1.0 only).
            ``None`` falls back to ImageNet mean.
        dataset_std: Per-channel std from EyePACS training set (mask=1.0 only).
            ``None`` falls back to ImageNet std.
        normalize_mean: ImageNet per-channel mean (fallback for baseline/imagenet mode).
        normalize_std: ImageNet per-channel std (fallback for baseline/imagenet mode).
        od_blur_sigma: Gaussian σ for OD detection blur.
        od_percentile: Intensity percentile for OD mask threshold.
        fovea_blur_sigma: Gaussian σ for fovea detection blur.
        fovea_inner_factor: Inner annulus boundary as multiple of OD diameter.
        fovea_outer_factor: Outer annulus boundary as multiple of OD diameter.
        adaptive_rotation_sigma: Use per-image adaptive σ (from OD/fovea uncertainty)
            instead of fixed rotation_sigma for augmentation.
        fallback_rotation_sigma: Rotation σ used when adaptive detection fails
            or adaptive_rotation_sigma is False.
    """

    # --- Mode ---
    mode: str = "full"                      # "full" or "baseline"

    # --- Toggles ---
    use_canonical_flip: bool = True
    use_od_fovea_rotation: bool = True
    use_flat_field: bool = True
    use_clahe: bool = True
    use_pca_color: bool = True
    use_brightness_contrast: bool = True
    use_shear: bool = True
    use_stretch: bool = True

    # --- Stage 2: Crop + Resize ---
    target_size: int = 512

    # --- Stage 4: Flat-Field Correction ---
    flat_field_mode: str = "adaptive"       # "adaptive" or "fixed"
    flat_field_sigma_factor: float = 0.07   # σ = factor × FOV diameter
    flat_field_sigma: float = 45.0
    flat_field_mask_only: bool = True       # apply only inside FOV mask

    # --- Stage 5: Upgraded CLAHE ---
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    clahe_clip_factor: float = 2.0
    clahe_global_threshold: float = 0.01
    clahe_train_prob: float = 0.8

    # --- Stage 6: Augmentation ---
    rotation_sigma: float = 13.0
    rotation_clip: float = 40.0
    zoom_range: tuple[float, float] = (0.9, 1.1)
    stretch_range: tuple[float, float] = (1 / 1.05, 1.05)
    shear_range: tuple[float, float] = (-2.0, 2.0)
    shear_prob: float = 0.3
    interp_weights: tuple[float, float, float] = (0.6, 0.3, 0.1)  # LINEAR, CUBIC, NEAREST
    border_mode: int = cv2.BORDER_REFLECT
    pca_color_sigma: float = 0.1
    pca_color_prob: float = 0.5
    brightness_alpha_range: tuple[float, float] = (0.9, 1.1)
    brightness_beta_range: tuple[float, float] = (-10.0, 10.0)
    bc_prob: float = 0.5

    # --- Stage 7: Normalise ---
    normalize_mode: str = "dataset_specific"  # "dataset_specific" or "imagenet"
    dataset_mean: tuple[float, float, float] | None = None
    dataset_std: tuple[float, float, float] | None = None
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # --- Stage 1: OD-Fovea Rotation Detection ---
    od_blur_sigma: float = 15.0
    od_percentile: float = 97.0
    fovea_blur_sigma: float = 25.0
    fovea_inner_factor: float = 1.5
    fovea_outer_factor: float = 3.5
    adaptive_rotation_sigma: bool = True
    fallback_rotation_sigma: float = 13.0

    # ------------------------------------------------------------------
    # Factory: from dict
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PreprocessingV5Config":
        """
        Build a :class:`PreprocessingV5Config` from a plain dict.

        Keys absent from *d* retain their dataclass defaults.  List values
        are coerced to tuples for tuple-annotated fields (e.g. when loading
        from a YAML file where sequences are represented as lists).

        Args:
            d: Dict of field overrides, typically from a YAML ``preprocessing_v5``
               section.

        Returns:
            PreprocessingV5Config instance.
        """
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            if f.name not in d:
                continue
            val = d[f.name]
            # Coerce list → tuple for tuple-annotated fields.
            # With `from __future__ import annotations`, f.type is always a str.
            type_str: str = f.type if isinstance(f.type, str) else str(f.type)
            if type_str.startswith("tuple") and isinstance(val, list):
                val = tuple(val)
            kwargs[f.name] = val
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Factory: from preset
    # ------------------------------------------------------------------

    @classmethod
    def from_preset(cls, preset_name: str) -> "PreprocessingV5Config":
        """
        Build a :class:`PreprocessingV5Config` from a named preset.

        The preset supplies a partial dict of overrides applied on top of
        dataclass defaults; unspecified fields keep their default values.

        Args:
            preset_name: Key into :data:`PIPELINE_PRESETS`
                (currently ``"resnet"`` or ``"efficientnet"``).

        Returns:
            PreprocessingV5Config instance.

        Raises:
            ValueError: If *preset_name* is not found in :data:`PIPELINE_PRESETS`.
        """
        if preset_name not in PIPELINE_PRESETS:
            available = list(PIPELINE_PRESETS)
            raise ValueError(
                f"Unknown preset '{preset_name}'. Available presets: {available}"
            )
        return cls.from_dict(PIPELINE_PRESETS[preset_name])


# ---------------------------------------------------------------------------
# Pipeline presets
# ---------------------------------------------------------------------------

PIPELINE_PRESETS: dict[str, dict[str, Any]] = {
    "resnet": {
        "use_flat_field": True,
        "use_clahe": True,
        "clahe_train_prob": 0.8,
        "use_pca_color": True,
        "pca_color_prob": 0.5,
        "use_brightness_contrast": True,
        "bc_prob": 0.5,
        "use_shear": True,
        "shear_prob": 0.3,
        "use_stretch": True,
        "use_od_fovea_rotation": True,
        "adaptive_rotation_sigma": True,
    },
    "efficientnet": {
        "use_flat_field": True,
        "use_clahe": True,
        "clahe_train_prob": 0.5,
        "use_pca_color": False,
        "use_brightness_contrast": True,
        "bc_prob": 0.3,
        "use_shear": False,
        "use_stretch": True,
        "use_od_fovea_rotation": True,
        "adaptive_rotation_sigma": True,
    },
}


