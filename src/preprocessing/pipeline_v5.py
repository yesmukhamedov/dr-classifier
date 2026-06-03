"""
V5 Pipeline Orchestrator.

:class:`PreprocessingPipelineV5` chains all eight V5 stages in the correct order.
It is the main public interface for the V5 preprocessing stack.

Stage execution order
---------------------
Preprocessing (train + inference):
  0. Canonical flip      — left→right eye orientation
  1. OD-Fovea rotation   — always (classical CV, fallback on low confidence)
  2. FOV crop + resize   — always (512×512, centered zero-padding)
  3. FOV mask            — implicit, returned from Stage 2 as 4th channel
  4. Flat-field          — always (adaptive σ=0.07·D, inside mask only)
  5. Upgraded CLAHE      — always (stochastic p=0.8 at train time)
  7. Dataset-specific normalize — always last, outputs ``torch.Tensor``

Augmentation (train only, inserted before Stage 7):
  6. Unified affine + brightness/contrast + PCA colour jitter
     (rotation σ adaptive from Stage 1)

Baseline mode (``create_baseline_v5``):
  Simple stretch-resize to 512×512 + ImageNet normalize. 3 channels (no mask).
  Replicates what most competitors do in the literature.

The V3 :class:`~src.preprocessing.pipeline.PreprocessingPipeline` is left
intact for backward compatibility with ablation experiments.
"""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import torch

from .canonical_orientation import canonical_flip, canonical_orientation
from .od_fovea_detect import ODFoveaResult
from .config import PreprocessingV5Config
from .crop_resize import crop_and_resize
from .flat_field import apply_flat_field
from .imagenet_normalize import imagenet_normalize
from .upgraded_clahe import ClaheParams, maybe_apply_clahe
# NOTE: FundusAugmentationV4 is imported lazily inside __init__ to avoid
# circular imports (src.data imports from src.preprocessing and vice-versa).


class PreprocessingPipelineV5:
    """
    V5 8-stage preprocessing + augmentation pipeline.

    Preprocessing stages 0–5 and 7 are applied identically at train and
    inference (except Stage 5 CLAHE, which is stochastic at train time).
    Stage 6 augmentation is applied only during training, inserted *before*
    Stage 7 normalisation so that it operates on uint8 images.

    All internal processing is done in **RGB**.  The ``input_color_space``
    parameter controls whether the input image is converted before any stage
    runs:

    - ``"bgr"`` (default) — converts BGR→RGB on entry.  Use this when images
      are loaded with :func:`cv2.imread`, which returns BGR.
    - ``"rgb"`` — no conversion.  Use this when the caller already provides
      an RGB array (e.g. loaded via PIL or a dataset that returns RGB).

    Passing an already-RGB image to a ``"bgr"`` pipeline silently swaps R and
    B channels, corrupting the image.  Always match this parameter to how the
    caller loads images.

    Args:
        config: :class:`PreprocessingV5Config` controlling all parameters
            and toggle flags.
        is_training: ``True`` enables stochastic CLAHE and augmentation.
        pca_eigvecs: PCA eigenvectors of shape ``(3, 3)`` for colour
            augmentation.  ``None`` disables PCA colour jitter.
        pca_eigvals: PCA eigenvalues of shape ``(3,)``.  ``None`` disables
            PCA colour jitter.
        input_color_space: ``"bgr"`` or ``"rgb"``.  Controls BGR→RGB
            conversion at the pipeline entry point.  Default ``"bgr"``
            matches :func:`cv2.imread` output.
    """

    def __init__(
        self,
        config: PreprocessingV5Config,
        is_training: bool = False,
        pca_eigvecs: np.ndarray | None = None,
        pca_eigvals: np.ndarray | None = None,
        input_color_space: str = "bgr",
    ) -> None:
        if input_color_space not in ("bgr", "rgb"):
            raise ValueError(
                f"input_color_space must be 'bgr' or 'rgb', got {input_color_space!r}"
            )
        self.config = config
        self.is_training = is_training
        self._input_color_space = input_color_space

        self._clahe_params = ClaheParams(
            tile_grid_size=config.clahe_tile_grid_size,
            clip_factor=config.clahe_clip_factor,
            global_threshold=config.clahe_global_threshold,
        )
        # Lazy import to break the circular dependency with src.data
        from src.data.augmentation_v4 import FundusAugmentationV4  # noqa: PLC0415
        self._augmentation = FundusAugmentationV4(
            config=config,
            pca_eigvecs=pca_eigvecs,
            pca_eigvals=pca_eigvals,
        )

    # ------------------------------------------------------------------
    # Core callable
    # ------------------------------------------------------------------

    def __call__(
        self,
        image: np.ndarray,
        eye_side: str = "unknown",
    ) -> torch.Tensor:
        """
        Apply the full V5 pipeline to one image.

        Args:
            image: Raw image as a uint8 NumPy array of shape ``(H, W, 3)``.
                Must match ``input_color_space`` set at construction:
                ``"bgr"`` for :func:`cv2.imread` output,
                ``"rgb"`` for PIL/already-converted arrays.
            eye_side: ``"left"``, ``"right"``, or ``"unknown"``.

        Returns:
            Normalised float32 tensor of shape
            ``(4, target_size, target_size)`` for full V5 mode, or
            ``(3, target_size, target_size)`` for baseline mode.
            In full mode, channel 3 is a binary FOV mask
            (1.0 = real data, 0.0 = padding).
        """
        # Convert to RGB if input is BGR (e.g. from cv2.imread).
        if self._input_color_space == "bgr":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Baseline mode: simple stretch-resize + ImageNet normalize (3 channels)
        if self.config.mode == "baseline":
            image = cv2.resize(
                image,
                (self.config.target_size, self.config.target_size),
                interpolation=cv2.INTER_LINEAR,
            )
            return imagenet_normalize(
                image,
                mean=self.config.normalize_mean,
                std=self.config.normalize_std,
            )

        # Full V5: deterministic Stages 0–4, then stochastic Stages 5–7.
        # Split into two helpers so the precompute-and-cache path
        # (precompute_deterministic + finish_from_cache) is provably identical
        # to the live path — both finish through the same self._finish.
        flat_rgb, fov_mask, od_fovea_result = self._precompute_rgb(image, eye_side)
        return self._finish(flat_rgb, fov_mask, od_fovea_result)

    # ------------------------------------------------------------------
    # Stage split: deterministic 0–4  vs  stochastic 5–7
    # ------------------------------------------------------------------

    def _precompute_rgb(
        self,
        image: np.ndarray,
        eye_side: str,
    ) -> tuple[np.ndarray, np.ndarray, "ODFoveaResult | None"]:
        """Run the **deterministic** Stages 0–4 on an already-RGB image.

        These stages carry no train-time randomness, so their output is safe to
        cache once and reuse every epoch (the V5 throughput fix). Stage 5 CLAHE
        is the first stochastic stage and is intentionally excluded here.

        Args:
            image: RGB uint8 array of shape ``(H, W, 3)`` (already colour-
                converted — callers handle ``input_color_space``).
            eye_side: ``"left"``, ``"right"``, or ``"unknown"``.

        Returns:
            Tuple of:
              - ``flat_rgb``: RGB uint8 ``(target_size, target_size, 3)`` after
                Stage 4 flat-field (padding zeroed).
              - ``fov_mask``: float32 ``(target_size, target_size)`` binary mask
                (1.0 inside FOV, 0.0 padding).
              - ``od_fovea_result``: :class:`ODFoveaResult` from Stage 0/1, or
                ``None`` if orientation is disabled.
        """
        # Stage 0: canonical orientation (flip + OD–fovea rotation)
        od_fovea_result: ODFoveaResult | None = None
        if self.config.use_canonical_flip or self.config.use_od_fovea_rotation:
            image, od_fovea_result = canonical_orientation(
                image,
                eye_side=eye_side if self.config.use_canonical_flip else "unknown",
                enable_rotation=self.config.use_od_fovea_rotation,
            )

        # Stage 2: FOV crop + isotropic resize (always) — returns (image, mask)
        image, fov_mask = crop_and_resize(image, self.config.target_size)

        # Compute FOV diameter from mask for adaptive flat-field
        fov_rows = np.any(fov_mask > 0, axis=1)
        fov_diameter = float(np.sum(fov_rows))  # height of FOV region in pixels
        adaptive_sigma = self.config.flat_field_sigma_factor * fov_diameter

        # Stage 4: flat-field correction (adaptive σ)
        if self.config.use_flat_field:
            if self.config.flat_field_mode == "adaptive":
                sigma = adaptive_sigma
            else:
                sigma = self.config.flat_field_sigma
            image = apply_flat_field(
                image,
                sigma=sigma,
                mask=fov_mask if self.config.flat_field_mask_only else None,
            )

        return image, fov_mask, od_fovea_result

    def _finish(
        self,
        image: np.ndarray,
        fov_mask: np.ndarray,
        od_fovea_result: "ODFoveaResult | SimpleNamespace | None",
    ) -> torch.Tensor:
        """Run the **stochastic** Stages 5–7 on a Stage-4 (flat-field) image.

        Shared by both the live :meth:`__call__` path and the cached
        :meth:`finish_from_cache` path, guaranteeing identical train/inference
        behaviour regardless of whether Stages 0–4 came from disk.

        Args:
            image: RGB uint8 array (Stage-4 flat-field output).
            fov_mask: float32 binary FOV mask (Stage 2 output).
            od_fovea_result: object exposing ``confident`` and
                ``rotation_sigma_deg`` (only fields Stage 6 reads), or ``None``.

        Returns:
            Normalised float32 tensor ``(4, target_size, target_size)``;
            channel 3 is the (un-augmented) FOV mask.
        """
        # Stage 5: upgraded CLAHE (stochastic at train time)
        if self.config.use_clahe:
            image = maybe_apply_clahe(
                image,
                params=self._clahe_params,
                is_training=self.is_training,
                train_prob=self.config.clahe_train_prob,
            )

        # Stage 6: augmentation (train only, uint8, before normalize)
        if self.is_training:
            image = self._augmentation(image, od_fovea_result=od_fovea_result)

        # Stage 7: dataset-specific or ImageNet normalize → tensor (always last)
        if self.config.normalize_mode == "dataset_specific" and \
                self.config.dataset_mean is not None and \
                self.config.dataset_std is not None:
            mean = self.config.dataset_mean
            std = self.config.dataset_std
        else:
            mean = self.config.normalize_mean
            std = self.config.normalize_std

        rgb_tensor = imagenet_normalize(image, mean=mean, std=std)

        # Append FOV mask as 4th channel (un-augmented, mirrors live behaviour)
        mask_tensor = torch.from_numpy(fov_mask).unsqueeze(0)  # (1, H, W)
        return torch.cat([rgb_tensor, mask_tensor], dim=0)      # (4, H, W)

    # ------------------------------------------------------------------
    # Precompute-and-cache public API (V5 throughput fix)
    # ------------------------------------------------------------------

    def precompute_deterministic(
        self,
        image: np.ndarray,
        eye_side: str = "unknown",
    ) -> tuple[np.ndarray, np.ndarray, bool, float]:
        """Stages 0–4 for offline caching, from a raw (decoded) image.

        Handles ``input_color_space`` then runs :meth:`_precompute_rgb`,
        returning the two cacheable scalars from the OD/fovea result rather than
        the full object (all Stage 6 needs).

        Args:
            image: Raw uint8 array matching ``input_color_space`` (``"bgr"`` for
                :func:`cv2.imread` output).
            eye_side: ``"left"``, ``"right"``, or ``"unknown"``.

        Returns:
            ``(flat_rgb_uint8, fov_mask_float32, confident, rotation_sigma_deg)``.
        """
        if self._input_color_space == "bgr":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        flat_rgb, fov_mask, od = self._precompute_rgb(image, eye_side)
        confident = bool(od.confident) if od is not None else False
        rotation_sigma_deg = float(od.rotation_sigma_deg) if od is not None else 0.0
        return flat_rgb, fov_mask, confident, rotation_sigma_deg

    def finish_from_cache(
        self,
        flat_rgb: np.ndarray,
        fov_mask: np.ndarray,
        confident: bool,
        rotation_sigma_deg: float,
    ) -> torch.Tensor:
        """Stages 5–7 from cached Stage-4 output — the per-epoch train path.

        Equivalent to :meth:`__call__` for the full pipeline, but skips the
        expensive deterministic Stages 0–4 by consuming a cached flat-field
        image + FOV mask + the two OD/fovea scalars.

        Args:
            flat_rgb: cached RGB uint8 Stage-4 image.
            fov_mask: cached float32 binary FOV mask.
            confident: cached ``ODFoveaResult.confident``.
            rotation_sigma_deg: cached ``ODFoveaResult.rotation_sigma_deg``.

        Returns:
            Normalised float32 tensor ``(4, target_size, target_size)``.
        """
        od = SimpleNamespace(
            confident=bool(confident),
            rotation_sigma_deg=float(rotation_sigma_deg),
        )
        return self._finish(flat_rgb, fov_mask, od)

    # ------------------------------------------------------------------
    # Stage breakdown (for the demo "what preprocessing does" panel)
    # ------------------------------------------------------------------

    def stage_breakdown(
        self,
        image: np.ndarray,
        eye_side: str = "unknown",
    ) -> dict:
        """Run preprocessing Stages 0–5 capturing each intermediate.

        Mirrors :meth:`__call__` up to (but excluding) augmentation and Stage 7
        normalize, returning the labelled intermediate images for the demo's
        V5 preview strip (TASK-Demo §C.6) plus the FOV mask and OD/fovea result.
        Deterministic (inference mode): CLAHE is applied with no stochasticity.

        Args:
            image: Raw uint8 array matching ``input_color_space``.
            eye_side: ``"left"``, ``"right"``, or ``"unknown"``.

        Returns:
            Dict with:
              - ``stages``: list of ``(label, rgb_uint8)`` from original →
                flip → rotation → FOV crop+resize → flat-field → CLAHE.
              - ``fov_mask``: float32 ``(H, W)`` mask (1.0 inside FOV).
              - ``od_fovea``: :class:`ODFoveaResult` or ``None``.
        """
        if self._input_color_space == "bgr":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        stages: list[tuple[str, np.ndarray]] = [("original", image.copy())]

        # Stage 0: canonical flip only (for a dedicated panel).
        flipped = (
            canonical_flip(image, eye_side)
            if self.config.use_canonical_flip else image
        )
        stages.append(("canonical_flip", flipped.copy()))

        # Stage 0+1: flip + OD-fovea rotation (the real path), with OD result.
        od_fovea_result: ODFoveaResult | None = None
        if self.config.use_canonical_flip or self.config.use_od_fovea_rotation:
            oriented, od_fovea_result = canonical_orientation(
                image,
                eye_side=eye_side if self.config.use_canonical_flip else "unknown",
                enable_rotation=self.config.use_od_fovea_rotation,
            )
        else:
            oriented = image
        stages.append(("od_fovea_rotation", oriented.copy()))

        # Stage 2/3: FOV crop + isotropic resize → (image, mask).
        cropped, fov_mask = crop_and_resize(oriented, self.config.target_size)
        stages.append(("fov_crop_resize", cropped.copy()))

        # Stage 4: flat-field correction (adaptive σ).
        flat = cropped
        if self.config.use_flat_field:
            if self.config.flat_field_mode == "adaptive":
                fov_rows = np.any(fov_mask > 0, axis=1)
                sigma = self.config.flat_field_sigma_factor * float(np.sum(fov_rows))
            else:
                sigma = self.config.flat_field_sigma
            flat = apply_flat_field(
                cropped, sigma=sigma,
                mask=fov_mask if self.config.flat_field_mask_only else None,
            )
        stages.append(("flat_field", flat.copy()))

        # Stage 5: CLAHE (deterministic at inference).
        clahed = flat
        if self.config.use_clahe:
            clahed = maybe_apply_clahe(
                flat, params=self._clahe_params,
                is_training=False, train_prob=self.config.clahe_train_prob,
            )
        stages.append(("clahe", clahed.copy()))

        return {"stages": stages, "fov_mask": fov_mask, "od_fovea": od_fovea_result}

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        """Return ``True`` if main preprocessing components are enabled."""
        return self.config.use_flat_field and self.config.use_clahe

    def is_absent(self) -> bool:
        """Return ``True`` if pipeline is in baseline mode."""
        return self.config.mode == "baseline"

    # ------------------------------------------------------------------
    # Factory class-methods
    # ------------------------------------------------------------------

    @classmethod
    def create_for_training(
        cls,
        config: PreprocessingV5Config,
        pca_eigvecs: np.ndarray | None = None,
        pca_eigvals: np.ndarray | None = None,
        input_color_space: str = "bgr",
    ) -> "PreprocessingPipelineV5":
        """
        Create a pipeline in training mode (stochastic CLAHE + augmentation).

        Args:
            config: :class:`PreprocessingV5Config` instance.
            pca_eigvecs: Optional PCA eigenvectors for colour jitter.
            pca_eigvals: Optional PCA eigenvalues for colour jitter.
            input_color_space: ``"bgr"`` (default) or ``"rgb"``.

        Returns:
            :class:`PreprocessingPipelineV5` with ``is_training=True``.
        """
        return cls(
            config,
            is_training=True,
            pca_eigvecs=pca_eigvecs,
            pca_eigvals=pca_eigvals,
            input_color_space=input_color_space,
        )

    @classmethod
    def create_for_inference(
        cls,
        config: PreprocessingV5Config,
        input_color_space: str = "bgr",
    ) -> "PreprocessingPipelineV5":
        """
        Create a pipeline in inference mode (deterministic, no augmentation).

        Args:
            config: :class:`PreprocessingV5Config` instance.
            input_color_space: ``"bgr"`` (default) or ``"rgb"``.

        Returns:
            :class:`PreprocessingPipelineV5` with ``is_training=False``.
        """
        return cls(config, is_training=False, input_color_space=input_color_space)

    @classmethod
    def create_baseline_v5(
        cls,
        target_size: int = 512,
        is_training: bool = False,
        input_color_space: str = "bgr",
    ) -> "PreprocessingPipelineV5":
        """Baseline: stretch-resize + ImageNet normalize (3 channels, no mask).

        This replicates what most competitors do in the literature.

        Args:
            target_size: Output spatial resolution in pixels.
            is_training: Unused (baseline has no stochastic components).
            input_color_space: ``"bgr"`` (default) or ``"rgb"``.

        Returns:
            :class:`PreprocessingPipelineV5` in baseline mode.
        """
        config = PreprocessingV5Config(
            mode="baseline",
            use_canonical_flip=False,
            use_od_fovea_rotation=False,
            use_flat_field=False,
            use_clahe=False,
            use_pca_color=False,
            use_brightness_contrast=False,
            use_shear=False,
            use_stretch=False,
            target_size=target_size,
            normalize_mode="imagenet",
        )
        return cls(config, is_training=is_training, input_color_space=input_color_space)

    @classmethod
    def create_baseline(
        cls,
        target_size: int = 512,
        is_training: bool = False,
        input_color_space: str = "bgr",
    ) -> "PreprocessingPipelineV5":
        """Alias for :meth:`create_baseline_v5` (backward compatibility)."""
        return cls.create_baseline_v5(
            target_size=target_size,
            is_training=is_training,
            input_color_space=input_color_space,
        )
