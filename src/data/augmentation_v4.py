"""
Stage 5 (V4): Augmentation — Unified Affine + PCA Color + Brightness/Contrast.

Replaces :class:`FundusAugmentation` (V3) with a V4 augmentation pipeline that
applies a single unified affine transform (rotation + zoom + stretch + shear),
stochastic interpolation, PCA colour jitter, and brightness/contrast scaling.

Applied to **RGB uint8** images *before* ImageNet normalisation (Stage 4).
All sub-transforms are individually toggleable via :class:`PreprocessingV5Config`.

References
----------
Work 2: kaggle_diabetic, team o_O, 2nd place Kaggle DR 2015.
Wang et al., "Rotation Has Two Sides", ICLR 2024 (rotation distribution).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.preprocessing.od_fovea_detect import ODFoveaResult

import cv2
import numpy as np

from src.preprocessing.config import PreprocessingV5Config


class FundusAugmentationV4:
    """
    V4 augmentation: unified affine transform + PCA colour + brightness/contrast.

    Applied to uint8 RGB images before ImageNet normalisation.  All
    sub-transforms are controlled by the boolean toggles in *config*.

    Args:
        config: :class:`PreprocessingV5Config` controlling all augmentation
            parameters and toggle flags.
        pca_eigvecs: PCA eigenvectors of shape ``(3, 3)`` computed offline
            from the training set.  ``None`` disables PCA colour jitter
            regardless of ``config.use_pca_color``.
        pca_eigvals: PCA eigenvalues of shape ``(3,)``.  ``None`` disables
            PCA colour jitter regardless of ``config.use_pca_color``.
    """

    def __init__(
        self,
        config: PreprocessingV5Config,
        pca_eigvecs: np.ndarray | None = None,
        pca_eigvals: np.ndarray | None = None,
    ) -> None:
        self.config = config
        self.pca_eigvecs = pca_eigvecs
        self.pca_eigvals = pca_eigvals

    # ------------------------------------------------------------------
    # Public callable
    # ------------------------------------------------------------------

    def __call__(
        self,
        image: np.ndarray,
        od_fovea_result: ODFoveaResult | None = None,
    ) -> np.ndarray:
        """
        Apply the full V4 augmentation pipeline to one image.

        Application order:
        1. Unified affine (rotation + zoom + stretch + shear)
        2. Brightness / contrast
        3. PCA colour jitter

        Args:
            image: RGB uint8 NumPy array of shape ``(H, W, 3)``.
            od_fovea_result: Optional detection result from Stage 0b.
                When provided and confident, the rotation σ is derived
                from OD/fovea localization uncertainty instead of the
                fixed ``config.rotation_sigma``.

        Returns:
            Augmented RGB uint8 NumPy array of the same shape.
        """
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)

        # 1. Unified affine
        params = self._sample_affine_params(od_fovea_result=od_fovea_result)
        M = self._build_affine_matrix(
            theta=params["theta"],
            sx=params["sx"],
            sy=params["sy"],
            shear_rad=params["shear_rad"],
            center=center,
        )
        interp = self._sample_interpolation()
        image = self._apply_affine(image, M, interp)

        # 2. Brightness / contrast
        if self.config.use_brightness_contrast:
            image = self._apply_brightness_contrast(image)

        # 3. PCA colour jitter
        if self.config.use_pca_color:
            image = self._apply_pca_color(image)

        return image

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_affine_params(
        self,
        od_fovea_result: ODFoveaResult | None = None,
    ) -> dict[str, float]:
        """
        Sample affine transform parameters.

        Args:
            od_fovea_result: Optional OD/fovea detection result.
                When provided, confident, and ``config.adaptive_rotation_sigma``
                is enabled, the rotation σ is derived from localization
                uncertainty.  Otherwise ``config.fallback_rotation_sigma``
                (or ``config.rotation_sigma`` if adaptive is disabled) is used.

        Returns:
            Dict with keys ``theta`` (°), ``sx``, ``sy``, ``shear_rad`` (rad).
        """
        cfg = self.config

        # Rotation σ: adaptive (per-image) or fixed (fallback)
        if (
            cfg.adaptive_rotation_sigma
            and od_fovea_result is not None
            and od_fovea_result.confident
        ):
            rotation_sigma = od_fovea_result.rotation_sigma_deg
        elif cfg.adaptive_rotation_sigma:
            # Adaptive enabled but detection failed → use fallback
            rotation_sigma = cfg.fallback_rotation_sigma
        else:
            # Adaptive disabled → use original fixed σ
            rotation_sigma = cfg.rotation_sigma

        # Rotation: truncated Gaussian, clipped at ±rotation_clip
        theta = float(np.random.normal(0.0, rotation_sigma))
        theta = float(np.clip(theta, -cfg.rotation_clip, cfg.rotation_clip))

        # Zoom: isotropic, log-uniform in zoom_range
        log_zoom = np.random.uniform(
            np.log(cfg.zoom_range[0]),
            np.log(cfg.zoom_range[1]),
        )
        zoom = float(np.exp(log_zoom))

        # Stretch: anisotropic, log-uniform in stretch_range
        if cfg.use_stretch:
            log_stretch = np.random.uniform(
                np.log(cfg.stretch_range[0]),
                np.log(cfg.stretch_range[1]),
            )
            stretch = float(np.exp(log_stretch))
        else:
            stretch = 1.0

        sx = zoom * stretch
        sy = zoom / stretch

        # Shear: conditional
        if cfg.use_shear and np.random.rand() < cfg.shear_prob:
            shear_deg = float(np.random.uniform(cfg.shear_range[0], cfg.shear_range[1]))
        else:
            shear_deg = 0.0
        shear_rad = float(np.deg2rad(shear_deg))

        return {"theta": theta, "sx": sx, "sy": sy, "shear_rad": shear_rad}

    def _build_affine_matrix(
        self,
        theta: float,
        sx: float,
        sy: float,
        shear_rad: float,
        center: tuple[float, float],
    ) -> np.ndarray:
        """
        Build a unified 2×3 affine matrix combining rotation, scale, and shear.

        Args:
            theta: Rotation angle in degrees.
            sx: Horizontal scale factor (zoom × stretch).
            sy: Vertical scale factor (zoom / stretch).
            shear_rad: Shear angle in radians.
            center: Image centre ``(cx, cy)`` in pixels.

        Returns:
            2×3 float64 affine matrix for :func:`cv2.warpAffine`.
        """
        M = cv2.getRotationMatrix2D(center, theta, 1.0)   # (2, 3)
        M_full = np.array([
            [M[0, 0] * sx,  M[0, 1] + np.tan(shear_rad),  M[0, 2]],
            [M[1, 0],        M[1, 1] * sy,                  M[1, 2]],
        ], dtype=np.float64)
        return M_full

    def _sample_interpolation(self) -> int:
        """
        Sample a cv2 interpolation flag stochastically.

        Probabilities: 60 % LINEAR, 30 % CUBIC, 10 % NEAREST (from
        ``config.interp_weights``).

        Returns:
            One of ``cv2.INTER_LINEAR``, ``cv2.INTER_CUBIC``,
            ``cv2.INTER_NEAREST``.
        """
        flags = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST]
        weights = list(self.config.interp_weights)
        # numpy choice requires integer indices; use cumulative sum
        r = np.random.rand()
        cumulative = 0.0
        for flag, w in zip(flags, weights):
            cumulative += w
            if r < cumulative:
                return flag
        return cv2.INTER_LINEAR  # fallback

    def _apply_affine(
        self,
        image: np.ndarray,
        M: np.ndarray,
        interp: int,
    ) -> np.ndarray:
        """
        Warp *image* with affine matrix *M*.

        Args:
            image: RGB uint8 array of shape ``(H, W, 3)``.
            M: 2×3 affine matrix.
            interp: cv2 interpolation flag.

        Returns:
            Warped RGB uint8 array of the same shape.
        """
        h, w = image.shape[:2]
        return cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=interp,
            borderMode=self.config.border_mode,
        )

    def _apply_pca_color(self, image: np.ndarray) -> np.ndarray:
        """
        Apply PCA colour jitter with probability ``config.pca_color_prob``.

        No-op if ``pca_eigvecs`` or ``pca_eigvals`` are ``None``.

        Args:
            image: RGB uint8 array.

        Returns:
            Jittered RGB uint8 array (or original if skipped).
        """
        if self.pca_eigvecs is None or self.pca_eigvals is None:
            return image
        if np.random.rand() >= self.config.pca_color_prob:
            return image

        alpha = np.random.normal(0.0, self.config.pca_color_sigma, size=3)
        noise = self.pca_eigvecs @ (alpha * self.pca_eigvals)
        img_float = image.astype(np.float32) + noise
        return np.clip(img_float, 0, 255).astype(np.uint8)

    def _apply_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Apply brightness and contrast scaling with probability ``config.bc_prob``.

        Args:
            image: RGB uint8 array.

        Returns:
            Adjusted RGB uint8 array (or original if skipped).
        """
        if np.random.rand() >= self.config.bc_prob:
            return image

        alpha = float(np.random.uniform(*self.config.brightness_alpha_range))
        beta = float(np.random.uniform(*self.config.brightness_beta_range))
        img_float = image.astype(np.float32) * alpha + beta
        return np.clip(img_float, 0, 255).astype(np.uint8)
