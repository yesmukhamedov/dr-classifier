"""Drift-regression tests for the V5 Stage 0–4 cache (throughput fix, TASK §2).

The cache is only safe if the cached path (``precompute_deterministic`` →
PNG round-trip → ``finish_from_cache``) reproduces the live pipeline
(``__call__``) exactly. These tests lock that invariant in both inference and
seeded-training mode, plus the lossless PNG round-trip of the binary FOV mask.
"""

import os
import tempfile

import cv2
import numpy as np
import torch

from src.preprocessing.config import PreprocessingV5Config
from src.preprocessing.pipeline_v5 import PreprocessingPipelineV5


def _make_synthetic_fundus(seed: int = 0) -> np.ndarray:
    """Landscape BGR image (w>1.2h so FOV detection engages) with a disc + OD."""
    rng = np.random.default_rng(seed)
    img = np.zeros((600, 800, 3), np.uint8)
    cv2.circle(img, (400, 300), 280, (160, 110, 80), -1)
    cv2.circle(img, (500, 250), 40, (240, 220, 200), -1)  # bright OD-ish region
    noise = rng.integers(-8, 8, img.shape)
    return (img.astype(np.int16) + noise).clip(0, 255).astype(np.uint8)


def _png_roundtrip(flat_rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Save flat_rgb+mask as a 4-channel PNG and reload — mirrors the cache I/O."""
    bgr = cv2.cvtColor(flat_rgb, cv2.COLOR_RGB2BGR)
    alpha = (mask > 0.5).astype(np.uint8) * 255
    bgra = np.dstack([bgr, alpha])
    path = os.path.join(tempfile.gettempdir(), "v5_cache_test_tile.png")
    cv2.imwrite(path, bgra, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    back = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    rgb2 = cv2.cvtColor(back[:, :, :3], cv2.COLOR_BGR2RGB)
    mask2 = back[:, :, 3].astype(np.float32) / 255.0
    return rgb2, mask2


def _config() -> PreprocessingV5Config:
    cfg = PreprocessingV5Config.from_preset("efficientnet")
    cfg.dataset_mean = (0.42, 0.31, 0.22)  # exercise dataset-specific Stage 7
    cfg.dataset_std = (0.27, 0.20, 0.16)
    return cfg


def test_mask_png_roundtrip_is_lossless():
    """The binary FOV mask survives the 0↔255 alpha round-trip exactly."""
    pipe = PreprocessingPipelineV5.create_for_inference(_config())
    flat, mask, _, _ = pipe.precompute_deterministic(_make_synthetic_fundus(), "left")
    _, mask2 = _png_roundtrip(flat, mask)
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    assert np.array_equal(mask, mask2)


def test_cache_matches_live_inference():
    """Cached path == live ``__call__`` in deterministic inference mode."""
    pipe = PreprocessingPipelineV5.create_for_inference(_config())
    img = _make_synthetic_fundus()
    t_live = pipe(img.copy(), eye_side="left")
    flat, mask, conf, rot = pipe.precompute_deterministic(img.copy(), "left")
    flat2, mask2 = _png_roundtrip(flat, mask)
    t_cache = pipe.finish_from_cache(flat2, mask2, conf, rot)
    assert t_live.shape == t_cache.shape == (4, 512, 512)
    assert torch.allclose(t_live, t_cache, atol=1e-6)


def test_cache_matches_live_training_seeded():
    """Cached path == live in training mode under identical RNG seed.

    Stages 0–4 consume no ``np.random``, so seeding before each path makes the
    stochastic Stages 5–7 (CLAHE + augmentation) draw identically.
    """
    pipe = PreprocessingPipelineV5.create_for_training(_config())
    img = _make_synthetic_fundus()

    np.random.seed(0)
    t_live = pipe(img.copy(), eye_side="left")

    flat, mask, conf, rot = pipe.precompute_deterministic(img.copy(), "left")
    flat2, mask2 = _png_roundtrip(flat, mask)

    np.random.seed(0)
    t_cache = pipe.finish_from_cache(flat2, mask2, conf, rot)

    assert torch.allclose(t_live, t_cache, atol=1e-6)
