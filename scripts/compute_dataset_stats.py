"""Compute dataset-specific mean and std for V5 normalization (Stage 7).

Computes channel-wise mean and std from the EyePACS training set AFTER
applying V5 Stages 0–4 (canonical flip, OD-fovea rotation, FOV crop+resize,
FOV mask, flat-field correction) — but NOT CLAHE (Stage 5) and NOT
augmentation (Stage 6). Only pixels where the FOV mask == 1.0 are included
(D-2 design decision). Statistics are computed in the [0, 1] scale, matching
``torchvision.transforms.ToTensor`` used by Stage 7.

This mirrors ``PreprocessingPipelineV5.__call__`` stages 0–4 exactly (it reuses
``from_preset("efficientnet")`` so the flat-field/rotation parameters are
identical to those Config D trains with).

Usage:
    python scripts/compute_dataset_stats.py \
        --images-root /path/to/EyePACS/train \
        --labels-csv  /path/to/EyePACS/trainLabels.csv \
        --output-dir  data/processed \
        --n-samples   5000

Output:
    Writes ``<output-dir>/eyepacs_norm_stats.json`` with keys ``mean``/``std``
    (consumed automatically by ``src/experiments/exp1_factorial.py``) and
    prints the values for pasting into ``configs/default.yaml`` if desired.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import cv2
import numpy as np

# Allow running from repo root without installing the package.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.preprocessing.canonical_orientation import canonical_orientation
from src.preprocessing.config import PreprocessingV5Config
from src.preprocessing.crop_resize import crop_and_resize
from src.preprocessing.flat_field import apply_flat_field


def _discover_eyepacs(
    images_root: pathlib.Path,
    labels_csv: pathlib.Path,
    rng: np.random.Generator,
    n_samples: int,
) -> list[tuple[pathlib.Path, str]]:
    """Return ``(path, eye_side)`` pairs for EyePACS train images.

    Mirrors the indexing used by ``exp1_factorial`` (``<name>.jpeg`` under
    ``images_root``; eye side parsed from the ``_left``/``_right`` suffix).

    Args:
        images_root: Directory containing ``<name>.jpeg`` files.
        labels_csv: ``trainLabels.csv`` with an ``image`` column.
        rng: NumPy generator for reproducible sampling.
        n_samples: Max images to sample; ``0`` means use all.

    Returns:
        List of ``(image_path, eye_side)`` tuples.
    """
    rows: list[tuple[pathlib.Path, str]] = []
    with open(labels_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = row.get("image", "")
            if not name:
                continue
            path = images_root / f"{name}.jpeg"
            if not path.exists():
                continue
            side = "left" if "_left" in name else "right" if "_right" in name else "unknown"
            rows.append((path, side))

    if n_samples and len(rows) > n_samples:
        idx = rng.choice(len(rows), n_samples, replace=False)
        rows = [rows[i] for i in idx]
    return rows


def _preprocess_stages_0_to_4(
    image_bgr: np.ndarray,
    eye_side: str,
    config: PreprocessingV5Config,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply V5 Stages 0–4 to one image (no CLAHE, no augmentation).

    Args:
        image_bgr: BGR uint8 array as returned by ``cv2.imread``.
        eye_side: ``"left"``, ``"right"``, or ``"unknown"``.
        config: Pipeline config (uses target_size + flat-field parameters).

    Returns:
        Tuple ``(image_rgb_uint8, fov_mask)`` where ``image_rgb_uint8`` has
        shape ``(S, S, 3)`` and ``fov_mask`` shape ``(S, S)`` with 1.0 inside
        the field of view, 0.0 in zero-padding.
    """
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Stage 0 + 1: canonical flip + OD-fovea rotation.
    if config.use_canonical_flip or config.use_od_fovea_rotation:
        image, _ = canonical_orientation(
            image,
            eye_side=eye_side if config.use_canonical_flip else "unknown",
            enable_rotation=config.use_od_fovea_rotation,
        )

    # Stage 2 + 3: FOV crop + isotropic resize → (image, mask).
    image, fov_mask = crop_and_resize(image, config.target_size)

    # Stage 4: flat-field correction (adaptive σ = factor · FOV diameter).
    if config.use_flat_field:
        if config.flat_field_mode == "adaptive":
            fov_rows = np.any(fov_mask > 0, axis=1)
            fov_diameter = float(np.sum(fov_rows))
            sigma = config.flat_field_sigma_factor * fov_diameter
        else:
            sigma = config.flat_field_sigma
        image = apply_flat_field(
            image,
            sigma=sigma,
            mask=fov_mask if config.flat_field_mask_only else None,
        )

    return image, fov_mask


def compute_stats(
    samples: list[tuple[pathlib.Path, str]],
    config: PreprocessingV5Config,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Compute masked per-channel mean/std over preprocessed images.

    Uses single-pass running sums in float64 (mean = Σx/N,
    std = sqrt(Σx²/N − mean²)) on pixels in the [0, 1] scale.

    Args:
        samples: ``(path, eye_side)`` pairs.
        config: Pipeline config controlling Stages 0–4.

    Returns:
        Tuple ``(mean, std, n_images, n_pixels)`` where ``mean`` and ``std``
        are float32 arrays of shape ``(3,)``.
    """
    ch_sum = np.zeros(3, dtype=np.float64)
    ch_sumsq = np.zeros(3, dtype=np.float64)
    n_pixels = 0
    n_used = 0
    n_failed = 0

    for i, (path, eye_side) in enumerate(samples):
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            n_failed += 1
            continue

        image_rgb, fov_mask = _preprocess_stages_0_to_4(img_bgr, eye_side, config)

        sel = fov_mask > 0.5  # (S, S) boolean — real FOV pixels only
        pixels = image_rgb[sel].astype(np.float64) / 255.0  # (P, 3) in [0, 1]
        if pixels.size == 0:
            continue

        ch_sum += pixels.sum(axis=0)
        ch_sumsq += (pixels ** 2).sum(axis=0)
        n_pixels += pixels.shape[0]
        n_used += 1

        if (i + 1) % 500 == 0 or (i + 1) == len(samples):
            print(f"  Processed {i + 1}/{len(samples)} images "
                  f"({n_failed} failed)…", flush=True)

    if n_pixels == 0:
        raise RuntimeError("No FOV pixels collected — check --images-root/--labels-csv.")

    mean = ch_sum / n_pixels
    var = ch_sumsq / n_pixels - mean ** 2
    std = np.sqrt(np.clip(var, 1e-12, None))
    return mean.astype(np.float32), std.astype(np.float32), n_used, n_pixels


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute EyePACS dataset-specific mean/std for V5 Stage 7.",
    )
    parser.add_argument("--images-root", required=True, type=pathlib.Path,
                        help="Directory of <name>.jpeg EyePACS train images.")
    parser.add_argument("--labels-csv", required=True, type=pathlib.Path,
                        help="Path to trainLabels.csv (columns: image, level).")
    parser.add_argument("--output-dir", default="data/processed", type=pathlib.Path,
                        help="Directory to write eyepacs_norm_stats.json.")
    parser.add_argument("--n-samples", default=5000, type=int,
                        help="Images to sample (0 = use all). A few thousand "
                             "gives a stable estimate; default 5000.")
    parser.add_argument("--seed", default=42, type=int, help="Sampling seed.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    # Use the exact config Config D trains with, so Stages 0–4 match.
    config = PreprocessingV5Config.from_preset("efficientnet")

    print(f"Images root : {args.images_root}")
    print(f"Labels CSV  : {args.labels_csv}")
    print(f"n_samples   : {args.n_samples if args.n_samples else 'ALL'}")
    print(f"target_size : {config.target_size}")
    print()

    samples = _discover_eyepacs(args.images_root, args.labels_csv, rng, args.n_samples)
    print(f"Discovered {len(samples)} image(s).\n")
    if not samples:
        print("ERROR: No images found. Check paths.", file=sys.stderr)
        sys.exit(1)

    print("Computing stats (Stages 0–4, mask=1.0 pixels only, no CLAHE)…")
    mean, std, n_used, n_pixels = compute_stats(samples, config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "eyepacs_norm_stats.json"
    payload = {
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
        "n_images": int(n_used),
        "n_pixels": int(n_pixels),
        "scale": "[0,1]",
        "stages": "0-4 (no CLAHE), mask=1.0 only",
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print()
    print("─" * 56)
    print(f"dataset_mean: [{mean[0]:.5f}, {mean[1]:.5f}, {mean[2]:.5f}]")
    print(f"dataset_std : [{std[0]:.5f}, {std[1]:.5f}, {std[2]:.5f}]")
    print(f"images used : {n_used}  |  pixels: {n_pixels:,}")
    print(f"Saved → {out_path}")
    print("─" * 56)
    print("exp1_factorial.py loads this file automatically for Configs B/D.")


if __name__ == "__main__":
    main()
