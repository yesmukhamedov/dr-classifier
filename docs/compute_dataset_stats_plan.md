# Implementation Plan: `compute_dataset_stats.py`

**Issue resolved:** B-6  
**Implemented by:** ChatGPT (this plan) → Claude Code (review)  
**Script location:** `experiments/scripts/compute_dataset_stats.py`

---

## 1. Specification

### Purpose

Compute EyePACS training-set channel-wise mean and standard deviation for
V5 Stage 7 normalization. The output is pasted into `configs/default.yaml`
under `preprocessing_v5.dataset_mean` and `preprocessing_v5.dataset_std`.

### Input

- **Dataset:** EyePACS full training set (~35,126 images), loaded from
  `trainLabels.csv` at `{config.paths.eyepacs}/trainLabels.csv`.
  Images at `{config.paths.eyepacs}/train/{name}.jpeg`.
- **Scope:** ALL images in the CSV (all folds combined — stats are computed
  on the full training corpus, not per-fold).

### Pipeline stages to apply

| Stage | Name | Apply? |
|-------|------|--------|
| 0 | Canonical flip (left→right) | YES |
| 1 | OD-fovea rotation normalization | YES |
| 2 | FOV crop + isotropic resize 512×512 | YES |
| 3 | FOV mask generation (returned by Stage 2) | YES (implicit) |
| 4 | Flat-field correction (adaptive σ=0.07·D) | YES |
| 5 | CLAHE | **NO** — stochastic; computing stats after stochastic Stage 5 is methodologically unsound |
| 6 | Augmentation | **NO** — train-only, stochastic |
| 7 | Normalize → tensor | **NO** — this is what we are computing |

**Design decision D-2:** Stats are computed after the deterministic Stages 0–4,
before stochastic CLAHE (Stage 5). Computing stats after a stochastic transform
would produce non-reproducible statistics, violating the requirement for a stable
normalization constant. See also `INVARIANTS.md` OD-3 Stage 7 description.

### Mask handling

For channels 0–2 (RGB), include **only pixels where `fov_mask == 1.0`**.  
Zero-padded background pixels (`fov_mask == 0.0`) must NOT enter the computation.

### Output format

Three-element float lists in [0, 1] range (float32), suitable for pasting
directly into `default.yaml`:

```yaml
dataset_mean: [R, G, B]   # computed by compute_dataset_stats.py
dataset_std:  [R, G, B]   # computed by compute_dataset_stats.py
```

---

## 2. Implementation Approach

### Memory strategy: Welford's online algorithm

Do NOT collect all pixels in memory. EyePACS has ~35k images × ~160k masked
pixels per image × 3 channels × 4 bytes = ~67 GB. Instead, use Welford's
online algorithm to compute running mean and variance in a single pass.

Welford's update per pixel-batch (vectorised across N pixels):
```
For a new batch of N pixel values p[0..N-1]:
  count  += N
  delta   = p - mean           # broadcast over N
  mean   += delta.sum(axis=0) / count
  delta2  = p - mean           # recomputed after mean update
  M2     += (delta * delta2).sum(axis=0)

Final:
  variance = M2 / count        # population variance
  std      = sqrt(variance)
```

Batch-update Welford (more numerically stable than single-sample iteration)
is preferred — process all masked pixels of one image as one batch.

### Why call stage functions directly, not through the pipeline

`PreprocessingPipelineV5.__call__()` always runs Stage 7 (normalization).
There is no flag to disable it. Therefore, call the stage functions directly
rather than constructing a pipeline object. This avoids any risk of accidentally
including CLAHE or normalization.

---

## 3. Code Structure (Pseudocode with Real Function Names)

### Script docstring

```python
"""Compute dataset-specific mean and std for V5 normalization (Stage 7).

Stats computed after deterministic Stages 0–4, before stochastic CLAHE
(Stage 5). See INVARIANTS OD-3 and design decision D-2 for rationale.

Only pixels where FOV mask = 1.0 are included in the computation (background
zero-padding is excluded). Uses Welford's online algorithm to avoid loading
all images into memory simultaneously.

Usage:
    python scripts/compute_dataset_stats.py --config configs/default.yaml

Output:
    Prints mean and std values (float32, [0,1] range) to paste into
    default.yaml under preprocessing_v5.dataset_mean / dataset_std.
"""
```

### Imports

```python
import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from src.preprocessing.canonical_orientation import canonical_orientation
from src.preprocessing.crop_resize import crop_and_resize
from src.preprocessing.flat_field import apply_flat_field
from src.preprocessing.splits import extract_patient_id  # for eye_side detection
```

### Config loading

```python
def load_config(config_path: str) -> dict:
    """Load YAML config and return as dict.

    Args:
        config_path: Path to configs/default.yaml.

    Returns:
        Parsed YAML as a plain dict.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)
```

### EyePACS image discovery

```python
def discover_eyepacs_images(eyepacs_root: Path) -> list[tuple[Path, str]]:
    """Discover all EyePACS training images and their eye sides.

    Reads trainLabels.csv and resolves absolute image paths.
    Returns only paths that exist on disk.

    Args:
        eyepacs_root: Path to EyePACS dataset root (e.g. E:/datasets/EyePACS).

    Returns:
        List of (image_path, eye_side) tuples where eye_side is
        "left", "right", or "unknown".
    """
    labels_csv = eyepacs_root / "trainLabels.csv"
    df = pd.read_csv(labels_csv)

    result = []
    for _, row in df.iterrows():
        name: str = str(row["image"])
        img_path = eyepacs_root / "train" / f"{name}.jpeg"
        if not img_path.exists():
            continue
        eye_side = "left" if "_left" in name else "right"
        result.append((img_path, eye_side))
    return result
```

### Welford accumulator (per-channel, over pixels)

```python
class WelfordAccumulator:
    """Online mean/variance accumulator using Welford's algorithm.

    Processes batches of pixel values (float32, [0, 1]).
    Channel dimension is tracked independently.

    Args:
        n_channels: Number of channels (3 for RGB).
    """

    def __init__(self, n_channels: int = 3) -> None:
        self.count: int = 0
        self.mean: np.ndarray = np.zeros(n_channels, dtype=np.float64)
        self.M2: np.ndarray = np.zeros(n_channels, dtype=np.float64)

    def update(self, pixels: np.ndarray) -> None:
        """Update running statistics with a batch of pixels.

        Args:
            pixels: float32 array of shape (N, C) where N is the number
                of masked pixels and C is the number of channels.
                Values must be in [0, 1].
        """
        pixels = pixels.astype(np.float64)  # use float64 for numerical stability
        n = pixels.shape[0]
        if n == 0:
            return

        # Batch Welford update
        new_count = self.count + n
        batch_mean = pixels.mean(axis=0)           # (C,)
        delta = batch_mean - self.mean              # (C,)
        self.mean += delta * n / new_count
        # Within-batch sum of squared deviations
        batch_var = pixels.var(axis=0) * n         # (C,)  population variance × n
        # Cross-term correction
        delta2 = batch_mean - self.mean             # (C,) recomputed after mean update
        self.M2 += batch_var + delta * delta2 * n
        self.count = new_count

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute final mean and standard deviation.

        Returns:
            Tuple of (mean, std) as float32 arrays of shape (C,).

        Raises:
            RuntimeError: If no pixels have been accumulated.
        """
        if self.count == 0:
            raise RuntimeError("No pixels accumulated — check dataset path.")
        variance = self.M2 / self.count            # population variance
        std = np.sqrt(variance)
        return self.mean.astype(np.float32), std.astype(np.float32)
```

### Per-image processing (stages 0–4 only)

```python
def process_image_stages_0_to_4(
    image_bgr: np.ndarray,
    eye_side: str,
    target_size: int,
    flat_field_sigma_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply V5 Stages 0–4 to one image. Does NOT apply CLAHE, augmentation, or normalization.

    Args:
        image_bgr: BGR uint8 array from cv2.imread, shape (H, W, 3).
        eye_side: "left", "right", or "unknown".
        target_size: Output spatial resolution (512).
        flat_field_sigma_factor: σ = factor × FOV diameter (default 0.07).

    Returns:
        Tuple of:
          - image: RGB uint8 array of shape (target_size, target_size, 3) after Stages 0–4.
          - fov_mask: float32 array of shape (target_size, target_size),
                      1.0 = real data, 0.0 = padding.
    """
    # Convert BGR → RGB (cv2.imread returns BGR)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Stage 0 + 1: canonical flip + OD-fovea rotation
    image, _ = canonical_orientation(image, eye_side=eye_side, enable_rotation=True)

    # Stage 2 + 3: FOV crop + isotropic resize → image + FOV mask
    image, fov_mask = crop_and_resize(image, target_size=target_size)

    # Compute adaptive σ from FOV diameter (height of mask region)
    fov_rows = np.any(fov_mask > 0, axis=1)
    fov_diameter = float(np.sum(fov_rows))
    adaptive_sigma = flat_field_sigma_factor * fov_diameter

    # Stage 4: flat-field correction (inside mask only)
    image = apply_flat_field(image, sigma=adaptive_sigma, mask=fov_mask)

    return image, fov_mask
```

### Pixel extraction and normalization to [0, 1]

```python
def extract_masked_pixels(
    image: np.ndarray,
    fov_mask: np.ndarray,
) -> np.ndarray:
    """Extract only the pixels inside the FOV mask, normalized to [0, 1].

    Args:
        image: RGB uint8 array of shape (H, W, 3) after Stages 0–4.
        fov_mask: float32 array of shape (H, W), 1.0 = real data.

    Returns:
        float32 array of shape (N_masked_pixels, 3) with values in [0, 1].
    """
    mask_bool = fov_mask == 1.0               # (H, W) boolean
    pixels = image[mask_bool]                 # (N, 3) uint8
    return pixels.astype(np.float32) / 255.0  # (N, 3) float32 in [0, 1]
```

### Main function

```python
def main() -> None:
    """Entry point: compute EyePACS dataset mean and std for Stage 7 normalization."""
    parser = argparse.ArgumentParser(
        description="Compute EyePACS dataset mean/std for V5 Stage 7 normalization."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML config file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    eyepacs_root = Path(cfg["paths"]["eyepacs"])
    target_size = cfg["preprocessing_v5"].get("target_size", 512)
    sigma_factor = cfg["preprocessing_v5"].get("flat_field_sigma_factor", 0.07)

    print(f"EyePACS root: {eyepacs_root}")
    print(f"Target size: {target_size}")
    print(f"Flat-field sigma factor: {sigma_factor}")
    print("Discovering images...")

    image_list = discover_eyepacs_images(eyepacs_root)
    print(f"Found {len(image_list)} images.")

    accumulator = WelfordAccumulator(n_channels=3)
    errors = 0

    for i, (img_path, eye_side) in enumerate(image_list):
        if (i + 1) % 500 == 0:
            print(f"  [{i + 1}/{len(image_list)}] processed so far...")

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"  [WARN] Could not read: {img_path}")
            errors += 1
            continue

        try:
            image_rgb, fov_mask = process_image_stages_0_to_4(
                image_bgr,
                eye_side=eye_side,
                target_size=target_size,
                flat_field_sigma_factor=sigma_factor,
            )
        except Exception as exc:
            print(f"  [WARN] Stage 0-4 failed for {img_path}: {exc}")
            errors += 1
            continue

        pixels = extract_masked_pixels(image_rgb, fov_mask)
        accumulator.update(pixels)

    mean, std = accumulator.finalize()

    print("\n" + "=" * 60)
    print("RESULTS (paste into default.yaml):")
    print("=" * 60)
    print(f"  dataset_mean: [{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}]")
    print(f"  dataset_std:  [{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]")
    print("=" * 60)
    print(f"Total pixels processed: {accumulator.count:,}")
    print(f"Images with errors: {errors}")
    print("\nSanity checks:")
    print(f"  Mean in [0, 1]: {all(0.0 <= m <= 1.0 for m in mean)}")
    print(f"  Std in (0.01, 0.5): {all(0.01 < s < 0.5 for s in std)}")
    imagenet_mean = [0.485, 0.456, 0.406]
    differs = any(abs(float(m) - im) > 0.01 for m, im in zip(mean, imagenet_mean))
    print(f"  Differs from ImageNet mean: {differs}")


if __name__ == "__main__":
    main()
```

---

## 4. Script Docstring

The top-level module docstring must be exactly:

```python
"""Compute dataset-specific mean and std for V5 normalization (Stage 7).

Stats computed after deterministic Stages 0–4, before stochastic CLAHE
(Stage 5). See INVARIANTS OD-3 and design decision D-2 for rationale.

Only pixels where FOV mask = 1.0 are included in the computation (background
zero-padding is excluded). Uses Welford's online algorithm to avoid loading
all images into memory simultaneously.

Usage:
    python scripts/compute_dataset_stats.py --config configs/default.yaml

Output:
    Prints mean and std values (float32, [0,1] range) to paste into
    default.yaml under preprocessing_v5.dataset_mean / dataset_std.
"""
```

---

## 5. YAML Update Instructions

After running the script successfully, locate the `preprocessing_v5` section in
`configs/default.yaml` and replace the `null` placeholders with the printed values:

```yaml
preprocessing_v5:
  # --- Stage 7: Dataset-Specific Normalize ---
  normalize_mode: dataset_specific        # "dataset_specific" or "imagenet"
  # Dataset-specific stats computed from EyePACS train set (mask=1.0 pixels only)
  dataset_mean: [R, G, B]   # computed by compute_dataset_stats.py
  dataset_std:  [R, G, B]   # computed by compute_dataset_stats.py
  # ImageNet fallback (used for baseline configs)
  imagenet_mean: [0.485, 0.456, 0.406]
  imagenet_std: [0.229, 0.224, 0.225]
```

Replace `[R, G, B]` with the exact values printed by the script, e.g.:

```yaml
  dataset_mean: [0.381245, 0.241830, 0.156781]
  dataset_std:  [0.273912, 0.182044, 0.120357]
```

The values above are illustrative only — do not use them. Use the values
output by your actual run.

---

## 6. Verification Steps

### After running the script

1. **Mean in [0, 1] range:** All three mean values must be in `[0.0, 1.0]`.  
   Fundus images are predominantly dark (optic structures against background);
   expect mean values roughly in `[0.10, 0.50]`.

2. **Std in reasonable range:** All three std values must be well above 0.01
   (not near-zero) and well below 0.5 (not near full-range).  
   Typical fundus std values are in `[0.05, 0.35]`.

3. **Differs from ImageNet stats:** The computed mean must differ from ImageNet
   mean `[0.485, 0.456, 0.406]` by more than 0.01 in at least one channel.
   If values are identical to ImageNet, the normalization fallback has been
   applied — investigate the config loading or the image loading pipeline.

4. **Pixel count sanity:** The script prints the total number of pixels
   processed. With ~35k images at 512×512 and ~80% mask coverage:
   expected pixel count ≈ 35,000 × 512 × 512 × 0.80 ≈ 7.3 billion pixels.
   A significantly lower count indicates images were skipped due to errors.

### After updating default.yaml

5. Run any experiment that loads the config and check that Stage 7
   normalization picks up the dataset stats rather than the ImageNet fallback.
   In `pipeline_v5.py`, the relevant check is:
   ```python
   if self.config.normalize_mode == "dataset_specific" and \
           self.config.dataset_mean is not None and \
           self.config.dataset_std is not None:
   ```
   Log the mean/std values used during a training run to confirm.

---

## 7. Edge Cases and Implementation Notes

- **Images that fail Stage 0–4:** Skip with a warning, increment error counter.
  Do not crash the entire run.
- **Images cv2 cannot read:** Skip with a warning. Check path and encoding.
- **float64 accumulators:** Use `float64` inside `WelfordAccumulator` to
  minimize floating-point drift over ~7 billion pixel accumulations.
  Cast to `float32` only at `finalize()`.
- **Progress logging:** Print progress every 500 images. With ~35k images
  and Stage 0–4 running ~200 ms per image (including OD-fovea rotation),
  total runtime is approximately 2 hours on the RTX 3060.
- **Eye side detection for EyePACS:** From filename — "left" if `"_left"` in
  `name` else `"right"`. This matches `EyePACSDataset.from_directory()`.
- **No shuffle needed:** Order of images does not affect Welford's result.
- **Script working directory:** Run from `experiments/` so that relative
  imports (`from src.preprocessing...`) resolve correctly.

---

*Plan ready for ChatGPT implementation. Implementation output: replace the
`raise NotImplementedError(...)` block in `experiments/scripts/compute_dataset_stats.py`
with the full implementation described above.*
