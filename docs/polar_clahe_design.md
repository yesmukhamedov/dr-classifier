# Polar CLAHE — Design Document

Research note for anatomy-aware contrast enhancement on fundus images.

---

## Problem Statement

Standard CLAHE uses rectangular 8×8 grid. On fundus images this causes:
- Visible tile boundaries ("grid artifacts")
- Tiles split across vessels/lesions
- Tiles at FOV edges are partially empty (wasted)
- No adaptation to radial fundus geometry

Our `upgraded_clahe.py` adds dual clip constraint but lacks bilinear interpolation between tiles → produces visible grid squares.

---

## Proposed Architecture: Polar Vessel-Adaptive CLAHE

### Pipeline

```
Input (L-channel from LAB)
    ↓
1. Vessel detection (Frangi filter)
    ↓
2. Vessel density map
    ↓
3. Polar grid (fovea-centered, density-adaptive)
    ↓
4. Per-sector dual-constraint CLAHE (LUT computation)
    ↓
5. Polar bilinear interpolation (4-neighbor blending)
    ↓
Output (enhanced L-channel)
```

---

## Step-by-Step Design

### Step 1. LAB Conversion

```python
lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
L, A, B = cv2.split(lab)
```

### Step 2. Polar Coordinates

Center = Fovea (from `coords.json`, already detected in stage_1).

```python
dx = x - fovea_x
dy = y - fovea_y
r = sqrt(dx^2 + dy^2)
theta = atan2(dy, dx)

r_norm = r / r_max        # [0, 1]
theta_norm = theta / 2pi   # [0, 1)
```

### Step 3. Grid Discretization

Base grid:
- Nr = 6–10 radial rings
- Nθ = 12–24 angular sectors

```python
r_bin = int(r_norm * Nr)
theta_bin = int(theta_norm * Nθ)
```

#### Adaptive variant (vessel-density-driven)

Radial bins — logarithmic spacing (finer at center, coarser at periphery):
```python
r_bins = logspace(...)  # more rings near fovea
```

Angular bins — proportional to vessel density in each ring:
```python
# high vessel density → more angular sectors
n_sectors[ring_i] = base_sectors * (1 + beta * vessel_density[ring_i])
```

### Step 4. Per-Sector CLAHE (Dual Constraint)

For each polar sector (r_bin, theta_bin):
1. Collect pixels belonging to this sector
2. Compute histogram (256 bins)
3. Apply dual clip constraint:
   ```python
   clip_limit = clip_factor * tile_area / 256
   clip_limit = min(clip_limit, global_threshold * tile_area)
   ```
4. Redistribute excess counts uniformly
5. Build CDF → LUT (lookup table)

#### Adaptive clipLimit as function of radius

```python
clipLimit(r) = base * (1 + alpha * (1 - r_norm))
# center → higher contrast (details matter)
# periphery → lower contrast (reduce noise)
```

### Step 5. Polar Bilinear Interpolation (CRITICAL)

Without interpolation → radial seams + ring artifacts.

Each pixel gets contribution from 4 neighboring sectors:
- 2 adjacent rings (inner/outer)
- 2 adjacent angular sectors (left/right)

Weights:
```python
w_r = fractional part of (r_norm * Nr)       # radial weight
w_theta = fractional part of (theta_norm * Nθ)  # angular weight

# 4 neighbors
sectors = [(r_lo, t_lo), (r_lo, t_hi), (r_hi, t_lo), (r_hi, t_hi)]
weights = [(1-w_r)*(1-w_theta), (1-w_r)*w_theta, w_r*(1-w_theta), w_r*w_theta]

L_out = sum(w_i * LUT_i[L_in] for w_i, LUT_i in zip(weights, sector_LUTs))
```

This is the standard bilinear scheme but in polar coordinates.

---

## Design Variants Considered

### A. Radial CLAHE (selected for implementation)

Polar segmentation centered on fovea. Most organic fit for fundus:
- Vessels radiate from OD
- Illumination changes radially
- Macula/fovea is the contrast center
- No wasted rectangular tiles outside FOV circle

Formalization: *"Polar-domain contrast limited adaptive histogram equalization aligned with retinal geometry"*

### B. Vessel-Aware CLAHE

Use vessel map as weight: higher clipLimit on vessels, lower on background.
- Vessels are the key diagnostic signal
- Standard CLAHE amplifies noise and signal equally
- Formalization: *"Structure-guided contrast enhancement"*

### C. Illumination-Compensated CLAHE

Adaptive clipLimit based on local illumination level:
```python
clipLimit(x,y) = base * (1 + alpha * illumination(x,y))
```
Prevents over-enhancement on dark periphery. We already do flat-field (stage 4), so this is partially redundant.

### D. Disc-Centered Adaptive Tiling

OD as center instead of fovea. Content-adaptive tile sizes:
- Center → small tiles (details)
- Periphery → large tiles (noise reduction)

### E. Seamless Interpolation

Not a variant but a mandatory requirement for any tiling scheme. Without it, any "smart tiles" still produce artifacts. Options:
- Bilinear interpolation between tile LUTs
- Overlap + blending (more expensive)

---

## Chosen Approach

**Polar CLAHE + vessel-density adaptive sectors + dual-constraint clipping + polar bilinear interpolation**

Why:
- Uses fundus geometry (polar = natural for circular FOV)
- Uses anatomy (fovea center, vessel density)
- Preserves our dual-constraint contribution
- Bilinear interpolation eliminates tile artifacts
- Implementable as ~100-150 lines of Python

Novelty:
1. Polar tiling (vs rectangular) — matches retinal geometry
2. Vessel-density adaptive sector size — content-aware
3. Dual clip constraint — inherited from upgraded_clahe
4. Radial clipLimit decay — reduces peripheral noise

---

## Relation to Current Pipeline

```
Current:  flat-field → upgraded_clahe (8×8 rect, dual clip, NO interpolation)
Proposed: flat-field → polar_clahe (Nr×Nθ polar, dual clip + vessel adaptive, WITH interpolation)
```

Stage 5 in V5 pipeline would use polar_clahe instead of upgraded_clahe.
All other stages unchanged.

---

## Implementation Estimate

- Frangi vessel detection: existing (OpenCV/scikit-image) — ~20 lines
- Polar coordinate mapping: ~15 lines
- Adaptive grid construction: ~30 lines
- Per-sector CLAHE (reuse upgraded_clahe internals): ~20 lines
- Polar bilinear interpolation: ~50 lines
- Integration + testing: ~30 lines
- **Total: ~150 lines, ~2-3 hours**

---

## Key Constraint

Interpolation is mandatory. Without it, polar tiling produces radial seams and ring artifacts — same problem as rectangular tiles but in polar form. The bilinear interpolation step (Step 5) is not optional.

---

## References

- Original dual-constraint CLAHE: `experiments/src/preprocessing/upgraded_clahe.py`
- Stage 4 flat-field: `experiments/src/preprocessing/flat_field.py`
- ChatGPT analysis: variants A–E evaluation, bilinear interpolation scheme
- Coords (fovea centers): `demo/public/pipeline/helpers/coords.json`
