"""Precompute and cache V5 Stages 0–4 for the full EyePACS set (throughput fix).

The V5 bottleneck (TASK.md §2): the entire 8-stage pipeline runs on CPU inside
the DataLoader, per image, **every epoch**. The heaviest stages (OD/fovea
detection, FOV crop+resize, flat-field) are the deterministic Stages 0–4 — they
carry no train-time randomness (the first stochastic stage is CLAHE, Stage 5).
So they can be computed **once** and cached; training then loads the cache and
runs only the cheap stochastic Stages 5–7, turning a CPU-bound epoch into a
GPU-bound one.

This script mirrors ``PreprocessingPipelineV5.precompute_deterministic`` exactly
(it *is* that call) so the cache is bit-identical to what the live pipeline would
have produced. Output per image is a 4-channel PNG:

    R,G,B = Stage-4 flat-field output (uint8, padding zeroed)
    A     = binary FOV mask (0 or 255)  ← lossless: the mask is strictly {0,1}

plus a sidecar ``cache_meta.csv`` with the two OD/fovea scalars Stage 6 reads
(``confident``, ``rotation_sigma_deg``). ``trainLabels.csv`` is copied into the
cache dir so the cache is a self-contained training input.

Usage:
    python scripts/precompute_v5_cache.py \
        --images-root /path/to/EyePACS/train \
        --labels-csv  /path/to/EyePACS/trainLabels.csv \
        --output-dir  /path/to/v5_cache \
        --num-workers 8 \
        --preset      efficientnet

Resumable: re-running skips images already in ``cache_meta.csv`` with a PNG on
disk, so an interrupted build continues where it stopped.

Output:
    <output-dir>/<name>.png        — one 4-channel PNG per image
    <output-dir>/cache_meta.csv    — image,confident,rotation_sigma_deg
    <output-dir>/trainLabels.csv   — copy of the labels CSV (image,level)
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import shutil
import sys

import cv2
import numpy as np

# Allow running from repo root without installing the package.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Module-global pipeline, built once per worker process (avoids re-import cost
# and keeps cv2/pipeline objects out of the pickled task payloads).
_PIPELINE = None
_OUT_DIR: pathlib.Path | None = None
_PNG_COMPRESSION = 6


def _init_worker(preset: str, out_dir: str, png_compression: int) -> None:
    """Pool initializer: build one inference pipeline per worker.

    Args:
        preset: ``PreprocessingV5Config`` preset name (must match the training
            config — Config D uses ``"efficientnet"``).
        out_dir: Cache output directory.
        png_compression: cv2 PNG compression level (0–9).
    """
    global _PIPELINE, _OUT_DIR, _PNG_COMPRESSION
    from src.preprocessing.config import PreprocessingV5Config
    from src.preprocessing.pipeline_v5 import PreprocessingPipelineV5

    _PIPELINE = PreprocessingPipelineV5.create_for_inference(
        PreprocessingV5Config.from_preset(preset)
    )
    _OUT_DIR = pathlib.Path(out_dir)
    _PNG_COMPRESSION = png_compression


def _process_one(task: tuple[str, str, str]) -> tuple[str, str, float]:
    """Cache one image's Stages 0–4. Runs in a worker process.

    Args:
        task: ``(name, image_path, eye_side)``.

    Returns:
        ``(name, status, rotation_sigma_deg)`` where ``status`` is ``"ok"``,
        ``"ok:True"``/``"ok:False"`` encoding ``confident``, or an error string.
    """
    name, image_path, eye_side = task
    assert _PIPELINE is not None and _OUT_DIR is not None

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return (name, "ERROR_READ", 0.0)

    flat_rgb, fov_mask, confident, rotation_sigma_deg = (
        _PIPELINE.precompute_deterministic(img_bgr, eye_side)
    )

    # 4-channel PNG: BGR (for cv2 round-trip) + binary mask as alpha (0/255).
    bgr = cv2.cvtColor(flat_rgb, cv2.COLOR_RGB2BGR)
    alpha = (fov_mask > 0.5).astype(np.uint8) * 255
    bgra = np.dstack([bgr, alpha])

    out_png = _OUT_DIR / f"{name}.png"
    ok = cv2.imwrite(str(out_png), bgra, [cv2.IMWRITE_PNG_COMPRESSION, _PNG_COMPRESSION])
    if not ok:
        return (name, "ERROR_WRITE", 0.0)

    return (name, f"ok:{bool(confident)}", float(rotation_sigma_deg))


def _discover(
    images_root: pathlib.Path,
    labels_csv: pathlib.Path,
    limit: int,
) -> list[tuple[str, str, str]]:
    """Build ``(name, image_path, eye_side)`` tasks from the labels CSV.

    Mirrors ``exp1_factorial`` indexing: ``<name>.jpeg`` under *images_root*,
    eye side parsed from the ``_left``/``_right`` suffix.

    Args:
        images_root: Directory of ``<name>.jpeg`` images.
        labels_csv: ``trainLabels.csv`` with an ``image`` column.
        limit: Max images (0 = all) — for quick tests.

    Returns:
        List of ``(name, image_path, eye_side)`` tuples.
    """
    tasks: list[tuple[str, str, str]] = []
    with open(labels_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            name = row.get("image", "")
            if not name:
                continue
            path = images_root / f"{name}.jpeg"
            if not path.exists():
                continue
            side = "left" if "_left" in name else "right" if "_right" in name else "unknown"
            tasks.append((name, str(path), side))
            if limit and len(tasks) >= limit:
                break
    return tasks


def _load_done(meta_path: pathlib.Path, out_dir: pathlib.Path) -> set[str]:
    """Names already cached (present in meta AND with a PNG on disk) — for resume."""
    if not meta_path.exists():
        return set()
    done: set[str] = set()
    with open(meta_path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = row.get("image", "")
            if name and (out_dir / f"{name}.png").exists():
                done.add(name)
    return done


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute V5 Stages 0–4 cache for EyePACS (throughput fix).",
    )
    parser.add_argument("--images-root", required=True, type=pathlib.Path,
                        help="Directory of <name>.jpeg EyePACS train images.")
    parser.add_argument("--labels-csv", required=True, type=pathlib.Path,
                        help="Path to trainLabels.csv (columns: image, level).")
    parser.add_argument("--output-dir", required=True, type=pathlib.Path,
                        help="Cache output directory (PNGs + cache_meta.csv).")
    parser.add_argument("--preset", default="efficientnet",
                        help="PreprocessingV5Config preset; must match training "
                             "(Config D = 'efficientnet'). Default: efficientnet.")
    parser.add_argument("--num-workers", default=max(1, os.cpu_count() or 1), type=int,
                        help="Parallel worker processes (default: all CPU cores).")
    parser.add_argument("--png-compression", default=_PNG_COMPRESSION, type=int,
                        help="cv2 PNG compression 0–9 (higher = smaller/slower). "
                             "Default 6.")
    parser.add_argument("--limit", default=0, type=int,
                        help="Cap number of images (0 = all) — for quick tests.")
    return parser.parse_args()


def main() -> None:
    import multiprocessing as mp

    args = _parse_args()
    out_dir: pathlib.Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "cache_meta.csv"

    print(f"Images root : {args.images_root}")
    print(f"Labels CSV  : {args.labels_csv}")
    print(f"Output dir  : {out_dir}")
    print(f"Preset      : {args.preset}  |  workers: {args.num_workers}")
    print()

    tasks = _discover(args.images_root, args.labels_csv, args.limit)
    print(f"Discovered {len(tasks)} image(s).")
    if not tasks:
        print("ERROR: No images found. Check --images-root/--labels-csv.", file=sys.stderr)
        sys.exit(1)

    done = _load_done(meta_path, out_dir)
    todo = [t for t in tasks if t[0] not in done]
    print(f"Already cached: {len(done)}  |  to process: {len(todo)}\n")

    # Append-mode meta so resume keeps prior rows; write header if new.
    write_header = not meta_path.exists()
    n_ok = 0
    n_err = 0
    errors: list[str] = []

    with open(meta_path, "a", newline="") as meta_fh:
        writer = csv.writer(meta_fh)
        if write_header:
            writer.writerow(["image", "confident", "rotation_sigma_deg"])

        if todo:
            ctx = mp.get_context("spawn")  # safe with cv2/OpenMP across platforms
            with ctx.Pool(
                processes=max(1, args.num_workers),
                initializer=_init_worker,
                initargs=(args.preset, str(out_dir), args.png_compression),
            ) as pool:
                for i, (name, status, rot) in enumerate(
                    pool.imap_unordered(_process_one, todo, chunksize=8), start=1
                ):
                    if status.startswith("ok"):
                        confident = status.split(":", 1)[1]  # "True"/"False"
                        writer.writerow([name, confident, f"{rot:.6f}"])
                        n_ok += 1
                    else:
                        n_err += 1
                        if len(errors) < 20:
                            errors.append(f"{name}: {status}")
                    if i % 500 == 0 or i == len(todo):
                        meta_fh.flush()
                        print(f"  {i}/{len(todo)} processed "
                              f"(ok={n_ok}, err={n_err})…", flush=True)

    # Copy labels CSV so the cache is a self-contained training input.
    labels_dst = out_dir / "trainLabels.csv"
    if not labels_dst.exists():
        shutil.copyfile(args.labels_csv, labels_dst)

    print()
    print("─" * 56)
    print(f"Cache dir   : {out_dir}")
    print(f"Newly cached: {n_ok}  |  errors: {n_err}  |  total target: {len(tasks)}")
    if errors:
        print("First errors:")
        for e in errors:
            print(f"  - {e}")
    print(f"Sidecar     : {meta_path.name}  +  trainLabels.csv")
    print("─" * 56)
    print("Point training at this dir via  paths.v5_cache_dir=<output-dir>.")


if __name__ == "__main__":
    main()
