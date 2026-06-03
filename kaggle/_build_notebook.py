#!/usr/bin/env python3
"""Generate train_config_d.ipynb. One-shot authoring helper (kept for reproducibility)."""
import json
from pathlib import Path

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}
def code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}

cells = []

cells.append(md(
"""# Train Config D (Full V5 + EfficientNet-B3) on EyePACS

Single-fold-per-session training to fit Kaggle's 12 h limit. Change `FOLD`
between sessions: `0 -> 1 -> 2 -> 3 -> 4`, attaching the previous session's
output as an input each time so `--resume` can pick up.

**Before running:** Add Input ->
1. Competition `diabetic-retinopathy-detection` (the 2015 EyePACS DR detection),
   **or** any attached dataset that contains EyePACS `trainLabels.csv` + `*.jpeg`
   train images, **or** the `aptos2019-blindness-detection` competition for a
   quick smaller-dataset functionality check (~3.7k images — auto-converted to
   the EyePACS layout by the adapter cell below).
2. (folds 1-4 only) the previous fold's saved output, so checkpoints resume.

Source repo: https://github.com/yesmukhamedov/dr-classifier
"""))

cells.append(md("## 1. Clone repo + install deps"))
cells.append(code(
"""!git clone --depth 1 https://github.com/yesmukhamedov/dr-classifier.git /kaggle/working/repo
%cd /kaggle/working/repo
!pip install -q -r requirements.txt
!pip install -q timm        # required by src/models/efficientnet.py, not in requirements.txt
"""))

cells.append(md("## 2. Sanity checks"))
cells.append(code(
"""!nvidia-smi
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
!ls /kaggle/input
"""))

cells.append(md(
"""## 3. Dataset layout adapter (EyePACS native **or** APTOS 2019)

`exp1` expects a root dir containing `trainLabels.csv` (columns: `image`,`level`)
and a `train/` subfolder of `<image>.jpeg` files. This cell auto-detects the
attached input and assembles `EYEPACS_ROOT` (no copy, no extra disk):

* **EyePACS** — uses `trainLabels.csv` + `*.jpeg` directly via symlinks.
* **APTOS 2019** — synthesizes the same layout from `train.csv`
  (`id_code`,`diagnosis`) + `train_images/*.png`: writes a derived
  `trainLabels.csv` (`image`=`id_code`, `level`=`diagnosis`) and symlinks each
  `<id_code>.png` as `<id_code>.jpeg` (`cv2.imread` decodes by content, so the
  suffix is cosmetic). This is a quick smaller-dataset functionality check; the
  rest of the notebook is unchanged.

If only the EyePACS competition's multi-part zips are present it falls back to
extraction (slow + disk-heavy — see README)."""))
cells.append(code(
r'''import os, sys, subprocess
import pandas as pd
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK  = Path("/kaggle/working/eyepacs")   # synthesized EyePACS-shaped root (both sources)

def first(gen):
    for x in gen:
        return x
    return None

# 1) Native EyePACS layout: trainLabels.csv (image,level) + *.jpeg train images.
labels     = first(p for p in INPUT.rglob("trainLabels.csv") if p.is_file())
train_jpeg = first(p for p in INPUT.rglob("*.jpeg") if p.is_file())
train_dir  = train_jpeg.parent if train_jpeg else None
SOURCE     = "eyepacs" if (labels is not None and train_dir is not None) else None

# 2) APTOS 2019 layout: train.csv (id_code,diagnosis) + *.png -> synthesize EyePACS layout.
if SOURCE is None:
    def _is_aptos_csv(p):
        try:
            cols = set(pd.read_csv(p, nrows=0).columns)
        except Exception:
            return False
        return {"id_code", "diagnosis"}.issubset(cols)

    aptos_csv = first(p for p in INPUT.rglob("train.csv") if p.is_file() and _is_aptos_csv(p))
    if aptos_csv is not None:
        df  = pd.read_csv(aptos_csv)
        png = first(p for p in aptos_csv.parent.rglob("*.png") if p.is_file()) \
              or first(p for p in INPUT.rglob("*.png") if p.is_file())
        assert png is not None, "APTOS train.csv found but no .png images under /kaggle/input."
        src_dir = png.parent
        WORK.mkdir(parents=True, exist_ok=True)
        tgt_train = WORK / "train"
        tgt_train.mkdir(exist_ok=True)
        n_linked = 0
        for code_id in df["id_code"].astype(str):
            src = src_dir / f"{code_id}.png"
            lnk = tgt_train / f"{code_id}.jpeg"   # cv2 decodes PNG bytes regardless of suffix
            if src.exists() and not lnk.exists():
                os.symlink(src, lnk)
                n_linked += 1
        (df.rename(columns={"id_code": "image", "diagnosis": "level"})[["image", "level"]]
           .to_csv(WORK / "trainLabels.csv", index=False))
        labels    = WORK / "trainLabels.csv"
        train_dir = tgt_train
        SOURCE    = "aptos2019"
        print(f"APTOS detected: linked {n_linked} png->jpeg from {src_dir}")

print("source   :", SOURCE)
print("labels   :", labels)
print("train dir:", train_dir)

# 3) Fallback: extract the EyePACS competition's multi-part zips. Heavy on disk/time.
if labels is None or train_dir is None:
    comp = INPUT / "diabetic-retinopathy-detection"
    print("\nNo ready-to-use layout found. Attempting extraction from", comp)
    WORK.mkdir(parents=True, exist_ok=True)
    if labels is None and (comp / "trainLabels.csv.zip").exists():
        subprocess.run(f"unzip -o -q '{comp}/trainLabels.csv.zip' -d '{WORK}'", shell=True, check=True)
        labels = WORK / "trainLabels.csv"
    parts = sorted(comp.glob("train.zip.0*"))
    if parts and not (WORK / "train").exists():
        print(f"Concatenating {len(parts)} train.zip parts and extracting (this is slow)...")
        subprocess.run(f"cat {comp}/train.zip.0* > /kaggle/working/train.zip", shell=True, check=True)
        subprocess.run(f"unzip -o -q /kaggle/working/train.zip -d '{WORK}'", shell=True, check=True)
        os.remove("/kaggle/working/train.zip")
        train_dir = WORK / "train"
    train_jpeg = first(p for p in WORK.rglob("*.jpeg") if p.is_file())
    train_dir  = train_jpeg.parent if train_jpeg else train_dir

assert labels is not None and train_dir is not None, \
    "Could not locate a usable dataset. Check the attached inputs (see README)."

# Assemble EYEPACS_ROOT = {WORK}/ with a 'train' subdir + trainLabels.csv via symlinks.
# (For APTOS these already exist as real dir/file, so the guards just skip.)
WORK.mkdir(parents=True, exist_ok=True)
tgt_train  = WORK / "train"
tgt_labels = WORK / "trainLabels.csv"
if not tgt_train.exists():
    os.symlink(train_dir, tgt_train)
if not tgt_labels.exists():
    os.symlink(labels, tgt_labels)

EYEPACS_ROOT = str(WORK)
n_imgs = sum(1 for _ in tgt_train.glob("*.jpeg"))
print(f"\nsource = {SOURCE}")
print(f"EYEPACS_ROOT = {EYEPACS_ROOT}")
print(f"train images visible: {n_imgs}")
'''))

cells.append(md("## 4. Build merged config\n\nDeep-merges `default.yaml` <- `kaggle_paths.yaml` and injects the resolved EyePACS path. Writes `configs/kaggle_merged.yaml`, used by every command below."))
cells.append(code(
r'''subprocess.run([
    sys.executable, "kaggle/merge_config.py",
    "--base", "configs/default.yaml",
    "--override", "kaggle/kaggle_paths.yaml",
    "--out", "configs/kaggle_merged.yaml",
    f"paths.eyepacs={EYEPACS_ROOT}",
    # If you hit CUDA OOM at 512px (EfficientNet runs fp32), uncomment:
    # "training.batch_size=8",
], check=True)
'''))

cells.append(md(
"""## 5. (Optional) PCA eigenvectors for colour augmentation

One-shot. If skipped, colour augmentation is simply disabled (exp1 prints a
notice and continues). Output dir matches where exp1 looks:
`/kaggle/working/data/processed`. Comment out the cell to skip."""))
cells.append(code(
r'''subprocess.run([
    sys.executable, "scripts/compute_pca_eigvecs.py",
    "--dataset", "eyepacs",
    "--images-root", str(Path(EYEPACS_ROOT) / "train"),
    "--labels-csv", str(Path(EYEPACS_ROOT) / "trainLabels.csv"),
    "--output-dir", "/kaggle/working/data/processed",
    "--n-samples", "5000",
    "--seed", "42",
], check=True)
'''))

cells.append(md(
"""## 6. Dataset-specific normalization stats (Stage 7, D-2) — REQUIRED for Config D

The thesis (`methods/preprocessing-pipeline.md`, Stage 7) specifies that the
full configs (B/D) normalize with mean/std **computed from the EyePACS training
set after Stages 0–4, mask=1.0 pixels only** — *not* ImageNet (that is the
baseline-only fallback). This cell computes those stats and writes
`/kaggle/working/data/processed/eyepacs_norm_stats.json`, which
`exp1_factorial.py` loads automatically for Config D.

One-shot per dataset. `--n-samples 5000` gives a stable estimate; use `0` for
the full train set (much slower — Stage 1 OD-fovea rotation is the bottleneck)."""))
cells.append(code(
r'''subprocess.run([
    sys.executable, "scripts/compute_dataset_stats.py",
    "--images-root", str(Path(EYEPACS_ROOT) / "train"),
    "--labels-csv", str(Path(EYEPACS_ROOT) / "trainLabels.csv"),
    "--output-dir", "/kaggle/working/data/processed",
    "--n-samples", "5000",
    "--seed", "42",
], check=True)
'''))

cells.append(md("## 7. Train one fold\n\nSet `FOLD` per session. `--resume` continues from `last_checkpoint.pt` if the previous output is attached."))
cells.append(code(
r'''FOLD = 0   # <-- change to 1, 2, 3, 4 in subsequent sessions

subprocess.run([
    sys.executable, "run_experiment.py", "exp1",
    "--config", "configs/kaggle_merged.yaml",
    "--configs", "D",
    "--fold", str(FOLD),
    "--resume",
], check=True)
'''))

cells.append(md("## 8. Inspect outputs\n\nOutputs already live under `/kaggle/working/outputs`, which persists with *Save Version* — no copy needed. Best checkpoint is `best_model.pt`."))
cells.append(code(
r'''ckpt_dir = Path("/kaggle/working/outputs/exp1/checkpoints") / f"D_fold{FOLD}"
print("checkpoint dir:", ckpt_dir)
!ls -la {ckpt_dir}
print("\n--- metrics.csv (tail) ---")
!tail -n 15 /kaggle/working/outputs/exp1/metrics.csv
'''))

cells.append(md(
"""## 9. Multi-fold strategy

Run this notebook five times, once per fold:

| Session | `FOLD` | Add Input |
|---|---|---|
| 1 | 0 | competition (or EyePACS dataset) |
| 2 | 1 | + session-1 output dataset |
| 3 | 2 | + session-2 output dataset |
| 4 | 3 | + session-3 output dataset |
| 5 | 4 | + session-4 output dataset |

After each session: *Save Version -> Save & Run All (Commit)*. The committed
output (`/kaggle/working/outputs/...`) becomes a dataset you attach to the next
session so the adapter sees prior checkpoints and `--resume` continues.

When all five folds are done, download `/kaggle/working/outputs/exp1/` and place
it locally at `experiments/outputs/exp1/`, then run
`python scripts/verify_exp1.py`."""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "train_config_d.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out)
