#!/usr/bin/env python3
"""Generate train_config_d_colab.ipynb.

One-shot authoring helper (kept for reproducibility). Targets Google Colab with
two design goals that make the **real EyePACS Config-D run** actually feasible:

1. **V5 Stage 0–4 cache (the throughput fix).** The bottleneck is CPU-side V5
   preprocessing recomputed per image every epoch (TASK §2). This notebook builds
   the deterministic Stages 0–4 cache **once** (scripts/precompute_v5_cache.py),
   then trains from it so each epoch is GPU-bound. Built once, reused every fold.

2. **Kaggle-Dataset persistence (replaces Google Drive entirely).** No Drive, no
   OAuth, no paid 15 GB tier. The ~8 GB cache and the checkpoints live as private
   Kaggle Datasets, pushed/pulled with the same KGAT_ token used to download
   EyePACS. A background uploader versions the checkpoint dataset every N minutes
   so a Colab session killed at the 12 h / 24 h limit loses ≤ N minutes — the next
   session pulls the latest checkpoint and `--resume` continues.

Two MODES (set in cell 1):
  * MODE="build_cache" — one-time: download EyePACS → run Stages 0–4 + dataset
    stats + PCA → push the bundle as a Kaggle Dataset.
  * MODE="train"       — per fold: pull the cache + last checkpoints → train one
    fold (GPU-bound) → push checkpoints back.

Run:  python experiments/colab/_build_colab_notebook.py
"""
import json
from pathlib import Path


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


cells = []

cells.append(md(
"""# Train Config D (Full V5 + EfficientNet-B3) on Colab — cached + Kaggle-persisted

The real Experiment-1 / Config-D run (Full V5 + EfficientNet-B3, 4-channel, Focal
Loss, 5-fold patient-level CV), engineered to actually finish on Colab.

## Why this version exists
* **Throughput fix.** The V5 pipeline's deterministic Stages 0–4 (OD/fovea, FOV
  crop+resize, flat-field) are the expensive part and carry no train-time
  randomness, so we cache them **once**. Training then runs only Stages 5–7
  (CLAHE + augmentation + normalize) per epoch → the epoch becomes **GPU-bound**
  (~30–40 min on a T4) instead of CPU-starved (no epoch in 12 h before).
* **Kaggle-Dataset persistence.** Google Drive is **not used**. The cache and the
  checkpoints persist as private **Kaggle Datasets**, pushed/pulled with the same
  KGAT_ token used to download EyePACS. This sidesteps Drive's 15 GB free cap.

## Two MODES (cell 1)
| MODE | What it does | When |
|------|--------------|------|
| `"build_cache"` | Download EyePACS → Stages 0–4 cache + dataset-stats + PCA → push as a Kaggle Dataset. | **Once.** ~1–2 h (EyePACS) / minutes (APTOS test). |
| `"train"` | Pull cache + last checkpoint → train one fold → push checkpoints. | **Per fold** (`FOLD=0..4`). |

## 24 h limit / unfinished runs
Free Colab caps ~12 h (no background exec); **Pro+** allows ~24 h continuous in the
background. With the cache, **fold 0 ≈ 12 h** fits one session. If a session dies
mid-fold, the background uploader has already versioned the checkpoint dataset
(every `CKPT_UPLOAD_EVERY_MIN` min) — just rerun MODE="train" with the same `FOLD`:
the cell pulls the latest checkpoint and `--resume` continues. **One fold per
session; the cache is built only once.**

Source repo (cloned below): https://github.com/yesmukhamedov/dr-classifier
"""))

cells.append(md("## 0. Runtime check\n\n`Runtime → Change runtime type → T4 GPU` first."))
cells.append(code(
"""!nvidia-smi
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
"""))

cells.append(md(
"""## 1. Config switches

* `DATASET` — `"eyepacs"` (real run) or `"aptos"` (quick functional test).
* `MODE`    — `"build_cache"` (one-time) then `"train"` (per fold).
* `FOLD`    — change per training session: `0 → 1 → 2 → 3 → 4`.
* `KAGGLE_USERNAME` — **required**, your Kaggle username (the owner of the
  persistence datasets). Find it on your Kaggle profile URL `kaggle.com/<username>`.
* `CKPT_UPLOAD_EVERY_MIN` — background checkpoint-versioning cadence (0 disables)."""))
cells.append(code(
r'''DATASET   = "eyepacs"      # "eyepacs" (real) | "aptos" (quick functional test)
MODE      = "train"        # "build_cache" (once) | "train" (per fold)
FOLD      = 0              # training session: 0,1,2,3,4

KAGGLE_USERNAME = ""       # <-- REQUIRED: your Kaggle username (kaggle.com/<username>)
CKPT_UPLOAD_EVERY_MIN = 30 # background checkpoint upload cadence (0 = disable)

assert KAGGLE_USERNAME, "Set KAGGLE_USERNAME (your Kaggle username) in cell 1."

# Persistence datasets (private, owned by you). Created automatically on first push.
CACHE_SLUG = f"{KAGGLE_USERNAME}/eyepacs-v5-cache-{DATASET}"
CKPT_SLUG  = f"{KAGGLE_USERNAME}/dr-config-d-ckpts-{DATASET}"

# Ephemeral /content working dirs (re-created each session).
from pathlib import Path
DATA        = Path("/content/data")        # raw dataset download (build mode only)
CACHE_LOCAL = Path("/content/v5_cache")    # extracted Stage 0–4 cache (train reads this)
OUT_LOCAL   = Path("/content/outputs")     # exp1 writes checkpoints here
PROC_LOCAL  = Path("/content/data_proc")   # data/processed (stats + PCA) for exp1
for d in (CACHE_LOCAL, OUT_LOCAL, PROC_LOCAL):
    d.mkdir(parents=True, exist_ok=True)

# Kaggle source per dataset (download handled in build mode).
KAGGLE_SOURCE = {
    "aptos":   ("competition", "aptos2019-blindness-detection"),  # needs 1-time rule accept
    "eyepacs": ("dataset",     "dreamer07/eyepacs"),              # no rule accept
}[DATASET]
print(f"DATASET={DATASET} | MODE={MODE} | FOLD={FOLD}")
print(f"CACHE_SLUG={CACHE_SLUG} | CKPT_SLUG={CKPT_SLUG}")
'''))

cells.append(md(
"""## 2. Kaggle credentials

Accepts **both** Kaggle token formats:

* **New (`KGAT_…`)** — Colab Secret `KAGGLE_API_TOKEN`. Written to
  `~/.kaggle/access_token` + `KAGGLE_API_TOKEN` env var.
* **Legacy** — Colab Secret `KAGGLE_USERNAME`+`KAGGLE_KEY`, or a `kaggle.json`
  upload. Written to `~/.kaggle/kaggle.json` (chmod 600).

The same token both **downloads** EyePACS and **pushes/pulls** the cache +
checkpoint datasets (no Drive, no extra auth). Order: new-format Secret → legacy
Secret → upload."""))
cells.append(code(
r'''import os, json
from pathlib import Path

KDIR = Path.home() / ".kaggle"
KDIR.mkdir(exist_ok=True)

def _install_new_token(tok):
    """New Kaggle token (KGAT_...): ~/.kaggle/access_token + KAGGLE_API_TOKEN env."""
    tok = tok.strip()
    (KDIR / "access_token").write_text(tok)
    os.chmod(KDIR / "access_token", 0o600)
    os.environ["KAGGLE_API_TOKEN"] = tok
    print("Kaggle new-format token (KGAT_) installed.")

def _install_legacy(creds):
    """Legacy kaggle.json {'username','key'}: ~/.kaggle/kaggle.json + env."""
    (KDIR / "kaggle.json").write_text(json.dumps(creds))
    os.chmod(KDIR / "kaggle.json", 0o600)
    os.environ["KAGGLE_USERNAME"] = creds["username"]
    os.environ["KAGGLE_KEY"] = creds["key"]
    print("Kaggle legacy creds installed for user:", creds["username"])

done = False
# 1) Colab Secrets — prefer the new KGAT_ token, then legacy username/key.
try:
    from google.colab import userdata
    try:
        tok = userdata.get("KAGGLE_API_TOKEN")
    except Exception:
        tok = None
    if tok:
        _install_new_token(tok); done = True
    else:
        try:
            u, k = userdata.get("KAGGLE_USERNAME"), userdata.get("KAGGLE_KEY")
        except Exception:
            u = k = None
        if u and k:
            _install_legacy({"username": u, "key": k}); done = True
    if done:
        print("Loaded from Colab Secrets.")
except Exception:
    pass

# 2) Fallback: upload. Accepts a kaggle.json OR a text file holding a KGAT_ token.
if not done:
    from google.colab import files
    print("No Colab Secret found. Upload kaggle.json, OR a text file containing your")
    print("new KGAT_ access token (Kaggle -> Settings -> API -> Create New Token):")
    up = files.upload()
    raw = up[next(iter(up))].decode().strip()
    if raw.startswith("KGAT_"):
        _install_new_token(raw)
    else:
        _install_legacy(json.loads(raw))

print("Kaggle auth ready.")
'''))

cells.append(md("## 3. Clone repo + install deps\n\nThe repo must contain the up-to-date `experiments/` — in particular `scripts/precompute_v5_cache.py`, the `paths.v5_cache_dir` wiring in `exp1_factorial.py`, and the EyePACS adapter `is_file()` fix. Mirror the latest `experiments/` into `yesmukhamedov/dr-classifier` before running."))
cells.append(code(
"""!git clone --depth 1 https://github.com/yesmukhamedov/dr-classifier.git /content/repo
%cd /content/repo
!pip install -q -r requirements.txt
!pip install -q timm kaggle    # timm for efficientnet; kaggle CLI for download + dataset push/pull
"""))

cells.append(md(
"""## 4. Kaggle-Dataset persistence helpers

Push/pull helpers used by both modes. A dataset is **created** on first push and
**versioned** thereafter. Large folders are tarred into a single archive first
(35k PNGs → one `v5_cache.tar`) for fast, reliable upload/download."""))
cells.append(code(
r'''import subprocess, json, time, threading
from pathlib import Path

def _run(args, check=True):
    r = subprocess.run(args, capture_output=True, text=True)
    if r.stdout.strip(): print(r.stdout.strip())
    if r.returncode != 0 and r.stderr.strip(): print("[stderr]", r.stderr.strip())
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(args)}")
    return r

def dataset_exists(slug):
    """True if the Kaggle dataset already exists (so we version, not create)."""
    r = subprocess.run(["kaggle", "datasets", "files", slug], capture_output=True, text=True)
    blob = (r.stdout + r.stderr).lower()
    return r.returncode == 0 and "404" not in blob and "not found" not in blob

def _write_meta(folder, slug, title):
    (Path(folder) / "dataset-metadata.json").write_text(json.dumps(
        {"title": title, "id": slug, "licenses": [{"name": "CC0-1.0"}]}))

def push_dataset(folder, slug, title, message="update"):
    """Create (first time) or version (subsequent) a Kaggle dataset from *folder*."""
    _write_meta(folder, slug, title)
    if dataset_exists(slug):
        _run(["kaggle", "datasets", "version", "-p", str(folder), "-m", message, "-d"], check=False)
    else:
        _run(["kaggle", "datasets", "create", "-p", str(folder), "-d"], check=False)

def pull_dataset(slug, dest):
    """Download + unwrap a Kaggle dataset into *dest* (leaves inner .tar files)."""
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    _run(["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"])

def tar_dir(src_dir, tar_path):
    Path(tar_path).parent.mkdir(parents=True, exist_ok=True)
    _run(["tar", "-cf", str(tar_path), "-C", str(src_dir), "."])

def untar(tar_path, dest_dir):
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    _run(["tar", "-xf", str(tar_path), "-C", str(dest_dir)])

print("persistence helpers ready.")
'''))

cells.append(md(
"""## 5. BUILD MODE — download EyePACS, run Stages 0–4 cache, push to Kaggle

Runs **only** when `MODE="build_cache"`. Downloads the raw dataset, adapts the
layout, computes dataset-stats (Stage 7) + PCA (Stage 6 colour), runs
`precompute_v5_cache.py` over all images, bundles everything, and pushes it as the
`CACHE_SLUG` dataset. Skip straight to §6 if you already built the cache."""))
cells.append(code(
r'''import os, sys, subprocess, glob
import pandas as pd
from pathlib import Path

if MODE == "build_cache":
    # ---- 5a. Download raw dataset into /content/data ----
    kind, slug = KAGGLE_SOURCE
    if not any(DATA.iterdir()):
        if kind == "competition":
            subprocess.run(["kaggle", "competitions", "download", "-c", slug, "-p", str(DATA)], check=True)
        else:
            subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(DATA)], check=True)
        for zf in glob.glob(str(DATA / "*.zip")):
            subprocess.run(["unzip", "-q", "-o", zf, "-d", str(DATA)], check=True)
            os.remove(zf)
    print("raw data ready under", DATA)
else:
    print(f'MODE={MODE!r} — skipping build. (Set MODE="build_cache" to (re)build the cache.)')
'''))
cells.append(code(
r'''# ---- 5b. Dataset-layout adapter (EyePACS native OR APTOS 2019) ----
if MODE == "build_cache":
    WORK = Path("/content/eyepacs")   # EyePACS-shaped root: train/ + trainLabels.csv

    def first(gen):
        for x in gen:
            return x
        return None

    labels     = first(p for p in DATA.rglob("trainLabels.csv") if p.is_file())
    train_jpeg = first(p for p in DATA.rglob("*.jpeg") if p.is_file())
    train_dir  = train_jpeg.parent if train_jpeg else None
    SOURCE     = "eyepacs" if (labels is not None and train_dir is not None) else None

    if SOURCE is None:   # APTOS 2019: synthesize the EyePACS layout
        def _is_aptos_csv(p):
            try:
                cols = set(pd.read_csv(p, nrows=0).columns)
            except Exception:
                return False
            return {"id_code", "diagnosis"}.issubset(cols)

        aptos_csv = first(p for p in DATA.rglob("train.csv") if p.is_file() and _is_aptos_csv(p))
        if aptos_csv is not None:
            df  = pd.read_csv(aptos_csv)
            png = first(p for p in aptos_csv.parent.rglob("*.png") if p.is_file()) \
                  or first(p for p in DATA.rglob("*.png") if p.is_file())
            assert png is not None, "APTOS train.csv found but no .png images."
            src_dir = png.parent
            (WORK / "train").mkdir(parents=True, exist_ok=True)
            n = 0
            for code_id in df["id_code"].astype(str):
                src = src_dir / f"{code_id}.png"
                lnk = WORK / "train" / f"{code_id}.jpeg"
                if src.exists() and not lnk.exists():
                    os.symlink(src, lnk); n += 1
            (df.rename(columns={"id_code": "image", "diagnosis": "level"})[["image", "level"]]
               .to_csv(WORK / "trainLabels.csv", index=False))
            labels, train_dir, SOURCE = WORK / "trainLabels.csv", WORK / "train", "aptos2019"
            print(f"APTOS detected: linked {n} png->jpeg")

    assert labels is not None and train_dir is not None, "No usable dataset under /content/data."
    WORK.mkdir(parents=True, exist_ok=True)
    if not (WORK / "train").exists():        os.symlink(train_dir, WORK / "train")
    if not (WORK / "trainLabels.csv").exists(): os.symlink(labels, WORK / "trainLabels.csv")
    EYEPACS_ROOT = str(WORK)
    print("source =", SOURCE, "| EYEPACS_ROOT =", EYEPACS_ROOT,
          "| images:", sum(1 for _ in (WORK / "train").glob("*.jpeg")))
'''))
cells.append(code(
r'''# ---- 5c. dataset-stats (Stage 7) + PCA (Stage 6) + Stage 0–4 cache, then push ----
if MODE == "build_cache":
    PROC_LOCAL.mkdir(parents=True, exist_ok=True)

    # Stage 7 dataset-specific normalize stats (REQUIRED for Config D).
    subprocess.run([sys.executable, "scripts/compute_dataset_stats.py",
        "--images-root", str(Path(EYEPACS_ROOT) / "train"),
        "--labels-csv",  str(Path(EYEPACS_ROOT) / "trainLabels.csv"),
        "--output-dir",  str(PROC_LOCAL), "--n-samples", "5000", "--seed", "42"], check=True)

    # PCA eigenvectors for colour augmentation (optional; best-effort).
    try:
        subprocess.run([sys.executable, "scripts/compute_pca_eigvecs.py",
            "--dataset", "eyepacs",
            "--images-root", str(Path(EYEPACS_ROOT) / "train"),
            "--labels-csv",  str(Path(EYEPACS_ROOT) / "trainLabels.csv"),
            "--output-dir",  str(PROC_LOCAL), "--n-samples", "5000", "--seed", "42"], check=True)
    except Exception as e:
        print("PCA step skipped (colour aug will be disabled):", e)

    # The heavy one-shot: V5 Stages 0–4 cache for ALL images (multi-process).
    import os as _os
    subprocess.run([sys.executable, "scripts/precompute_v5_cache.py",
        "--images-root", str(Path(EYEPACS_ROOT) / "train"),
        "--labels-csv",  str(Path(EYEPACS_ROOT) / "trainLabels.csv"),
        "--output-dir",  str(CACHE_LOCAL),
        "--num-workers", str(max(2, (_os.cpu_count() or 2))),
        "--preset", "efficientnet"], check=True)

    # Bundle stats + PCA INTO the cache dir so they travel with the cache.
    import shutil
    for f in PROC_LOCAL.glob("*"):
        if f.is_file():
            shutil.copy(f, CACHE_LOCAL / f.name)

    # Tar the whole cache dir into one archive and push as a Kaggle Dataset.
    push_dir = Path("/content/cache_push"); push_dir.mkdir(exist_ok=True)
    tar_dir(CACHE_LOCAL, push_dir / "v5_cache.tar")
    push_dataset(push_dir, CACHE_SLUG, f"EyePACS V5 Stage0-4 cache ({DATASET})", "build")
    print("\\nCache pushed ->", CACHE_SLUG, "\\nNow set MODE='train', FOLD=0 and Run all.")
'''))

cells.append(md(
"""## 6. TRAIN MODE — pull cache + last checkpoint, train one fold, push checkpoints

Runs **only** when `MODE="train"`. Pulls the Stage 0–4 cache (fast, ~8 GB), restores
the latest checkpoint for `--resume`, trains `FOLD`, and versions the checkpoint
dataset both periodically (background) and at the end."""))
cells.append(code(
r'''import subprocess, sys, shutil
from pathlib import Path

if MODE == "train":
    # ---- 6a. Pull the Stage 0–4 cache and extract ----
    if not any(CACHE_LOCAL.glob("*.png")):
        dl = Path("/content/cache_dl")
        pull_dataset(CACHE_SLUG, dl)
        tar = next(iter(dl.glob("*.tar")), None)
        assert tar is not None, f"No .tar in {CACHE_SLUG} download — was build mode run?"
        untar(tar, CACHE_LOCAL)
    n_png = sum(1 for _ in CACHE_LOCAL.glob("*.png"))
    print("cache images:", n_png)
    assert n_png > 0, "Cache empty — run MODE='build_cache' first."

    # ---- 6b. Restore dataset-stats + PCA where exp1 looks (output_dir.parent/data/processed) ----
    proc = OUT_LOCAL.parent / "data" / "processed"   # exp1: Path(output_dir).parent/data/processed
    proc.mkdir(parents=True, exist_ok=True)
    for name in ("eyepacs_norm_stats.json", "eyepacs_pca_eigvecs.npy", "eyepacs_pca_eigvals.npy"):
        src = CACHE_LOCAL / name
        if src.exists():
            shutil.copy(src, proc / name)
    print("restored stats/PCA ->", proc, "|", [p.name for p in proc.glob('*')])

    # ---- 6c. Restore the latest checkpoint for --resume ----
    if dataset_exists(CKPT_SLUG):
        dl = Path("/content/ckpt_dl")
        pull_dataset(CKPT_SLUG, dl)
        tar = next(iter(dl.glob("*.tar")), None)
        if tar is not None:
            untar(tar, OUT_LOCAL)
            print("restored checkpoints ->", OUT_LOCAL)
    else:
        print("no checkpoint dataset yet (fresh start).")
'''))
cells.append(code(
r'''# ---- 6d. Build merged config (points training at the cache) ----
if MODE == "train":
    merge_cmd = [
        sys.executable, "kaggle/merge_config.py",
        "--base", "configs/default.yaml",
        "--out",  "configs/colab_merged.yaml",
        f"paths.eyepacs={CACHE_LOCAL}",        # unused for indexing in cache mode, but required key
        f"paths.v5_cache_dir={CACHE_LOCAL}",   # <-- the throughput fix: read cached Stages 0–4
        f"paths.output_dir={OUT_LOCAL}",
        "training.num_workers=2",
        # If T4 OOMs at 512px (EfficientNet runs fp32), uncomment:
        # "training.batch_size=8",
    ]
    subprocess.run(merge_cmd, check=True)
    print("merged config -> configs/colab_merged.yaml")
'''))
cells.append(code(
r'''# ---- 6e. Train one fold, with a background checkpoint uploader (24 h-limit safety) ----
if MODE == "train":
    stop_evt = threading.Event()

    def push_ckpts(message):
        if not (OUT_LOCAL / "exp1").exists():
            return
        pd_ = Path("/content/ckpt_push"); pd_.mkdir(exist_ok=True)
        tar_dir(OUT_LOCAL, pd_ / "outputs.tar")
        push_dataset(pd_, CKPT_SLUG, f"DR Config-D checkpoints ({DATASET})", message)

    def _uploader(every_min):
        # Periodic snapshot so a session killed at the 12 h/24 h cap loses <= every_min.
        while not stop_evt.wait(every_min * 60):
            try:
                push_ckpts(f"auto {time.strftime('%Y-%m-%d %H:%M')} fold{FOLD}")
            except Exception as e:
                print("[uploader] error:", e)

    if CKPT_UPLOAD_EVERY_MIN > 0:
        threading.Thread(target=_uploader, args=(CKPT_UPLOAD_EVERY_MIN,), daemon=True).start()
        print(f"background checkpoint uploader every {CKPT_UPLOAD_EVERY_MIN} min -> {CKPT_SLUG}")

    try:
        subprocess.run([sys.executable, "run_experiment.py", "exp1",
            "--config", "configs/colab_merged.yaml",
            "--configs", "D", "--fold", str(FOLD), "--resume"], check=True)
    finally:
        stop_evt.set()
        push_ckpts(f"final {time.strftime('%Y-%m-%d %H:%M')} fold{FOLD}")
        print("final checkpoints pushed ->", CKPT_SLUG)
'''))

cells.append(md("## 7. Inspect outputs"))
cells.append(code(
r'''ckpt_dir = OUT_LOCAL / "exp1" / "checkpoints" / f"D_fold{FOLD}"
print("checkpoint dir:", ckpt_dir)
!ls -la "{ckpt_dir}" 2>/dev/null || echo "(none yet)"
print("\n--- metrics.csv (tail) ---")
!tail -n 15 "{OUT_LOCAL}/exp1/metrics.csv" 2>/dev/null || echo "(no metrics yet)"
'''))

cells.append(md(
"""## 8. Multi-fold strategy & retrieval

* **Build once:** `MODE="build_cache"`, `DATASET="eyepacs"` → pushes `CACHE_SLUG`.
  (For a quick end-to-end smoke test first, do the same with `DATASET="aptos"`.)
* **Train per fold:** `MODE="train"`, `FOLD=0..4`, Run all each time. The cache and
  the last checkpoint are pulled automatically; `--resume` continues a partial fold.
* **fold 0 alone unblocks the demo.** Folds 1–4 only feed `verify_exp1.py`'s
  dominance check (which also needs a Config C run — out of scope here).
* **24 h / disconnect safety:** the background uploader versions `CKPT_SLUG` every
  `CKPT_UPLOAD_EVERY_MIN` min, and a final push runs in a `finally`. A killed
  session loses ≤ that cadence; the next `MODE="train"` session resumes from it.
* **Retrieve for the demo:** download `CKPT_SLUG`, extract `outputs.tar`, copy
  `exp1/checkpoints/D_fold0/best_model.pt` **plus** `eyepacs_norm_stats.json` (from
  `CACHE_SLUG`) into the FastAPI server together (both, to avoid preprocessing drift).

**Notes**
* EfficientNet runs **fp32** (mixed precision disabled — fp16 overflow). If the T4
  OOMs at 512px, uncomment `training.batch_size=8` in cell 6d.
* The cache is **preset-specific for Stages 0–4** (`efficientnet`). Configs B and D
  share Stages 0–4, so the same cache serves both; baseline configs (A/C) must run
  **without** `v5_cache_dir` (they use a different preprocessing path)."""))

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "train_config_d_colab.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out)
