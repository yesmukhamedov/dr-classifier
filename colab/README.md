# Google Colab — Train Config D (Full V5 + EfficientNet-B3), cached + Kaggle-persisted

The Colab runner for the real Experiment-1 / Config-D run (Full V5 +
EfficientNet-B3, 4-channel, Focal Loss, 5-fold patient-level CV), engineered so it
actually finishes on Colab.

Two engineering decisions make it viable (see `TASK.md` §2):

1. **V5 Stage 0–4 cache (throughput fix).** The deterministic Stages 0–4 (OD/fovea,
   FOV crop+resize, flat-field) are the expensive part and carry no train-time
   randomness, so they are computed **once** and cached as 4-channel PNGs. Training
   then runs only the stochastic Stages 5–7 per epoch → the epoch becomes
   **GPU-bound** (~30–40 min on a T4) instead of CPU-starved (no epoch in 12 h before).
2. **Kaggle-Dataset persistence (no Google Drive).** The ~8 GB cache and the
   checkpoints persist as **private Kaggle Datasets**, pushed/pulled with the same
   `KGAT_` token used to download EyePACS — sidestepping Drive's 15 GB free cap and
   OAuth entirely.

The notebook clones the standalone repo
**https://github.com/yesmukhamedov/dr-classifier** and imports nothing from this
folder at runtime — it is self-contained.

## Files

| File | Purpose |
|---|---|
| `train_config_d_colab.ipynb` | Main notebook (two MODES: `build_cache` once, `train` per fold). |
| `_build_colab_notebook.py` | One-shot generator for the `.ipynb`. Re-run after edits. |

## Two MODES (cell 1)

| MODE | What it does | When |
|---|---|---|
| `"build_cache"` | Download dataset → V5 Stages 0–4 cache + dataset-stats + PCA → push as the Kaggle dataset `<user>/eyepacs-v5-cache-<dataset>`. | **Once.** ~1–2 h (EyePACS) / minutes (APTOS test). |
| `"train"` | Pull the cache + last checkpoint → train one fold (GPU-bound) → push checkpoints to `<user>/dr-config-d-ckpts-<dataset>`. | **Per fold** (`FOLD=0..4`). |

`DATASET` switches the source: `"eyepacs"` (real, `dreamer07/eyepacs`, ~35k) or
`"aptos"` (quick functional test, competition `aptos2019-blindness-detection`, ~3.7k).
The layout adapter converts APTOS into the EyePACS layout; everything else is identical.

## What you must provide (one-time, minimal)

1. **Kaggle API token** — for the dataset download **and** the cache/checkpoint
   push/pull. Two formats accepted:
   * **New (recommended, `KGAT_…`):** add a Colab Secret named `KAGGLE_API_TOKEN`
     (🔑 left sidebar) whose value is the `KGAT_…` token from Kaggle → *Settings* →
     *API* → *Create New Token*.
   * **Legacy:** add `KAGGLE_USERNAME` + `KAGGLE_KEY` to Colab Secrets, or upload
     `kaggle.json` when cell 2 prompts.
2. **Your Kaggle username** — set `KAGGLE_USERNAME` in cell 1 (owner of the
   persistence datasets; it's the `kaggle.com/<username>` on your profile).
3. **APTOS only:** accept the competition rules once at
   <https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules>.
   (`dreamer07/eyepacs` needs no rule acceptance.)

No Google Drive, no OAuth. Nothing secret is handed to the repo or to Claude —
credentials live only in your Colab session.

## How to run

1. Open the notebook in Colab. `Runtime → Change runtime type → T4 GPU`.
2. **Cell 1:** set `KAGGLE_USERNAME`, `DATASET="eyepacs"`.
3. **Build once:** `MODE="build_cache"` → *Run all*. Pushes `<user>/eyepacs-v5-cache-eyepacs`.
4. **Train per fold:** `MODE="train"`, `FOLD=0` → *Run all*. Repeat with `FOLD=1..4`.
   * **fold 0 alone unblocks the demo.** Folds 1–4 only feed `verify_exp1.py`'s
     dominance check (which also needs a Config C run — out of scope here).

> Quick smoke test first: do steps 3–4 with `DATASET="aptos"` — a full build+train
> pass in minutes proves the pipeline before committing to the ~35 GB EyePACS build.

## Persistence & the 24 h limit

* **Cache** persists as a Kaggle Dataset; pulled (~8 GB tar, a few min) each training
  session instead of re-downloading 35 GB EyePACS.
* **Checkpoints** persist as a Kaggle Dataset, **versioned** every
  `CKPT_UPLOAD_EVERY_MIN` (default 30) by a background thread + a final push in a
  `finally`. A session killed at the 12 h / 24 h cap loses ≤ that cadence; the next
  `MODE="train"` session pulls the latest and `--resume` continues.
* Free Colab ≈ 12 h, no background execution; **Pro+** ≈ 24 h continuous in the
  background. With the cache, **fold 0 ≈ 12 h fits one session**. Pro+ removes the
  babysitting but is **not required** — free works with the one-fold-per-session loop.

## Retrieve for the demo

Download `<user>/dr-config-d-ckpts-eyepacs`, extract `outputs.tar`, and copy
`exp1/checkpoints/D_fold0/best_model.pt` **together with** `eyepacs_norm_stats.json`
(from the cache dataset) into the FastAPI server — both, to avoid preprocessing drift.
Only this EyePACS checkpoint is thesis-faithful.

## Notes & caveats

* EfficientNet runs **fp32** (mixed precision disabled — fp16 overflow). If the T4
  OOMs at 512px, uncomment `training.batch_size=8` in cell 6d.
* The cache is **preset-specific for Stages 0–4** (`efficientnet`). Configs B and D
  share Stages 0–4, so one cache serves both; baseline configs (A/C) must run
  **without** `v5_cache_dir`.
* The cached path is **bit-identical** to the live pipeline (verified max|Δ| = 0.0,
  inference + seeded training) — the same pipeline object finishes both paths via
  `finish_from_cache`, so there is no train/inference drift.

## ⚠ Prerequisite: the cloned repo must be current

The notebook clones `yesmukhamedov/dr-classifier`. That repo must contain this
branch's `experiments/` — in particular `scripts/precompute_v5_cache.py`, the
`precompute_deterministic`/`finish_from_cache` split in
`src/preprocessing/pipeline_v5.py`, `CachedEyePACSDataset` in `src/data/datasets.py`,
and the `paths.v5_cache_dir` wiring in `src/experiments/exp1_factorial.py`. If it is
stale, mirror the latest `experiments/` tree into it first.
