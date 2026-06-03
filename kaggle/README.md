# Kaggle — Train Config D (Full V5 + EfficientNet-B3)

Trains Experiment 1 / Config D on EyePACS, one fold per Kaggle session, working
around the 12 h limit. Mirrors `experiments/` and is consumed from the
standalone repo **https://github.com/yesmukhamedov/dr-classifier** (the notebook
clones that repo, code lives at its root).

## Files

| File | Purpose |
|---|---|
| `train_config_d.ipynb` | Main notebook (clone → adapt data → merge config → train one fold). |
| `kaggle_paths.yaml` | Override layer over `configs/default.yaml` (paths, num_workers, batch_size). |
| `merge_config.py` | Deep-merges `default.yaml ← kaggle_paths.yaml` (+ dotted overrides) → `configs/kaggle_merged.yaml`. The CLIs accept only one `--config`, so we pre-merge instead of patching them. |
| `_build_notebook.py` | One-shot generator for the `.ipynb` (kept for reproducibility). |

## How to run

1. Kaggle → New Notebook → **Add Input**:
   - The competition **`diabetic-retinopathy-detection`** (2015 EyePACS DR
     detection), **or** any dataset that already contains EyePACS
     `trainLabels.csv` + `*.jpeg` train images.
   - For folds 1–4: also attach the **previous fold's saved output** so
     `--resume` picks up the checkpoints.
2. Copy `train_config_d.ipynb` in (or open it from the cloned repo).
3. Set `FOLD` in the train cell, then **Save Version → Save & Run All (Commit)**.
4. Repeat for `FOLD = 0,1,2,3,4`.

## Account handles (already wired)

- GitHub source: `yesmukhamedov/dr-classifier`
- Kaggle: `yesmukhamedov`

## Two things to know before you trust the run

### 1. Stage 7 uses dataset-specific normalize (thesis-faithful)
The thesis (`thesis/methods/preprocessing-pipeline.md`, Stage 7) requires the
full configs (B/D) to normalize with mean/std **computed from EyePACS train after
Stages 0–4, mask=1.0 pixels only** (design decision D-2) — ImageNet is the
baseline-only fallback. This is now wired in:
- `scripts/compute_dataset_stats.py` computes the stats (Stages 0–4, no CLAHE,
  masked) and writes `data/processed/eyepacs_norm_stats.json`.
- `src/experiments/exp1_factorial.py` loads that file automatically and injects
  the stats into Config D's pipeline (the preset path otherwise leaves them
  `None`). Notebook **Cell 6** runs the computation before training.

If `eyepacs_norm_stats.json` is missing, exp1 prints a warning and falls back to
ImageNet (so a run still completes), but that is **not** thesis-faithful — don't
ship a Config D checkpoint trained that way. Default `--n-samples 5000` is a
stable estimate; use `0` for the full train set (slower).

### 2. Disk / extraction
The raw competition ships train images as multi-part zips
(`train.zip.001…005`). Decompressing all ~35k images can exceed the writable
`/kaggle/working` quota. The notebook's adapter:
- first tries to **symlink** an already-extracted layout from any attached input
  (zero extra disk) — preferred;
- only falls back to concatenating + unzipping the parts if nothing usable is found.

**Recommendation:** attach a dataset that already has the train `*.jpeg` + labels
extracted, so the adapter just symlinks. PCA (Cell 5) is optional and can be
skipped to save time.

## Outputs

- Persist automatically under `/kaggle/working/outputs` (survives *Save Version*).
- Per-epoch metrics: `/kaggle/working/outputs/exp1/metrics.csv`.
- Best checkpoint per fold: `outputs/exp1/checkpoints/D_fold{0..4}/best_model.pt`
  (monitored on `weighted_f1`). Note: `best_model.pt`, under `D_fold{N}/` — not
  `fold_N/best.pt`.

When all folds finish, download `outputs/exp1/` → place at
`experiments/outputs/exp1/` → run `python scripts/verify_exp1.py`.

## Knobs

- **OOM at 512px:** EfficientNet runs fp32 (mixed precision must stay disabled —
  fp16 overflow). If the T4 OOMs, drop batch size: uncomment
  `training.batch_size=8` in Cell 4.
- **Skip PCA:** comment out Cell 5; colour augmentation is then disabled.
