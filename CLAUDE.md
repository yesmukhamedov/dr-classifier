# experiments/ — DR-Classifier

Automated Diabetic Retinopathy classification via fundus image preprocessing + CNN.
PhD dissertation project (Yesmukhamedov N.S.).

## Stack

- Python 3.10, PyTorch 2.5, timm (EfficientNet), OpenCV, scikit-learn
- Conda env: `conda activate dr-classifier`
- GPU: NVIDIA RTX 3060 12GB VRAM
- OS: WSL2 Ubuntu on Windows

## Governance

All governance documents (INVARIANTS, HYPOTHESIS, ARGUMENT_MAP, etc.) live in `../thesis/governance/` — that is the single source of truth. Do not create or maintain governance copies here. The only local doc is `docs/experimental_protocol.md` (quick-start guide).

## Project Structure

```
configs/default.yaml          — all hyperparameters and experiment specs
run_experiment.py             — CLI entry point
```

### Preprocessing

```
src/preprocessing/pipeline_v5.py        — V5 8-stage pipeline orchestrator (canonical)
src/preprocessing/config.py             — PreprocessingV5Config dataclass + presets
src/preprocessing/flat_field.py         — Stage 4: adaptive Gaussian flat-field correction
src/preprocessing/upgraded_clahe.py     — Stage 5: dual-constraint CLAHE (L-channel)
src/preprocessing/imagenet_normalize.py — Stage 7: dataset-specific mean/std → tensor
```

### Data

```
src/data/datasets.py            — dataset loaders (EyePACS, IDRiD, APTOS, Messidor-2, DDR, ODIR, RFMiD, Clinical)
src/data/clinical_dataset.py    — ClinicalDataset stub
src/data/messidor2_dataset.py   — Messidor2Dataset stub
src/data/splits.py              — PatientLevelKFold (patient-level stratified k-fold CV)
src/data/augmentation_v4.py     — V5 augmentation (unified affine + PCA color)
src/data/label_harmonization.py — taxonomy mapping for Messidor-2, RFMiD, ODIR
```

### Models

```
src/models/factory.py        — create_model()
src/models/resnet.py         — ResNet-50 with configurable in_channels and 5-class head
src/models/efficientnet.py   — EfficientNet-B0/B3/B4 via timm
src/models/two_stage.py      — two-stage fine-tuning protocol utility
```

### Training

```
src/training/trainer.py         — main training loop (mixed precision, early stopping, k-fold CV)
src/training/losses.py          — FocalLoss (γ=2) + inverse-frequency class weights
src/training/checkpoint.py      — CheckpointManager (save/load/keep-best)
src/training/train.py           — standalone training script
```

### Evaluation

```
src/evaluation/metrics.py           — primary (F1, AUC, κ, acc) + secondary + clinical + dominance check
src/evaluation/calibration.py       — ECE, Brier score
src/evaluation/statistical_tests.py — McNemar, DeLong, bootstrap CI
```

### Explainability

```
src/explainability/gradcam.py       — Grad-CAM activation maps
src/explainability/iou.py           — IoU/ALO vs IDRiD lesion masks
src/explainability/visualization.py — Grad-CAM overlay rendering
```

### Experiments

```
src/experiments/exp1_factorial.py         — Exp 1: 2×2 factorial (H-1)
src/experiments/exp2_ablation.py          — Exp 2: V5 stage ablation + CLAHE sweep (H-2)
src/experiments/exp3_transferability.py   — Exp 3: APTOS transferability (H-4)
src/experiments/exp4_explainability.py    — Exp 4: Grad-CAM on IDRiD + Clinical (H-5)
src/experiments/exp5_clinical_degradation.py — Exp 5: clinical degradation resistance (H-7)
src/experiments/exp6_device_shift.py      — Exp 6: device domain shift (H-6)
src/experiments/exp7_clinical.py          — Exp 7: small data training IDRiD → Clinical
src/experiments/_eval_utils.py            — shared inference helpers
```

### Utilities

```
src/utils/config.py        — YAML config loading and merging
src/utils/seed.py           — reproducibility (fixed seeds, deterministic mode)
src/utils/image_quality.py  — CNR, entropy, SSIM quality metrics
```

## V5 Preprocessing Pipeline (8 stages, all always-on except Stage 6)

Stage 0: Canonical flip (left→right eye orientation) — always on
Stage 1: OD-fovea rotation normalization — always on
Stage 2: FOV crop + isotropic resize to 512×512 + zero-padding — always on
Stage 3: FOV mask generation (binary 1.0/0.0 → 4th channel) — always on
Stage 4: Flat-field correction (adaptive Gaussian σ=0.07·D) — always on
Stage 5: CLAHE (dual-constraint, LAB L-channel, stochastic p=0.8 at train) — always on
Stage 6: Augmentation (unified affine + brightness/contrast + PCA color) — train only
Stage 7: Dataset-specific normalize → tensor (mean/std from training set) — always last

Baseline (Configs A/C) = stretch-resize 512×512 + ImageNet normalize (3 channels).
Full V5 (Configs B/D) = all 8 stages (4 channels: RGB + FOV mask).

## Experiment 1: 2×2 Factorial

Dataset: EyePACS 100% (~35,126 images). 4 configurations:
- A: baseline (3ch) + ResNet-50
- B: full V5 pipeline (4ch) + ResNet-50
- C: baseline (3ch) + EfficientNet-B3
- D: full V5 pipeline (4ch) + EfficientNet-B3

Dominance criterion (EH-3) — ALL THREE must hold:
- ΔF1 ≥ 5 percentage points
- ΔAUC ≥ 0.02
- No Cohen κ degradation

## Cross-Validation

5-fold patient-level stratified split. No patient images in both train and val within any fold.

## Hardware Constraints

```yaml
batch_size: 16
image_size: 512×512
input_channels: 4           # RGB + FOV mask (baseline: 3)
mixed_precision: true        # DISABLED for EfficientNet (fp16 overflow)
loss: Focal Loss (γ=2, α=inverse-frequency)
optimizer: Adam (lr=1e-4, weight_decay=1e-4)
early_stopping: patience=5, monitor=val_weighted_f1
scheduler: ReduceOnPlateau (factor=0.5, patience=3, min_lr=1e-6)
max_epochs: 20
seed: 42, deterministic=true
num_workers: 4
```

## Primary Metrics (descending importance)

1. Weighted F1-score
2. ROC-AUC (one-vs-rest, macro)
3. Cohen Kappa (quadratic weights)
4. Accuracy

## Commands

```bash
conda activate dr-classifier
python run_experiment.py exp1 --config configs/default.yaml
python run_experiment.py exp1 --config configs/default.yaml --configs A,B
python run_experiment.py exp2 --config configs/default.yaml
```

## Code Rules

- Type hints on all function signatures
- Docstrings with Args/Returns for every public function
- No hardcoded paths — all paths from config YAML
- Every module independently testable
- Use pathlib.Path, not os.path
- All internal image processing in RGB (cv2.imread returns BGR — convert on entry)

## Known Issues and Fixes

- EfficientNet-B3 fp16 overflow → mixed precision DISABLED for EfficientNet configs
- Gradient explosion in early training → gradient clipping added
