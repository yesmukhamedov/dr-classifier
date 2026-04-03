# dr-classifier

Research codebase for the PhD dissertation **"Automated Diabetic Retinopathy Diagnosis via Fundus Image Enhancement and CNN Classification"** (Yesmukhamedov N.S., IITU, Almaty).

Classifies fundus photographs into five DR severity grades (0–4) per the International Clinical DR Disease Severity Scale. The core thesis: an integrated 6-stage preprocessing pipeline is the primary driver of classification improvement, not architectural complexity.

## Pipeline Architecture

The V4 preprocessing pipeline processes each fundus image through six stages:

| Stage | Operation | Always On | Description |
|-------|-----------|-----------|-------------|
| 0 | Canonical orientation | Toggleable | Left→right eye flip + OD–fovea rotation normalization |
| 1 | FOV crop + resize | Yes | PIL-based FOV detection, isotropic resize to 512×512, zero-padded with binary mask |
| 2 | Flat-field correction | Toggleable | Gaussian blur subtraction (σ=45) to normalize illumination gradients |
| 3 | Dual-constraint CLAHE | Toggleable | LAB L-channel, stochastic at train time (p=0.8), clip_factor × tile_area/256 capped by global_threshold |
| 4 | ImageNet normalize | Yes | Channel-wise mean/std normalization → float32 tensor |
| 5 | Augmentation | Train only | Unified affine (truncated Gaussian rotation σ=13°, zoom, shear, stretch) + brightness/contrast + PCA colour jitter |

All configurations output a 4-channel tensor `(4, 512, 512)`: RGB + binary FOV mask. The mask channel informs the CNN where real image data exists vs. zero-padding from isotropic resize. The first Conv2d layer of both architectures is replaced with a 4-channel variant (pretrained RGB weights copied, channel 4 initialized with RGB mean).

## Experiments

Six experiments map directly to six hypotheses (H-1 through H-6):

| Experiment | Hypothesis | Design | Dataset |
|------------|-----------|--------|---------|
| **Exp 1** — Factorial Ablation | H-1: Preprocessing Dominance | 2×3 factorial: {ResNet-50, EfficientNet-B3} × {baseline, full V4, V4+PatientHead}, configs A–F | EyePACS (~14k images) |
| **Exp 2** — Component Ablation | H-2: CLAHE Sensitivity | CLAHE clip-limit sweep + pipeline component ablation | EyePACS + IDRiD |
| **Exp 3** — Robustness | — | Synthetic degradation (noise, blur, low illumination) | APTOS 2019 (DROPPED) |
| **Exp 4** — Explainability | H-5: Grad-CAM ALO | Attention–Lesion Overlap with pixel-level lesion masks | IDRiD |
| **Exp 5** — Generalization | H-4: Cross-dataset transfer | Train on EyePACS → evaluate on external sets without retraining | Messidor, Messidor-2, IDRiD |
| **Exp 6** — Device Shift | H-6: Camera robustness | Cross-device evaluation: Canon → Topcon, Kowa, Zeiss | RFMiD, DDR, ODIR-5K |

Dominance criterion (EH-3): Δ weighted-F1 ≥ 5 pp **AND** Δ ROC-AUC ≥ 0.02 **AND** no Cohen's κ degradation, validated independently for both architectures.

## Project Structure

```
dr-classifier/
├── run_experiment.py            # CLI entry point for all experiments
├── configs/
│   ├── default.yaml             # Master config (pipeline, training, evaluation)
│   └── smoke_test_1pct.yaml     # 1% subset for quick validation
├── src/
│   ├── preprocessing/           # V4 pipeline (14 modules)
│   │   ├── pipeline_v4.py       #   Pipeline orchestrator
│   │   ├── canonical_orientation.py
│   │   ├── crop_resize.py       #   FOV crop + isotropic resize + mask
│   │   ├── flat_field.py        #   Gaussian blur subtraction
│   │   ├── upgraded_clahe.py    #   Dual-constraint stochastic CLAHE
│   │   ├── od_fovea_detect.py   #   OD/fovea detection for Stage 0b
│   │   ├── imagenet_normalize.py
│   │   └── config.py            #   PreprocessingV4Config dataclass
│   ├── data/                    # Dataset classes & augmentation
│   │   ├── datasets.py          #   EyePACS, IDRiD, Messidor, RFMiD, DDR, ODIR
│   │   ├── augmentation_v4.py   #   Unified affine + PCA colour jitter
│   │   ├── splits.py            #   Patient-level stratified k-fold
│   │   └── label_harmonization.py
│   ├── models/                  # CNN architectures
│   │   ├── factory.py           #   Model factory (ResNet-50, EfficientNet-B0/B3/B4)
│   │   ├── patient_model.py     #   Binocular fusion: Φ(fL,fR) = [fL ‖ fR ‖ |fL−fR|]
│   │   ├── resnet.py
│   │   └── efficientnet.py
│   ├── training/                # Training infrastructure
│   │   ├── trainer.py           #   AMP, early stopping, 3-fold CV, metrics CSV
│   │   ├── patient_trainer.py   #   Per-patient binocular training loop
│   │   ├── losses.py            #   FocalLoss(γ=2, α=inv-freq) + weighted CE
│   │   └── checkpoint.py        #   Best/last checkpoint management
│   ├── evaluation/              # Metrics & statistical testing
│   │   ├── metrics.py           #   Primary (F1, AUC, κ, acc), secondary, clinical
│   │   ├── statistical_tests.py #   McNemar, DeLong, bootstrap CI, Holm-Bonferroni
│   │   └── calibration.py
│   ├── experiments/             # Experiment runners (exp1–exp6)
│   │   ├── exp1_factorial.py
│   │   ├── exp2_ablation.py
│   │   ├── exp3_robustness.py
│   │   ├── exp4_explainability.py
│   │   ├── exp5_generalization.py
│   │   └── exp6_device_shift.py
│   ├── explainability/          # Grad-CAM + ALO/IoU
│   │   ├── gradcam.py
│   │   ├── iou.py
│   │   └── visualization.py
│   ├── degradation/             # Synthetic image perturbations (Exp 3)
│   └── utils/                   # Config loader, seed, image quality
├── docs/                        # Governance system
│   ├── INVARIANTS.md            #   Binding constraint system (v4.0)
│   ├── ARGUMENT_MAP.md          #   Claim-evidence-dependency structure
│   ├── RESEARCH_ARCHITECTURE.md #   Methodological blueprint
│   ├── HYPOTHESIS.md            #   H-1 through H-6
│   └── VERSION_SYNC.md
├── scripts/                     # Utilities & verification
│   ├── generate_report.py       #   Auto-generate dissertation-ready report
│   ├── compute_pca_eigvecs.py   #   PCA eigenvectors for colour augmentation
│   ├── verify_*.py              #   Integration tests (datasets, models, experiments)
│   └── smoke_test_v4.py
├── tests/                       # Unit tests
├── logs/                        # Training logs
├── environment.yml              # Conda environment (Python 3.10, PyTorch 2.5.1, CUDA 12.1)
└── requirements.txt             # Pip dependencies
```

**Codebase:** 56 Python modules, ~10,000 lines of source code.

## Quickstart

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.1 (tested on RTX 3060 12GB)
- Datasets placed according to paths in `configs/default.yaml`

### Installation

```bash
# Option A: Conda (recommended)
conda env create -f environment.yml
conda activate dr-classifier

# Option B: Pip
pip install -r requirements.txt
pip install timm>=0.9.12 pytorch-grad-cam>=1.4.8 scipy>=1.11.0 statsmodels>=0.14.0
```

### Running Experiments

```bash
# Full Experiment 1 (all 6 configs, 3-fold CV)
python run_experiment.py exp1

# Single config
python run_experiment.py exp1 --configs D

# Single fold
python run_experiment.py exp1 --configs D --fold 0

# Resume from checkpoint
python run_experiment.py exp1 --configs D --resume

# Smoke test (1% subset)
python run_experiment.py exp1 --config configs/smoke_test_1pct.yaml

# Other experiments
python run_experiment.py exp2
python run_experiment.py exp4
python run_experiment.py exp5
python run_experiment.py exp6
```

### Generate Report

```bash
python scripts/generate_report.py
# → outputs/final_report.md (maps results to H-1..H-6 and PC-1..PC-9)
```

### Verify Installation

```bash
python scripts/verify_datasets.py     # Check dataset paths and label distributions
python scripts/verify_models.py       # Verify model creation and forward pass
python scripts/smoke_test_v4.py       # End-to-end pipeline smoke test
```

## Datasets

| Dataset | Role | Size | Camera | Tier |
|---------|------|------|--------|------|
| **EyePACS** | Primary training (Exp 1, 2) | ~35,126 (40% subset: ~14,050) | Canon CR-1 | Training |
| **IDRiD** | Lesion localization (Exp 4), generalization (Exp 5) | ~516 (81 with pixel masks) | Kowa | Clinical |
| **Messidor / Messidor-2** | External generalization (Exp 5) | ~1,200 / ~1,748 | Topcon | External |
| **RFMiD** | Device domain shift (Exp 6) | ~3,200 | Multiple | Device |
| **DDR** | Device domain shift (Exp 6) | ~12,522 | Canon, Topcon | Device |
| **ODIR-5K** | Device domain shift (Exp 6) | ~5,000 | Canon, Zeiss | Device |
| **APTOS 2019** | Robustness (Exp 3 — DROPPED) | ~3,662 | Mixed | Reserved |

All datasets use five-class DR staging (Grade 0–4). Cross-validation: 3-fold, patient-level stratified split (no data leakage between eyes of the same patient).

## Evaluation

**Primary metrics** (EH-1 priority order): weighted F1-score → ROC-AUC (macro, OvR) → Cohen's κ (quadratic) → accuracy.

**Statistical testing** (built into experiment runners): McNemar test (paired classifier comparison), DeLong test (AUC comparison), bootstrap 95% CI (≥1,000 iterations), Holm-Bonferroni correction (multiple comparisons).

**Clinical metrics**: sensitivity, specificity, PPV, NPV at referable DR threshold (Grade ≥ 2).

## Governance

The `docs/` directory contains a formal constraint system that governs all dissertation claims:

- **INVARIANTS.md** — Immutable thesis formulation, hypotheses, dominance criteria, non-claim boundaries, source interpretation rules, version control rules.
- **ARGUMENT_MAP.md** — Formal claim-evidence-dependency structure mapping each experimental result to a specific claim (PC-1 through PC-9).
- **RESEARCH_ARCHITECTURE.md** — Full methodological blueprint: data architecture, experimental protocols, statistical requirements.
- **HYPOTHESIS.md** — Operational formulations of H-1 through H-6 with the causal argument structure.

## Hardware

Developed and tested on: WSL2 Ubuntu, NVIDIA RTX 3060 12GB, conda environment `dr-classifier`. Datasets stored on `/mnt/d/datasets/`.

## License

This repository is part of a doctoral dissertation and is not licensed for redistribution.
