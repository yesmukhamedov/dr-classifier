# Experimental Protocol

See `docs/RESEARCH_ARCHITECTURE.md` for the full experimental design.

## Quick Start
```bash
conda activate dr-classifier

# Run Experiment 1 (2×2 factorial A-D, V5 pipeline)
python run_experiment.py --experiment exp1 --config configs/default.yaml

# Run specific configs only
python run_experiment.py --experiment exp1 --config configs/default.yaml --configs A,B

# Run Experiment 2 (component ablation, V5 pipeline stages + CLAHE/flat-field sweeps)
python run_experiment.py --experiment exp2 --config configs/default.yaml

# Run Experiment 3 (cross-dataset transferability, APTOS 2019)
python run_experiment.py --experiment exp3 --config configs/default.yaml

# Run Experiment 4 (Grad-CAM explainability, IDRiD + Clinical)
python run_experiment.py --experiment exp4 --config configs/default.yaml

# Run Experiment 5 (clinical degradation resistance, IDRiD + Messidor-2)
python run_experiment.py --experiment exp5 --config configs/default.yaml

# Run Experiment 6 (device domain shift, DDR + ODIR-5K + RFMiD)
python run_experiment.py --experiment exp6 --config configs/default.yaml

# Run Experiment 7 (small data training, IDRiD → Clinical)
python run_experiment.py --experiment exp7 --config configs/default.yaml
```

## Datasets

All datasets stored at `/mnt/d/datasets/` (Windows NTFS, accessed via WSL2).
Paths configured in `configs/default.yaml`.

| Dataset | ~Images | Classes | Camera | Role |
|---------|---------|---------|--------|------|
| EyePACS | ~35,126 | 5-class (DR 0–4) | Canon CR-1 | Primary training (Exp 1,2,3,4,5,6) |
| APTOS 2019 | ~3,662 | 5-class | Mixed | Cross-dataset transferability (Exp 3) |
| IDRiD | 516 | 5-class + lesion masks | Kowa | Explainability, degradation (Exp 4,5,7 train) |
| Messidor-2 | ~1,748 | Referable/non-referable | Topcon | Clinical degradation (Exp 5) |
| DDR | ~13,673 | 5-class | Canon, Topcon | Device domain shift (Exp 6) |
| ODIR-5K | ~5,000 bilateral | Multi-disease → DR subset | Canon, Zeiss | Device domain shift (Exp 6) |
| RFMiD | ~3,200 | Binary DR | Topcon, Kowa | Device domain shift (Exp 6) |
| Clinical (Kazakh) | 60 | 5-class balanced | TBD | Clinical validation (Exp 4,5,7 test) |

## Cross-Validation

5-fold CV with patient-level stratified split. No patient leakage across folds.

## Grading Scale

| Grade | Clinical Stage | Approximate Prevalence (EyePACS) |
|-------|---------------|----------------------------------|
| 0 | No DR | ~73% |
| 1 | Mild NPDR | ~7% |
| 2 | Moderate NPDR | ~15% |
| 3 | Severe NPDR | ~2% |
| 4 | Proliferative DR | ~3% |

Class imbalance addressed via Focal Loss (γ=2, inverse-frequency α weights).

## Reproducibility
- seed: 42, deterministic=true
- All configs committed in `configs/`
- Checkpoints saved per fold per config
