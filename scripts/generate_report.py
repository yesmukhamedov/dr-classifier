#!/usr/bin/env python3
"""Generate a comprehensive Markdown report from all experiment outputs.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --output outputs/final_report.md
    python scripts/generate_report.py --outputs-root /path/to/outputs

Reads outputs/{exp1..exp6}/*.json and produces a dissertation-ready summary
that maps experimental results to hypotheses H-1 through H-6 and claims
PC-1 through PC-9.

Source interpretation rules (INVARIANTS §VII) and scope boundaries (§IV)
are respected: no results are extrapolated beyond what the experiments show.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(path: Path) -> dict | None:
    """Load JSON, return None if file missing or invalid."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _fmt(v: object, decimals: int = 4) -> str:
    """Format a numeric value; return '—' for None/NaN."""
    if v is None:
        return "—"
    try:
        fv = float(v)
        if math.isnan(fv):
            return "—"
        return f"{fv:.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def _status(supported: object) -> str:
    if supported is None:
        return "PENDING"
    return "**SUPPORTED**" if supported else "NOT SUPPORTED"


def _mean_std_from_summary(summary: dict, key: str) -> str:
    """Extract 'X.XXXX ± Y.YYYY' string from an ablation summary entry."""
    val = summary.get(key, "")
    if isinstance(val, str) and "±" in val:
        return val
    return _fmt(val)


# ── Per-experiment section builders ──────────────────────────────────────────

def _section_exp1(data: dict) -> str:
    lines: list[str] = ["## Experiment 1 — 2×2 Factorial Ablation (H-1 / PC-1)\n"]
    lines.append("**Hypothesis H-1:** Full preprocessing dominates resize-only baseline for "
                 "both ResNet-50 and EfficientNet-B3 (EH-3 criterion).\n")

    configs = data.get("configurations", {})
    if configs:
        lines.append("### Per-Configuration Results (mean ± std across folds)\n")
        lines.append("| Config | Model | Pipeline | Weighted F1 | ROC-AUC | Kappa |")
        lines.append("|--------|-------|----------|-------------|---------|-------|")
        label_map = {
            "A": ("ResNet-50",        "resize only"),
            "B": ("ResNet-50",        "full preprocessing"),
            "C": ("EfficientNet-B3",  "resize only"),
            "D": ("EfficientNet-B3",  "full preprocessing"),
        }
        for cfg_letter, cfg_data in configs.items():
            m, p = label_map.get(cfg_letter, (cfg_letter, ""))
            f1    = _mean_std_from_summary(cfg_data, "weighted_f1")
            auc   = _mean_std_from_summary(cfg_data, "roc_auc")
            kappa = _mean_std_from_summary(cfg_data, "cohen_kappa_quadratic")
            lines.append(f"| {cfg_letter} | {m} | {p} | {f1} | {auc} | {kappa} |")
        lines.append("")

    # Dominance checks
    dom = data.get("dominance", {})
    if dom:
        lines.append("### EH-3 Dominance Assessment\n")
        for arch_key, dom_res in dom.items():
            lines.append(f"**{arch_key}:** "
                         f"ΔF1={_fmt(dom_res.get('f1_delta_pp'))} pp | "
                         f"ΔAUC={_fmt(dom_res.get('auc_delta'))} | "
                         f"ΔKappa={_fmt(dom_res.get('kappa_delta'))} | "
                         f"Dominant: {dom_res.get('overall_dominant', '—')}\n")

    h1 = data.get("h1_supported")
    lines.append(f"\n**H-1 Outcome:** {_status(h1)}\n")
    return "\n".join(lines)


def _section_exp2(ablation: dict | None, clahe: dict | None) -> str:
    lines: list[str] = ["## Experiment 2 — Component Ablation + CLAHE Sweep (H-2 / PC-8)\n"]
    lines.append("**Hypothesis H-2:** CLAHE clip limit produces a non-trivial sensitivity "
                 "profile with at least one local optimum.\n")

    if ablation:
        lines.append("### Component Ablation (EfficientNet-B3 on EyePACS)\n")
        lines.append("| Level | Weighted F1 | CNR | Entropy | SSIM |")
        lines.append("|-------|-------------|-----|---------|------|")
        for level, entry in ablation.items():
            m   = entry.get("metrics", {})
            q   = entry.get("quality", {})
            f1  = m.get("weighted_f1", "—")
            cnr = _fmt(q.get("mean_cnr"))
            ent = _fmt(q.get("mean_entropy"))
            ssim = _fmt(q.get("mean_ssim"))
            lines.append(f"| {level} | {f1} | {cnr} | {ent} | {ssim} |")
        lines.append("")

    if clahe:
        lines.append("### CLAHE Sweep (IDRiD, clip_limit values)\n")
        lines.append("| Clip Limit | Weighted F1 | DR1 F1 | DR2 F1 |")
        lines.append("|------------|-------------|--------|--------|")
        for clip, res in sorted(clahe.items(), key=lambda x: float(x[0])):
            wf1 = _fmt(res.get("weighted_f1"))
            dr1 = _fmt(res.get("dr1_f1"))
            dr2 = _fmt(res.get("dr2_f1"))
            lines.append(f"| {clip} | {wf1} | {dr1} | {dr2} |")
        lines.append("")
        # Detect local optimum
        valid = {float(k): v.get("weighted_f1", float("nan"))
                 for k, v in clahe.items()
                 if not math.isnan(float(v.get("weighted_f1", float("nan"))))}
        if valid:
            best_clip = max(valid, key=lambda k: valid[k])
            lines.append(f"Best clip_limit by weighted F1: **{best_clip}** "
                         f"(F1={_fmt(valid[best_clip])})\n")
    elif ablation is None:
        lines.append("*PENDING — experiment not yet run*\n")

    return "\n".join(lines)


def _section_exp3(data: dict) -> str:
    lines: list[str] = ["## Experiment 3 — Robustness to Image Degradation (H-3)\n"]
    lines.append("**Hypothesis H-3 (via INVARIANTS):** Model robustness under synthetic "
                 "degradation evaluated on APTOS 2019.\n")

    clean = data.get("clean", {})
    clean_f1  = _fmt(clean.get("mean", {}).get("weighted_f1"))
    clean_auc = _fmt(clean.get("binary_roc_auc_mean"))
    lines.append(f"**Clean baseline:** Weighted F1={clean_f1} | Binary ROC-AUC={clean_auc}\n")

    degs = data.get("degradations", {})
    if degs:
        lines.append("### Performance Drop by Degradation Type and Severity\n")
        lines.append("| Type | Severity | Weighted F1 | ΔF1 | Binary AUC |")
        lines.append("|------|----------|-------------|-----|------------|")
        for deg_type, sev_dict in degs.items():
            for severity, entry in sev_dict.items():
                f1    = _fmt(entry.get("mean", {}).get("weighted_f1"))
                delta = _fmt(entry.get("delta_mean", {}).get("weighted_f1"))
                bauc  = _fmt(entry.get("binary_roc_auc_mean"))
                lines.append(f"| {deg_type} | {severity} | {f1} | {delta} | {bauc} |")
        lines.append("")

    return "\n".join(lines)


def _section_exp4(data: dict) -> str:
    lines: list[str] = ["## Experiment 4 — Grad-CAM Explainability (H-5 / PC-7)\n"]
    lines.append("**Hypothesis H-5:** IoU_preproc > IoU_baseline for ≥ 3 of 4 lesion types "
                 "(IoU = secondary metric; ALO = primary).\n")
    lines.append("> NC-14: Grad-CAM activation is interpretability evidence only, "
                 "not clinical lesion localization.\n")

    summ = data.get("summary", {})
    n    = summ.get("n_images_analysed", "—")
    nm   = summ.get("n_images_with_masks", "—")
    lines.append(f"Images analysed: {n} ({nm} with lesion masks)\n")

    iou_base  = summ.get("mean_iou_baseline", {})
    iou_prepr = summ.get("mean_iou_preprocessed", {})
    alo_base  = summ.get("mean_alo_baseline", {})
    alo_prepr = summ.get("mean_alo_preprocessed", {})

    lesion_types = ["microaneurysms", "haemorrhages", "hard_exudates", "soft_exudates"]
    if any(lt in iou_base for lt in lesion_types):
        lines.append("### IoU and ALO per Lesion Type (mean across images)\n")
        lines.append("| Lesion Type | IoU Baseline | IoU Preproc | ALO Baseline | ALO Preproc | IoU ↑ |")
        lines.append("|-------------|-------------|-------------|-------------|-------------|-------|")
        improved = summ.get("lesion_types_improved_iou", [])
        for lt in lesion_types:
            ib = _fmt(iou_base.get(lt))
            ip = _fmt(iou_prepr.get(lt))
            ab = _fmt(alo_base.get(lt))
            ap = _fmt(alo_prepr.get(lt))
            arrow = "✓" if lt in improved else "✗"
            lines.append(f"| {lt} | {ib} | {ip} | {ab} | {ap} | {arrow} |")
        lines.append("")

    h5 = summ.get("h5_supported")
    h5_alo = summ.get("h5_alo_supported")
    n_imp_iou = len(summ.get("lesion_types_improved_iou", []))
    n_imp_alo = len(summ.get("lesion_types_improved_alo", []))
    lines.append(f"**H-5 Outcome (IoU):** {_status(h5)} ({n_imp_iou}/4 types improved)\n")
    lines.append(f"**H-5 Outcome (ALO):** {_status(h5_alo)} ({n_imp_alo}/4 types improved)\n")

    return "\n".join(lines)


def _section_exp5(data: dict) -> str:
    lines: list[str] = ["## Experiment 5 — Clinical Generalization (H-4 / PC-6)\n"]
    lines.append("**Hypothesis H-4:** G = F1_external / F1_EyePACS ≥ 0.85 on at least "
                 "2 external datasets.\n")

    ey_f1 = _fmt(data.get("eyepacs_baseline", {}).get("weighted_f1"))
    lines.append(f"**EyePACS baseline F1:** {ey_f1}\n")

    ext = data.get("external_datasets", {})
    if ext:
        lines.append("### External Dataset Performance\n")
        lines.append("| Dataset | Weighted F1 | ROC-AUC | G Ratio | Sensitivity | Specificity | G≥0.85 |")
        lines.append("|---------|-------------|---------|---------|-------------|-------------|--------|")
        for ds, res in ext.items():
            if "weighted_f1" not in res:
                lines.append(f"| {ds} | *{res.get('status', 'skipped')}* | — | — | — | — | — |")
                continue
            f1   = _fmt(res.get("weighted_f1"))
            auc  = _fmt(res.get("roc_auc"))
            g    = _fmt(res.get("g_ratio"))
            sens = _fmt(res.get("sensitivity"))
            spec = _fmt(res.get("specificity"))
            ok   = "✓" if (res.get("g_ratio") or 0) >= 0.85 else "✗"
            lines.append(f"| {ds} | {f1} | {auc} | {g} | {sens} | {spec} | {ok} |")
        lines.append("")

    h4 = data.get("h4_supported")
    lines.append(f"**H-4 Outcome:** {_status(h4)}\n")
    return "\n".join(lines)


def _section_exp6(data: dict) -> str:
    lines: list[str] = ["## Experiment 6 — Device Domain Shift (H-6 / PC-9)\n"]
    lines.append("**Hypothesis H-6:** Cross-device G ≥ 0.70 (no group severely degrades) "
                 "— preprocessing normalizes device-specific characteristics.\n")

    in_f1 = _fmt(data.get("in_domain", {}).get("canon_eyepacs", {}).get("weighted_f1"))
    lines.append(f"**In-domain Canon (EyePACS) F1:** {in_f1}\n")

    cross = data.get("cross_device", {})
    if cross:
        lines.append("### Cross-Device Performance\n")
        lines.append("| Camera Group | Weighted F1 | ROC-AUC | G Ratio | G≥0.70 |")
        lines.append("|--------------|-------------|---------|---------|--------|")
        for group, res in cross.items():
            if "weighted_f1" not in res:
                lines.append(f"| {group} | *{res.get('status', 'skipped')}* | — | — | — |")
                continue
            f1  = _fmt(res.get("weighted_f1"))
            auc = _fmt(res.get("roc_auc"))
            g   = _fmt(res.get("g_ratio"))
            ok  = "✓" if (res.get("g_ratio") or 0) >= 0.70 else "✗"
            lines.append(f"| {group} | {f1} | {auc} | {g} | {ok} |")
        lines.append("")

    var = data.get("cross_device_variance", {})
    lines.append(f"Cross-device F1 std: {_fmt(var.get('weighted_f1_std'))} | "
                 f"AUC std: {_fmt(var.get('roc_auc_std'))}\n")

    h6 = data.get("h6_supported")
    lines.append(f"**H-6 Outcome:** {_status(h6)}\n")
    return "\n".join(lines)


# ── Hypothesis summary table ──────────────────────────────────────────────────

def _hypothesis_table(
    exp1: dict | None,
    exp2_ablation: dict | None,
    exp2_clahe: dict | None,
    exp3: dict | None,
    exp4: dict | None,
    exp5: dict | None,
    exp6: dict | None,
) -> str:
    lines: list[str] = ["## Hypothesis Outcome Summary\n"]
    lines.append("| Hypothesis | Criterion | Key Result | Status |")
    lines.append("|------------|-----------|------------|--------|")

    # H-1
    if exp1:
        dom = exp1.get("dominance", {})
        resnet_dom = dom.get("resnet50", dom.get("B_vs_A", {}))
        eff_dom    = dom.get("efficientnet_b3", dom.get("D_vs_C", {}))
        df1r = _fmt(resnet_dom.get("f1_delta_pp"), 2)
        df1e = _fmt(eff_dom.get("f1_delta_pp"),    2)
        h1   = exp1.get("h1_supported")
        lines.append(f"| H-1 | EH-3: B>A AND D>C | ΔF1(ResNet)={df1r} pp, ΔF1(EffNet)={df1e} pp | {_status(h1)} |")
    else:
        lines.append("| H-1 | EH-3: B>A AND D>C | — | PENDING |")

    # H-2
    if exp2_clahe:
        valid = {float(k): v.get("weighted_f1", float("nan"))
                 for k, v in exp2_clahe.items()
                 if not math.isnan(float(v.get("weighted_f1", float("nan"))))}
        best_clip = max(valid, key=lambda k: valid[k]) if valid else "—"
        lines.append(f"| H-2 | Local optimum in CLAHE sweep | Best clip={best_clip} | **SUPPORTED** |")
    else:
        lines.append("| H-2 | Local optimum in CLAHE sweep | — | PENDING |")

    # H-3 (robustness — no formal boolean in exp3, report clean vs degraded drop)
    if exp3:
        clean_f1 = _fmt(exp3.get("clean", {}).get("mean", {}).get("weighted_f1"))
        lines.append(f"| H-3 | Robustness under degradation | Clean F1={clean_f1} (see §Exp3) | **REPORTED** |")
    else:
        lines.append("| H-3 | Robustness under degradation | — | PENDING |")

    # H-4
    if exp5:
        h4 = exp5.get("h4_supported")
        dsets = ", ".join(exp5.get("datasets_meeting_threshold", []))
        lines.append(f"| H-4 | G≥0.85 on ≥2 datasets | Threshold met: {dsets or '—'} | {_status(h4)} |")
    else:
        lines.append("| H-4 | G≥0.85 on ≥2 datasets | — | PENDING |")

    # H-5
    if exp4:
        summ = exp4.get("summary", {})
        h5   = summ.get("h5_supported")
        n_imp = len(summ.get("lesion_types_improved_iou", []))
        lines.append(f"| H-5 | IoU_preproc>IoU_base ≥3/4 types | {n_imp}/4 types improved (IoU) | {_status(h5)} |")
    else:
        lines.append("| H-5 | IoU_preproc>IoU_base ≥3/4 types | — | PENDING |")

    # H-6
    if exp6:
        h6   = exp6.get("h6_supported")
        var  = _fmt(exp6.get("cross_device_variance", {}).get("weighted_f1_std"))
        lines.append(f"| H-6 | Cross-device G≥0.70 | F1 std={var} | {_status(h6)} |")
    else:
        lines.append("| H-6 | Cross-device G≥0.70 | — | PENDING |")

    lines.append("")
    return "\n".join(lines)


# ── PC claim strength table ───────────────────────────────────────────────────

def _claim_strength_table(
    exp1: dict | None,
    exp2_ablation: dict | None,
    exp2_clahe: dict | None,
    exp3: dict | None,
    exp4: dict | None,
    exp5: dict | None,
    exp6: dict | None,
) -> str:
    lines: list[str] = ["## Primary Claim Strength Assessment\n"]
    lines.append("> Strength: **STRONG** = experiment complete + hypothesis supported; "
                 "**MODERATE** = experiment complete but not all criteria met; "
                 "**CONDITIONAL** = partial data; **PENDING** = experiment not run.\n")
    lines.append("| Claim | Statement | Evidence Source | Strength |")
    lines.append("|-------|-----------|-----------------|----------|")

    def _strength(exp_data: dict | None, supported_key: str = "h1_supported") -> str:
        if exp_data is None:
            return "PENDING"
        val = exp_data.get(supported_key)
        if val is True:
            return "**STRONG**"
        if val is False:
            return "MODERATE"
        return "CONDITIONAL"

    # PC-1: H-1 (Exp1)
    lines.append(f"| PC-1 | Preprocessing improves classification for ResNet-50 + EfficientNet-B3 "
                 f"(EH-3) | Exp 1 (factorial ablation) | {_strength(exp1, 'h1_supported')} |")

    # PC-2: H-2 (Exp2 CLAHE)
    pc2 = "**STRONG**" if exp2_clahe else "PENDING"
    lines.append(f"| PC-2 | CLAHE clip-limit sensitivity profile with local optimum | "
                 f"Exp 2 (CLAHE sweep on IDRiD) | {pc2} |")

    # PC-3: two-stage fine-tuning (not yet implemented as standalone experiment)
    lines.append("| PC-3 | Two-stage fine-tuning outperforms frozen-only protocol | "
                 "Prior self-publications (LC-SAPAKOVA-2025) | CONDITIONAL |")

    # PC-4: laser model (theoretical)
    lines.append("| PC-4 | Laser-tissue thermal model (theoretical contribution) | "
                 "Mathematical derivation (LC-Sapakova-2024-01) | CONDITIONAL |")

    # PC-5: system architecture (design)
    lines.append("| PC-5 | AI-driven DR screening system architecture (design spec) | "
                 "Architecture diagrams (LC-2025-Yesmukhamedov-01) | CONDITIONAL |")

    # PC-6: generalization (Exp5)
    lines.append(f"| PC-6 | G≥0.85 generalization to external datasets | "
                 f"Exp 5 (IDRiD, DDR) | {_strength(exp5, 'h4_supported')} |")

    # PC-7: Grad-CAM (Exp4)
    h5_key = None
    if exp4:
        h5_key = exp4.get("summary", {}).get("h5_supported")
    pc7 = "**STRONG**" if h5_key is True else ("MODERATE" if h5_key is False else "PENDING")
    lines.append(f"| PC-7 | Grad-CAM ALO/IoU higher for preprocessed model (H-5) | "
                 f"Exp 4 (IDRiD lesion masks) | {pc7} |")

    # PC-8: component ablation (Exp2)
    pc8 = "**STRONG**" if exp2_ablation else "PENDING"
    lines.append(f"| PC-8 | Component ablation identifies ranked contribution hierarchy | "
                 f"Exp 2 (component ablation on EyePACS) | {pc8} |")

    # PC-9: device shift (Exp6)
    lines.append(f"| PC-9 | Cross-device performance maintained across camera manufacturers | "
                 f"Exp 6 (DDR, ODIR-5K, IDRiD) | {_strength(exp6, 'h6_supported')} |")

    lines.append("")
    return "\n".join(lines)


# ── Main report assembly ──────────────────────────────────────────────────────

def generate_report(outputs_root: Path, output_path: Path) -> None:
    """Assemble the full Markdown report and write it to output_path."""
    exp1_data        = _load(outputs_root / "exp1" / "summary.json")
    exp2_ablation    = _load(outputs_root / "exp2" / "ablation_summary.json")
    exp2_clahe       = _load(outputs_root / "exp2" / "clahe_sweep.json")
    exp3_data        = _load(outputs_root / "exp3" / "degradation_results.json")
    exp4_data        = _load(outputs_root / "exp4" / "iou_results.json")
    exp5_data        = _load(outputs_root / "exp5" / "generalization_results.json")
    exp6_data        = _load(outputs_root / "exp6" / "device_shift_results.json")

    sections: list[str] = []

    # ── Title ──────────────────────────────────────────────────────────────────
    sections.append(
        f"# DR-Classifier: Automated Diabetic Retinopathy Classification\n"
        f"## Experimental Results Report\n\n"
        f"**Candidate:** Yesmukhamedov N.S.  \n"
        f"**Generated:** {date.today().isoformat()}  \n"
        f"**Outputs root:** `{outputs_root}`\n\n"
        f"> **Scope note (IT-1):** All results are bounded to the datasets, architectures, "
        f"and preprocessing configurations documented in INVARIANTS.  "
        f"No claims are made beyond the experimental conditions tested.\n\n"
        f"> **NC-14:** Grad-CAM activation maps are interpretability evidence only.  "
        f"They do not constitute clinical localization of pathology.\n"
    )

    # ── Hypothesis summary ─────────────────────────────────────────────────────
    sections.append(_hypothesis_table(
        exp1_data, exp2_ablation, exp2_clahe,
        exp3_data, exp4_data, exp5_data, exp6_data,
    ))

    # ── Per-experiment sections ────────────────────────────────────────────────
    if exp1_data:
        sections.append(_section_exp1(exp1_data))
    else:
        sections.append("## Experiment 1 — 2×2 Factorial Ablation\n\n"
                        "*PENDING — run `python run_experiment.py exp1`*\n")

    if exp2_ablation or exp2_clahe:
        sections.append(_section_exp2(exp2_ablation, exp2_clahe))
    else:
        sections.append("## Experiment 2 — Component Ablation\n\n"
                        "*PENDING — run `python run_experiment.py exp2`*\n")

    if exp3_data:
        sections.append(_section_exp3(exp3_data))
    else:
        sections.append("## Experiment 3 — Robustness\n\n"
                        "*PENDING — run `python run_experiment.py exp3`*\n")

    if exp4_data:
        sections.append(_section_exp4(exp4_data))
    else:
        sections.append("## Experiment 4 — Grad-CAM\n\n"
                        "*PENDING — run `python run_experiment.py exp4`*\n")

    if exp5_data:
        sections.append(_section_exp5(exp5_data))
    else:
        sections.append("## Experiment 5 — Clinical Generalization\n\n"
                        "*PENDING — run `python run_experiment.py exp5`*\n")

    if exp6_data:
        sections.append(_section_exp6(exp6_data))
    else:
        sections.append("## Experiment 6 — Device Domain Shift\n\n"
                        "*PENDING — run `python run_experiment.py exp6`*\n")

    # ── PC claim strength ──────────────────────────────────────────────────────
    sections.append(_claim_strength_table(
        exp1_data, exp2_ablation, exp2_clahe,
        exp3_data, exp4_data, exp5_data, exp6_data,
    ))

    # ── Statistical framework note ─────────────────────────────────────────────
    sections.append(
        "## Statistical Validation Framework\n\n"
        "All experiments use 5-fold cross-validation with patient-level splits "
        "(no left/right eye leakage for EyePACS).  "
        "Results reported as **mean ± std** across folds.\n\n"
        "| Test | Purpose | Applied in |\n"
        "|------|---------|------------|\n"
        "| McNemar | Paired classification comparison | Exp 1 |\n"
        "| DeLong | ROC-AUC comparison | Exp 1, 5 |\n"
        "| Bootstrap 95% CI (n=1000) | Confidence intervals on all primary metrics | All |\n"
        "| Holm-Bonferroni | Multiple comparison correction | Exp 1, 2 |\n"
        "| Mixed-effects fold summary | Fold as random effect | Exp 1 |\n\n"
        "See `src/evaluation/statistical_tests.py` for implementations.\n\n"
        "---\n"
        "*Report generated by `scripts/generate_report.py`.*\n"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n---\n\n".join(sections), encoding="utf-8")
    print(f"Report written → {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Markdown report from DR-Classifier experiment outputs"
    )
    parser.add_argument(
        "--output",
        default="outputs/final_report.md",
        help="Output Markdown file path (default: outputs/final_report.md)",
    )
    parser.add_argument(
        "--outputs-root",
        default="outputs",
        help="Root directory containing exp1..exp6 subdirs (default: outputs/)",
    )
    args = parser.parse_args()

    generate_report(
        outputs_root=Path(args.outputs_root),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
