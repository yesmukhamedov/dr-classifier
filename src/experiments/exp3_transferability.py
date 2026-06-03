"""Experiment 3: Cross-Dataset Transferability (H-4).

Train on EyePACS with full V5 pipeline, evaluate zero-shot on APTOS 2019.
Compute generalization ratio G = F1_APTOS / F1_EyePACS.
Success criterion: G >= 0.85.
"""

from typing import Any


def run(config: dict[str, Any], *, fold: int | None = None,
        resume: bool = False, _configs_to_run: list[str] | None = None) -> None:
    """Run Experiment 3: Cross-Dataset Transferability."""
    raise NotImplementedError(
        "Experiment 3 (transferability) not yet implemented. "
        "Train on EyePACS, test zero-shot on APTOS 2019."
    )
