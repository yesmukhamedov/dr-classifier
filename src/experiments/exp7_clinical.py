"""Experiment 7: Small Data Training.

5-fold CV on IDRiD as training data, Clinical dataset as held-out test.
Report mean +/- std across folds.
Tests trainability on small clinical datasets.
"""

from typing import Any


def run(config: dict[str, Any], *, fold: int | None = None,
        resume: bool = False, _configs_to_run: list[str] | None = None) -> None:
    """Run Experiment 7: Small Data Training."""
    raise NotImplementedError(
        "Experiment 7 (small data clinical) not yet implemented. "
        "5-fold CV on IDRiD, test on Clinical held-out."
    )
