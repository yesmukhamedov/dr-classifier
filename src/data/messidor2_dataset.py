"""Messidor-2 dataset for DR classification.

Messidor-2 provides ~1,748 fundus images with referable/non-referable DR grading.
Used in Experiment 5 (Clinical Degradation Resistance, H-7).

Camera: Topcon.
Labels: Referable (DR >= 2) / Non-referable (DR < 2).
For 5-class experiments, map to binary or use available severity grades.
"""

from pathlib import Path

from src.data.datasets import BaseFundusDataset


class Messidor2Dataset(BaseFundusDataset):
    """Dataset class for Messidor-2 fundus images.

    Args:
        root: Path to the Messidor-2 dataset directory.
        preprocessing: Optional preprocessing callable.
        augmentation: Optional augmentation callable.
    """

    def __init__(
        self,
        root: str | Path,
        preprocessing=None,
        augmentation=None,
    ) -> None:
        root = Path(root)
        # TODO: Implement label loading from Messidor-2 CSV/Excel files
        # Messidor-2 uses referable/non-referable grading
        raise NotImplementedError(
            "Messidor2Dataset not yet implemented. "
            "Need to parse Messidor-2 label files and map to DR grades."
        )
