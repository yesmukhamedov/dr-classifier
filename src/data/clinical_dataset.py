"""Clinical (Kazakh) dataset for DR classification.

60 fundus images from 30 patients (2 eyes each), 5-class DR staging.
Balanced: 12 images per class. PNG format.
Located at E:/datasets/clinical/

Used in Experiments 4 (qualitative Grad-CAM), 5 (clinical degradation), 7 (test set).
"""

from pathlib import Path

from src.data.datasets import BaseFundusDataset


class ClinicalDataset(BaseFundusDataset):
    """Dataset class for Kazakh clinical fundus images.

    Args:
        root: Path to the clinical dataset directory.
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
        # TODO: Implement image loading from clinical dataset
        # Expected structure: root/grade_0/, root/grade_1/, ... root/grade_4/
        # Or root/images/ + root/labels.csv with patient_id, eye_side, grade
        raise NotImplementedError(
            "ClinicalDataset not yet implemented. "
            "Need to define directory structure and label format."
        )
