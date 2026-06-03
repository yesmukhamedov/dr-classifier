"""Fundus image dataset classes for DR classification."""

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# PreprocessingPipelineV5 is imported lazily inside methods to avoid circular
# imports (src.preprocessing.pipeline_v5 imports from src.data.augmentation_v4,
# which is a sibling of this module).
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.preprocessing.pipeline_v5 import PreprocessingPipelineV5


class BaseFundusDataset(Dataset):
    """Base dataset for fundus images.

    Handles image loading, optional preprocessing, optional augmentation,
    and conversion to float32 CHW tensors.

    Args:
        image_paths: List of absolute paths to image files.
        labels: List of integer DR grade labels (0–4).
        preprocessing: Optional callable applied to the raw BGR uint8 image.
            Expected to return a numpy array (uint8 or float32 [0,1]).
        augmentation: Optional callable applied after preprocessing.
            Expected to return a numpy array of the same dtype/range.
    """

    def __init__(
        self,
        image_paths: list[str],
        labels: list[int],
        preprocessing: Callable | None = None,
        augmentation: Callable | None = None,
    ) -> None:
        if len(image_paths) != len(labels):
            raise ValueError(
                f"image_paths and labels must have equal length, "
                f"got {len(image_paths)} and {len(labels)}"
            )
        self.image_paths = image_paths
        self.labels = labels
        self.preprocessing = preprocessing
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Load and return one sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, label) where image_tensor is float32
            with shape (C, H, W) and values in [0, 1].
        """
        image = cv2.imread(str(self.image_paths[idx]))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {self.image_paths[idx]}")

        if self.preprocessing is not None:
            image = self.preprocessing(image)

        if self.augmentation is not None:
            image = self.augmentation(image)

        # Normalise to [0, 1] float32 only if still uint8
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)

        # HWC → CHW
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        return tensor, self.labels[idx]


def _apply_subset(
    image_paths: list[str],
    labels: list[int],
    patient_ids: list[str],
    subset_indices: list[int] | None,
) -> tuple[list[str], list[int], list[str]]:
    """Filter lists to subset_indices if provided."""
    if subset_indices is None:
        return image_paths, labels, patient_ids
    return (
        [image_paths[i] for i in subset_indices],
        [labels[i] for i in subset_indices],
        [patient_ids[i] for i in subset_indices],
    )


class EyePACSDataset(BaseFundusDataset):
    """EyePACS fundus dataset (~88k images, 5-class DR grading).

    Patient ID = numeric prefix before _left / _right in the filename.
    Left and right eyes of the same patient share a patient_id, ensuring
    they are always assigned to the same CV fold.

    Args:
        image_paths: Absolute paths to .jpeg images.
        labels: DR grade labels (0–4).
        patient_ids: Numeric patient strings (e.g. "10").
        preprocessing: Optional preprocessing callable.
        augmentation: Optional augmentation callable.
    """

    def __init__(
        self,
        image_paths: list[str],
        labels: list[int],
        patient_ids: list[str],
        preprocessing: Callable | None = None,
        augmentation: Callable | None = None,
        eye_sides: list[str] | None = None,
    ) -> None:
        super().__init__(image_paths, labels, preprocessing, augmentation)
        self.patient_ids = patient_ids
        self.eye_sides: list[str] = (
            eye_sides if eye_sides is not None
            else ["unknown"] * len(image_paths)
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Load and return one sample, routing through V4 or V3 pipeline.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, label) where image_tensor is float32
            CHW and values are ImageNet-normalised (V4) or in [0, 1] (V3).
        """
        image = cv2.imread(str(self.image_paths[idx]))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {self.image_paths[idx]}")

        eye_side = self.eye_sides[idx]

        if self.preprocessing is not None:
            from src.preprocessing.pipeline_v5 import PreprocessingPipelineV5  # noqa: PLC0415
            if isinstance(self.preprocessing, PreprocessingPipelineV5):
                # V4 pipeline handles BGR→RGB conversion, all stages, and
                # returns a normalised tensor directly.
                tensor = self.preprocessing(image, eye_side=eye_side)
                return tensor, self.labels[idx]
            else:
                # V3 legacy path
                image = self.preprocessing(image)

        if self.augmentation is not None:
            image = self.augmentation(image)

        # Legacy normalisation: uint8 → float32 [0, 1]
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            else:
                image = image.astype(np.float32)
            tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        else:
            tensor = image  # already a tensor

        return tensor, self.labels[idx]

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        labels_csv: str | Path,
        subset_indices: list[int] | None = None,
        preprocessing: Callable | None = None,
        augmentation: Callable | None = None,
    ) -> "EyePACSDataset":
        """Build dataset from EyePACS directory layout.

        Args:
            root: Path to the train/ image directory.
            labels_csv: Path to trainLabels.csv (columns: image, level).
                        Image names have no extension (e.g. "10_left").
            subset_indices: Optional list of row indices to load from the CSV.
                            If None, loads all rows.
            preprocessing: Optional preprocessing callable.
            augmentation: Optional augmentation callable.

        Returns:
            Constructed EyePACSDataset with patient_ids attribute.
        """
        root = Path(root)
        df = pd.read_csv(labels_csv)
        if subset_indices is not None:
            df = df.iloc[subset_indices].reset_index(drop=True)

        image_paths, labels, patient_ids, eye_sides = [], [], [], []
        for _, row in df.iterrows():
            name: str = str(row["image"])
            img_path = root / f"{name}.jpeg"
            if not img_path.exists():
                continue
            image_paths.append(str(img_path))
            labels.append(int(row["level"]))
            patient_ids.append(name.split("_")[0])
            eye_sides.append("left" if "_left" in name else "right")

        return cls(image_paths, labels, patient_ids, preprocessing, augmentation, eye_sides)


def load_cache_meta(cache_dir: str | Path) -> dict[str, tuple[bool, float]]:
    """Load the V5 cache sidecar mapping image name → (confident, rotation_sigma_deg).

    These are the only two OD/fovea scalars Stage 6 augmentation reads; they are
    cached alongside the Stage-4 PNGs by ``scripts/precompute_v5_cache.py``.

    Args:
        cache_dir: Directory containing ``cache_meta.csv`` (columns:
            ``image``, ``confident``, ``rotation_sigma_deg``).

    Returns:
        Dict mapping image name (no extension) → ``(confident, rotation_sigma_deg)``.
    """
    meta_path = Path(cache_dir) / "cache_meta.csv"
    df = pd.read_csv(meta_path)
    return {
        str(row["image"]): (bool(row["confident"]), float(row["rotation_sigma_deg"]))
        for _, row in df.iterrows()
    }


class CachedEyePACSDataset(EyePACSDataset):
    """EyePACS dataset reading a precomputed V5 Stage 0–4 cache (the throughput fix).

    Each sample is a 4-channel PNG written by ``scripts/precompute_v5_cache.py``
    (RGB = Stage-4 flat-field output, alpha = binary FOV mask) plus two cached
    OD/fovea scalars. Only the stochastic Stages 5–7 (CLAHE + augmentation +
    normalize) run per epoch via :meth:`PreprocessingPipelineV5.finish_from_cache`,
    so the DataLoader stops being the bottleneck and the GPU no longer starves.

    The ``preprocessing`` pipeline MUST be a **full** (non-baseline)
    :class:`PreprocessingPipelineV5`; baseline configs have no cacheable
    Stage 0–4 and must use :class:`EyePACSDataset` on raw images instead.

    Args:
        image_paths: Absolute paths to the cached ``{name}.png`` files.
        labels: DR grade labels (0–4).
        patient_ids: Numeric patient strings.
        preprocessing: Full V5 pipeline (its ``finish_from_cache`` is called).
        cache_meta: Mapping from :func:`load_cache_meta`.
        eye_sides: ``"left"``/``"right"`` per image (unused at read time — the
            cache already encodes Stage-0 orientation — kept for parity).
    """

    def __init__(
        self,
        image_paths: list[str],
        labels: list[int],
        patient_ids: list[str],
        preprocessing: Callable,
        cache_meta: dict[str, tuple[bool, float]],
        eye_sides: list[str] | None = None,
    ) -> None:
        super().__init__(
            image_paths, labels, patient_ids,
            preprocessing=preprocessing, augmentation=None, eye_sides=eye_sides,
        )
        self._cache_meta = cache_meta

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Load one cached sample and finish Stages 5–7.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, label); image_tensor is float32 ``(4, H, W)``,
            normalised exactly as the live pipeline would produce.
        """
        path = self.image_paths[idx]
        bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if bgra is None:
            raise FileNotFoundError(f"Could not load cached image: {path}")
        if bgra.ndim != 3 or bgra.shape[2] != 4:
            raise ValueError(
                f"Cached image {path} must be 4-channel BGRA, got shape {bgra.shape}"
            )

        flat_rgb = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2RGB)
        fov_mask = bgra[:, :, 3].astype(np.float32) / 255.0  # {0,255} → {0.0,1.0}

        name = Path(path).stem
        confident, rotation_sigma_deg = self._cache_meta[name]

        tensor = self.preprocessing.finish_from_cache(
            flat_rgb, fov_mask, confident, rotation_sigma_deg
        )
        return tensor, self.labels[idx]


class APTOS2019Dataset(BaseFundusDataset):
    """APTOS 2019 Blindness Detection dataset (~3,662 images, 5-class DR).

    Each image corresponds to a unique patient.

    Args:
        image_paths: Absolute paths to .png images.
        labels: DR grade labels (0–4).
        patient_ids: id_code strings (one per image, equals patient ID).
        preprocessing: Optional preprocessing callable.
        augmentation: Optional augmentation callable.
    """

    def __init__(
        self,
        image_paths: list[str],
        labels: list[int],
        patient_ids: list[str],
        preprocessing: Callable | None = None,
        augmentation: Callable | None = None,
    ) -> None:
        super().__init__(image_paths, labels, preprocessing, augmentation)
        self.patient_ids = patient_ids

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        labels_csv: str | Path,
        subset_indices: list[int] | None = None,
        preprocessing: Callable | None = None,
        augmentation: Callable | None = None,
    ) -> "APTOS2019Dataset":
        """Build dataset from APTOS 2019 directory layout.

        Args:
            root: Path to train_images/ directory.
            labels_csv: Path to train.csv (columns: id_code, diagnosis).
            subset_indices: Optional list of row indices to load from the CSV.
            preprocessing: Optional preprocessing callable.
            augmentation: Optional augmentation callable.

        Returns:
            Constructed APTOS2019Dataset with patient_ids attribute.
        """
        root = Path(root)
        df = pd.read_csv(labels_csv)
        if subset_indices is not None:
            df = df.iloc[subset_indices].reset_index(drop=True)

        image_paths, labels, patient_ids = [], [], []
        for _, row in df.iterrows():
            id_code = str(row["id_code"])
            img_path = root / f"{id_code}.png"
            if not img_path.exists():
                continue
            image_paths.append(str(img_path))
            labels.append(int(row["diagnosis"]))
            patient_ids.append(id_code)

        return cls(image_paths, labels, patient_ids, preprocessing, augmentation)


class IDRiDDataset(BaseFundusDataset):
    """IDRiD disease grading dataset (~413 training images, 5-class DR).

    Also provides access to pixel-level lesion masks for the 54-image
    segmentation subset (Microaneurysms, Haemorrhages, Hard Exudates,
    Soft Exudates).

    Mask naming convention:
        Grading image IDRiD_XXX.jpg (3-digit) corresponds to mask
        IDRiD_YY_<suffix>.tif (2-digit) where YY = int(XXX).

    Args:
        image_paths: Absolute paths to .jpg images.
        labels: DR grade labels (0–4).
        patient_ids: Image name stems used as patient IDs.
        image_stems: Image name stems (e.g. "IDRiD_001") for mask lookup.
        masks_root: Root of segmentation groundtruths directory, or None.
        preprocessing: Optional preprocessing callable.
        augmentation: Optional augmentation callable.
    """

    LESION_DIRS: dict[str, str] = {
        "microaneurysms": "1. Microaneurysms",
        "haemorrhages": "2. Haemorrhages",
        "hard_exudates": "3. Hard Exudates",
        "soft_exudates": "4. Soft Exudates",
    }
    LESION_SUFFIXES: dict[str, str] = {
        "microaneurysms": "MA",
        "haemorrhages": "HE",
        "hard_exudates": "EX",
        "soft_exudates": "SE",
    }

    def __init__(
        self,
        image_paths: list[str],
        labels: list[int],
        patient_ids: list[str],
        image_stems: list[str],
        masks_root: Path | None,
        preprocessing: Callable | None = None,
        augmentation: Callable | None = None,
    ) -> None:
        super().__init__(image_paths, labels, preprocessing, augmentation)
        self.patient_ids = patient_ids
        self.image_stems = image_stems
        self.masks_root = masks_root

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        labels_csv: str | Path,
        subset_indices: list[int] | None = None,
        lesion_mask_dir: str | Path | None = None,
        preprocessing: Callable | None = None,
        augmentation: Callable | None = None,
    ) -> "IDRiDDataset":
        """Build dataset from IDRiD directory layout.

        Args:
            root: Path to grading images directory
                  (.../B. Disease Grading/1. Original Images/a. Training Set/).
            labels_csv: Path to grading training labels CSV.
            subset_indices: Optional list of row indices to load from the CSV.
            lesion_mask_dir: Path to segmentation groundtruths directory
                  (.../A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/).
                  If None, mask loading is disabled.
            preprocessing: Optional preprocessing callable.
            augmentation: Optional augmentation callable.

        Returns:
            Constructed IDRiDDataset with lesion mask support.
        """
        root = Path(root)
        labels_csv = Path(labels_csv)

        # CSV has trailing commas — use only the first 2 columns
        df = pd.read_csv(labels_csv, usecols=[0, 1])
        df.columns = ["image_name", "retinopathy_grade"]
        if subset_indices is not None:
            df = df.iloc[subset_indices].reset_index(drop=True)

        image_paths, labels, patient_ids, image_stems = [], [], [], []
        for _, row in df.iterrows():
            stem = str(row["image_name"]).strip()
            img_path = root / f"{stem}.jpg"
            if not img_path.exists():
                continue
            image_paths.append(str(img_path))
            labels.append(int(row["retinopathy_grade"]))
            patient_ids.append(stem)
            image_stems.append(stem)

        masks_root: Path | None = None
        if lesion_mask_dir is not None:
            masks_root = Path(lesion_mask_dir)
            if not masks_root.exists():
                masks_root = None

        return cls(
            image_paths, labels, patient_ids, image_stems,
            masks_root, preprocessing, augmentation,
        )

    def get_lesion_masks(self, idx: int) -> dict[str, np.ndarray] | None:
        """Load available lesion masks for a given sample index.

        Masks exist only for the 54-image segmentation subset.
        Image names use 3-digit IDs (IDRiD_001); mask files use 2-digit IDs
        (IDRiD_01_MA.tif).

        Args:
            idx: Sample index.

        Returns:
            Dict mapping lesion type key to binary mask (H×W uint8), or None
            if no masks are available for this image.
        """
        if self.masks_root is None:
            return None

        stem = self.image_stems[idx]        # e.g. "IDRiD_001"
        num_str = stem.split("_")[1]        # "001"
        mask_num = f"{int(num_str):02d}"    # "01"
        mask_stem = f"IDRiD_{mask_num}"     # "IDRiD_01"

        masks: dict[str, np.ndarray] = {}
        for lesion_key, lesion_dir in self.LESION_DIRS.items():
            suffix = self.LESION_SUFFIXES[lesion_key]
            mask_path = self.masks_root / lesion_dir / f"{mask_stem}_{suffix}.tif"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    masks[lesion_key] = mask

        return masks if masks else None

    def count_images_with_masks(self) -> int:
        """Count images that have at least one lesion mask available.

        Returns:
            Number of images for which get_lesion_masks returns non-None.
        """
        return sum(
            1 for i in range(len(self)) if self.get_lesion_masks(i) is not None
        )


class DDRDataset(BaseFundusDataset):
    """DDR dataset for DR grading (~13,673 images, 5-class DR).

    Supports loading train, test, and valid splits.

    Args:
        image_paths: Absolute paths to .jpg images.
        labels: DR grade labels (0–4).
        patient_ids: Filename stems used as patient IDs.
        preprocessing: Optional preprocessing callable.
        augmentation: Optional augmentation callable.
    """

    def __init__(
        self,
        image_paths: list[str],
        labels: list[int],
        patient_ids: list[str],
        preprocessing: Callable | None = None,
        augmentation: Callable | None = None,
    ) -> None:
        super().__init__(image_paths, labels, preprocessing, augmentation)
        self.patient_ids = patient_ids

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        split: str = "train",
        subset_indices: list[int] | None = None,
        preprocessing: Callable | None = None,
        augmentation: Callable | None = None,
    ) -> "DDRDataset":
        """Build dataset from DDR DR_grading directory layout.

        Args:
            root: Path to DR_grading/ directory (contains train/, test/,
                  valid/ subdirs and train.txt, test.txt, valid.txt).
            split: One of "train", "test", "valid".
            subset_indices: Optional list of row indices within the split file.
            preprocessing: Optional preprocessing callable.
            augmentation: Optional augmentation callable.

        Returns:
            Constructed DDRDataset.
        """
        root = Path(root)
        split_txt = root / f"{split}.txt"
        images_dir = root / split

        with open(split_txt, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        if subset_indices is not None:
            lines = [lines[i] for i in subset_indices]

        image_paths, labels, patient_ids = [], [], []
        for line in lines:
            parts = line.split()
            filename, grade = parts[0], int(parts[1])
            img_path = images_dir / filename
            if not img_path.exists():
                continue
            image_paths.append(str(img_path))
            labels.append(grade)
            patient_ids.append(Path(filename).stem)

        return cls(image_paths, labels, patient_ids, preprocessing, augmentation)


# ---------------------------------------------------------------------------
# DR keyword → grade mapping for ODIR-5K
# ---------------------------------------------------------------------------
_DR_KEYWORD_GRADE: list[tuple[str, int]] = [
    ("proliferative diabetic retinopathy", 4),
    ("very severe", 4),
    ("severe non proliferative retinopathy", 3),
    ("severe diabetic retinopathy", 3),
    ("moderate non proliferative retinopathy", 2),
    ("moderate diabetic retinopathy", 2),
    ("mild non proliferative retinopathy", 1),
    ("mild diabetic retinopathy", 1),
    ("diabetic retinopathy", 2),  # unqualified → treat as moderate
    ("laser spot", 4),            # laser treatment implies previous PDR
    ("non proliferative retinopathy", 2),
]


def _keyword_to_grade(keyword: str) -> int | None:
    """Map ODIR diagnostic keyword string to DR grade (0–4) or None.

    Args:
        keyword: Lowercased diagnostic keyword string for one eye.

    Returns:
        Estimated DR grade, or None if the keyword contains no DR signal.
    """
    kw = keyword.lower()
    for phrase, grade in _DR_KEYWORD_GRADE:
        if phrase in kw:
            return grade
    return None


class ODIR5KDataset(BaseFundusDataset):
    """ODIR-5K fundus dataset — DR subset (5-class DR grading).

    ODIR-5K is a bilateral multi-disease dataset. This class extracts only
    the DR-relevant eyes by filtering on diagnostic keywords, then maps
    keyword descriptions to DR grades 0–4.

    Non-DR eyes of patients that also have a DR eye are excluded to avoid
    label ambiguity.

    Args:
        image_paths: Absolute paths to .jpg images (left/right mixed).
        labels: DR grade labels (0–4).
        patient_ids: Patient ID strings (same for left and right eye).
        preprocessing: Optional preprocessing callable.
        augmentation: Optional augmentation callable.
    """

    def __init__(
        self,
        image_paths: list[str],
        labels: list[int],
        patient_ids: list[str],
        preprocessing: Callable | None = None,
        augmentation: Callable | None = None,
    ) -> None:
        super().__init__(image_paths, labels, preprocessing, augmentation)
        self.patient_ids = patient_ids

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        subset_indices: list[int] | None = None,
        preprocessing: Callable | None = None,
        augmentation: Callable | None = None,
    ) -> "ODIR5KDataset":
        """Build DR-subset dataset from ODIR-5K Training Set layout.

        Args:
            root: Path to the ODIR-5K Training Set directory (contains
                  Images/ and Annotation/).
            subset_indices: Optional list of sample indices after DR filtering.
            preprocessing: Optional preprocessing callable.
            augmentation: Optional augmentation callable.

        Returns:
            Constructed ODIR5KDataset with patient_ids attribute.
        """
        root = Path(root)
        images_dir = root / "Images"
        annotation_path = root / "Annotation" / "training annotation (English).xlsx"

        df = pd.read_excel(annotation_path, engine="openpyxl")

        image_paths, labels, patient_ids = [], [], []

        for _, row in df.iterrows():
            pid = str(int(row["ID"]))

            for side, fname_col, kw_col in [
                ("left", "Left-Fundus", "Left-Diagnostic Keywords"),
                ("right", "Right-Fundus", "Right-Diagnostic Keywords"),
            ]:
                keyword = str(row[kw_col]).lower().strip()
                grade = _keyword_to_grade(keyword)

                # Include only eyes with a DR signal; non-DR eyes → grade 0
                # Only include if D flag=1 or keyword maps to a grade
                is_dr_patient = int(row.get("D", 0)) == 1
                if not is_dr_patient and grade is None:
                    continue

                final_grade = grade if grade is not None else 0

                fname = str(row[fname_col]).strip()
                img_path = images_dir / fname
                if not img_path.exists():
                    continue

                image_paths.append(str(img_path))
                labels.append(final_grade)
                patient_ids.append(pid)

        if subset_indices is not None:
            image_paths = [image_paths[i] for i in subset_indices]
            labels = [labels[i] for i in subset_indices]
            patient_ids = [patient_ids[i] for i in subset_indices]

        return cls(image_paths, labels, patient_ids, preprocessing, augmentation)


class RFMiDDataset(BaseFundusDataset):
    """RFMiD (Retinal Fundus Multi-disease) dataset — binary DR labels.

    Uses the DR column (0/1) from the ground-truth CSV as the label:
      0 = No DR present
      1 = DR present (any severity)

    Each image corresponds to a unique patient; patient ID = image ID number.

    Args:
        image_paths: Absolute paths to .png images.
        labels: Binary DR labels (0 or 1).
        patient_ids: Image ID strings (e.g. "1", "42").
        preprocessing: Optional preprocessing callable.
        augmentation: Optional augmentation callable.
    """

    _SPLIT_DIRS: dict[str, tuple[str, str]] = {
        "train":      ("a. Training Set",   "a. RFMiD_Training_Labels.csv"),
        "validation": ("b. Validation Set", "b. RFMiD_Validation_Labels.csv"),
        "test":       ("c. Testing Set",    "c. RFMiD_Testing_Labels.csv"),
    }

    def __init__(
        self,
        image_paths: list[str],
        labels: list[int],
        patient_ids: list[str],
        preprocessing: Callable | None = None,
        augmentation: Callable | None = None,
    ) -> None:
        super().__init__(image_paths, labels, preprocessing, augmentation)
        self.patient_ids = patient_ids

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        split: str = "train",
        subset_indices: list[int] | None = None,
        preprocessing: Callable | None = None,
        augmentation: Callable | None = None,
    ) -> "RFMiDDataset":
        """Build dataset from RFMiD_All_Classes_Dataset/ directory layout.

        Args:
            root: Path to RFMiD_All_Classes_Dataset/ (contains
                  ``1. Original Images/`` and ``2. Groundtruths/``).
            split: One of "train", "validation", "test".
            subset_indices: Optional list of row indices within the CSV.
            preprocessing: Optional preprocessing callable.
            augmentation: Optional augmentation callable.

        Returns:
            Constructed RFMiDDataset with binary DR labels and patient_ids.

        Raises:
            ValueError: If split is not one of the recognised split names.
        """
        root = Path(root)
        if split not in cls._SPLIT_DIRS:
            raise ValueError(
                f"split must be one of {list(cls._SPLIT_DIRS)}, got {split!r}"
            )

        img_subdir, csv_name = cls._SPLIT_DIRS[split]
        images_dir = root / "1. Original Images" / img_subdir
        labels_csv = root / "2. Groundtruths" / csv_name

        df = pd.read_csv(labels_csv)
        if subset_indices is not None:
            df = df.iloc[subset_indices].reset_index(drop=True)

        image_paths, labels, patient_ids = [], [], []
        for _, row in df.iterrows():
            img_id   = int(row["ID"])
            dr_label = int(row["DR"])
            img_path = images_dir / f"{img_id}.png"
            if not img_path.exists():
                continue
            image_paths.append(str(img_path))
            labels.append(dr_label)
            patient_ids.append(str(img_id))

        return cls(image_paths, labels, patient_ids, preprocessing, augmentation)


# ---------------------------------------------------------------------------
# EyePACS patient-pair dataset (per-patient blending)
# ---------------------------------------------------------------------------

class EyePACSPatientPairDataset(Dataset):
    """EyePACS dataset returning ``(img_L, img_R, label)`` per patient.

    Each sample represents one patient; a zero tensor is used when an eye
    is absent, accompanied by a mask flag.  Intended for per-patient
    blending experiments.

    Args:
        patient_data: Dict mapping ``patient_id`` to a dict with keys
            ``"left_path"``, ``"right_path"``, ``"left_label"``,
            ``"right_label"`` (each is a ``str | None`` / ``int | None``).
        preprocessing: :class:`PreprocessingPipelineV5` applied to each eye.
        label_strategy: How to derive a single label per patient.
            ``"max"`` uses the maximum grade across both eyes (default).
    """

    def __init__(
        self,
        patient_data: dict[str, dict],
        preprocessing: "PreprocessingPipelineV5",
        label_strategy: str = "max",
    ) -> None:
        self.patient_data = patient_data
        self.patient_ids: list[str] = list(patient_data.keys())
        self.preprocessing = preprocessing
        self.label_strategy = label_strategy

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> dict:
        """Return one patient sample.

        Returns:
            Dict with keys:

            - ``"left"``         — float32 tensor ``(3, H, W)``; zero if absent
            - ``"right"``        — float32 tensor ``(3, H, W)``; zero if absent
            - ``"label"``        — int DR grade (0–4)
            - ``"left_mask"``    — 1 if left eye present, else 0
            - ``"right_mask"``   — 1 if right eye present, else 0
            - ``"patient_id"``   — str
        """
        pid = self.patient_ids[idx]
        data = self.patient_data[pid]

        target_size = self.preprocessing.config.target_size
        zero = torch.zeros(3, target_size, target_size, dtype=torch.float32)

        def _load(path: str | None, side: str) -> tuple[torch.Tensor, int]:
            if path is None:
                return zero, 0
            img = cv2.imread(path)
            if img is None:
                return zero, 0
            return self.preprocessing(img, eye_side=side), 1

        left_tensor, left_mask = _load(data.get("left_path"), "left")
        right_tensor, right_mask = _load(data.get("right_path"), "right")

        # Derive label
        left_label: int | None = data.get("left_label")
        right_label: int | None = data.get("right_label")
        grades = [g for g in (left_label, right_label) if g is not None]
        label = max(grades) if grades else 0

        return {
            "left": left_tensor,
            "right": right_tensor,
            "label": label,
            "left_mask": left_mask,
            "right_mask": right_mask,
            "patient_id": pid,
        }

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        labels_csv: str | Path,
        subset_indices: list[int] | None = None,
        preprocessing: "PreprocessingPipelineV5 | None" = None,
    ) -> "EyePACSPatientPairDataset":
        """Build a patient-pair dataset from the EyePACS directory layout.

        Rows are optionally filtered by *subset_indices* first, then grouped
        by patient ID.

        Args:
            root: Path to the train/ image directory.
            labels_csv: Path to ``trainLabels.csv`` (columns: image, level).
            subset_indices: Optional list of CSV row indices to include.
            preprocessing: :class:`PreprocessingPipelineV5` instance.
                Defaults to an inference-mode pipeline with default config.

        Returns:
            :class:`EyePACSPatientPairDataset` instance.
        """
        from src.preprocessing.config import PreprocessingV5Config
        from src.preprocessing.pipeline_v5 import PreprocessingPipelineV5  # noqa: PLC0415

        root = Path(root)
        df = pd.read_csv(labels_csv)
        if subset_indices is not None:
            df = df.iloc[subset_indices].reset_index(drop=True)

        if preprocessing is None:
            preprocessing = PreprocessingPipelineV5.create_for_inference(
                PreprocessingV5Config()
            )

        patient_data: dict[str, dict] = {}
        for _, row in df.iterrows():
            name: str = str(row["image"])
            img_path = root / f"{name}.jpeg"
            grade = int(row["level"])
            pid = name.split("_")[0]
            side = "left" if "_left" in name else "right"

            if pid not in patient_data:
                patient_data[pid] = {
                    "left_path": None,
                    "right_path": None,
                    "left_label": None,
                    "right_label": None,
                }

            path_str = str(img_path) if img_path.exists() else None
            patient_data[pid][f"{side}_path"] = path_str
            patient_data[pid][f"{side}_label"] = grade

        return cls(patient_data, preprocessing)


def patient_pair_collate(batch: list[dict]) -> dict:
    """Collate a list of patient-pair sample dicts into batched tensors.

    Args:
        batch: List of dicts as returned by
            :meth:`EyePACSPatientPairDataset.__getitem__`.

    Returns:
        Dict with the same keys; tensor values are stacked along dim 0;
        ``"patient_id"`` is a list of strings.
    """
    return {
        "left":        torch.stack([b["left"] for b in batch]),
        "right":       torch.stack([b["right"] for b in batch]),
        "label":       torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "left_mask":   torch.tensor([b["left_mask"] for b in batch], dtype=torch.long),
        "right_mask":  torch.tensor([b["right_mask"] for b in batch], dtype=torch.long),
        "patient_id":  [b["patient_id"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Dataset registry — maps dataset name to class for factory access
# ---------------------------------------------------------------------------
DATASET_REGISTRY: dict[str, type] = {
    "eyepacs":   EyePACSDataset,
    "aptos2019": APTOS2019Dataset,
    "idrid":     IDRiDDataset,
    "ddr":       DDRDataset,
    "odir5k":    ODIR5KDataset,
    "rfmid":     RFMiDDataset,
}
