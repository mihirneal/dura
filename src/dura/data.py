import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to
from torch.utils.data import Dataset

from dura.models.base import ModelTransform

# released datasets in DATA_ROOT/<name>, see datasets/<name>/make_dataset.py
DATA_ROOT = Path(os.getenv("DURA_DATA_ROOT", "/data/mihir-stuff/dura"))


class ImageDataset(Dataset):
    """Raw nii.gz images, preprocessed on the fly by the model's transform."""

    def __init__(self, images: list[Path], transform: ModelTransform):
        self.images = images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, Any]:
        transformed = self.transform(nib.load(self.images[index]))
        return {
            "image": torch.as_tensor(transformed["image"], dtype=torch.float32),
            "mask": torch.as_tensor(transformed["mask"], dtype=torch.float32),
            "affine": np.asarray(transformed["affine"], dtype=np.float64),
        }


class LabelDataset(Dataset):
    """
    Segmentation labels (nii.gz paths), mapped to a binary array by label_fn and resampled
    (nearest) onto each image's model grid (shape, affine) as "target". Their native geometry is
    kept as "label_shape", "label_affine" and "label_positive" (flat indices of the positive
    voxels) so predictions can be scored in native space.
    """

    def __init__(
        self,
        labels: list[Path],
        grids: list[tuple[tuple[int, ...], np.ndarray]],
        label_fn: Callable[[np.ndarray], np.ndarray],
    ):
        self.labels = labels
        self.grids = grids
        self.label_fn = label_fn

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        label_img = nib.load(self.labels[index])
        label = self.label_fn(np.asarray(label_img.dataobj)).astype(np.uint8)
        label_img = nib.Nifti1Image(label, label_img.affine)
        fit_label = resample_from_to(label_img, self.grids[index], order=0)
        return {
            "target": torch.as_tensor(np.asarray(fit_label.dataobj), dtype=torch.uint8),
            "label_shape": label.shape,
            "label_affine": label_img.affine,
            "label_positive": np.flatnonzero(label),
        }


def fit_transform(transform: ModelTransform, images: list[Path]) -> None:
    """Fit the transform's global parameters on the task's raw images, if it has any."""
    if hasattr(transform, "fit"):
        transform.fit(nib.load(image) for image in images)


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack the tensors, keep everything else (native label geometry, affines) as lists."""
    return {
        key: torch.stack([item[key] for item in batch])
        if isinstance(batch[0][key], torch.Tensor)
        else [item[key] for item in batch]
        for key in batch[0]
    }
