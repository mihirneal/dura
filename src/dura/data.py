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


class NiftiDataset(Dataset):
    """Raw nii.gz images, preprocessed on the fly by the model's transform.

    Each sample is a dict with "subject", "image" (a nii.gz path) and either a scalar "target"
    or a "label" nii.gz path for segmentation. Labels are mapped to a binary array by label_fn
    and resampled (nearest) onto the transformed image grid as "target". Their native geometry
    is kept as "label_shape", "label_affine" and "label_positive" (flat indices of the
    positive voxels) so predictions can be scored in native space.
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],
        transform: ModelTransform,
        label_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self.samples = samples
        self.transform = transform
        self.label_fn = label_fn

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        transformed = self.transform(nib.load(sample["image"]))
        image = torch.as_tensor(transformed["image"], dtype=torch.float32)
        affine = np.asarray(transformed["affine"], dtype=np.float64)
        item = {
            "subject": sample["subject"],
            "image": image,
            "mask": torch.as_tensor(transformed["mask"], dtype=torch.float32),
            "affine": affine,
        }
        if "label" in sample:
            label_img = nib.load(sample["label"])
            label = self.label_fn(np.asarray(label_img.dataobj)).astype(np.uint8)
            label_img = nib.Nifti1Image(label, label_img.affine)
            fit_label = resample_from_to(label_img, (image.shape[-3:], affine), order=0)
            item["target"] = torch.as_tensor(np.asarray(fit_label.dataobj), dtype=torch.uint8)
            item["label_shape"] = label.shape
            item["label_affine"] = label_img.affine
            item["label_positive"] = np.flatnonzero(label)
        else:
            item["target"] = torch.as_tensor(sample["target"])
        return item


def fit_transform(transform: ModelTransform, samples: list[dict[str, Any]]) -> None:
    """Fit the transform's global parameters on the task's raw images, if it has any."""
    if hasattr(transform, "fit"):
        transform.fit(nib.load(sample["image"]) for sample in samples)


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack the tensors, keep everything else (native label geometry, affines) as lists."""
    return {
        key: torch.stack([item[key] for item in batch])
        if isinstance(batch[0][key], torch.Tensor)
        else [item[key] for item in batch]
        for key in batch[0]
    }
