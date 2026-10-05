from collections.abc import Callable, Iterable
from typing import Any

import nibabel as nib
from jaxtyping import Float
from torch import Tensor, nn


class ModelWrapper(nn.Module):
    """
    Wrap an sMRI encoder model. Takes a batch of transformed images and returns embeddings.
    """

    def global_embed(
        self,
        images: Float[Tensor, "B ... X Y Z"],
        mask: Float[Tensor, "B X Y Z"],
    ) -> Float[Tensor, "B D"]: ...

    def dense_embed(
        self,
        images: Float[Tensor, "B ... X Y Z"],
        mask: Float[Tensor, "B X Y Z"],
    ) -> Float[Tensor, "B Gx Gy Gz C"]:
        """Patch embeddings. The patch grid must tile X Y Z exactly, each patch covering
        (X/Gx, Y/Gy, Z/Gz) voxels."""


class ModelTransform:
    """
    Model specific data transform. dura loads the released nii.gz as is, and the transform
    does all of the model's preprocessing (reorienting, resampling, cropping, masking,
    intensity normalization) on the fly. It runs in data loader workers, so it must not touch
    the GPU, and its outputs are batched, so they must have the same shape for every image.
    """

    def __call__(self, img: nib.Nifti1Image) -> dict[str, Any]:
        """Preprocess one raw image. Returns a dict with:

        - "image": [X Y Z] or [C X Y Z] tensor, the model input
        - "mask": [X Y Z] tensor, the foreground. Dense probes skip patches with no foreground
        - "affine": [4 4] array, the voxel to world affine of the X Y Z grid. Labels are
          resampled onto this grid for training, and predictions are mapped back through it to
          score in the labels' native space
        """

    def fit(self, images: Iterable[nib.Nifti1Image]) -> None:
        """
        Precompute global transform parameters (e.g. intensity statistics) on a task's raw
        images. Label free, so it sees every subject of the task.

        Optional, doesn't have to be defined.
        """


ModelTransformPair = tuple[ModelTransform, ModelWrapper]

ModelFn = Callable[..., ModelTransformPair]
