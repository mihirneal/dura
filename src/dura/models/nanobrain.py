"""
nanobrain model wrapper

nanobrain ViT MAEs (https://github.com/MedARC-AI/nanobrain), loaded from a training
checkpoint. Install the optional nanobrain extra.

- nanobrain: pretrained checkpoint
- nanobrain_random: the checkpoint's architecture with random init, a no-pretraining baseline

References:
    nanobrain/src/nanobrain/eval/preprocessing.py
    nanobrain/src/nanobrain/eval/prepare.py
"""

import importlib
import math

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import torch
from jaxtyping import Float
from nibabel.processing import resample_from_to, resample_to_output, smooth_image
from torch import Tensor, nn

from dura.models.base import ModelTransform, ModelWrapper
from dura.models.registry import register_model

try:
    import nanobrain.model
except ImportError as exc:
    raise ImportError(
        "nanobrain not installed. Please install the optional nanobrain extra."
    ) from exc

GRID_FOV = (192.0, 240.0, 192.0)
VMIN_QUANTILE = 0.005
VMAX_QUANTILE = 0.995
DEFAULT_MODEL_CLASS = "nanobrain.model:ViTMAE3D"


class NanobrainTransform(ModelTransform):
    """
    Output: image [192 240 192], raw values in [0, 2**16) zero outside the head mask, and the
    head mask on the same grid.
    """

    def __init__(self, grid_fov: tuple[float, float, float] = GRID_FOV):
        self.grid_fov = grid_fov

    def __call__(self, img: nib.Nifti1Image) -> dict:
        mask_img = threshold_mask(img)
        fit_img = conform_image(img, mask_img, max_fov=self.grid_fov, fixed_grid=True)
        fit_mask = np.asarray(resample_from_to(mask_img, fit_img, order=0).dataobj) > 0
        image = truncate_image(fit_img.get_fdata(dtype=np.float32), fit_mask)
        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(fit_mask.astype(np.float32)),
            "affine": fit_img.affine,
        }


class NanobrainModelWrapper(ModelWrapper):
    """Input [B 192 240 192], dense output on the 8mm patch grid [B 24 30 24 D]."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def dense_embed(
        self, images: Float[Tensor, "B X Y Z"], mask: Float[Tensor, "B X Y Z"]
    ) -> Float[Tensor, "B Gx Gy Gz D"]:
        return self.backbone.dense_embed(images, mask)

    def global_embed(
        self, images: Float[Tensor, "B X Y Z"], mask: Float[Tensor, "B X Y Z"]
    ) -> Float[Tensor, "B D"]:
        return self.backbone.global_embed(images, mask)


def build_backbone(ckpt_path: str, load_weights: bool = True) -> nn.Module:
    """Build the checkpoint's model from its training config. Older checkpoints have no
    model_class and are all ViTMAE3D."""
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    module_name, class_name = checkpoint.get("model_class", DEFAULT_MODEL_CLASS).split(":")
    model_class = getattr(importlib.import_module(module_name), class_name)
    model = model_class.from_config(checkpoint["args"])
    if load_weights:
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def nanobrain(ckpt_path: str) -> tuple[NanobrainTransform, NanobrainModelWrapper]:
    """A pretrained nanobrain checkpoint."""
    return NanobrainTransform(), NanobrainModelWrapper(build_backbone(ckpt_path))


@register_model
def nanobrain_random(
    ckpt_path: str, seed: str = "42"
) -> tuple[NanobrainTransform, NanobrainModelWrapper]:
    """The checkpoint's architecture with random init weights, as a no-pretraining baseline."""
    torch.manual_seed(int(seed))
    backbone = build_backbone(ckpt_path, load_weights=False)
    return NanobrainTransform(), NanobrainModelWrapper(backbone)


def truncate_image(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Clip to the mask's intensity quantiles and scale to [0, 2**16) as uint16 values, zero
    outside the mask."""
    values = data[mask]
    vmin, vmax = np.quantile(values, (VMIN_QUANTILE, VMAX_QUANTILE))
    assert vmax > vmin, "degenerate value range"
    values = np.clip((values - vmin) / (vmax - vmin), 0, 1)
    values = (values * (2**16 - 1)).astype(np.uint16)
    out = np.zeros(data.shape, dtype=np.float32)
    out[mask] = values
    return out


def threshold_mask(
    img: nib.Nifti1Image,
    resolution: float | None = 2.0,
    sigma: float | None = 4.0,
    remove_islands: bool = True,
    fill_holes: bool = True,
) -> nib.Nifti1Image:
    """Threshold based head masking.

    Args:
        resolution: target resolution to compute the mask at.
        sigma: gaussian smoothing sigma in mm.
        remove_islands: keep only largest connected component.
        fill_holes: fill holes enclosed by the mask.

    Returns:
        mask nibabel image

    Notes:
        The recipe of smoothing + mean threshold seemed pretty robust on OpenNeuro test
        images. Also tried median filtering instead of gaussian, otsu/minimum threshold
        instead of mean, which weren't any better.

    References:
        https://github.com/dipy/dipy/blob/1.12.1/dipy/segment/mask.py#L132
    """
    orig_img = img

    if resolution is not None:
        voxel_sizes = 3 * (resolution,)
        img = resample_to_output(img, voxel_sizes=voxel_sizes, order=1)

    data = img.get_fdata(dtype=np.float32)

    # filter to remove high frequency noise
    if sigma is not None:
        spacing = np.array(img.header.get_zooms()[:3])
        voxel_sigmas = sigma / spacing
        data = ndi.gaussian_filter(data, sigma=voxel_sigmas)

    # threshold, simple mean works fine, esp after smoothing
    mask = data > data.mean()

    # morphology cleanup
    if remove_islands:
        mask = largest_component(mask)
    if fill_holes:
        mask = ndi.binary_fill_holes(mask)

    mask_img = nib.Nifti1Image(mask.astype(np.uint8), img.affine)
    mask_img = resample_from_to(mask_img, orig_img, order=0)
    return mask_img


def largest_component(mask: np.ndarray):
    label, count = ndi.label(mask)
    if count > 1:
        sizes = ndi.sum_labels(mask, label, range(1, count + 1))
        mask = label == (np.argmax(sizes) + 1)
    return mask


def conform_image(
    img: nib.Nifti1Image,
    mask_img: nib.Nifti1Image,
    min_voxel_size: float = 1.0,
    max_fov: tuple[float, float, float] = (208.0, 240.0, 208.0),
    top_margin: float = 5.0,
    order: int = 1,
    fixed_grid: bool = False,
):
    """Conform image to target minimum voxel size and max FOV.

    The x/y crop is centered on the head mask. The z crop is taken from the top of the
    head mask (plus top_margin mm) down.

    With fixed_grid, resample to exactly min_voxel_size spacing on a grid spanning max_fov,
    zero padding where the image does not reach.

    References:
        https://github.com/nipy/nibabel/blob/5.4.2/nibabel/processing.py#L318
    """
    assert img.ndim == 3, f"expected 3D image, got {img.ndim}"
    img = nib.as_closest_canonical(img)
    # resampling keeps the input dtype, so integer images would be rounded after interpolation
    # nb, this is a deviation from the preprocessing currently implemented for pretraining
    img = nib.Nifti1Image(img.get_fdata(dtype=np.float32), img.affine)
    mask_img = nib.as_closest_canonical(mask_img)

    voxel_sizes = img.header.get_zooms()
    if fixed_grid:
        new_voxel_sizes = 3 * [min_voxel_size]
        new_fov = list(max_fov)
    else:
        new_voxel_sizes = [max(min_voxel_size, sz) for sz in voxel_sizes]
        fov = [w * sz for w, sz in zip(img.shape, voxel_sizes)]
        new_fov = [min(w_, w) for w_, w in zip(max_fov, fov)]
    new_shape = [math.ceil(w / sz) for w, sz in zip(new_fov, new_voxel_sizes)]

    # smooth before downsampling. upsampled axes (fixed grid only) get no smoothing
    if any(sz_ > sz for sz_, sz in zip(new_voxel_sizes, voxel_sizes)):
        fwhm = [
            math.sqrt(max(sz_**2 - sz**2, 0.0)) for sz_, sz in zip(new_voxel_sizes, voxel_sizes)
        ]
        img = smooth_image(img, fwhm=fwhm)

    # initialize voxel center on mask bbox center
    mask = np.asarray(mask_img.dataobj) > 0
    mask_x_ids = np.nonzero(mask.any(axis=(1, 2)))[0]
    mask_y_ids = np.nonzero(mask.any(axis=(0, 2)))[0]
    mask_z_ids = np.nonzero(mask.any(axis=(0, 1)))[0]
    centroid = np.array(
        [round((ids.min() + ids.max()) / 2) for ids in [mask_x_ids, mask_y_ids, mask_z_ids]]
    )

    # anchor z to the top of the mask to cut out neck rather than brain
    top = mask_z_ids.max() + top_margin / voxel_sizes[2]
    top = min(top, img.shape[2])
    half_height = new_fov[2] / voxel_sizes[2] / 2
    centroid[2] = round(top - half_height)

    # resample with cropping
    new_affine = rescale_affine(
        img.affine, img.shape, new_voxel_sizes, new_shape=new_shape, centroid=centroid
    )
    new_img = resample_from_to(img, (new_shape, new_affine), order=order)
    return new_img


def rescale_affine(
    affine: np.ndarray,
    shape: tuple[int, int, int],
    zooms: tuple[float, float, float],
    new_shape: tuple[int, int, int] | None = None,
    centroid: tuple[int, int, int] | None = None,
):
    """Return a new affine matrix with updated voxel sizes.

    Accepts a new target shape (defaulting to input shape) and source centroid
    (defaulting to input voxel grid center).

    Reference:
        nibabel.affines.rescale_affine
    """
    shape = np.asarray(shape)
    new_shape = np.array(new_shape if new_shape is not None else shape)
    centroid = np.array(centroid if centroid is not None else (shape - 1) // 2)

    s = nib.affines.voxel_sizes(affine)
    rzs_out = affine[:3, :3] * zooms / s

    # Using xyz = A @ ijk, determine translation
    centroid = nib.affines.apply_affine(affine, centroid)
    t_out = centroid - rzs_out @ ((new_shape - 1) // 2)
    return nib.affines.from_matvec(rzs_out, t_out)
