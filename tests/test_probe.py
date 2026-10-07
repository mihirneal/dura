import nibabel as nib
import numpy as np
import pytest
import torch
from nibabel.processing import resample_from_to
from torch import Tensor

from dura.models.base import ModelTransform, ModelWrapper
from dura.probe import (
    patchify3d,
    probe_binary_classification,
    probe_binary_segmentation,
    probe_multiclass_classification,
    probe_regression,
    to_native,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="probes need cuda")

# native images are LAS with 2mm slices, the dummy model works on a RAS 1mm grid
NATIVE_SHAPE = (64, 64, 32)
NATIVE_AFFINE = np.array([[-1, 0, 0, 63], [0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]], dtype=float)
MODEL_SHAPE = (64, 64, 64)
PATCH_SIZE = (8, 8, 8)


@pytest.fixture(scope="module")
def samples(tmp_path_factory) -> list[dict]:
    root = tmp_path_factory.mktemp("data")
    n_samples = 40
    rng = np.random.default_rng(0)
    labels = rng.permutation(np.arange(n_samples) % 2)
    random_labels = rng.permutation(np.arange(n_samples) % 2)
    ages = rng.uniform(20, 80, n_samples)
    centers = rng.uniform(-12, 12, (n_samples, 3)) + 32

    # voxel centers in mm
    x, y, z = np.meshgrid(*[np.arange(size) for size in NATIVE_SHAPE], indexing="ij")
    x, z = 63 - x, 2 * z
    head = ((x - 32) / 28) ** 2 + ((y - 32) / 28) ** 2 + ((z - 32) / 28) ** 2 < 1

    samples = []
    for ii in range(n_samples):
        blob = (x - centers[ii, 0]) ** 2 + (y - centers[ii, 1]) ** 2 + (z - centers[ii, 2]) ** 2
        blob = blob < 8**2
        image = np.random.default_rng([0, ii]).normal(100 + ages[ii], 10, NATIVE_SHAPE)
        image[blob] += 200 if labels[ii] else 100
        image = (image * head).astype(np.float32)
        image_path = root / f"{ii}_image.nii.gz"
        label_path = root / f"{ii}_label.nii.gz"
        nib.save(nib.Nifti1Image(image, NATIVE_AFFINE), image_path)
        nib.save(nib.Nifti1Image(blob.astype(np.uint8), NATIVE_AFFINE), label_path)
        samples.append(
            {
                "subject": str(ii),
                "image": image_path,
                "label": label_path,
                "class": labels[ii],
                "random_class": random_labels[ii],
                "age": ages[ii],
                "age_group": int(np.digitize(ages[ii], [40, 60])),
            }
        )
    return samples


def with_target(samples: list[dict], key: str) -> list[dict]:
    return [{"subject": s["subject"], "image": s["image"], "target": s[key]} for s in samples]


class DummyTransform(ModelTransform):
    """Reorient to RAS and resample to 1mm."""

    def __call__(self, img: nib.Nifti1Image) -> dict:
        img = nib.as_closest_canonical(img)
        affine = img.affine @ np.diag([1.0, 1.0, 0.5, 1.0])
        img = resample_from_to(img, (MODEL_SHAPE, affine), order=1)
        image = np.asarray(img.dataobj, dtype=np.float32)
        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy((image > 0).astype(np.float32)),
            "affine": affine,
        }


class DummyEncoder(ModelWrapper):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        quantiles = torch.tensor([0.5, 0.9, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999, 1.0])
        self.register_buffer("quantiles", quantiles)

    def global_embed(self, images: Tensor, mask: Tensor) -> Tensor:
        embeddings = []
        for image, image_mask in zip(images, mask):
            values = image[image_mask > 0].float()
            embeddings.append(torch.quantile(values, self.quantiles))
        return self.scale * torch.stack(embeddings)

    def dense_embed(self, images: Tensor, mask: Tensor) -> Tensor:
        patches = patchify3d(images, PATCH_SIZE)
        grid = [size // patch for size, patch in zip(MODEL_SHAPE, PATCH_SIZE)]
        return self.scale * patches.reshape(len(images), *grid, -1)


def test_probe_binary_classification(samples):
    record, _ = probe_binary_classification(
        DummyEncoder().cuda(), DummyTransform(), with_target(samples, "class")
    )
    assert record["auroc"] > 0.95
    assert record["n_samples"] == 40
    assert record["embed_dim"] == 9
    assert len(record["probabilities"]) == 40


def test_probe_binary_classification_random_labels(samples):
    record, _ = probe_binary_classification(
        DummyEncoder().cuda(), DummyTransform(), with_target(samples, "random_class")
    )
    low, high = record["auroc_ci"]
    assert low < 0.5 < high


def test_probe_regression(samples):
    record, _ = probe_regression(
        DummyEncoder().cuda(), DummyTransform(), with_target(samples, "age")
    )
    assert record["r"] > 0.9
    assert record["mae"] < 10


def test_probe_multiclass_classification(samples):
    record, _ = probe_multiclass_classification(
        DummyEncoder().cuda(), DummyTransform(), with_target(samples, "age_group")
    )
    assert record["auroc"] > 0.9
    assert len(record["class_auroc"]) == 3
    assert np.array(record["probabilities"]).shape == (40, 3)


def test_probe_binary_segmentation(samples):
    samples = [
        {"subject": s["subject"], "image": s["image"], "label": s["label"]} for s in samples[:20]
    ]
    record, state = probe_binary_segmentation(
        DummyEncoder().cuda(), DummyTransform(), samples, label_fn=lambda x: x
    )
    # dice is limited by partial volume at the blob edge, upsampled from 2mm slices
    assert record["model_grid_dice"] > 0.85
    # scored on the native 2mm LAS grid
    assert record["dice"] > 0.85
    assert record["average_precision"] > 0.9
    assert record["voxel_auroc"] > 0.9
    assert record["embed_dim"] == 512
    assert len(record["subject_dice"]) == 20
    assert state["geometry"][0]["label_shape"] == NATIVE_SHAPE


def test_to_native_flip():
    # model grid RAS, native grid LAS: mapping back flips x
    grid_shape, patch_size = (2, 2, 2), (4, 4, 4)
    probabilities = torch.rand(8, 64)
    volume = probabilities.reshape(2, 2, 2, 4, 4, 4).permute(0, 3, 1, 4, 2, 5).reshape(8, 8, 8)
    geometry = {
        "affine": np.eye(4),
        "grid_shape": grid_shape,
        "patch_size": patch_size,
        "label_shape": (8, 8, 8),
        "label_affine": np.array([[-1, 0, 0, 7], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        "label_positive": np.array([0, 5]),
    }
    native, labels = to_native(probabilities, torch.ones(8, dtype=torch.bool), geometry)
    assert torch.allclose(native, volume.flip(0).flatten(), atol=1e-5)
    assert labels.nonzero().flatten().tolist() == [0, 5]
