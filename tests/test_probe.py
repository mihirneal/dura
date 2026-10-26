import nibabel as nib
import numpy as np
import pytest
import torch
from nibabel.processing import resample_from_to
from torch import Tensor

import dura.probe
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
N_BINS = 64
BIN_WIDTH = 10.0
EMBED_DIM = 8**3 + 2 * N_BINS


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
    """Each patch's voxels, plus its mean and max intensity as soft one-hots over intensity bins:
    the attention probe layer-norms each token, which removes a level shared by its voxels."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.register_buffer("bins", torch.linspace(0, 500, N_BINS))

    def dense_embed(self, images: Tensor, mask: Tensor) -> Tensor:
        patches = patchify3d(images, PATCH_SIZE)
        levels = [patches.mean(dim=2, keepdim=True), patches.amax(dim=2, keepdim=True)]
        bins = [torch.exp(-(((level - self.bins) / BIN_WIDTH) ** 2)) for level in levels]
        grid = [size // patch for size, patch in zip(MODEL_SHAPE, PATCH_SIZE)]
        return self.scale * torch.cat([patches, *bins], dim=2).reshape(len(images), *grid, -1)


@pytest.fixture(autouse=True)
def fast_light(monkeypatch):
    # few subjects are one optimizer step per epoch: a higher learning rate trains in time
    monkeypatch.setattr(dura.probe, "LIGHT_LR", 1e-2)
    monkeypatch.setattr(dura.probe, "LIGHT_SEEDS", (0,))


def test_probe_binary_classification(samples):
    record, _ = probe_binary_classification(
        DummyEncoder().cuda(), DummyTransform(), with_target(samples, "class")
    )
    assert record["auroc"] > 0.95
    assert record["n_samples"] == 40
    assert record["embed_dim"] == EMBED_DIM
    assert len(record["probabilities"]) == 40
    assert len(record["fold_choice"]) == 5
    assert {"auroc_ci", "balanced_accuracy", "probe_seconds", "embed_seconds"} <= set(record)


def test_probe_binary_classification_random_labels(samples):
    record, _ = probe_binary_classification(
        DummyEncoder().cuda(), DummyTransform(), with_target(samples, "random_class")
    )
    # no signal to find: not above chance. Below chance is just noise, at 40 subjects
    assert record["auroc_ci"][0] < 0.5


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
    # 8mm radius blobs, 2.1 ml
    assert np.allclose(record["subject_label_volume_ml"], 4 / 3 * np.pi * 0.8**3, rtol=0.05)
    # no bias overall, though one threshold over- and under-segments blobs of either contrast
    volume_ratio = np.mean(record["subject_volume_ml"]) / np.mean(record["subject_label_volume_ml"])
    assert abs(volume_ratio - 1) < 0.1
    assert record["embed_dim"] == EMBED_DIM
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
