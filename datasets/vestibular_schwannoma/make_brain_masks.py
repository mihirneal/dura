"""Brain masks for the vestibular_schwannoma scans from SynthStrip (mri_synthstrip, FreeSurfer's or
the freesurfer/synthstrip:1.8 image's, on PATH) on CPU, widened by 10 mm and cut to the scan's
nonzero voxels with their holes filled. The tumor sits in the ear canal and the cistern beside the
brainstem, up to 8 mm outside SynthStrip's mask, and the widening keeps all of it. Written on the
scan's grid to DATA_ROOT/brain_masks/vestibular_schwannoma/images/<sub>/<ses>/t1c.nii.gz, where
the models' transforms read them. Existing masks are kept. --shard i --shards n strips every n-th
scan from the i-th, to split the work across jobs.

    uv run python datasets/vestibular_schwannoma/make_brain_masks.py [--shard i --shards n]
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from dura.data import DATA_ROOT

MARGIN_MM = 10.0


def write_mask(image: Path) -> None:
    out = DATA_ROOT / "brain_masks" / image.relative_to(DATA_ROOT)
    with tempfile.TemporaryDirectory() as tmp:
        stripped = Path(tmp) / "mask.nii.gz"
        subprocess.run(
            ["mri_synthstrip", "-i", str(image), "-m", str(stripped)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        img = nib.load(image)
        nonzero = ndimage.binary_fill_holes(np.asarray(img.dataobj) != 0)
        brain = np.asarray(nib.load(stripped).dataobj) > 0
        zooms = img.header.get_zooms()[:3]
        mask = (ndimage.distance_transform_edt(~brain, sampling=zooms) <= MARGIN_MM) & nonzero
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name.replace(".nii.gz", ".tmp.nii.gz"))
    nib.Nifti1Image(mask.astype(np.uint8), img.affine).to_filename(tmp)
    tmp.rename(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()

    root = DATA_ROOT / "vestibular_schwannoma" / "images"
    images = sorted(root.glob("*/ses-*/t1c.nii.gz"))
    assert images, f"no scans in {root}"
    for image in images[args.shard :: args.shards]:
        if not (DATA_ROOT / "brain_masks" / image.relative_to(DATA_ROOT)).exists():
            write_mask(image)
    print(f"{len(images[args.shard :: args.shards])} scans -> {DATA_ROOT / 'brain_masks'}")
