"""Build the ADNI eval data from manifest.tsv.

The nii.gz stay exactly as converted from the release, symlinked in as
images/sub-<id>/ses-<date>/<name>. Each task gets <task>.tsv: the subject, its image path relative
to the output dir, and the target, recomputed from the tables (see adni.py).

    uv run --group datasets python datasets/adni/make_dataset.py /data/smri-datasets/ADNI \
        /data/mihir-stuff/dura/adni
"""

import argparse
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from adni import cohorts, image_path
from nibabel.filebasedimages import ImageFileError

MANIFEST = Path(__file__).parent / "manifest.tsv"


def read_error(path: Path) -> str | None:
    """Decompress the whole image, so a truncated download fails here rather than mid eval."""
    try:
        np.asarray(nib.load(path).dataobj)
    except (OSError, EOFError, ValueError, zlib.error, ImageFileError) as exc:
        return repr(exc)
    return None


def link_path(name: str) -> str:
    sub, ses = name.split("_")[:2]
    return f"images/{sub}/{ses}/{name}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_root", type=Path, help="ADNI release, with nifti/ and metadata/")
    parser.add_argument("out_root", type=Path)
    args = parser.parse_args()

    assert not args.out_root.exists(), f"{args.out_root} exists, delete it first"
    manifest = pd.read_csv(MANIFEST, sep="\t", dtype=str, keep_default_na=False)
    tables = cohorts(args.raw_root)
    for task, table in tables.items():
        frozen = manifest[manifest.task == task].drop(columns="task").to_numpy()
        current = table.drop(columns="target").astype(str).to_numpy()
        assert frozen.shape == current.shape and (frozen == current).all(), (
            f"{task} no longer matches {MANIFEST}: the release or its tables changed"
        )

    names = sorted(set(manifest.image))
    with ThreadPoolExecutor(16) as executor:
        errors = list(executor.map(lambda name: read_error(image_path(args.raw_root, name)), names))
    unreadable = {name: error for name, error in zip(names, errors) if error}
    assert not unreadable, f"unreadable images: {unreadable}"

    for name in names:
        link = args.out_root / link_path(name)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(image_path(args.raw_root, name))
    for task, table in tables.items():
        table = table.assign(image=table.image.map(link_path))
        table.to_csv(args.out_root / f"{task}.tsv", sep="\t", index=False)
    print(f"{len(names)} images, {manifest.subject.nunique()} subjects -> {args.out_root}")
