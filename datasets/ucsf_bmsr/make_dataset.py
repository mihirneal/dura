"""Build the ucsf_bmsr eval data.

The visits in manifest.txt, one per patient. Visit 100101A is stored as sub-100101/ses-A. The
nii.gz stay exactly as released, symlinked in: T1post (as t1c), FLAIR and the BraTS labels
(1 non-enhancing core, 2 edema, 3 enhancing tumor), all on the T1post grid. labels.json has the
visit id and the edema and enhancing tumor volumes.

    uv run python datasets/ucsf_bmsr/make_dataset.py /data/smri-datasets/UCSF-BMSR \
        /data/mihir-stuff/dura/ucsf_bmsr
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

MANIFEST = Path(__file__).parent / "manifest.txt"
IMAGES = {"t1c.nii.gz": "T1post", "flair.nii.gz": "FLAIR"}


def add_subject(raw_dir: Path, out_root: Path, visit: str):
    sub, ses = f"sub-{visit[:-1]}", f"ses-{visit[-1]}"
    img_dir = out_root / "images" / sub / ses
    label_dir = out_root / "labels" / sub / ses
    img_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for name, source in IMAGES.items():
        path = raw_dir / visit / f"{visit}_{source}.nii.gz"
        assert path.exists(), f"missing {path}"
        (img_dir / name).symlink_to(path)
    seg = raw_dir / visit / f"{visit}_BraTS-seg.nii.gz"
    (label_dir / "seg.nii.gz").symlink_to(seg)

    seg_img = nib.load(seg)
    voxel_ml = np.prod(seg_img.header.get_zooms()[:3]) / 1000
    labels = np.asarray(seg_img.dataobj).round()
    info = {
        "visit": visit,
        "edema_ml": float((labels == 2).sum() * voxel_ml),
        "enhancing_ml": float((labels == 3).sum() * voxel_ml),
    }
    (label_dir / "labels.json").write_text(json.dumps(info) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_root", type=Path, help="extracted UCSF-BMSR v1.3 release")
    parser.add_argument("out_root", type=Path)
    args = parser.parse_args()

    assert not args.out_root.exists(), f"{args.out_root} exists, delete it first"
    raw_dir = args.raw_root / "UCSF_BrainMetastases_TRAIN"
    visits = MANIFEST.read_text().split()
    for visit in visits:
        add_subject(raw_dir, args.out_root, visit)
    print(f"{len(visits)} subjects -> {args.out_root}")
