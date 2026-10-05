"""Freeze the UCSF-BMSR visits of the ucsf_bmsr eval as manifest.txt.

One visit per patient, so CV folds never share a patient: the earliest visit (visits are named
patient id + A, B, ... in scan order) whose BraTS labels have at least MIN_EDEMA_ML of edema
(label 2) and some enhancing tumor (label 3). The floor drops trace edema annotations, whose dice
would be noise. The non-enhancing core (label 1) is not required: it is a trace in most visits.

    uv run python datasets/ucsf_bmsr/make_manifest.py /data/smri-datasets/UCSF-BMSR
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import nibabel as nib
import numpy as np

MIN_EDEMA_ML = 0.1
MANIFEST = Path(__file__).parent / "manifest.txt"


def has_edema_and_enhancing(visit_dir: Path, visit: str) -> bool:
    seg_img = nib.load(visit_dir / visit / f"{visit}_BraTS-seg.nii.gz")
    seg = np.asarray(seg_img.dataobj).round()
    edema_ml = (seg == 2).sum() * np.prod(seg_img.header.get_zooms()) / 1000
    return edema_ml >= MIN_EDEMA_ML and (seg == 3).any()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_root", type=Path, help="extracted UCSF-BMSR v1.3 release")
    args = parser.parse_args()

    visit_dir = args.raw_root / "UCSF_BrainMetastases_TRAIN"
    visits = sorted(
        path.name
        for path in visit_dir.iterdir()
        if (path / f"{path.name}_BraTS-seg.nii.gz").exists()
    )
    with ProcessPoolExecutor() as executor:
        keep = list(executor.map(partial(has_edema_and_enhancing, visit_dir), visits))
    patient_visits = {}
    for visit, ok in zip(visits, keep):
        if ok:
            patient_visits.setdefault(visit[:-1], visit)
    subjects = sorted(patient_visits.values())
    MANIFEST.write_text("\n".join(subjects) + "\n")
    print(f"{len(subjects)} visits ({len(visits)} with BraTS labels) -> {MANIFEST}")
