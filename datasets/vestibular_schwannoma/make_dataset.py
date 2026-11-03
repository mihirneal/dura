"""Build the vestibular_schwannoma eval data.

Contrast enhanced T1 scans with vestibular schwannoma masks from two releases. Each scan is
symlinked in as released, images/<subject>/<session>/t1c.nii.gz, and its tumor mask (uint8,
1 = tumor) written to labels/<subject>/<session>/seg.nii.gz on the image grid.

- crossMoDA 2022 training source: one ceT1 per person, subjects crossmoda-ldn<n> (London) and
  crossmoda-etz<n> (Tilburg), session ses-1. Its cochlea label is dropped.
- VS-MC-RC2: every T1C with a mask, subjects vsmcrc2-<n>, sessions named by the scan's (offset)
  date.

scans.tsv lists every scan with its tumor volume, vs_segmentation.tsv one scan per person: every
crossMoDA scan and each VS-MC-RC2 patient's first.

    uv run python datasets/vestibular_schwannoma/make_dataset.py \
        /data/mihir-stuff/crossmoda2022 /data/mihir-stuff/tumor_raw/Vestibular-Schwannoma-MC-RC2 \
        /data/mihir-stuff/dura/vestibular_schwannoma
"""

import argparse
import csv
import re
from pathlib import Path

import nibabel as nib
import numpy as np

VS_LABEL = 1


def add_scan(out_root: Path, subject: str, session: str, image: Path, seg: Path) -> dict[str, str]:
    """Symlink the image, write the tumor mask and return the scan's row."""
    image_img, seg_img = nib.load(image), nib.load(seg)
    assert seg_img.shape == image_img.shape, f"{seg} is off the image grid"
    assert np.allclose(seg_img.affine, image_img.affine, atol=1e-3), f"{seg} is off the image grid"
    tumor = np.asarray(seg_img.dataobj).round() == VS_LABEL
    assert tumor.any(), f"no tumor in {seg}"
    img_dir = out_root / "images" / subject / session
    label_dir = out_root / "labels" / subject / session
    img_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (img_dir / "t1c.nii.gz").symlink_to(image)
    nib.Nifti1Image(tumor.astype(np.uint8), seg_img.affine).to_filename(label_dir / "seg.nii.gz")
    voxel_ml = np.prod(seg_img.header.get_zooms()[:3]) / 1000
    return {
        "subject": subject,
        "session": session,
        "image": f"images/{subject}/{session}/t1c.nii.gz",
        "label": f"labels/{subject}/{session}/seg.nii.gz",
        "tumor_ml": f"{tumor.sum() * voxel_ml:.4f}",
    }


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("crossmoda_root", type=Path, help="crossMoDA 2022 training data")
    parser.add_argument("vs_mc_rc2_root", type=Path, help="VS-MC-RC2 release")
    parser.add_argument("out_root", type=Path)
    args = parser.parse_args()

    assert not args.out_root.exists(), f"{args.out_root} exists, delete it first"
    scans = []
    crossmoda = args.crossmoda_root / "training_source"
    with (args.crossmoda_root / "infos_source_training.csv").open() as f:
        names = [row["crossmoda_name"] for row in csv.DictReader(f)]
    for name in names:
        site, number = re.fullmatch(r"crossmoda202[12]_(ldn|etz)_(\d+)", name).groups()
        subject = f"crossmoda-{site}{int(number):03d}"
        image, seg = crossmoda / f"{name}_ceT1.nii.gz", crossmoda / f"{name}_Label.nii.gz"
        scans.append(add_scan(args.out_root, subject, "ses-1", image, seg))
    # sorted by patient, then date
    for seg in sorted(args.vs_mc_rc2_root.glob("VS_MC_RC2_*_T1C_seg.nii.gz")):
        patient, date = re.fullmatch(
            r"VS_MC_RC2_(\d+)_(\d{4}-\d{2}-\d{2})_T1C_seg\.nii\.gz", seg.name
        ).groups()
        image = seg.with_name(seg.name.replace("_seg", ""))
        session = f"ses-{date.replace('-', '')}"
        scans.append(add_scan(args.out_root, f"vsmcrc2-{patient}", session, image, seg))

    first = {}
    for scan in scans:
        first.setdefault(scan["subject"], scan)
    write_tsv(args.out_root / "scans.tsv", scans)
    write_tsv(args.out_root / "vs_segmentation.tsv", list(first.values()))
    print(f"{len(scans)} scans, {len(first)} people -> {args.out_root}")
