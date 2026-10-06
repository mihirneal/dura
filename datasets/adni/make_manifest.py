"""Freeze each task's subjects and images as manifest.tsv.

uv run --group datasets python datasets/adni/make_manifest.py /data/smri-datasets/ADNI
"""

import argparse
from pathlib import Path

import pandas as pd
from adni import cohorts

MANIFEST = Path(__file__).parent / "manifest.tsv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_root", type=Path, help="ADNI release, with nifti/ and metadata/")
    args = parser.parse_args()

    tables = cohorts(args.raw_root)
    manifest = pd.concat(
        [table.drop(columns="target").assign(task=task) for task, table in tables.items()]
    )
    manifest = manifest[["task", "subject", "image"]]
    manifest.to_csv(MANIFEST, sep="\t", index=False)
    for task, table in tables.items():
        print(f"{task}: {len(table)} subjects, target {table.target.describe().round(3).to_dict()}")
    print(f"{manifest.subject.nunique()} subjects -> {MANIFEST}")
