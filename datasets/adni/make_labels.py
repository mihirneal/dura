"""SynthSeg labels of the adni_anatomy images.

    uv run --group datasets python datasets/adni/make_labels.py /data/mihir-stuff/dura/adni

SynthSeg 2.0 robust, run on each T1 as released. The labels are on SynthSeg's 1 mm grid,
aligned with the T1 in world space, with SynthSeg's QC scores next to each as <label>_qc.csv.
--shard i --shards n segments every n-th image from the i-th, to split the work across jobs.
"""

import argparse
import shlex
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

SYNTHSEG = (
    "uvx --python 3.11 --from "
    "git+https://github.com/MedARC-AI/SynthSeg.git@8a2812bac64ae9dc34a79075a3b86bd901041957 "
    "SynthSeg"
)
MIN_QC = 0.65

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="DURA_DATA_ROOT/adni, from make_dataset.py")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    table = pd.read_csv(args.root / "adni_anatomy.tsv", sep="\t", dtype=str)
    table = table.iloc[args.shard :: args.shards]
    images = [args.root / image for image in table.image]
    labels = [args.root / label for label in table.label]
    assert not any(label.exists() for label in labels), "labels exist, delete them first"
    for label in labels:
        label.parent.mkdir(parents=True, exist_ok=True)
    qcs = [label.with_name(label.name.replace(".nii.gz", "_qc.csv")) for label in labels]

    with tempfile.TemporaryDirectory() as tmp:
        lists = {name: Path(tmp) / f"{name}.txt" for name in ("i", "o", "qc")}
        for name, paths in zip(lists, (images, labels, qcs)):
            lists[name].write_text("".join(f"{path}\n" for path in paths))
        command = [
            *shlex.split(SYNTHSEG),
            *("--i", lists["i"], "--o", lists["o"], "--qc", lists["qc"], "--robust"),
            *("--threads", args.threads, *(["--cpu"] if args.cpu else [])),
        ]
        subprocess.run([str(part) for part in command], check=True)

    missing = [label for label in labels if not label.exists()]
    assert not missing, f"SynthSeg wrote no labels for {missing}"
    # per structure group: low cortex QC alone doesn't affect the anatomy tasks
    scores = pd.concat([pd.read_csv(qc, index_col=0) for qc in qcs])
    print(f"{len(labels)} labels -> {args.root / 'labels'}")
    print(f"QC min: {scores.min().round(3).to_dict()}")
    print(f"QC below {MIN_QC}: {(scores < MIN_QC).sum().to_dict()}")
