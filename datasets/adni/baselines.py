"""Reference scores (age+sex, SynthSeg volumes, FreeSurfer WM hypointensities), probed like
the models.

    uv run --group datasets python datasets/adni/baselines.py /data/smri-datasets/ADNI
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from adni import REGIONS, demographics, nearest, read_table, session, synthseg_volumes

from dura.data import DATA_ROOT
from dura.probe import fit_score_binary, fit_score_multiclass, fit_score_regression
from dura.tasks import HEADLINE_METRICS

FREESURFER_MAX_DAYS = 60
FIT_SCORE = {
    "adni_diagnosis": fit_score_multiclass,
    "adni_mci_conversion": fit_score_binary,
    "adni_wmh_flair": fit_score_regression,
    "adni_wmh_t1": fit_score_regression,
    **{f"adni_{region}": fit_score_regression for region in REGIONS},
}


def freesurfer_wm_hypo(raw_root: Path) -> pd.DataFrame:
    fs = read_table(raw_root, "UCSFFSX7").dropna(subset=["EXAMDATE", "ST128SV", "ST10CV"])
    fs = fs[~fs.OVERALLQC.astype(str).isin(["Fail", "Hippocampus Only"])]
    fs = fs[fs.FIELD_STRENGTH.astype(str).str.startswith("3")]
    wm_hypo = np.log(100 * fs.ST128SV / fs.ST10CV + 0.01)
    return fs.assign(fs_date=pd.to_datetime(fs.EXAMDATE), wm_hypo=wm_hypo)[
        ["RID", "fs_date", "wm_hypo"]
    ]


def score(task: str, features: pd.DataFrame | np.ndarray, table: pd.DataFrame) -> str:
    features = np.asarray(features, dtype=np.float64)
    keep = ~np.isnan(features).any(axis=1)
    result = FIT_SCORE[task](features[keep], table.target.to_numpy()[keep])
    return f"{result[HEADLINE_METRICS[task]]:.2f} (n={keep.sum()})"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_root", type=Path, help="ADNI release, with synthseg/ and metadata/")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT / "adni")
    args = parser.parse_args()

    tables = {
        task: pd.read_csv(args.data_root / f"{task}.tsv", sep="\t", dtype={"subject": str})
        for task in FIT_SCORE
    }
    for task, table in tables.items():
        table["RID"] = table.subject.str.split("S").str[1].astype(int)
        table["date"] = pd.to_datetime([session(image)[1] for image in table.image])
    sessions = {
        (subject, date)
        for table in tables.values()
        for subject, date in zip(table.subject, table.date)
    }
    volumes = np.log(synthseg_volumes(args.raw_root, sessions).clip(lower=1))
    demo = demographics(args.raw_root)
    wm_hypo = freesurfer_wm_hypo(args.raw_root)

    rows = []
    for task, table in tables.items():
        table = table.merge(demo, on="RID", how="left")
        age_sex = np.c_[(table.date - table.dob).dt.days / 365.25, table.male]
        baseline = volumes.reindex(list(zip(table.subject, table.date)))
        row = {
            "task": task,
            "age+sex": score(task, age_sex, table),
            "SynthSeg": score(task, baseline, table),
        }
        if task.startswith("adni_wmh"):
            fs = nearest(table, wm_hypo, "date", "fs_date", FREESURFER_MAX_DAYS)
            fs = fs.set_index("subject").reindex(table.subject)
            row["FreeSurfer WM hypo"] = score(task, fs[["wm_hypo"]], table)
        rows.append(row)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    print("| " + " | ".join(columns) + " |\n|" + "---|" * len(columns))
    for row in rows:
        print("| " + " | ".join(row.get(column, "") for column in columns) + " |")
