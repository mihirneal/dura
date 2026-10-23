"""ADNI eval cohorts: 3T scans from ADNIGO on, labels from the 18 Jun 2026 ADNIMERGE2 tables.
One scan per subject, at most MAX_SUBJECTS per task."""

import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pyreadr
from nibabel.filebasedimages import ImageFileError

TABLES = Path("metadata/Tables_18Jun2026/ADNIMERGE2/data")
PHASES = ("ADNIGO", "ADNI2", "ADNI3", "ADNI4")
DIAGNOSES = {"CN": 0, "MCI": 1, "Dementia": 2}
MAX_SUBJECTS = 500
SAMPLE_SEED = 0
DX_MAX_DAYS = 90
CONVERSION_YEARS = 3.0
CONVERSION_MIN_FOLLOWUP_YEARS = CONVERSION_YEARS - 0.25
WMH_LOG_OFFSET = 0.01
MIN_SYNTHSEG_QC = 0.65

# noise, no brain
BAD_IMAGES = {"sub-019S4835_ses-20170719_run-02_FLAIR.nii.gz"}

IMAGE_NAME = re.compile(
    r"^sub-(?P<subject>\d+S(?P<rid>\d+))_ses-(?P<date>\d{8})(?:_run-(?P<run>\d+))?"
    r"_(?P<suffix>T1w|FLAIR)\.nii\.gz$"
)


def image_path(raw_root: Path, name: str) -> Path:
    """sub-002S0295_ses-20110602_run-01_T1w.nii.gz -> nifti/sub-002S0295/ses-20110602/anat/..."""
    sub, ses = name.split("_")[:2]
    return raw_root / "nifti" / sub / ses / "anat" / name


def read_table(raw_root: Path, name: str) -> pd.DataFrame:
    table = next(iter(pyreadr.read_r(raw_root / TABLES / f"{name}.rda").values()))
    return table.dropna(subset=["RID"]).assign(RID=lambda t: t.RID.astype(int))


def read_image(raw_root: Path, name: str) -> dict:
    """Scanner metadata from the dcm2niix sidecar, and whether a 3T image loads as 3D."""
    path = image_path(raw_root, name)
    try:
        meta = json.loads(path.with_name(name.replace(".nii.gz", ".json")).read_text())
    except (OSError, json.JSONDecodeError):
        meta = {}
    tesla = meta.get("MagneticFieldStrength", np.nan)
    # a few sidecars are in gauss
    tesla = tesla / 1e4 if tesla > 100 else tesla
    tesla = 3.0 if 2.5 < tesla < 3.5 else 1.5 if 1.0 < tesla < 2.0 else np.nan
    img = None
    if tesla == 3.0:
        try:
            img = nib.load(path)
        except (OSError, EOFError, ValueError, ImageFileError):
            img = None
    valid = img is not None and img.ndim == 3 and None not in nib.aff2axcodes(img.affine)
    description = meta.get("SeriesDescription", "").lower()
    return {
        "tesla": tesla,
        "acquisition": meta.get("MRAcquisitionType"),
        "nd": "ND" in meta.get("ImageType", []),
        "repeat": "repeat" in description,
        "accel": bool(re.search(r"accel|grappa|sense|\bcs\b", description)),
        "valid": valid,
    }


def index_images(raw_root: Path) -> pd.DataFrame:
    """Every valid 3T T1w and FLAIR, skipping series converted to several files."""
    log = pd.read_csv(raw_root / "nifti" / "conversion_log.tsv", sep="\t")
    log = log[log.status.isin(["ok", "skipped"]) & ~log.output.str.contains(";")]
    names = log.output.str.rsplit("/", n=1).str[-1]
    images = names.str.extract(IMAGE_NAME).assign(name=names, image_id=log.image_id)
    images = images.dropna(subset=["subject"]).drop_duplicates("name")
    images = images[~images["name"].isin(BAD_IMAGES)]
    images = images.assign(
        RID=images.rid.astype(int),
        date=pd.to_datetime(images.date, format="%Y%m%d"),
        run=images.run.fillna(1).astype(int),
    ).drop(columns="rid")
    with ThreadPoolExecutor(32) as executor:
        meta = list(executor.map(lambda name: read_image(raw_root, name), images["name"]))
    images = pd.concat([images.reset_index(drop=True), pd.DataFrame(meta)], axis=1)
    return images[(images.tesla == 3.0) & images.valid].reset_index(drop=True)


def select_t1(images: pd.DataFrame) -> pd.DataFrame:
    """One T1 per session: corrected over ND, first over repeat, full over accelerated."""
    t1 = images[images.suffix == "T1w"]
    t1 = t1.sort_values(["subject", "date", "nd", "repeat", "accel", "run"])
    return t1.drop_duplicates(["subject", "date"]).reset_index(drop=True)


def nearest(
    left: pd.DataFrame, right: pd.DataFrame, left_on: str, right_on: str, max_days: int
) -> pd.DataFrame:
    """Nearest right row of the same RID within max_days."""
    left = left.assign(**{left_on: left[left_on].astype("datetime64[ns]")})
    right = right.assign(**{right_on: right[right_on].astype("datetime64[ns]")})
    return pd.merge_asof(
        left.sort_values(left_on),
        right.sort_values(right_on),
        left_on=left_on,
        right_on=right_on,
        by="RID",
        direction="nearest",
        tolerance=pd.Timedelta(days=max_days),
    )


def diagnoses(raw_root: Path) -> pd.DataFrame:
    dx = read_table(raw_root, "DXSUM").dropna(subset=["DIAGNOSIS", "EXAMDATE"])
    dx = dx.assign(
        dx_date=pd.to_datetime(dx.EXAMDATE),
        DIAGNOSIS=dx.DIAGNOSIS.astype(str),
        COLPROT=dx.COLPROT.astype(str),
    )
    return dx[dx.DIAGNOSIS.isin(list(DIAGNOSES))][["RID", "dx_date", "DIAGNOSIS", "COLPROT"]]


def mci_conversion(sessions: pd.DataFrame, dx: pd.DataFrame) -> pd.DataFrame:
    """MCI sessions with dementia within CONVERSION_YEARS, or followed that long."""
    mci = sessions[sessions.DIAGNOSIS == "MCI"][["RID", "subject", "date"]]
    later = mci.merge(dx, on="RID")
    years = (later.dx_date - later.date).dt.days / 365.25
    later = later.assign(
        converted=(later.DIAGNOSIS == "Dementia") & (years > 0) & (years <= CONVERSION_YEARS),
        followed=years >= CONVERSION_MIN_FOLLOWUP_YEARS,
    )
    outcome = later.groupby(["subject", "date"])[["converted", "followed"]].any().reset_index()
    outcome = outcome[outcome.converted | outcome.followed]
    return outcome.assign(target=outcome.converted.astype(int))[["subject", "date", "target"]]


def wmh(raw_root: Path, images: pd.DataFrame, t1: pd.DataFrame) -> pd.DataFrame:
    """UC Davis WMH on the FLAIR it was measured on. 3D FLAIR only: 2D FLAIR inflates WMH."""
    ucd = read_table(raw_root, "UCD_WMH")
    ucd = ucd[ucd.STATUS.isna()].dropna(subset=["IMAGEUID", "TOTAL_WMH", "CEREBRUM_TCV"])
    ucd = ucd.assign(image_id="I" + ucd.IMAGEUID.astype(int).astype(str))
    flair = images[(images.suffix == "FLAIR") & (images.acquisition == "3D")]
    flair = flair[["image_id", "subject", "date", "name"]].merge(ucd, on="image_id")
    flair = flair.merge(t1[["subject", "date"]], on=["subject", "date"])
    flair = flair.sort_values("name").drop_duplicates(["subject", "date"])
    wmh_percent = 100 * flair.TOTAL_WMH / flair.CEREBRUM_TCV
    flair = flair.assign(flair=flair["name"], target=np.log(wmh_percent + WMH_LOG_OFFSET))
    return flair[["subject", "date", "flair", "target"]]


def demographics(raw_root: Path) -> pd.DataFrame:
    demo = read_table(raw_root, "PTDEMOG")
    demo = demo.assign(dob=pd.to_datetime(demo.PTDOB, format="%m/%Y", errors="coerce"))
    demo = demo.dropna(subset=["dob"]).sort_values("VISDATE").drop_duplicates("RID")
    return demo.assign(male=(demo.PTGENDER.astype(str) == "Male").astype(float))[
        ["RID", "dob", "male"]
    ]


def session(name: str) -> tuple[str, pd.Timestamp]:
    """(subject, date) of a sub-<subject>_ses-<date>_... name."""
    match = re.search(r"sub-(\w+?)_ses-(\d{8})_", name)
    return match[1], pd.Timestamp(match[2])


def synthseg_volumes(raw_root: Path, sessions: set[tuple[str, pd.Timestamp]]) -> pd.DataFrame:
    """Volumes of each session's best SynthSeg run, matched by session, not run number."""
    qc_files = defaultdict(list)
    for path in (raw_root / "synthseg").glob("sub-*_T1w*_qc.csv"):
        key = session(path.name)
        if key in sessions:
            qc_files[key].append(path)

    def best_run(paths: list[Path]) -> pd.Series | None:
        qc = {path: pd.read_csv(path, index_col=0).iloc[0].min() for path in paths}
        best = max(qc, key=qc.get)
        if qc[best] < MIN_SYNTHSEG_QC:
            return None
        return pd.read_csv(str(best).replace("_qc.csv", "_volumes.csv"), index_col=0).iloc[0]

    with ThreadPoolExecutor(32) as executor:
        volumes = dict(zip(qc_files, executor.map(best_run, qc_files.values())))
    volumes = {key: row for key, row in volumes.items() if row is not None}
    return pd.DataFrame.from_dict(volumes, orient="index")


def first_session(table: pd.DataFrame) -> pd.DataFrame:
    return table.sort_values(["subject", "date"]).drop_duplicates("subject")


def sample(table: pd.DataFrame, n: int) -> pd.DataFrame:
    """A seeded random n rows."""
    assert n <= len(table), f"can't sample {n} of {len(table)}"
    return table.sort_values("subject").sample(n=n, random_state=SAMPLE_SEED)


def balanced_sample(table: pd.DataFrame) -> pd.DataFrame:
    """MAX_SUBJECTS rows, as many CN as MCI and dementia."""
    per_class = np.diff(np.linspace(0, MAX_SUBJECTS, len(DIAGNOSES) + 1).round().astype(int))
    return pd.concat(
        [sample(table[table.DIAGNOSIS == dx], n) for dx, n in zip(DIAGNOSES, per_class)]
    )


def cohorts(raw_root: Path) -> dict[str, pd.DataFrame]:
    """Each task's subject, image and target."""
    images = index_images(raw_root)
    t1 = select_t1(images)
    dx = diagnoses(raw_root)
    sessions = nearest(t1, dx, "date", "dx_date", DX_MAX_DAYS).dropna(subset=["DIAGNOSIS"])
    sessions = sessions[sessions.COLPROT.isin(PHASES)]
    key = ["subject", "date"]
    t1_names = t1[[*key, "name"]]

    diagnosis = balanced_sample(first_session(sessions))
    diagnosis = diagnosis.assign(target=diagnosis.DIAGNOSIS.map(DIAGNOSES))
    # every converter
    conversion = first_session(sessions[[*key, "name"]].merge(mci_conversion(sessions, dx), on=key))
    converted = conversion[conversion.target == 1]
    stable = sample(conversion[conversion.target == 0], MAX_SUBJECTS - len(converted))
    conversion = pd.concat([converted, stable])
    wmh_sessions = first_session(t1_names.merge(wmh(raw_root, images, t1), on=key))
    wmh_sessions = sample(wmh_sessions, min(MAX_SUBJECTS, len(wmh_sessions)))

    tables = {
        "adni_diagnosis": diagnosis.assign(image=diagnosis["name"]),
        "adni_mci_conversion": conversion.assign(image=conversion["name"]),
        "adni_wmh_flair": wmh_sessions.assign(image=wmh_sessions.flair),
        "adni_wmh_t1": wmh_sessions.assign(image=wmh_sessions["name"]),
        # the diagnosis subjects, segmented by SynthSeg (make_labels.py). No target
        "adni_anatomy": diagnosis.assign(image=diagnosis["name"]).drop(columns="target"),
    }
    columns = ["subject", "image", "target"]
    return {
        task: table.reindex(columns=columns).sort_values("subject").reset_index(drop=True)
        for task, table in tables.items()
    }
