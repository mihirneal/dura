"""The ADNI eval cohorts, shared by make_manifest.py and make_dataset.py.

Images are the local DICOM to NIfTI conversion of the ADNI release (nifti/, with a dcm2niix sidecar
per image), labels come from the ADNIMERGE2 tables of the 18 Jun 2026 download. Only 3T scans
from ADNIGO on are used: ADNI1 is mostly 1.5T, with an older protocol and no FLAIR. Field strength
comes from the sidecars, because LONI's MRI key table has no row for about 2k of the T1s.

Each task has its own subjects, at most MAX_SUBJECTS, with one scan each, so CV folds never share
a subject. Subjects are sampled with a fixed seed.
"""

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
# stable MCI needs follow-up to the horizon, give or take a visit window
CONVERSION_MIN_FOLLOWUP_YEARS = CONVERSION_YEARS - 0.25
# log of WMH as a percent of cranial volume, the offset for the few scans with none
WMH_LOG_OFFSET = 0.01
# FreeSurfer 7 aseg volumes, left and right. The temporal horns are the inferior lateral ventricles
REGIONS = {
    "hippocampus": ("ST29SV", "ST88SV"),
    "amygdala": ("ST12SV", "ST71SV"),
    "temporal_horns": ("ST30SV", "ST89SV"),
    "lateral_ventricles": ("ST37SV", "ST96SV"),
}
# and their SynthSeg labels, to check FreeSurfer: a run is left out if any region's volume is off
# from SynthSeg's by more than MAX_DISAGREEMENT robust SDs of the usual ratio between the two
SYNTHSEG_REGIONS = {
    "hippocampus": ("left hippocampus", "right hippocampus"),
    "amygdala": ("left amygdala", "right amygdala"),
    "temporal_horns": ("left inferior lateral ventricle", "right inferior lateral ventricle"),
    "lateral_ventricles": ("left lateral ventricle", "right lateral ventricle"),
}
MAX_DISAGREEMENT = 5.0
MIN_SYNTHSEG_QC = 0.65
FREESURFER_MAX_DAYS = 30
AMYLOID_MAX_DAYS = 365

# broken on disk, found by eye: noise without a brain, though UC Davis has a WMH volume for its id
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
    """Scanner metadata from the dcm2niix sidecar. A 3T image is valid if nibabel reads it as 3D
    with a sane affine."""
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
        # no gradient distortion correction on the scanner. ADNI3 also stores the corrected one
        "nd": "ND" in meta.get("ImageType", []),
        "repeat": "repeat" in description,
        "accel": bool(re.search(r"accel|grappa|sense|\bcs\b", description)),
        "valid": valid,
    }


def index_images(raw_root: Path) -> pd.DataFrame:
    """Every valid 3T T1w and FLAIR, one row per nii.gz. Series that converted to several files
    (multi-echo and slice spacing variants: acq- names, or several outputs in the log) are left
    out."""
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
    """One T1 per session: distortion corrected before ND, first scan before repeat, full before
    accelerated (ADNIGO/2 acquired both), lowest run."""
    t1 = images[images.suffix == "T1w"]
    t1 = t1.sort_values(["subject", "date", "nd", "repeat", "accel", "run"])
    return t1.drop_duplicates(["subject", "date"]).reset_index(drop=True)


def nearest(
    left: pd.DataFrame, right: pd.DataFrame, left_on: str, right_on: str, max_days: int
) -> pd.DataFrame:
    """For each left row, the right row of the same RID nearest in date, if within max_days."""
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
    """MCI sessions with a known outcome. Converter: a dementia diagnosis within CONVERSION_YEARS
    of the scan. Stable: no dementia diagnosis within the horizon and a diagnosis visit at it or
    later, reverters to CN included. Sessions followed for less time are left out."""
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
    """Sessions with UC Davis WMH volumes, on the exact FLAIR they were measured on, and a
    selected T1. 3D FLAIR only: ADNIGO/2 FLAIR is 2D with 5mm slices, on which UC Davis measures
    about twice as much WMH at the same age, and a model can read the slice thickness off the
    image."""
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


def freesurfer_volumes(raw_root: Path) -> pd.DataFrame:
    """UCSF FreeSurfer 7 volumes of the REGIONS, left + right, and the ICV of 3T runs. Most runs
    were never rated visually, so only those rated as failed are left out."""
    fs = read_table(raw_root, "UCSFFSX7").dropna(subset=["EXAMDATE", "ST10CV"])
    fs = fs[fs.FIELD_STRENGTH.astype(str) == "3T"]
    ratings = fs[["OVERALLQC", "HIPPOQC", "VENTQC"]].astype(str)
    fs = fs[~ratings.isin(["Fail", "Hippocampus Only"]).any(axis=1)]
    volumes = {region: fs[left] + fs[right] for region, (left, right) in REGIONS.items()}
    volumes = pd.DataFrame(
        {"RID": fs.RID, "fs_date": pd.to_datetime(fs.EXAMDATE), "icv": fs.ST10CV, **volumes}
    )
    return volumes.dropna()


def session(name: str) -> tuple[str, pd.Timestamp]:
    """Subject and session date from a sub-<subject>_ses-<date>_... file name."""
    match = re.search(r"sub-(\w+?)_ses-(\d{8})_", name)
    return match[1], pd.Timestamp(match[2])


def synthseg_volumes(raw_root: Path, sessions: set[tuple[str, pd.Timestamp]]) -> pd.DataFrame:
    """Volumes of each session's best SynthSeg T1 run (synthseg/ in the release), if its QC
    passes, indexed by (subject, date). SynthSeg ran on an earlier conversion with its own run
    numbers, so runs are matched by session, not by image."""
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


def agrees_with_synthseg(raw_root: Path, volumes: pd.DataFrame) -> np.ndarray:
    """Whether each FreeSurfer run's regional volumes agree with SynthSeg on the same session.
    False without a SynthSeg run that passes QC."""
    keys = list(zip(volumes.subject, volumes.date))
    synthseg = synthseg_volumes(raw_root, set(keys)).reindex(keys)
    agrees = np.ones(len(volumes), dtype=bool)
    for region, labels in SYNTHSEG_REGIONS.items():
        log_ratio = np.log(
            volumes[region].to_numpy() / synthseg[list(labels)].sum(axis=1).to_numpy()
        )
        median = np.nanmedian(log_ratio)
        robust_sd = 1.4826 * np.nanmedian(np.abs(log_ratio - median))
        agrees &= np.abs(log_ratio - median) <= MAX_DISAGREEMENT * robust_sd
    return agrees


def amyloid_status(raw_root: Path) -> pd.DataFrame:
    amyloid = read_table(raw_root, "UCBERKELEY_AMY_6MM").dropna(
        subset=["SCANDATE", "AMYLOID_STATUS"]
    )
    # qc_flag 0 failed, -2 could not be processed
    amyloid = amyloid[~amyloid.qc_flag.isin([0, -2])]
    return amyloid.assign(pet_date=pd.to_datetime(amyloid.SCANDATE))[
        ["RID", "pet_date", "AMYLOID_STATUS"]
    ]


def w_scores(sessions: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Each region's log volume relative to normal aging: the residual of a linear fit on age,
    sex and log ICV in the reference sessions, in units of the reference residual SD."""

    def design(table: pd.DataFrame) -> np.ndarray:
        return np.c_[np.ones(len(table)), table.age, table.male, np.log(table.icv)]

    scores = {}
    for region in REGIONS:
        coef, *_ = np.linalg.lstsq(design(reference), np.log(reference[region]), rcond=None)
        residual = np.log(reference[region]) - design(reference) @ coef
        sd = residual.std(ddof=design(reference).shape[1])
        scores[f"{region}_w"] = (np.log(sessions[region]) - design(sessions) @ coef) / sd
    return pd.DataFrame(scores, index=sessions.index)


def regional_volumes(raw_root: Path, sessions: pd.DataFrame) -> pd.DataFrame:
    """First sessions with FreeSurfer volumes that agree with SynthSeg, and their regional
    W-scores against amyloid negative CN sessions (amyloid PET within AMYLOID_MAX_DAYS)."""
    volumes = nearest(
        sessions, freesurfer_volumes(raw_root), "date", "fs_date", FREESURFER_MAX_DAYS
    ).dropna(subset=list(REGIONS))
    volumes = volumes[agrees_with_synthseg(raw_root, volumes)]
    volumes = volumes.merge(demographics(raw_root), on="RID")
    volumes = volumes.assign(age=(volumes.date - volumes.dob).dt.days / 365.25)
    amyloid = nearest(volumes, amyloid_status(raw_root), "date", "pet_date", AMYLOID_MAX_DAYS)
    reference = first_session(amyloid[(amyloid.DIAGNOSIS == "CN") & (amyloid.AMYLOID_STATUS == 0)])
    volumes = first_session(volumes)
    return volumes.join(w_scores(volumes, reference))


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
    """Each task's subjects: subject, image as a nii.gz name, and target.
    Each subject's first eligible session, then at most MAX_SUBJECTS subjects per task."""
    images = index_images(raw_root)
    t1 = select_t1(images)
    dx = diagnoses(raw_root)
    sessions = nearest(t1, dx, "date", "dx_date", DX_MAX_DAYS).dropna(subset=["DIAGNOSIS"])
    sessions = sessions[sessions.COLPROT.isin(PHASES)]
    key = ["subject", "date"]
    t1_names = t1[[*key, "name"]]

    # balanced classes, the sanity check would otherwise be mostly CN and MCI
    diagnosis = balanced_sample(first_session(sessions))
    diagnosis = diagnosis.assign(target=diagnosis.DIAGNOSIS.map(DIAGNOSES))
    # every converter, the scarce class, and a sample of the stable
    conversion = first_session(sessions[[*key, "name"]].merge(mci_conversion(sessions, dx), on=key))
    converted = conversion[conversion.target == 1]
    stable = sample(conversion[conversion.target == 0], MAX_SUBJECTS - len(converted))
    conversion = pd.concat([converted, stable])
    wmh_sessions = first_session(t1_names.merge(wmh(raw_root, images, t1), on=key))
    wmh_sessions = sample(wmh_sessions, min(MAX_SUBJECTS, len(wmh_sessions)))
    # one sample for all regions, balanced so that disease spreads the volumes
    volumes = balanced_sample(regional_volumes(raw_root, sessions))

    tables = {
        "adni_diagnosis": diagnosis.assign(image=diagnosis["name"]),
        "adni_mci_conversion": conversion.assign(image=conversion["name"]),
        "adni_wmh_flair": wmh_sessions.assign(image=wmh_sessions.flair),
        "adni_wmh_t1": wmh_sessions.assign(image=wmh_sessions["name"]),
        **{
            f"adni_{region}": volumes.assign(image=volumes["name"], target=volumes[f"{region}_w"])
            for region in REGIONS
        },
    }
    columns = ["subject", "image", "target"]
    return {
        task: table.reindex(columns=columns).sort_values("subject").reset_index(drop=True)
        for task, table in tables.items()
    }
