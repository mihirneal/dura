import csv
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np

from dura.data import DATA_ROOT
from dura.probe import (
    probe_binary_classification,
    probe_binary_segmentation,
    probe_multiclass_classification,
    probe_regression,
)

Probe = Callable[..., tuple[dict[str, Any], dict[str, Any]]]

UCSF_EDEMA = 2
UCSF_ENHANCING = 3
MAX_NEGATIVE_RATIO = 10.0


def select_labels(label: np.ndarray, label_ids: tuple[int, ...]) -> np.ndarray:
    return np.isin(label.round(), label_ids)


def ucsf_bmsr_samples(modality: str) -> list[dict[str, Any]]:
    """One visit per patient, visit 100101A is stored as sub-100101/ses-A. The images and the
    BraTS labels (2 edema, 3 enhancing tumor) are all on the T1post grid."""
    root = DATA_ROOT / "ucsf_bmsr"
    sessions = sorted((root / "images").glob("sub-*/ses-*"))
    assert sessions, f"no sessions in {root / 'images'}"
    return [
        {
            "subject": f"{ses.parent.name.removeprefix('sub-')}{ses.name.removeprefix('ses-')}",
            "image": ses / f"{modality}.nii.gz",
            "label": root / "labels" / ses.parent.name / ses.name / "seg.nii.gz",
        }
        for ses in sessions
    ]


def ucsf_bmsr_label(modality: str, label_id: int) -> Probe:
    return partial(
        probe_binary_segmentation,
        samples=ucsf_bmsr_samples(modality),
        label_fn=partial(select_labels, label_ids=(label_id,)),
        max_negative_ratio=MAX_NEGATIVE_RATIO,
    )


def ucsf_bmsr_t1c_enhancing() -> Probe:
    return ucsf_bmsr_label("t1c", UCSF_ENHANCING)


def ucsf_bmsr_t1c_edema() -> Probe:
    return ucsf_bmsr_label("t1c", UCSF_EDEMA)


def ucsf_bmsr_flair_edema() -> Probe:
    return ucsf_bmsr_label("flair", UCSF_EDEMA)


def adni_samples(task: str, target_type: type = float) -> list[dict[str, Any]]:
    """One subject per row of DATA_ROOT/adni/<task>.tsv: its image, relative to DATA_ROOT/adni,
    and the target."""
    root = DATA_ROOT / "adni"
    with (root / f"{task}.tsv").open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert rows, f"no subjects in {root / f'{task}.tsv'}"
    return [
        {
            "subject": row["subject"],
            "image": root / row["image"],
            "target": target_type(row["target"]),
        }
        for row in rows
    ]


def adni_diagnosis() -> Probe:
    return partial(probe_multiclass_classification, samples=adni_samples("adni_diagnosis", int))


def adni_mci_conversion() -> Probe:
    return partial(probe_binary_classification, samples=adni_samples("adni_mci_conversion", int))


def adni_wmh_flair() -> Probe:
    return partial(probe_regression, samples=adni_samples("adni_wmh_flair"))


def adni_wmh_t1() -> Probe:
    return partial(probe_regression, samples=adni_samples("adni_wmh_t1"))


def adni_hippocampus() -> Probe:
    return partial(probe_regression, samples=adni_samples("adni_hippocampus"))


def adni_amygdala() -> Probe:
    return partial(probe_regression, samples=adni_samples("adni_amygdala"))


def adni_temporal_horns() -> Probe:
    return partial(probe_regression, samples=adni_samples("adni_temporal_horns"))


def adni_lateral_ventricles() -> Probe:
    return partial(probe_regression, samples=adni_samples("adni_lateral_ventricles"))


TASKS: dict[str, Callable[[], Probe]] = {
    task.__name__: task
    for task in [
        ucsf_bmsr_t1c_enhancing,
        ucsf_bmsr_t1c_edema,
        ucsf_bmsr_flair_edema,
        adni_diagnosis,
        adni_mci_conversion,
        adni_wmh_flair,
        adni_wmh_t1,
        adni_hippocampus,
        adni_amygdala,
        adni_temporal_horns,
        adni_lateral_ventricles,
    ]
}

HEADLINE_METRICS: dict[str, str] = {
    "ucsf_bmsr_t1c_enhancing": "dice",
    "ucsf_bmsr_t1c_edema": "dice",
    "ucsf_bmsr_flair_edema": "dice",
    # macro one-vs-rest
    "adni_diagnosis": "auroc",
    "adni_mci_conversion": "auroc",
    "adni_wmh_flair": "r2",
    "adni_wmh_t1": "r2",
    "adni_hippocampus": "r2",
    "adni_amygdala": "r2",
    "adni_temporal_horns": "r2",
    "adni_lateral_ventricles": "r2",
}
