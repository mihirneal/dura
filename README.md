# Dura

Dura is an evaluation suite for structural MRI foundation models on restricted-access datasets: ones that can't be redistributed publicly. The evals are model-agnostic: each model is a wrapper file in [src/dura/models/](src/dura/models/) with its own transform, and its code comes in as an optional dependency.

## Installation

```bash
git clone <dura repo> dura && cd dura
uv sync --extra nanobrain   # one extra per model you want to run
uv run pytest               # needs a GPU
```

Model extras point at local clones next to dura (see `[tool.uv.sources]` in [pyproject.toml](pyproject.toml)).

## Usage

```bash
uv run python -m dura.run <model> <output_dir> --model-kwargs ckpt_path=/path/to/ckpt.pt
```

Each task writes `<output_dir>/<task>.json`, and tasks that already have a result are skipped. `--tasks` runs a subset, and `--help` lists the registered models and tasks.

| Model | Extra | Kwargs |
|---|---|---|
| `nanobrain` | `nanobrain` | `ckpt_path` |
| `nanobrain_random` | `nanobrain` | `ckpt_path` (for the architecture), `seed` (default 42) |

## Datasets

The releases sit behind logins or data use agreements, so each one is downloaded by hand and nothing here downloads data. The data lives in `DURA_DATA_ROOT` (default `/data/mihir-stuff/dura`), a release's eval subjects in `<name>/`. The nii.gz are used exactly as released: dura does no preprocessing, each model's `transform` does its own on the fly.

| Dataset | Tasks | Metric |
|---|---|---|
| ucsf_bmsr | `ucsf_bmsr_t1c_enhancing`, `ucsf_bmsr_t1c_edema`, `ucsf_bmsr_flair_edema` | dice |
| adni | `adni_diagnosis`, `adni_mci_conversion`, `adni_wmh_flair`, `adni_wmh_t1`, `adni_hippocampus`, `adni_amygdala`, `adni_temporal_horns`, `adni_lateral_ventricles` | AUROC, R² |

**ucsf_bmsr**: [UCSF-BMSR](https://imagingdatasets.ucsf.edu/dataset/1) v1.3 ([Rudie et al. 2024](https://doi.org/10.1148/ryai.230126)), brain metastasis segmentation. 199 visits, one per patient, with T1post, FLAIR and the BraTS-METS 2023 labels. Each task segments one label: enhancing tumor on T1c, and edema on T1c and on FLAIR.

```bash
uv run python datasets/ucsf_bmsr/make_manifest.py /data/smri-datasets/UCSF-BMSR   # freeze the subjects
uv run python datasets/ucsf_bmsr/make_dataset.py /data/smri-datasets/UCSF-BMSR $DURA_DATA_ROOT/ucsf_bmsr
```

**adni**: [ADNI](https://adni.loni.usc.edu) 3T scans from ADNIGO to ADNI4, converted from the DICOMs with dcm2niix, with labels from the ADNIMERGE2 tables of the 18 Jun 2026 download. ADNI1 is left out: it is mostly 1.5T and has no FLAIR. Each task has its own 500 subjects with one scan each (the four regional volume tasks share theirs), 1,458 subjects in all. ADNI's diagnosis comes from the same cognitive tests a clinician already has, so it is only a sanity check. The other tasks are things MRI itself shows: prognosis, small vessel disease and atrophy of the regions that separate Alzheimer's disease from normal aging.

| Task | Input | Target | Subjects | Metric |
|---|---|---|---|---|
| `adni_diagnosis` | T1 | CN, MCI or dementia | 167, 166, 167 | macro one-vs-rest AUROC |
| `adni_mci_conversion` | T1 of an MCI subject | dementia within 3 years | 500, 119 convert | AUROC |
| `adni_wmh_flair` | 3D FLAIR | log white matter hyperintensity volume as a % of cranial volume (UC Davis) | 500 | R² |
| `adni_wmh_t1` | T1 of the same session | the same | 500 | R² |
| `adni_hippocampus` | T1 | hippocampal volume relative to normal aging (FreeSurfer 7) | 167, 166, 167 | R² |
| `adni_amygdala` | T1 | the same for the amygdala | the same | R² |
| `adni_temporal_horns` | T1 | the same for the temporal horns (inferior lateral ventricles) | the same | R² |
| `adni_lateral_ventricles` | T1 | the same for the lateral ventricles | the same | R² |

- Each task takes each subject's first eligible scan and samples 500 subjects with a fixed seed: diagnosis with balanced classes, MCI conversion keeping every converter.
- The regional targets are W-scores: each log volume's residual from a linear fit on age, sex and log intracranial volume in amyloid negative CN subjects, in SDs. They measure atrophy beyond normal aging: the lateral ventricles grow with age in everyone, the hippocampus, amygdala and temporal horns change much faster in Alzheimer's disease. The subjects are balanced across CN, MCI and dementia. Most FreeSurfer runs were never rated visually, so runs whose volumes disagree with SynthSeg's are left out.
- WMH uses 3D FLAIR only. ADNIGO/2 FLAIR is 2D with 5mm slices, on which UC Davis measures about twice as much WMH at the same age.
- Each session uses one T1: distortion corrected before the uncorrected copy, first scan before repeat, full before accelerated.

Classical features probed the same way give each task's reference scores ([baselines.py](datasets/adni/baselines.py)):

| Task | age+sex | SynthSeg volumes | Task specific |
|---|---|---|---|
| `adni_diagnosis` | 0.58 | 0.77 | |
| `adni_mci_conversion` | 0.56 | 0.71 | |
| `adni_wmh_flair`, `adni_wmh_t1` | 0.19 | 0.22 | FreeSurfer T1 WM hypointensities 0.71 |
| `adni_hippocampus` | 0.01 | 0.68 | |
| `adni_amygdala` | 0.02 | 0.60 | |
| `adni_temporal_horns` | 0.00 | 0.64 | |
| `adni_lateral_ventricles` | -0.01 | 0.54 | |

```bash
uv run --group datasets python datasets/adni/make_manifest.py /data/smri-datasets/ADNI   # freeze the subjects
uv run --group datasets python datasets/adni/make_dataset.py /data/smri-datasets/ADNI $DURA_DATA_ROOT/adni
uv run --group datasets python datasets/adni/baselines.py /data/smri-datasets/ADNI   # reference scores
```

## Adding a model

1. Add `src/dura/models/my_model.py`. Import your official model code rather than copying it, and add it as an optional dependency in [pyproject.toml](pyproject.toml):

    ```toml
    [project.optional-dependencies]
    my-model = ["my-model"]
    ```

2. Implement a `ModelTransform` and a `ModelWrapper` ([src/dura/models/base.py](src/dura/models/base.py)), and a `@register_model` constructor that takes string kwargs (`--model-kwargs key=value`) and returns both:

    ```python
    @register_model
    def my_model(ckpt_path: str) -> tuple[MyTransform, MyModelWrapper]:
        return MyTransform(), MyModelWrapper(load_my_model(ckpt_path))
    ```

    - `transform(img)`: the raw `nib.Nifti1Image` → `{"image": [X Y Z], "mask": [X Y Z], "affine": [4 4]}`. All of the model's preprocessing (reorienting, resampling, cropping, masking, normalization) happens here, on the fly in the data loader workers. The shape must be the same for every image, and the affine maps the grid to world space. An optional `fit(images)` precomputes global parameters on a task's raw images.
    - `dense_embed(images, mask)`: `[B X Y Z]` → `[B Gx Gy Gz C]`, a patch grid that tiles `X Y Z` exactly
    - `global_embed(images, mask)`: `[B X Y Z]` → `[B D]`

    Segmentation labels are resampled onto the model's grid to train the probe, and the predictions are mapped back to the labels' native grid for scoring, so every model is scored on the same voxels. [src/dura/models/nanobrain.py](src/dura/models/nanobrain.py) is a full example.

To keep a model out of this repo instead, put the same file in `my_repo/src/dura/models/` (no `__init__.py` files) and install dura in your repo's environment: models are found by [namespace package plugin discovery](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-namespace-packages).

## Adding a dataset

- **Curation scripts** live in [datasets/](datasets/), one folder per dataset: `make_manifest.py` picks the eval subjects and freezes them in the tracked `manifest.txt`, and `make_dataset.py` links them in from the release unchanged.
- **Tasks** live in [src/dura/tasks.py](src/dura/tasks.py): each lists its samples' nii.gz paths and binds them to a probe in [src/dura/probe.py](src/dura/probe.py).

## License

The code is under the [MIT License](LICENSE). Each dataset keeps its original license.
