import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader

from dura.data import NiftiDataset, collate, fit_transform
from dura.models.base import ModelTransform, ModelWrapper

CV_SEED = 0
N_BOOTSTRAP = 1000
LOGISTIC_CS = np.logspace(-4, 4, 9)
RIDGE_ALPHAS = np.logspace(-2, 6, 9)
SEGMENTATION_ALPHAS = (1e1, 1e2, 1e3, 1e4, 1e5)
SEGMENTATION_THRESHOLDS = torch.logspace(-3, -0.1, 30)


def probe_binary_classification(
    model: ModelWrapper,
    transform: ModelTransform,
    samples: list[dict[str, Any]],
    n_folds: int = 5,
    batch_size: int = 8,
    num_workers: int = 8,
    device: str = "cuda",
    amp: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Linear probe for binary classification on the global embedding. Logistic regression with
    the penalty tuned by inner CV. Scores pooled out-of-fold predictions with bootstrap CIs.
    """
    return probe_global(
        fit_score_binary, model, transform, samples, n_folds, batch_size, num_workers, device, amp
    )


def probe_multiclass_classification(
    model: ModelWrapper,
    transform: ModelTransform,
    samples: list[dict[str, Any]],
    n_folds: int = 5,
    batch_size: int = 8,
    num_workers: int = 8,
    device: str = "cuda",
    amp: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Linear probe for classification into labels 0..K-1 on the global embedding. Multinomial
    logistic regression with the penalty tuned by inner CV. Scores pooled out-of-fold predictions
    by macro one-vs-rest AUROC and balanced accuracy, with bootstrap CIs.
    """
    return probe_global(
        fit_score_multiclass,
        model,
        transform,
        samples,
        n_folds,
        batch_size,
        num_workers,
        device,
        amp,
    )


def probe_regression(
    model: ModelWrapper,
    transform: ModelTransform,
    samples: list[dict[str, Any]],
    n_folds: int = 5,
    batch_size: int = 8,
    num_workers: int = 8,
    device: str = "cuda",
    amp: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Linear probe for regression on the global embedding. Ridge regression with the penalty
    tuned by leave-one-out. Scores pooled out-of-fold predictions with bootstrap CIs.
    """
    return probe_global(
        fit_score_regression,
        model,
        transform,
        samples,
        n_folds,
        batch_size,
        num_workers,
        device,
        amp,
    )


def probe_global(
    fit_score: Callable[[np.ndarray, np.ndarray, int], dict[str, Any]],
    model: ModelWrapper,
    transform: ModelTransform,
    samples: list[dict[str, Any]],
    n_folds: int,
    batch_size: int,
    num_workers: int,
    device: str,
    amp: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Embed the samples with the global embedding and score them with fit_score."""
    features, targets, embed_seconds = embed_global(
        model, transform, samples, batch_size, num_workers, device, amp
    )
    result = fit_score(features, targets, n_folds)
    result["n_params"] = sum(p.numel() for p in model.parameters())
    result["embed_seconds"] = embed_seconds
    state = {"features": features}
    return result, state


def embed_global(
    model: ModelWrapper,
    transform: ModelTransform,
    samples: list[dict[str, Any]],
    batch_size: int,
    num_workers: int,
    device: str,
    amp: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Global embeddings [N D] and targets [N] of the samples in order, and the seconds spent in
    global_embed."""
    device_type = torch.device(device).type
    model.eval()
    fit_transform(transform, samples)
    dataset = NiftiDataset(samples, transform)
    loader = DataLoader(dataset, batch_size, num_workers=num_workers, collate_fn=collate)
    embeddings = []
    targets = []
    embed_seconds = 0.0
    with torch.inference_mode(), torch.autocast(device_type, dtype=torch.bfloat16, enabled=amp):
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            start = time.perf_counter()
            embedding = model.global_embed(images, masks).float().cpu()
            embed_seconds += time.perf_counter() - start
            assert embedding.ndim == 2 and len(embedding) == len(images), (
                f"unexpected global embedding shape {embedding.shape}"
            )
            embeddings.append(embedding)
            targets.append(batch["target"])
    return torch.cat(embeddings).numpy(), torch.cat(targets).numpy(), embed_seconds


def fit_score_binary(features: np.ndarray, labels: np.ndarray, n_folds: int = 5) -> dict[str, Any]:
    """Logistic regression on fixed features, fit in n_folds CV, pooled out-of-fold predictions
    scored with bootstrap CIs. The probe's scoring, also used for the classical baselines."""
    assert set(np.unique(labels)) == {0, 1}, "expected binary labels"
    n_samples = len(labels)
    probabilities = np.zeros(n_samples)
    fold_Cs = []
    folds = StratifiedKFold(n_folds, shuffle=True, random_state=CV_SEED)
    for train_ids, test_ids in folds.split(features, labels):
        classifier = logistic_classifier()
        classifier.fit(features[train_ids], labels[train_ids])
        probabilities[test_ids] = classifier.predict_proba(features[test_ids])[:, 1]
        fold_Cs.append(float(classifier[-1].C_))
    predictions = probabilities > 0.5

    rng = np.random.default_rng(0)
    bootstrap_auroc = []
    bootstrap_balanced_accuracy = []
    for _ in range(N_BOOTSTRAP):
        ids = rng.integers(0, n_samples, n_samples)
        # auroc is undefined when a resample has only one class
        if len(np.unique(labels[ids])) < 2:
            continue
        bootstrap_auroc.append(roc_auc_score(labels[ids], probabilities[ids]))
        bootstrap_balanced_accuracy.append(balanced_accuracy_score(labels[ids], predictions[ids]))

    return {
        "auroc": roc_auc_score(labels, probabilities),
        "auroc_ci": np.percentile(bootstrap_auroc, [2.5, 97.5]).tolist(),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "balanced_accuracy_ci": np.percentile(bootstrap_balanced_accuracy, [2.5, 97.5]).tolist(),
        "fold_C": fold_Cs,
        "labels": labels.tolist(),
        "probabilities": probabilities.tolist(),
        "n_samples": n_samples,
        "embed_dim": features.shape[1],
    }


def fit_score_multiclass(
    features: np.ndarray, labels: np.ndarray, n_folds: int = 5
) -> dict[str, Any]:
    """Multinomial logistic regression on fixed features for labels 0..K-1, fit in n_folds CV,
    pooled out-of-fold predictions scored by macro one-vs-rest AUROC and balanced accuracy with
    bootstrap CIs."""
    n_classes = len(np.unique(labels))
    assert n_classes > 2 and set(np.unique(labels)) == set(range(n_classes)), (
        "expected labels 0..K-1 with K > 2, use probe_binary_classification for two classes"
    )
    n_samples = len(labels)
    probabilities = np.zeros((n_samples, n_classes))
    fold_Cs = []
    folds = StratifiedKFold(n_folds, shuffle=True, random_state=CV_SEED)
    for train_ids, test_ids in folds.split(features, labels):
        classifier = logistic_classifier()
        classifier.fit(features[train_ids], labels[train_ids])
        probabilities[test_ids] = classifier.predict_proba(features[test_ids])
        fold_Cs.append(float(classifier[-1].C_))
    predictions = probabilities.argmax(axis=1)

    rng = np.random.default_rng(0)
    bootstrap_auroc = []
    bootstrap_balanced_accuracy = []
    for _ in range(N_BOOTSTRAP):
        ids = rng.integers(0, n_samples, n_samples)
        # one-vs-rest auroc is undefined when a resample misses a class
        if len(np.unique(labels[ids])) < n_classes:
            continue
        bootstrap_auroc.append(roc_auc_score(labels[ids], probabilities[ids], multi_class="ovr"))
        bootstrap_balanced_accuracy.append(balanced_accuracy_score(labels[ids], predictions[ids]))

    return {
        "auroc": roc_auc_score(labels, probabilities, multi_class="ovr"),
        "auroc_ci": np.percentile(bootstrap_auroc, [2.5, 97.5]).tolist(),
        "class_auroc": [roc_auc_score(labels == k, probabilities[:, k]) for k in range(n_classes)],
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "balanced_accuracy_ci": np.percentile(bootstrap_balanced_accuracy, [2.5, 97.5]).tolist(),
        "fold_C": fold_Cs,
        "labels": labels.tolist(),
        "probabilities": probabilities.tolist(),
        "n_samples": n_samples,
        "embed_dim": features.shape[1],
    }


def fit_score_regression(
    features: np.ndarray, targets: np.ndarray, n_folds: int = 5
) -> dict[str, Any]:
    """Ridge regression on fixed features with the penalty tuned by leave-one-out, fit in
    n_folds CV, pooled out-of-fold predictions scored with bootstrap CIs."""
    targets = np.asarray(targets, dtype=np.float64)
    n_samples = len(targets)
    predictions = np.zeros(n_samples)
    fold_alphas = []
    folds = KFold(n_folds, shuffle=True, random_state=CV_SEED)
    for train_ids, test_ids in folds.split(features):
        regressor = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
        regressor.fit(features[train_ids], targets[train_ids])
        predictions[test_ids] = regressor.predict(features[test_ids])
        fold_alphas.append(float(regressor[-1].alpha_))

    rng = np.random.default_rng(0)
    bootstrap_mae = []
    bootstrap_r = []
    bootstrap_r2 = []
    for _ in range(N_BOOTSTRAP):
        ids = rng.integers(0, n_samples, n_samples)
        bootstrap_mae.append(mean_absolute_error(targets[ids], predictions[ids]))
        bootstrap_r.append(np.corrcoef(targets[ids], predictions[ids])[0, 1])
        bootstrap_r2.append(r2_score(targets[ids], predictions[ids]))

    return {
        "mae": mean_absolute_error(targets, predictions),
        "mae_ci": np.percentile(bootstrap_mae, [2.5, 97.5]).tolist(),
        "r": np.corrcoef(targets, predictions)[0, 1],
        "r_ci": np.percentile(bootstrap_r, [2.5, 97.5]).tolist(),
        "r2": r2_score(targets, predictions),
        "r2_ci": np.percentile(bootstrap_r2, [2.5, 97.5]).tolist(),
        "fold_alpha": fold_alphas,
        "targets": targets.tolist(),
        "predictions": predictions.tolist(),
        "n_samples": n_samples,
        "embed_dim": features.shape[1],
    }


def logistic_classifier() -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=LOGISTIC_CS,
            l1_ratios=(0.0,),
            scoring="neg_log_loss",
            # so that 0.5 (argmax for multiclass) is a sensible cutoff for balanced accuracy
            class_weight="balanced",
            max_iter=1000,
            use_legacy_attributes=False,
        ),
    )


def probe_binary_segmentation(
    model: ModelWrapper,
    transform: ModelTransform,
    samples: list[dict[str, Any]],
    label_fn: Callable[[np.ndarray], np.ndarray],
    n_folds: int = 5,
    n_inner_folds: int = 5,
    max_negative_ratio: float | None = None,
    batch_size: int = 4,
    num_workers: int = 8,
    device: str = "cuda",
    amp: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Linear probe for binary segmentation on the dense embedding. Each patch embedding predicts
    the voxel labels inside its patch, on the model's input grid. The penalty and threshold are
    tuned by inner CV on that grid. The out-of-fold predictions are then mapped back to each
    label's native grid, where per-subject dice, average precision and voxel auroc are scored
    with bootstrap CIs, so models with different grids are scored on the same voxels.
    """
    device_type = torch.device(device).type
    model.eval()
    fit_transform(transform, samples)
    dataset = NiftiDataset(samples, transform, label_fn)
    loader = DataLoader(dataset, batch_size, num_workers=num_workers, collate_fn=collate)
    patch_features = []
    patch_labels = []
    patch_in_mask = []
    geometry = []
    embed_seconds = 0.0
    with torch.inference_mode(), torch.autocast(device_type, dtype=torch.bfloat16, enabled=amp):
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            labels = batch["target"].to(device)
            assert labels.max() <= 1, "expected binary labels"
            start = time.perf_counter()
            embedding = model.dense_embed(images, masks).float()
            if device_type == "cuda":
                torch.cuda.synchronize()
            embed_seconds += time.perf_counter() - start
            assert embedding.ndim == 5 and len(embedding) == len(images), (
                f"unexpected dense embedding shape {embedding.shape}"
            )
            grid_shape = tuple(embedding.shape[1:4])
            patch_size = patch_size_for(tuple(masks.shape[1:]), grid_shape)

            embedding = embedding.flatten(1, 3)
            mask_patches = patchify3d(masks, patch_size)
            label_patches = patchify3d(labels, patch_size)
            # train on the patches that overlap the mask. labels outside it count as missed
            for ii in range(len(images)):
                in_mask = mask_patches[ii].any(dim=1)
                patch_in_mask.append(in_mask.cpu())
                patch_features.append(embedding[ii, in_mask])
                patch_labels.append(label_patches[ii, in_mask])
                geometry.append(
                    {
                        "affine": batch["affine"][ii],
                        "grid_shape": grid_shape,
                        "patch_size": patch_size,
                        "label_shape": batch["label_shape"][ii],
                        "label_affine": batch["label_affine"][ii],
                        "label_positive": batch["label_positive"][ii],
                    }
                )
    embed_dim = patch_features[0].shape[1]

    n_subjects = len(patch_features)
    n_thresholds = len(SEGMENTATION_THRESHOLDS)
    subject_dice = np.zeros(n_subjects)
    subject_model_grid_dice = np.zeros(n_subjects)
    subject_average_precision = np.zeros(n_subjects)
    subject_voxel_auroc = np.zeros(n_subjects)
    subject_probabilities_out_of_fold = [None] * n_subjects
    subject_fold = np.zeros(n_subjects, dtype=int)
    fold_alphas = []
    fold_thresholds = []
    folds = KFold(n_folds, shuffle=True, random_state=CV_SEED)
    for fold, (train_ids, test_ids) in enumerate(folds.split(np.arange(n_subjects))):
        inner_dice = torch.zeros(len(SEGMENTATION_ALPHAS), n_subjects, n_thresholds)
        inner_folds = KFold(n_inner_folds, shuffle=True, random_state=CV_SEED)
        for inner_train, inner_val in inner_folds.split(train_ids):
            inner_train_ids = train_ids[inner_train]
            inner_val_ids = train_ids[inner_val]
            for alpha_id, alpha in enumerate(SEGMENTATION_ALPHAS):
                probabilities = fit_predict_segmentation(
                    patch_features,
                    patch_labels,
                    inner_train_ids,
                    inner_val_ids,
                    alpha,
                    max_negative_ratio,
                )
                for ii, subject_probabilities in zip(inner_val_ids, probabilities):
                    inner_dice[alpha_id, ii] = dice_by_threshold(
                        subject_probabilities, patch_labels[ii]
                    )
        # pick the alpha and threshold with the best mean dice over inner validation subjects
        mean_inner_dice = inner_dice[:, train_ids].mean(dim=1)
        alpha_id, threshold_id = np.unravel_index(
            mean_inner_dice.argmax().item(), (len(SEGMENTATION_ALPHAS), n_thresholds)
        )
        alpha = SEGMENTATION_ALPHAS[alpha_id]
        threshold = SEGMENTATION_THRESHOLDS[threshold_id : threshold_id + 1]
        fold_alphas.append(alpha)
        fold_thresholds.append(threshold.item())

        probabilities = fit_predict_segmentation(
            patch_features, patch_labels, train_ids, test_ids, alpha, max_negative_ratio
        )
        for ii, subject_probabilities in zip(test_ids, probabilities):
            subject_probabilities_out_of_fold[ii] = subject_probabilities.cpu()
            subject_fold[ii] = fold
            subject_model_grid_dice[ii] = dice_by_threshold(
                subject_probabilities, patch_labels[ii], threshold
            ).item()
            native_probabilities, native_labels = to_native(
                subject_probabilities, patch_in_mask[ii], geometry[ii]
            )
            subject_dice[ii] = dice_by_threshold(
                native_probabilities, native_labels, threshold
            ).item()
            subject_average_precision[ii] = average_precision(native_probabilities, native_labels)
            subject_voxel_auroc[ii] = voxel_auroc(native_probabilities, native_labels)

    rng = np.random.default_rng(0)
    bootstrap_dice = []
    bootstrap_average_precision = []
    bootstrap_voxel_auroc = []
    for _ in range(N_BOOTSTRAP):
        ids = rng.integers(0, n_subjects, n_subjects)
        bootstrap_dice.append(subject_dice[ids].mean())
        bootstrap_average_precision.append(np.nanmean(subject_average_precision[ids]))
        bootstrap_voxel_auroc.append(np.nanmean(subject_voxel_auroc[ids]))

    result = {
        "dice": subject_dice.mean(),
        "dice_ci": np.percentile(bootstrap_dice, [2.5, 97.5]).tolist(),
        "average_precision": np.nanmean(subject_average_precision),
        "average_precision_ci": np.nanpercentile(bootstrap_average_precision, [2.5, 97.5]).tolist(),
        "voxel_auroc": np.nanmean(subject_voxel_auroc),
        "voxel_auroc_ci": np.nanpercentile(bootstrap_voxel_auroc, [2.5, 97.5]).tolist(),
        "model_grid_dice": subject_model_grid_dice.mean(),
        "fold_alpha": fold_alphas,
        "fold_threshold": fold_thresholds,
        "subject_dice": subject_dice.tolist(),
        "subject_average_precision": subject_average_precision.tolist(),
        "subject_voxel_auroc": subject_voxel_auroc.tolist(),
        "n_samples": n_subjects,
        "embed_dim": embed_dim,
        "n_params": sum(p.numel() for p in model.parameters()),
        "embed_seconds": embed_seconds,
    }
    state = {
        "in_mask": patch_in_mask,
        "probabilities": subject_probabilities_out_of_fold,
        "fold": subject_fold,
        "geometry": geometry,
    }
    return result, state


def fit_predict_segmentation(
    patch_features: list[Tensor],
    patch_labels: list[Tensor],
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    alpha: float,
    max_negative_ratio: float | None = None,
) -> list[Tensor]:
    """Fit on the train subjects' patches, return per-patch probabilities for each test subject."""
    train_features = torch.cat([patch_features[ii] for ii in train_ids])
    train_labels = torch.cat([patch_labels[ii] for ii in train_ids]).float()
    if max_negative_ratio is not None:
        keep_ids = subsample_negatives(train_labels, max_negative_ratio)
        train_features, train_labels = train_features[keep_ids], train_labels[keep_ids]
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0, correction=0).clamp_min(1e-6)
    coef, intercept = fit_logistic((train_features - mean) / std, train_labels, alpha)

    probabilities = []
    for ii in test_ids:
        logits = (patch_features[ii] - mean) / std @ coef + intercept
        probabilities.append(torch.sigmoid(logits))
    return probabilities


def subsample_negatives(
    labels: Tensor,
    max_negative_ratio: float,
) -> tuple[Tensor, Tensor]:
    """Keep every row with a positive label and at most max_negative_ratio negatives per positive."""
    positive = labels.sum(dim=1) > 0
    positive_ids = positive.nonzero()[:, 0]
    negative_ids = (~positive).nonzero()[:, 0]
    n_negative = min(len(negative_ids), int(max_negative_ratio * len(positive_ids)))
    generator = torch.Generator().manual_seed(CV_SEED)
    order = torch.randperm(len(negative_ids), generator=generator).to(negative_ids.device)
    keep_ids = torch.cat([positive_ids, negative_ids[order[:n_negative]]])
    return keep_ids


def dice_by_threshold(
    probabilities: Tensor, labels: Tensor, thresholds: Tensor = SEGMENTATION_THRESHOLDS
) -> Tensor:
    """Dice at each threshold. Empty prediction and empty label counts as 1."""
    thresholds = thresholds.to(probabilities.device)
    predicted = probabilities.flatten()[None, :] >= thresholds[:, None]
    overlap = (predicted & labels.flatten().bool()[None, :]).sum(dim=1)
    denominator = predicted.sum(dim=1) + labels.sum()
    dice = torch.where(denominator > 0, 2 * overlap / denominator.clamp_min(1), 1.0)
    return dice.cpu()


def to_native(
    probabilities: Tensor, in_mask: Tensor, geometry: dict[str, Any]
) -> tuple[Tensor, Tensor]:
    """
    Map one subject's patch probabilities from the model grid to the label's native grid,
    trilinear. Returns the probabilities and labels of the native voxels covered by a scored
    patch, plus every positive voxel: positives the model cropped or masked out score 0.
    """
    device = probabilities.device
    grid_shape, patch_size = geometry["grid_shape"], geometry["patch_size"]
    patches = probabilities.new_zeros(math.prod(grid_shape), math.prod(patch_size), 2)
    patches[in_mask.to(device), :, 0] = probabilities.float()
    patches[in_mask.to(device), :, 1] = 1.0
    volume = rearrange(
        patches,
        "(gx gy gz) (px py pz) c -> c (gx px) (gy py) (gz pz)",
        gx=grid_shape[0], gy=grid_shape[1], gz=grid_shape[2],
        px=patch_size[0], py=patch_size[1], pz=patch_size[2],
    )  # fmt: skip

    # native voxel -> world -> model voxel, normalized to [-1, 1] in grid_sample's (z, y, x) order
    native_to_model = np.linalg.inv(geometry["affine"]) @ geometry["label_affine"]
    native_to_model = torch.as_tensor(native_to_model, dtype=torch.float32, device=device)
    axes = [
        torch.arange(size, dtype=torch.float32, device=device) for size in geometry["label_shape"]
    ]
    ijk = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    xyz = ijk @ native_to_model[:3, :3].T + native_to_model[:3, 3]
    size = torch.tensor(volume.shape[1:], dtype=torch.float32, device=device)
    grid = (2 * xyz / (size - 1) - 1).flip(-1)
    native = F.grid_sample(
        volume[None], grid.view(1, 1, 1, -1, 3), mode="bilinear", align_corners=True
    ).view(2, -1)

    labels = torch.zeros(len(ijk), dtype=torch.bool, device=device)
    labels[torch.as_tensor(geometry["label_positive"], device=device)] = True
    keep = (native[1] > 0) | labels
    return native[0, keep], labels[keep]


def voxel_auroc(probabilities: Tensor, labels: Tensor) -> float:
    """Voxel AUROC from ranks, NaN when there are no positives. Ties are not merged."""
    positive = labels.flatten().bool()
    n_positive = positive.sum().item()
    if n_positive == 0:
        return np.nan
    n_negative = len(positive) - n_positive
    ranks = torch.empty(len(positive), dtype=torch.float64, device=positive.device)
    ranks[probabilities.flatten().argsort()] = torch.arange(
        1, len(positive) + 1, dtype=torch.float64, device=positive.device
    )
    rank_sum = ranks[positive].sum().item()
    return (rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def average_precision(probabilities: Tensor, labels: Tensor) -> float:
    """Voxel average precision, NaN when there are no positives. Ties are not merged."""
    order = probabilities.flatten().argsort(descending=True)
    sorted_labels = labels.flatten()[order].float()
    n_positive = sorted_labels.sum().item()
    if n_positive == 0:
        return np.nan
    ranks = torch.arange(1, len(sorted_labels) + 1, device=sorted_labels.device)
    precision = sorted_labels.cumsum(dim=0) / ranks
    return (precision * sorted_labels).sum().item() / n_positive


def fit_logistic(
    features: Tensor,
    targets: Tensor,
    alpha: float,
    max_iter: int = 1000,
) -> tuple[Tensor, Tensor]:
    """
    L2 penalized logistic regression fit with L-BFGS. Each target column is a separate binary
    problem sharing the same features. Targets can be soft (between 0 and 1).
    """
    n, d = features.shape
    assert targets.ndim == 2 and len(targets) == n, (
        f"targets {tuple(targets.shape)} do not match {n} samples of {d} features"
    )
    n_outputs = targets.shape[1]

    coef = torch.zeros(d, n_outputs, device=features.device, dtype=features.dtype)
    intercept = torch.logit(targets.mean(dim=0).clamp(1e-6, 1 - 1e-6))
    coef.requires_grad_(True)
    intercept.requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [coef, intercept], max_iter=max_iter, history_size=10, line_search_fn="strong_wolfe"
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        logits = features @ coef + intercept
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss = loss + alpha * coef.square().sum() / (n * n_outputs)
        loss.backward()
        return loss

    optimizer.step(closure)

    # l-bfgs stops on its own gradient and step tolerances; exhausting the budget means neither met
    n_iter = optimizer.state[coef]["n_iter"]
    assert n_iter < max_iter, f"l-bfgs used all {max_iter} iterations without converging"
    return coef.detach(), intercept.detach()


def patch_size_for(shape: tuple[int, ...], grid_shape: tuple[int, ...]) -> tuple[int, ...]:
    assert all(size % grid == 0 for size, grid in zip(shape, grid_shape)), (
        f"patch grid {grid_shape} does not tile the image grid {shape}"
    )
    return tuple(size // grid for size, grid in zip(shape, grid_shape))


def patchify3d(x: Tensor, patch_size: tuple[int, int, int]) -> Tensor:
    px, py, pz = patch_size
    x = rearrange(x, "b (gx px) (gy py) (gz pz) -> b (gx gy gz) (px py pz)", px=px, py=py, pz=pz)
    return x
