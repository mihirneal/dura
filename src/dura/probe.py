import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from torch import Tensor, nn
from torch.utils.data import DataLoader

from dura.data import NiftiDataset, collate, fit_transform
from dura.models.base import ModelTransform, ModelWrapper

CV_SEED = 0
N_BOOTSTRAP = 1000
SEGMENTATION_ALPHAS = (1e1, 1e2, 1e3, 1e4, 1e5)
SEGMENTATION_THRESHOLDS = torch.logspace(-3, -0.1, 30)
# rows per chunk of the logistic loss, so large training sets fit in memory
LOGISTIC_CHUNK = 2**18
# light attention probe (the global tasks): one learned query pools the model's dense tokens.
# Weight decay and epochs by inner CV, then refit with each seed
LIGHT_LR = 1e-3
LIGHT_WEIGHT_DECAYS = (0.05, 0.5, 5.0)
LIGHT_MAX_EPOCHS = 150
# inner CV stops a fit after this many epochs without a better validation loss
LIGHT_PATIENCE = 20
LIGHT_WARMUP_EPOCHS = 3
LIGHT_BATCH_SIZE = 32
# subjects per forward pass, which bounds memory; the gradient is the whole batch's
LIGHT_MICRO_BATCH = 8
# fraction of each subject's tokens dropped per training step
LIGHT_TOKEN_DROP = 0.5
LIGHT_SEEDS = (0, 1, 2)


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
    Light attention probe for binary classification. Scores pooled out-of-fold probabilities by
    AUROC with bootstrap CIs.
    """
    return probe_global(
        "binary", model, transform, samples, n_folds, batch_size, num_workers, device, amp
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
    Light attention probe for labels 0..K-1, scored by macro one-vs-rest AUROC.
    """
    return probe_global(
        "multiclass", model, transform, samples, n_folds, batch_size, num_workers, device, amp
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
    Light attention probe for regression. Scores pooled out-of-fold predictions by R2 with
    bootstrap CIs.
    """
    return probe_global(
        "regression", model, transform, samples, n_folds, batch_size, num_workers, device, amp
    )


def probe_global(
    kind: str,
    model: ModelWrapper,
    transform: ModelTransform,
    samples: list[dict[str, Any]],
    n_folds: int,
    batch_size: int,
    num_workers: int,
    device: str,
    amp: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The light attention probe of the given kind on the model's dense tokens."""
    tokens, targets, embed_seconds = embed_tokens(
        model, transform, samples, batch_size, num_workers, device, amp
    )
    result = fit_score_light(tokens, targets, kind, n_folds)
    result["n_params"] = sum(p.numel() for p in model.parameters())
    result["embed_seconds"] = embed_seconds
    return result, {}


def embed_tokens(
    model: ModelWrapper,
    transform: ModelTransform,
    samples: list[dict[str, Any]],
    batch_size: int,
    num_workers: int,
    device: str,
    amp: bool,
) -> tuple[list[Tensor], np.ndarray, float]:
    """Each subject's dense tokens of the patches that overlap the mask ([N_i D] bf16, on the
    device), targets [N] and seconds spent embedding."""
    device_type = torch.device(device).type
    model.eval()
    fit_transform(transform, samples)
    dataset = NiftiDataset(samples, transform)
    loader = DataLoader(dataset, batch_size, num_workers=num_workers, collate_fn=collate)
    tokens = []
    targets = []
    embed_seconds = 0.0
    with torch.inference_mode(), torch.autocast(device_type, dtype=torch.bfloat16, enabled=amp):
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            start = time.perf_counter()
            embedding = model.dense_embed(images, masks).float()
            if device_type == "cuda":
                torch.cuda.synchronize()
            embed_seconds += time.perf_counter() - start
            assert embedding.ndim == 5 and len(embedding) == len(images), (
                f"unexpected dense embedding shape {embedding.shape}"
            )
            patch_size = patch_size_for(tuple(masks.shape[1:]), tuple(embedding.shape[1:4]))
            in_mask = patchify3d(masks, patch_size).any(dim=2)
            embedding = embedding.flatten(1, 3)
            for ii in range(len(images)):
                tokens.append(embedding[ii, in_mask[ii]].to(torch.bfloat16))
            targets.append(batch["target"])
    return tokens, torch.cat(targets).numpy(), embed_seconds


class LightAttention(nn.Module):
    """
    Light attention probe: a learned query scores each layer-normed token, a softmax over the
    tokens weights their average, and a linear head reads it out. No key or value projections:
    with one query they fold into the query and the head. The prediction splits exactly into
    per-token terms, so the softmax weights are an attention map.
    """

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Parameter(torch.randn(1, dim) * dim**-0.5)
        self.head = nn.Linear(dim, out_dim)

    def forward(self, tokens: Tensor, pad: Tensor) -> Tensor:
        """tokens [B L D], pad [B L] (True where there is no token) -> [B out_dim]."""
        tokens = self.norm(tokens)
        logits = torch.einsum("bld,hd->bhl", tokens, self.query).masked_fill(
            pad[:, None], -torch.inf
        )
        pooled = torch.einsum("bhl,bld->bhd", logits.softmax(-1), tokens)
        return self.head(pooled.flatten(1))


class TokenStore:
    """All subjects' standardized tokens in one flat [T D] bf16 tensor, so subjects with many
    tokens (e.g. patch 4) don't pad every subject to the longest one."""

    def __init__(self, tokens: list[Tensor], train_ids: np.ndarray):
        # standardized by the train subjects' token mean and std, per channel
        total = torch.zeros(tokens[0].shape[1], dtype=torch.float64, device=tokens[0].device)
        total_sq = torch.zeros_like(total)
        count = 0
        for ii in train_ids:
            t = tokens[ii].double()
            total += t.sum(0)
            total_sq += (t * t).sum(0)
            count += len(t)
        mean = total / count
        std = (total_sq / count - mean**2).clamp_min(1e-12).sqrt().float()
        mean = mean.float()
        self.lengths = torch.tensor([len(t) for t in tokens], device=tokens[0].device)
        self.offsets = torch.cat([self.lengths.new_zeros(1), self.lengths.cumsum(0)[:-1]])
        self.flat = torch.empty(
            int(self.lengths.sum()),
            tokens[0].shape[1],
            dtype=torch.bfloat16,
            device=tokens[0].device,
        )
        start = 0
        for t in tokens:
            self.flat[start : start + len(t)] = ((t.float() - mean) / std).bfloat16()
            start += len(t)
        self.dim = self.flat.shape[1]

    def batch(self, ids: Tensor, drop: float = 0.0) -> tuple[Tensor, Tensor]:
        """Padded tokens [B L D] and their padding mask [B L], dropping a random drop fraction."""
        lengths = self.lengths[ids]
        positions = torch.arange(int(lengths.max()), device=ids.device)[None]
        pad = positions >= lengths[:, None]
        index = (self.offsets[ids][:, None] + positions).clamp_max(len(self.flat) - 1)
        tokens = self.flat[index]
        if drop > 0:
            pad = pad | (torch.rand(pad.shape, device=ids.device) < drop)
            # keep at least one token per subject
            pad[:, 0] = pad[:, 0] & (lengths == 0)
        return tokens, pad


def light_loss(kind: str, y: Tensor, train_ids: np.ndarray) -> Callable[[Tensor, Tensor], Tensor]:
    """MSE on standardized targets, or class-balanced binary / multiclass cross-entropy (like the
    logistic probes' class_weight="balanced")."""
    if kind == "regression":
        return lambda output, target: F.mse_loss(output.float().squeeze(-1), target)
    labels = y[torch.as_tensor(train_ids, device=y.device)].long()
    counts = torch.bincount(labels).float()
    class_weight = counts.sum() / (len(counts) * counts)
    if kind == "binary":

        def binary_loss(output: Tensor, target: Tensor) -> Tensor:
            weight = class_weight[target.long()]
            loss = F.binary_cross_entropy_with_logits(
                output.float().squeeze(-1), target, reduction="none"
            )
            return (loss * weight).sum() / weight.sum()

        return binary_loss
    return lambda output, target: F.cross_entropy(
        output.float(), target.long(), weight=class_weight
    )


def train_light(
    store: TokenStore,
    y: Tensor,
    train_ids: np.ndarray,
    weight_decay: float,
    epochs: int,
    seed: int,
    kind: str,
    eval_ids: np.ndarray | None = None,
) -> tuple[LightAttention, list[float]]:
    """Train on train_ids; with eval_ids also the eval loss after every epoch, stopping after
    LIGHT_PATIENCE epochs without improvement (the curve then holds its last value)."""
    device = store.flat.device
    torch.manual_seed(seed)
    out_dim = int(y.max().item()) + 1 if kind == "multiclass" else 1
    model = LightAttention(store.dim, out_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LIGHT_LR, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(train_ids) / LIGHT_BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / (LIGHT_WARMUP_EPOCHS * steps_per_epoch))
    )
    loss_fn = light_loss(kind, y, train_ids)
    generator = torch.Generator().manual_seed(seed)
    curve = []
    for _ in range(epochs):
        stalled = len(curve) > LIGHT_PATIENCE and min(curve[-LIGHT_PATIENCE:]) > min(curve)
        if eval_ids is not None and stalled:
            curve += [curve[-1]] * (epochs - len(curve))
            break
        model.train()
        order = train_ids[torch.randperm(len(train_ids), generator=generator).numpy()]
        for start in range(0, len(order), LIGHT_BATCH_SIZE):
            ids = torch.as_tensor(order[start : start + LIGHT_BATCH_SIZE], device=device)
            optimizer.zero_grad(set_to_none=True)
            for micro in ids.split(LIGHT_MICRO_BATCH):
                tokens, pad = store.batch(micro, LIGHT_TOKEN_DROP)
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    output = model(tokens, pad)
                loss = loss_fn(output, y[micro]) * len(micro) / len(ids)
                loss.backward()
            optimizer.step()
            scheduler.step()
        if eval_ids is not None:
            output = predict_light(model, store, eval_ids)
            curve.append(loss_fn(output, y[torch.as_tensor(eval_ids, device=device)]).item())
    return model, curve


@torch.no_grad()
def predict_light(model: LightAttention, store: TokenStore, ids: np.ndarray) -> Tensor:
    """Raw outputs [N out_dim] for the subjects in ids."""
    model.eval()
    device = store.flat.device
    outputs = []
    for start in range(0, len(ids), LIGHT_MICRO_BATCH):
        micro = torch.as_tensor(ids[start : start + LIGHT_MICRO_BATCH], device=device)
        tokens, pad = store.batch(micro)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            outputs.append(model(tokens, pad).float())
    return torch.cat(outputs)


def light_folds(kind: str, n_folds: int) -> KFold | StratifiedKFold:
    """The probe's folds: stratified for classification."""
    if kind == "regression":
        return KFold(n_folds, shuffle=True, random_state=CV_SEED)
    return StratifiedKFold(n_folds, shuffle=True, random_state=CV_SEED)


def fit_score_light(
    tokens: list[Tensor], targets: np.ndarray, kind: str, n_folds: int = 5
) -> dict[str, Any]:
    """
    Light attention probe in n_folds CV: weight decay and epochs by inner CV on each outer train
    split (seed 0), then refit with each of LIGHT_SEEDS. Scores the seeds' mean out-of-fold
    prediction with bootstrap CIs.
    """
    n_samples = len(targets)
    if kind == "regression":
        targets = np.asarray(targets, dtype=np.float64)
        labels = None
        predictions = np.zeros(n_samples)
    else:
        targets = np.asarray(targets).astype(int)
        n_classes = len(np.unique(targets))
        assert set(np.unique(targets)) == set(range(n_classes)), "expected labels 0..K-1"
        assert (n_classes == 2) == (kind == "binary"), f"{n_classes} classes for a {kind} probe"
        labels = targets
        predictions = np.zeros(n_samples if kind == "binary" else (n_samples, n_classes))
    folds = light_folds(kind, n_folds)
    fold_choices = []
    start = time.perf_counter()
    for train_ids, test_ids in folds.split(np.arange(n_samples), labels):
        store = TokenStore(tokens, train_ids)
        # regression targets standardized by the train split, so one learning rate fits all tasks
        mu, sd = (
            (targets[train_ids].mean(), targets[train_ids].std())
            if kind == "regression"
            else (0.0, 1.0)
        )
        y = torch.as_tensor((targets - mu) / sd, dtype=torch.float32, device=store.flat.device)
        inner = light_folds(kind, n_folds)
        inner_labels = None if labels is None else labels[train_ids]
        best = (np.inf, None, None)
        for weight_decay in LIGHT_WEIGHT_DECAYS:
            curves = [
                train_light(
                    store,
                    y,
                    train_ids[fit],
                    weight_decay,
                    LIGHT_MAX_EPOCHS,
                    0,
                    kind,
                    train_ids[val],
                )[1]
                for fit, val in inner.split(train_ids, inner_labels)
            ]
            curve = np.mean(curves, axis=0)
            if curve.min() < best[0]:
                best = (float(curve.min()), weight_decay, int(curve.argmin()) + 1)
        inner_loss, weight_decay, epochs = best
        outputs = []
        for seed in LIGHT_SEEDS:
            model, _ = train_light(store, y, train_ids, weight_decay, epochs, seed, kind)
            outputs.append(predict_light(model, store, test_ids))
        output = torch.stack(outputs)
        if kind == "regression":
            predictions[test_ids] = (output.squeeze(-1).mean(0) * sd + mu).cpu().numpy()
        elif kind == "binary":
            predictions[test_ids] = output.squeeze(-1).sigmoid().mean(0).cpu().numpy()
        else:
            predictions[test_ids] = output.softmax(-1).mean(0).cpu().numpy()
        fold_choices.append(
            {"weight_decay": weight_decay, "epochs": epochs, "inner_loss": inner_loss}
        )
        del store
    score = {"binary": score_binary, "multiclass": score_multiclass, "regression": score_regression}
    return {
        **score[kind](targets, predictions),
        "fold_choice": fold_choices,
        "probe_seconds": time.perf_counter() - start,
        "n_probe_params": sum(p.numel() for p in model.parameters()),
        "embed_dim": tokens[0].shape[1],
    }


def score_binary(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """AUROC and balanced accuracy (at 0.5) of pooled out-of-fold probabilities, bootstrap CIs."""
    n_samples = len(labels)
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
        "labels": labels.tolist(),
        "probabilities": probabilities.tolist(),
        "n_samples": n_samples,
    }


def score_multiclass(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Macro one-vs-rest AUROC and balanced accuracy (argmax), bootstrap CIs."""
    n_samples, n_classes = probabilities.shape
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
        "labels": labels.tolist(),
        "probabilities": probabilities.tolist(),
        "n_samples": n_samples,
    }


def score_regression(targets: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    """MAE, Pearson r and R2 of pooled out-of-fold predictions, bootstrap CIs."""
    n_samples = len(targets)
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
        "targets": targets.tolist(),
        "predictions": predictions.tolist(),
        "n_samples": n_samples,
    }


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
    with bootstrap CIs, so models with different grids are scored on the same voxels. Also
    scores the predicted volumes against the label volumes across subjects (r2, r).
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
                # bf16 on the cpu: patch 4 features of a large cohort don't fit on the gpu
                patch_features.append(embedding[ii, in_mask].to(torch.bfloat16).cpu())
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
    subject_volume = np.zeros(n_subjects)
    subject_label_volume = np.zeros(n_subjects)
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
            voxel_ml = abs(np.linalg.det(geometry[ii]["label_affine"][:3, :3])) / 1000
            n_predicted = (native_probabilities >= threshold.to(native_probabilities.device)).sum()
            subject_volume[ii] = n_predicted.item() * voxel_ml
            subject_label_volume[ii] = native_labels.sum().item() * voxel_ml

    rng = np.random.default_rng(0)
    bootstrap_dice = []
    bootstrap_average_precision = []
    bootstrap_voxel_auroc = []
    bootstrap_volume_r2 = []
    for _ in range(N_BOOTSTRAP):
        ids = rng.integers(0, n_subjects, n_subjects)
        bootstrap_dice.append(subject_dice[ids].mean())
        bootstrap_average_precision.append(np.nanmean(subject_average_precision[ids]))
        bootstrap_voxel_auroc.append(np.nanmean(subject_voxel_auroc[ids]))
        bootstrap_volume_r2.append(r2_score(subject_label_volume[ids], subject_volume[ids]))

    result = {
        "dice": subject_dice.mean(),
        "dice_ci": np.percentile(bootstrap_dice, [2.5, 97.5]).tolist(),
        "average_precision": np.nanmean(subject_average_precision),
        "average_precision_ci": np.nanpercentile(bootstrap_average_precision, [2.5, 97.5]).tolist(),
        "voxel_auroc": np.nanmean(subject_voxel_auroc),
        "voxel_auroc_ci": np.nanpercentile(bootstrap_voxel_auroc, [2.5, 97.5]).tolist(),
        "model_grid_dice": subject_model_grid_dice.mean(),
        "volume_r2": r2_score(subject_label_volume, subject_volume),
        "volume_r2_ci": np.percentile(bootstrap_volume_r2, [2.5, 97.5]).tolist(),
        "volume_r": np.corrcoef(subject_label_volume, subject_volume)[0, 1],
        "fold_alpha": fold_alphas,
        "fold_threshold": fold_thresholds,
        "subject_dice": subject_dice.tolist(),
        "subject_average_precision": subject_average_precision.tolist(),
        "subject_voxel_auroc": subject_voxel_auroc.tolist(),
        "subject_volume_ml": subject_volume.tolist(),
        "subject_label_volume_ml": subject_label_volume.tolist(),
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
    device = patch_labels[train_ids[0]].device
    train_labels = torch.cat([patch_labels[ii] for ii in train_ids])
    keep = torch.ones(len(train_labels), dtype=torch.bool, device=device)
    if max_negative_ratio is not None:
        keep = subsample_negatives(train_labels, max_negative_ratio)
    # only each subject's kept rows go to the device, as float32
    keeps = keep.cpu().split([len(patch_labels[ii]) for ii in train_ids])
    train_features = torch.cat([patch_features[ii][k] for ii, k in zip(train_ids, keeps)])
    train_features = train_features.float().to(device)
    train_labels = train_labels[keep]
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0, correction=0).clamp_min(1e-6)
    # in place: train_features is a copy
    coef, intercept = fit_logistic(train_features.sub_(mean).div_(std), train_labels, alpha)

    probabilities = []
    for ii in test_ids:
        logits = (patch_features[ii].to(device).float() - mean) / std @ coef + intercept
        probabilities.append(torch.sigmoid(logits))
    return probabilities


def subsample_negatives(labels: Tensor, max_negative_ratio: float) -> Tensor:
    """Which rows to keep: every row with a positive label and at most max_negative_ratio
    negatives per positive."""
    positive = labels.sum(dim=1) > 0
    negative_ids = (~positive).nonzero()[:, 0]
    n_negative = min(len(negative_ids), int(max_negative_ratio * positive.sum()))
    generator = torch.Generator().manual_seed(CV_SEED)
    order = torch.randperm(len(negative_ids), generator=generator).to(negative_ids.device)
    keep = positive.clone()
    keep[negative_ids[order[:n_negative]]] = True
    return keep


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
    max_iter: int = 5000,
) -> tuple[Tensor, Tensor]:
    """
    L2 penalized logistic regression fit with L-BFGS. Each target column is a separate binary
    problem sharing the same features. Targets can be soft (between 0 and 1) or integer labels.
    The loss is summed over chunks of LOGISTIC_CHUNK rows.
    """
    n, d = features.shape
    assert targets.ndim == 2 and len(targets) == n, (
        f"targets {tuple(targets.shape)} do not match {n} samples of {d} features"
    )
    n_outputs = targets.shape[1]

    coef = torch.zeros(d, n_outputs, device=features.device, dtype=features.dtype)
    target_mean = targets.sum(dim=0, dtype=torch.float64).div(n).to(features.dtype)
    intercept = torch.logit(target_mean.clamp(1e-6, 1 - 1e-6))
    coef.requires_grad_(True)
    intercept.requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [coef, intercept], max_iter=max_iter, history_size=10, line_search_fn="strong_wolfe"
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = alpha * coef.square().sum() / (n * n_outputs)
        loss.backward()
        loss = loss.detach()
        for start in range(0, n, LOGISTIC_CHUNK):
            rows = slice(start, start + LOGISTIC_CHUNK)
            logits = features[rows] @ coef + intercept
            chunk_loss = F.binary_cross_entropy_with_logits(
                logits, targets[rows].to(features.dtype), reduction="sum"
            ) / (n * n_outputs)
            chunk_loss.backward()
            loss = loss + chunk_loss.detach()
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
