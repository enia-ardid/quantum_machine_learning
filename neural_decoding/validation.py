"""Shared fold construction, preprocessing and metric reporting."""

import warnings

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold

from .preprocessing import AreaPCA


def make_splits(y, groups=None, n_splits=3, random_state=42):
    y = np.asarray(y)
    if y.ndim != 1 or len(np.unique(y)) < 2:
        raise ValueError("At least two classes and a one-dimensional label array are required")
    if n_splits < 2:
        raise ValueError("Cross-validation requires at least two splits")
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != y.shape:
            raise ValueError("Trial IDs must match the label array")
        count = min(n_splits, len(np.unique(groups)))
        if count < 2:
            raise ValueError("Group cross-validation requires at least two trials")
        splitter = GroupKFold(n_splits=count)
        return list(splitter.split(np.arange(len(y)), y, groups))
    count = min(n_splits, np.unique(y, return_counts=True)[1].min())
    if count < 2:
        raise ValueError("Stratified cross-validation requires at least two events per class")
    splitter = StratifiedKFold(n_splits=count, shuffle=True, random_state=random_state)
    return list(splitter.split(np.arange(len(y)), y))


def fold_features(pfc, hpc, train, test, n_pcs_pfc, n_pcs_hpc, random_state):
    def subset(events, indices):
        return None if events is None else [events[i] for i in indices]

    train_areas = subset(pfc, train), subset(hpc, train)
    test_areas = subset(pfc, test), subset(hpc, test)
    projection = AreaPCA(n_pcs_pfc, n_pcs_hpc, random_state).fit(*train_areas)
    return projection.transform(*train_areas), projection.transform(*test_areas)


def usable_fold(y, train, test, *, require_all_classes):
    reason = None
    if len(train) < 15 or len(test) < 5:
        reason = "fewer than 15 training events or 5 test events"
    elif len(np.unique(y[train])) < (len(np.unique(y)) if require_all_classes else 2):
        reason = "too few training classes"
    if reason:
        warnings.warn(f"Skipping fold: {reason}", RuntimeWarning, stacklevel=2)
        return False
    return True


def summarize_scores(accuracies, f1s, n_evaluated):
    if not accuracies:
        raise ValueError("No valid folds; check trial counts, class coverage and event counts")
    return {
        "acc": float(np.mean(accuracies)), "acc_sd": float(np.std(accuracies)),
        "f1": float(np.mean(f1s)), "f1_sd": float(np.std(f1s)),
        "n_folds": len(accuracies), "n_evaluated": n_evaluated,
    }
