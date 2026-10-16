"""Event bootstrap intervals and a complete-CV label permutation test."""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score

from .baselines import make_classifiers
from .validation import fold_features, make_splits


def bootstrap_ci_metric(y_true, y_pred, metric_fn=accuracy_score, *,
                        B=5000, seed=0, alpha=0.05):
    """Percentile interval over paired events, conditional on fitted predictions.

    Resampling individual events does not account for within-trial dependence
    or training variability.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("Bootstrap requires nonempty, aligned prediction arrays")
    if B < 1 or not 0 < alpha < 1:
        raise ValueError("Use B >= 1 and 0 < alpha < 1")
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(B):
        indices = rng.randint(0, len(y_true), size=len(y_true))
        scores.append(metric_fn(y_true[indices], y_pred[indices]))
    scores = np.sort(scores)
    return float(scores[int(alpha / 2 * B)]), float(scores[int((1 - alpha / 2) * B)])


def permutation_test_baseline(pfc, hpc, y, n_pcs_pfc, n_pcs_hpc, clf_name, *,
                              n_splits=3, n_perms=100, random_state=0,
                              groups=None, n_estimators=100):
    """Compare mean CV accuracy with n_perms complete label-permuted CV scores.

    With groups, labels are permuted only within each trial, assuming those events
    are exchangeable under the null. Without groups, all events are exchangeable.
    """
    if n_perms < 1:
        raise ValueError("n_perms must be positive")
    models = make_classifiers(random_state, n_estimators)
    if clf_name not in models:
        raise ValueError(f"Unknown classifier: {clf_name}")
    y = np.asarray(y)
    if groups is not None:
        groups = np.asarray(groups)
    splits = make_splits(y, groups, n_splits, random_state)
    rng = np.random.RandomState(random_state)
    folds = []
    for train, test in splits:
        x_train, x_test = fold_features(
            pfc, hpc, train, test, n_pcs_pfc, n_pcs_hpc, random_state)
        x_train += rng.normal(0, 1e-6, x_train.shape)
        x_test += rng.normal(0, 1e-6, x_test.shape)
        folds.append((train, test, x_train, x_test))

    def score(labels):
        accuracies = []
        for train, test, x_train, x_test in folds:
            if len(np.unique(labels[train])) < 2:
                raise ValueError("A permuted training fold has fewer than two classes")
            model = clone(models[clf_name]).fit(x_train, labels[train])
            accuracies.append(accuracy_score(labels[test], model.predict(x_test)))
        return float(np.mean(accuracies))

    observed = score(y)
    permuted_scores = []
    for _ in range(n_perms):
        if groups is None:
            permuted = rng.permutation(y)
        else:
            permuted = y.copy()
            for group in np.unique(groups):
                indices = np.flatnonzero(groups == group)
                permuted[indices] = rng.permutation(y[indices])
        permuted_scores.append(score(permuted))
    null = np.asarray(permuted_scores)
    return {
        "accuracy": observed,
        "p_value": float((np.count_nonzero(null >= observed) + 1) / (n_perms + 1)),
        "permuted_accuracy_mean": float(null.mean()),
        "permuted_scores": null,
    }
