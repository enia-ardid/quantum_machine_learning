"""Classical classifiers for maze-arm decoding."""

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

from .validation import fold_features, make_splits, summarize_scores, usable_fold


def make_classifiers(random_state=42, n_estimators=100):
    svm_options = dict(C=1.0, tol=1e-3, class_weight="balanced",
                       decision_function_shape="ovr", random_state=random_state)
    return {
        "LR": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        "LinSVM": SVC(kernel="linear", **svm_options),
        "RBF_SVM": SVC(kernel="rbf", gamma="auto", **svm_options),
        "RF": RandomForestClassifier(n_estimators=n_estimators, min_samples_split=2,
                                     min_samples_leaf=1, class_weight="balanced",
                                     random_state=random_state),
    }


def evaluate_baselines(pfc, hpc, y, n_pcs_pfc, n_pcs_hpc, *, groups=None,
                       protocol="rewarded", n_splits=3, random_state=42,
                       shuffle_labels=False):
    """Return fold-mean metrics and concatenated held-out predictions per model.

    `rewarded` uses 100 trees and requires all classes in each training fold;
    `all` uses 200 trees and permits training folds missing rare classes.
    Passing groups keeps every trial entirely in either train or test.
    """
    if protocol not in {"rewarded", "all"}:
        raise ValueError("protocol must be 'rewarded' or 'all'")
    y = np.asarray(y)
    splits = make_splits(y, groups, n_splits, random_state)
    rng = np.random.RandomState(random_state)
    models = make_classifiers(random_state, 100 if protocol == "rewarded" else 200)
    results, predictions = {}, {}
    for name, estimator in models.items():
        accuracies, f1s, truths, predicted = [], [], [], []
        for train, test in splits:
            if not usable_fold(y, train, test, require_all_classes=protocol == "rewarded"):
                continue
            x_train, x_test = fold_features(
                pfc, hpc, train, test, n_pcs_pfc, n_pcs_hpc, random_state)
            if protocol == "rewarded":
                x_train += rng.normal(0, 1e-5, x_train.shape)
                x_test += rng.normal(0, 1e-5, x_test.shape)
            else:
                x_train = x_train + rng.normal(0, 1e-5, x_train.shape)
                x_test = x_test + rng.normal(0, 1e-5, x_test.shape)
            train_labels = rng.permutation(y[train]) if shuffle_labels else y[train]
            model = clone(estimator).fit(x_train, train_labels)
            y_hat = model.predict(x_test)
            accuracies.append(accuracy_score(y[test], y_hat))
            f1s.append(f1_score(y[test], y_hat, average="macro"))
            truths.extend(y[test])
            predicted.extend(y_hat)
        results[name] = summarize_scores(accuracies, f1s, len(truths))
        predictions[name] = (np.asarray(truths), np.asarray(predicted))
    return results, predictions
