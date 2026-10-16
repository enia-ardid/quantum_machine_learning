"""Confusion matrices and accuracy comparisons for neural decoding."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion(y_true, y_pred, *, labels, title):
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("Confusion matrices require nonempty, aligned predictions")
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    _, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
        ax=ax, cmap="Blues", values_format="d")
    ax.set_title(title)
    return ax


def plot_accuracy_comparison(accuracies, *, n_classes, title):
    _, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    ax.bar(list(accuracies), list(accuracies.values()))
    ax.axhline(1 / n_classes, color="tab:red", linestyle="--",
               label=f"Uniform random guess (1/{n_classes})")
    ax.set(ylabel="Mean fold accuracy", title=title, ylim=(0, 1.05))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend()
    return ax
