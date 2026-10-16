"""Benchmark decoding across brain areas and PCA configurations."""

import pandas as pd

from .baselines import evaluate_baselines
from .quantum import evaluate_qnn
from .statistics import bootstrap_ci_metric


def run_benchmark(data, *, subset="rewarded", random_state=42,
                  max_epochs=None, include_controls=True, bootstrap_samples=5000):
    """Return a results table and predictions keyed by (configuration, model).

    Rewarded-arm classical models use stratified event folds;
    QNNs and all-arm classical models use trial groups.
    These different rewarded protocols must not be interpreted as a matched comparison.
    """
    if subset not in {"rewarded", "all"}:
        raise ValueError("subset must be 'rewarded' or 'all'")
    if len(data.classes) != (3 if subset == "rewarded" else 8):
        raise ValueError("The dataset class count does not match the benchmark subset")
    configs = [("PFC", 5, 0, "ring", 6), ("HPC", 0, 5, "ring", 6),
               ("PFC + HPC", 4, 1, "ring", 6)] if subset == "rewarded" else [
               ("PFC", 8, 0, "all", 8), ("HPC", 0, 8, "ring", 6),
               ("PFC + HPC", 7, 1, "ring", 6)]
    if subset == "rewarded":
        configs.append(("PFC (30 PCs)", 30, 0, None, None))
        if include_controls:
            configs.append(("PFC (shuffled labels)", 5, 0, None, None))
    rows, predictions = [], {}
    for area, n_pfc, n_hpc, topology, layers in configs:
        pfc = data.pfc if n_pfc else None
        hpc = data.hpc if n_hpc else None
        common = dict(protocol=subset, random_state=random_state)
        metrics, preds = evaluate_baselines(
            pfc, hpc, data.y, n_pfc, n_hpc,
            groups=data.groups if subset == "all" else None,
            shuffle_labels=area == "PFC (shuffled labels)", **common)
        for model, values in metrics.items():
            rows.append(dict(Input=area, Model=model, n_PCs_PFC=n_pfc, n_PCs_HPC=n_hpc,
                             CV="trial groups" if subset == "all" else "stratified events",
                             Architecture="classical", **values))
            predictions[area, model] = preds[model]
        if topology is not None:
            values, preds = evaluate_qnn(
                pfc, hpc, data.y, n_pfc, n_hpc, groups=data.groups,
                topology=topology, n_layers=layers, max_epochs=max_epochs, **common)
            low, high = bootstrap_ci_metric(*preds, B=bootstrap_samples)
            rows.append(dict(Input=area, Model="QNN", n_PCs_PFC=n_pfc, n_PCs_HPC=n_hpc,
                             CV="trial groups",
                             Architecture=f"{topology}, {layers} layers, {n_pfc + n_hpc} qubits",
                             event_bootstrap_low=low, event_bootstrap_high=high, **values))
            predictions[area, "QNN"] = preds
    return pd.DataFrame(rows), predictions
