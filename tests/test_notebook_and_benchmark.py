import base64
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pytest

from neural_decoding.data import EventData
from neural_decoding.experiments import run_benchmark
from neural_decoding.plotting import plot_accuracy_comparison, plot_confusion

ROOT = Path(__file__).resolve().parents[1]


def test_all_28_original_notebook_images_are_preserved_byte_for_byte():
    notebook = nbformat.read(ROOT / "qnn.ipynb", as_version=4)
    nbformat.validate(notebook)
    hashes = []
    for cell in notebook.cells:
        for bundle in cell.get("attachments", {}).values():
            raw = base64.b64decode(bundle["image/png"])
            hashes.append(hashlib.sha256(raw).hexdigest())
    assert len(hashes) == 28
    # Digest of ordered PNG hashes from the original notebook at 1e9c0ee.
    assert hashlib.sha256("".join(hashes).encode()).hexdigest() == (
        "edcbef76941f1aefa610be31097fcf1d8a63652b0ebcfdf74081f1c85bf47193")
    preview = (ROOT / "docs/figures/historical-all-arms.png").read_bytes()
    assert hashlib.sha256(preview).hexdigest() == hashes[13]


def test_notebook_executes_without_private_data_or_replacing_the_gallery():
    notebook = nbformat.read(ROOT / "qnn.ipynb", as_version=4)
    namespace = {}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, "qnn.ipynb", "exec"), namespace)
    assert namespace["events"] is None
    assert namespace["qnn_predictions"] is None


@pytest.mark.parametrize("subset,expected_rows", [("rewarded", 23), ("all", 15)])
def test_complete_benchmark_and_plots_on_synthetic_inputs(events, subset, expected_rows):
    if subset == "all":
        events = EventData(events.pfc, events.hpc, np.tile(np.arange(8), 9),
                           np.repeat(np.arange(9), 8), np.arange(1, 9))
    table, predictions = run_benchmark(events, subset=subset, max_epochs=1,
                                       bootstrap_samples=10)
    assert len(table) == expected_rows
    assert table["n_evaluated"].eq(72).all()
    assert table["n_folds"].eq(3).all()
    assert table["acc"].between(0, 1).all()
    quantum = table[table.Model == "QNN"]
    assert quantum.CV.eq("trial groups").all()
    assert "6 layers" in quantum[quantum.Input == "HPC"].iloc[0].Architecture
    ax = plot_confusion(*predictions["HPC", "QNN"], labels=events.classes,
                        title="Synthetic software test")
    assert np.asarray(ax.images[0].get_array()).sum() == len(events.y)
    comparison = plot_accuracy_comparison({"QNN": quantum.iloc[0].acc},
                                          n_classes=len(events.classes),
                                          title="Synthetic software test")
    assert comparison.get_ylim()[1] >= 1
    plt.close("all")
