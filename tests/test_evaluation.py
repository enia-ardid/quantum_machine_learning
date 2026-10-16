import numpy as np
import pytest
import torch
from sklearn.dummy import DummyClassifier

from neural_decoding.baselines import evaluate_baselines
from neural_decoding.quantum import QNNClassifier, evaluate_qnn
from neural_decoding.statistics import bootstrap_ci_metric, permutation_test_baseline
from neural_decoding.validation import make_splits


def test_grouped_baselines_use_every_trial_once_and_share_splits(events):
    splits = make_splits(events.y, events.groups)
    for train, test in splits:
        assert set(events.groups[train]).isdisjoint(events.groups[test])
    expected = np.concatenate([events.y[test] for _, test in splits])
    metrics, predictions = evaluate_baselines(
        events.pfc, None, events.y, 2, 0, groups=events.groups)
    assert set(metrics) == {"LR", "LinSVM", "RBF_SVM", "RF"}
    for name in metrics:
        assert metrics[name]["n_evaluated"] == len(events.y)
        assert metrics[name]["n_folds"] == 3
        np.testing.assert_array_equal(predictions[name][0], expected)


@pytest.mark.parametrize("groups,labels,message", [
    (np.zeros(6), np.tile([0, 1], 3), "two trials"),
    (None, np.array([0, 0, 1]), "two events per class"),
])
def test_invalid_cv_fails_clearly(groups, labels, message):
    with pytest.raises(ValueError, match=message):
        make_splits(labels, groups)


def test_no_evaluated_folds_is_not_reported_as_zero_accuracy(events):
    with pytest.warns(RuntimeWarning, match="Skipping fold"):
        with pytest.raises(ValueError, match="No valid folds"):
            evaluate_baselines(events.pfc[:9], None, events.y[:9], 2, 0)


@pytest.mark.parametrize("topology,n_classes", [("ring", 3), ("all", 8)])
def test_qnn_gradients_reach_quantum_weights_and_classical_head(topology, n_classes):
    torch.manual_seed(17)
    model = QNNClassifier(3, 2, topology=topology, n_classes=n_classes)
    x = torch.randn(8, 3)
    logits = model(x)
    assert logits.shape == (8, n_classes)
    torch.nn.functional.cross_entropy(logits, torch.arange(8) % n_classes).backward()
    for parameter in (model.input_scale, model.qlayer.weights, model.postproc.weight):
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("protocol", ["rewarded", "all"])
def test_short_qnn_training_keeps_trial_folds_and_predicts_all_events(events, protocol):
    metrics, (truth, prediction) = evaluate_qnn(
        events.pfc, None, events.y, 2, 0, groups=events.groups,
        protocol=protocol, n_layers=1, max_epochs=2)
    assert metrics["n_folds"] == 3
    assert metrics["n_evaluated"] == len(events.y)
    assert truth.shape == prediction.shape == events.y.shape
    assert set(prediction).issubset({0, 1, 2})


def test_bootstrap_rejects_empty_results_and_handles_perfect_predictions():
    with pytest.raises(ValueError, match="nonempty"):
        bootstrap_ci_metric([], [])
    assert bootstrap_ci_metric([0, 1, 2], [0, 1, 2], B=20) == (1., 1.)


def test_permutation_null_uses_complete_cv_scores_and_all_requested_permutations(events, monkeypatch):
    from neural_decoding import statistics

    fits = []

    class RecordingDummy(DummyClassifier):
        def fit(self, x, y, **kwargs):
            fits.append(y.copy())
            return super().fit(x, y, **kwargs)

    monkeypatch.setattr(statistics, "make_classifiers", lambda *args: {
        "LR": RecordingDummy(strategy="most_frequent")})
    result = permutation_test_baseline(events.pfc, None, events.y, 2, 0, "LR",
                                       groups=events.groups, n_perms=7)
    assert len(fits) == 3 * (7 + 1)
    assert result["permuted_scores"].shape == (7,)
    # Every trial contains all three labels, including after within-trial shuffles.
    np.testing.assert_allclose(result["permuted_scores"], 1 / 3)
    assert result["accuracy"] == pytest.approx(1 / 3)
    assert result["p_value"] == 1.
    assert any(not np.array_equal(fits[0], permuted) for permuted in fits[3:])
