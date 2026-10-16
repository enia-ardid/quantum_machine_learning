import pickle

import numpy as np
import pytest

from neural_decoding.data import load_events
from neural_decoding.preprocessing import AreaPCA, NeuronZScore


@pytest.mark.parametrize("pre", [False, True])
@pytest.mark.parametrize("subset,classes", [("rewarded", [2, 4, 8]), ("all", list(range(1, 9)))])
def test_export_axis_order_label_mapping_and_event_alignment(tmp_path, pre, subset, classes):
    prefix = "pre_" if pre else ""
    labels = np.tile(classes, 3)
    tensor = np.arange(4 * 2 * len(labels)).reshape(4, 2, len(labels))
    exported = {
        f"{prefix}tensor_pfc_{subset}": tensor,
        f"{prefix}tensor_hpc_{subset}": tensor + 1,
        f"{prefix}event_arm_ids_{subset}": labels,
        f"{prefix}trial_ids_{subset}": np.repeat(np.arange(3), len(classes)),
    }
    path = tmp_path / "synthetic.pkl"
    path.write_bytes(pickle.dumps(exported))
    data = load_events(path, subset=subset, pre=pre)
    np.testing.assert_array_equal(data.pfc[1], tensor[:, :, 1])
    np.testing.assert_array_equal(data.classes[data.y], labels)
    swapped = load_events(path, subset=subset, pre=pre, tensor_order="time_neurons_events")
    np.testing.assert_array_equal(swapped.pfc[1], tensor[:, :, 1].T)
    assert data.pfc[1].shape != swapped.pfc[1].shape
    exported[f"{prefix}trial_ids_{subset}"] = np.array([0])
    path.write_bytes(pickle.dumps(exported))
    with pytest.raises(ValueError, match="matching event counts"):
        load_events(path, subset=subset, pre=pre)


def test_missing_private_export_explains_availability(tmp_path):
    with pytest.raises(FileNotFoundError, match="private"):
        load_events(tmp_path / "qnn_data.pkl")


def test_normalization_uses_only_training_events():
    train = [np.array([[1., 3.], [4., 4.]]), np.array([[5., 7.], [4., 4.]])]
    fitted = NeuronZScore().fit(train)
    np.testing.assert_allclose(fitted.mean_.ravel(), [4., 4.])
    held_out = [np.array([[100., 100.], [5., 5.]])]
    result = fitted.transform(held_out)
    assert result[0][0, 0] > 40
    assert np.isfinite(result).all()
    np.testing.assert_allclose(fitted.mean_.ravel(), [4., 4.])


def test_area_pca_does_not_refit_on_held_out_data(events):
    fitted = AreaPCA(4, 1).fit(events.pfc[:48], events.hpc[:48])
    train = fitted.transform(events.pfc[:48], events.hpc[:48])
    np.testing.assert_allclose(train.mean(axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(train.std(axis=0), 1, atol=1e-6)
    before = [(z.mean_.copy(), pca.components_.copy()) for z, pca in fitted.transforms_]
    held_out = fitted.transform([event + 100 for event in events.pfc[48:]], events.hpc[48:])
    assert held_out.shape == (24, 5)
    assert held_out.dtype == np.float32
    for (mean, components), (z, pca) in zip(before, fitted.transforms_):
        np.testing.assert_array_equal(mean, z.mean_)
        np.testing.assert_array_equal(components, pca.components_)


@pytest.mark.parametrize("count,events_count", [(41, 72), (6, 5)])
def test_pca_rejects_dimension_changes_instead_of_mismatching_qubits(events, count, events_count):
    with pytest.raises(ValueError, match="training fold supports"):
        AreaPCA(count, 0).fit(events.pfc[:events_count], None)


def test_requested_area_cannot_be_silently_omitted(events):
    with pytest.raises(ValueError, match="HPC events are required"):
        AreaPCA(4, 1).fit(events.pfc, None)
