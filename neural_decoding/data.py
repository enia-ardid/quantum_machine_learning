"""Load event-aligned neural activity and trial labels."""

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np


@dataclass
class EventData:
    pfc: list[np.ndarray]
    hpc: list[np.ndarray]
    y: np.ndarray
    groups: np.ndarray
    classes: np.ndarray


def tensor_to_event_list(tensor):
    """Split a (neurons, time bins, events) tensor into event matrices."""
    tensor = np.asarray(tensor)
    if tensor.ndim != 3 or 0 in tensor.shape:
        raise ValueError(f"Expected a nonempty 3D tensor, got {tensor.shape}")
    if not np.issubdtype(tensor.dtype, np.number) or not np.isfinite(tensor).all():
        raise ValueError("Neural tensors must contain finite numeric values")
    return [tensor[:, :, event] for event in range(tensor.shape[2])]


def load_events(path, *, subset="rewarded", pre=False, tensor_order="neurons_time_events"):
    """Load a trusted local pickle; see docs/methodology.md for the schema.

    Pickle can execute code. Only open exports from a trusted source.
    The default matches the upstream construct_tensors.ipynb exporter. Select
    'time_neurons_events' explicitly for exports with those first two axes reversed.
    """
    if subset not in {"rewarded", "all"}:
        raise ValueError("subset must be 'rewarded' or 'all'")
    if tensor_order not in {"neurons_time_events", "time_neurons_events"}:
        raise ValueError("tensor_order must be 'neurons_time_events' or 'time_neurons_events'")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is unavailable. The neural recordings are private and "
            "are not distributed with this repository. View qnn.ipynb for saved results."
        )
    with path.open("rb") as stream:
        data = pickle.load(stream)
    prefix = "pre_" if pre else ""
    keys = [f"{prefix}{name}_{subset}" for name in
            ("tensor_pfc", "tensor_hpc", "event_arm_ids", "trial_ids")]
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"Missing export keys: {', '.join(missing)}")
    tensors = [np.asarray(data[key]) for key in keys[:2]]
    if any(t.ndim != 3 for t in tensors):
        raise ValueError("Exported PFC and HPC tensors must have three axes")
    if tensor_order == "time_neurons_events":
        tensors = [tensor.transpose(1, 0, 2) for tensor in tensors]
    pfc, hpc = [tensor_to_event_list(tensor) for tensor in tensors]
    labels, groups = [np.asarray(data[key]) for key in keys[2:]]
    if labels.ndim != 1 or groups.ndim != 1:
        raise ValueError("Arm IDs and trial IDs must be one-dimensional")
    if not (len(pfc) == len(hpc) == len(labels) == len(groups)):
        raise ValueError("PFC, HPC, arm IDs and trial IDs must have matching event counts")
    classes, y = np.unique(labels, return_inverse=True)
    expected = 3 if subset == "rewarded" else 8
    if len(classes) != expected:
        raise ValueError(f"Expected {expected} classes for {subset}, found {len(classes)}")
    return EventData(pfc, hpc, y.astype(np.int32), groups, classes)
