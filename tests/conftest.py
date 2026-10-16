"""Synthetic inputs test software behavior, not decoding performance."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
import torch

from neural_decoding.data import EventData

torch.set_num_threads(1)


@pytest.fixture
def events():
    rng = np.random.RandomState(7)
    labels = np.tile(np.arange(3), 24)
    pfc = [rng.normal(size=(8, 5)) for _ in labels]
    hpc = [rng.normal(size=(8, 5)) for _ in labels]
    return EventData(pfc, hpc, labels, np.repeat(np.arange(24), 3), np.array([2, 4, 8]))
