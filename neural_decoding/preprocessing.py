"""Training-fold normalization and separate PCA projections for each brain area."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class NeuronZScore:
    """Estimate each neuron's mean and standard deviation over training events/time."""

    def fit(self, events):
        if not events:
            raise ValueError("Cannot fit normalization on an empty event list")
        concatenated = np.concatenate(events, axis=1)
        self.mean_ = concatenated.mean(axis=1, keepdims=True)
        self.std_ = concatenated.std(axis=1, keepdims=True) + 1e-9
        return self

    def transform(self, events):
        return [(event - self.mean_) / self.std_ for event in events]


def flatten_events(events):
    if not events:
        raise ValueError("Cannot flatten an empty event list")
    return np.stack([event.reshape(-1) for event in events])


class AreaPCA:
    """Z-score, flatten, project each area, concatenate, then standardize features.

    All fitted statistics come from the training fold. Requested component counts
    are enforced so that a QNN's qubit count matches its input dimension.
    """

    def __init__(self, n_pcs_pfc, n_pcs_hpc, random_state=42):
        if min(n_pcs_pfc, n_pcs_hpc) < 0 or n_pcs_pfc + n_pcs_hpc == 0:
            raise ValueError("Select a positive number of PCs for at least one area")
        self.counts = (n_pcs_pfc, n_pcs_hpc)
        self.random_state = random_state

    def fit(self, pfc, hpc):
        self.transforms_ = []
        features = []
        for name, count, events in zip(("PFC", "HPC"), self.counts, (pfc, hpc)):
            if count == 0:
                self.transforms_.append(None)
                continue
            if events is None or len(events) == 0:
                raise ValueError(f"{name} events are required for {count} PCs")
            normalizer = NeuronZScore().fit(events)
            flat = flatten_events(normalizer.transform(events))
            if count > min(flat.shape):
                raise ValueError(
                    f"{name}: requested {count} PCs but this training fold supports "
                    f"at most {min(flat.shape)}; reduce the requested count"
                )
            pca = PCA(n_components=count, random_state=self.random_state).fit(flat)
            self.transforms_.append((normalizer, pca))
            features.append(pca.transform(flat))
        self.scaler_ = StandardScaler().fit(np.hstack(features))
        return self

    def transform(self, pfc, hpc):
        features = []
        for events, transform in zip((pfc, hpc), self.transforms_):
            if transform is not None:
                if events is None:
                    raise ValueError("An area used during fit is missing during transform")
                normalizer, pca = transform
                features.append(pca.transform(flatten_events(normalizer.transform(events))))
        return self.scaler_.transform(np.hstack(features)).astype(np.float32)
