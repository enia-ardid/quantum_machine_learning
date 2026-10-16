"""PennyLane circuit and PyTorch training for maze-arm classification."""

import numpy as np
import pennylane as qml
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score

from .validation import fold_features, make_splits, summarize_scores, usable_fold


class QNNClassifier(nn.Module):
    """Angle encoding, variational rotations/CNOTs, and a classical output head."""

    def __init__(self, n_qubits, n_layers, topology="ring", n_classes=3):
        super().__init__()
        if n_qubits < 2 or n_layers < 1 or n_classes < 2:
            raise ValueError("Use at least two qubits/classes and one circuit layer")
        if topology not in {"ring", "all"}:
            raise ValueError("topology must be 'ring' or 'all'")
        self.input_scale = nn.Parameter(torch.tensor(1.0))
        self.norm = nn.LayerNorm(n_qubits)
        self.dropout = nn.Dropout(0.1)
        self.postproc = nn.Linear(n_qubits, n_classes)
        nn.init.kaiming_normal_(self.postproc.weight)
        nn.init.zeros_(self.postproc.bias)
        device = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(device, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Z")
            for layer in range(n_layers):
                for wire in range(n_qubits):
                    qml.RY(weights[layer, wire, 0], wires=wire)
                    qml.RZ(weights[layer, wire, 1], wires=wire)
                if topology == "ring":
                    for wire in range(n_qubits):
                        qml.CNOT(wires=[wire, (wire + 1) % n_qubits])
                else:
                    for source in range(n_qubits):
                        for target in range(source + 1, n_qubits):
                            qml.CNOT(wires=[source, target])
            return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

        self.circuit = circuit
        self.qlayer = qml.qnn.TorchLayer(circuit, {"weights": (n_layers, n_qubits, 2)})

    def forward(self, x):
        x = torch.clamp(x, -np.pi, np.pi) * self.input_scale
        return self.postproc(self.dropout(self.norm(self.qlayer(x))))


def _train(model, x, y, protocol, max_epochs):
    if protocol == "rewarded":
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=50)
    criterion = nn.CrossEntropyLoss()
    best_accuracy, patience_left, best_state = -1.0, 40, None
    for _ in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if protocol == "rewarded":
            scheduler.step()
        else:
            scheduler.step(loss.detach().item())
        # Training accuracy includes dropout and is measured before the update.
        accuracy = (logits.detach().argmax(dim=1) == y).float().mean().item()
        if accuracy > best_accuracy + 1e-4:
            best_accuracy = accuracy
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            patience_left = 40
        else:
            patience_left -= 1
            if patience_left == 0:
                break
    model.load_state_dict(best_state)


def evaluate_qnn(pfc, hpc, y, n_pcs_pfc, n_pcs_hpc, *, groups,
                 protocol="rewarded", topology="ring", n_layers=6,
                 n_splits=3, random_state=42, max_epochs=None):
    """Train on trial-grouped folds; return metrics and held-out predictions.

    The default limits are 600 epochs for rewarded arms and 800 for all arms.
    `max_epochs` permits short software checks; those runs are not benchmarks.
    """
    if protocol not in {"rewarded", "all"}:
        raise ValueError("protocol must be 'rewarded' or 'all'")
    if groups is None:
        raise ValueError("QNN evaluation requires trial IDs")
    max_epochs = (600 if protocol == "rewarded" else 800) if max_epochs is None else max_epochs
    if max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    y = np.asarray(y)
    n_classes = len(np.unique(y))
    if not np.array_equal(np.unique(y), np.arange(n_classes)):
        raise ValueError("QNN labels must be consecutive integers starting at zero")
    splits = make_splits(y, groups, n_splits, random_state)
    torch.manual_seed(random_state)
    accuracies, f1s, truths, predicted = [], [], [], []
    for train, test in splits:
        if not usable_fold(y, train, test, require_all_classes=True):
            continue
        x_train, x_test = fold_features(
            pfc, hpc, train, test, n_pcs_pfc, n_pcs_hpc, random_state)
        model = QNNClassifier(n_pcs_pfc + n_pcs_hpc, n_layers, topology, n_classes)
        _train(model, torch.as_tensor(x_train), torch.as_tensor(y[train], dtype=torch.long),
               protocol, max_epochs)
        model.eval()
        with torch.no_grad():
            y_hat = model(torch.as_tensor(x_test)).argmax(dim=1).numpy()
        accuracies.append(accuracy_score(y[test], y_hat))
        f1s.append(f1_score(y[test], y_hat, average="macro"))
        truths.extend(y[test])
        predicted.extend(y_hat)
    return (summarize_scores(accuracies, f1s, len(truths)),
            (np.asarray(truths), np.asarray(predicted)))
