"""
Quantum variational layers for PyTorch using PennyLane.

This module provides quantum encoder layers that can be integrated into
classical neural networks for hybrid quantum-classical architectures.
"""

import torch
import torch.nn as nn

# Optional: PennyLane for quantum layers
try:
    import pennylane as qml
    _HAS_PENNYLANE = True
except Exception as e:
    qml = None
    _HAS_PENNYLANE = False

class QuantumEncoder(nn.Module):
    """A small variational quantum circuit wrapped as a Torch layer.
    - Input features are linearly projected to n_qubits then angle-encoded.
    - Circuit uses StronglyEntanglingLayers L layers.
    - Outputs expectation of PauliZ on each qubit (vector in R^{n_qubits}).
    """
    def __init__(self, in_dim: int, n_qubits: int = 8, n_layers: int = 2, dev_name: str = "default.qubit"):
        super().__init__()
        assert _HAS_PENNYLANE, "PennyLane not installed. Please install pennylane."
        self.in_dim = in_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.proj = nn.Linear(in_dim, n_qubits)
        # Build QNode with trainable weights
        dev = qml.device(dev_name, wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def qnode(inputs, weights):
            # inputs shape: (n_qubits,)
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        x = self.proj(x)
        return self.qlayer(x)
