"""
Theory of Mind observer models.

This module implements different ToM-style observers that predict agent actions
using classical, quantum, or hybrid state representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantum_layer import QuantumEncoder

# Check if PennyLane is available
try:
    import pennylane as qml
    _HAS_PENNYLANE = True
except Exception as e:
    qml = None
    _HAS_PENNYLANE = False

class ClassicalStateEncoder(nn.Module):
    """Classical neural network state encoder."""
    def __init__(self, in_dim: int, hid: int = 64, out: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, out), nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)


class ToMObserver(nn.Module):
    """General ToM-style observer combining:
        - Character embedding from past episodes (classical GRU over summaries)
        - Mental-state embedding from recent window (classical MLP)
        - State encoder: classical, quantum, or both
    Predicts next action distribution (5-way softmax).
    """
    def __init__(self, char_dim=22, mental_dim=17, state_dim=17,
                 mode: str = "classical", n_qubits: int = 8, device="cpu"):
        super().__init__()
        assert mode in {"classical", "quantum", "hybrid"}
        self.mode = mode
        self.device = device

        # Character encoder: simple MLP (could be GRU if we kept sequence dims)
        self.char_enc = nn.Sequential(
            nn.Linear(char_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        # Mental encoder
        self.mental_enc = nn.Sequential(
            nn.Linear(mental_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        # State encoder(s)
        if mode == "classical":
            self.state_enc_c = ClassicalStateEncoder(state_dim, hid=64, out=32)
            fused_dim = 32 + 32 + 32
        elif mode == "quantum":
            if not _HAS_PENNYLANE:
                raise RuntimeError("Quantum mode requires pennylane to be installed.")
            self.state_enc_q = QuantumEncoder(state_dim, n_qubits=n_qubits, n_layers=2)
            # keep a small linear head to map qubits -> 32
            self.q_head = nn.Linear(n_qubits, 32)
            fused_dim = 32 + 32 + 32
        else:  # hybrid
            if not _HAS_PENNYLANE:
                raise RuntimeError("Hybrid mode requires pennylane to be installed.")
            self.state_enc_c = ClassicalStateEncoder(state_dim, hid=64, out=32)
            self.state_enc_q = QuantumEncoder(state_dim, n_qubits=n_qubits, n_layers=2)
            self.q_head = nn.Linear(n_qubits, 16)
            self.c_head = nn.Identity()
            fused_dim = 32 + 32 + (16+32)

        # Policy head
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, char, mental, state):
        # Inputs: (B, char_dim), (B, mental_dim), (B, state_dim)
        c = self.char_enc(char)
        m = self.mental_enc(mental)
        if self.mode == "classical":
            s = self.state_enc_c(state)
        elif self.mode == "quantum":
            s = self.q_head(self.state_enc_q(state))
        else:
            sq = self.q_head(self.state_enc_q(state))
            sc = self.state_enc_c(state)
            s = torch.cat([sc, sq], dim=-1)
        x = torch.cat([c, m, s], dim=-1)
        logits = self.head(x)
        return logits
