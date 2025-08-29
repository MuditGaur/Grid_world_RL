"""
Theory of Mind observer models.

This module implements a compact ToM-style observer that predicts an agent's
next action by fusing:
- Character embedding (summary of past episodes; behavioral prior)
- Mental embedding (recent-window context; short-term situational context)
- Belief-state representation over the current world state

Belief-state backends:
- classical: higher-dimensional classical belief embedding (e.g., 64-d)
- classical_matched: parameter-matched smaller classical embedding (e.g., 32-d)
- quantum: variational quantum circuit embedding (requires PennyLane)
- hybrid / hybrid_matched: fusion of classical + quantum with matched sizes

The observer outputs logits over 5 actions (UP/DOWN/LEFT/RIGHT/STAY).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .belief_states import create_belief_state

class ToMObserver(nn.Module):
    """General ToM-style observer.

    Inputs
    ------
    - char:   (B, char_dim) character summary input
    - mental: (B, mental_dim) mental-state window input
    - state:  (B, state_dim) raw state features to be mapped into belief

    Config
    ------
    - belief_type: 'classical' | 'classical_matched' | 'quantum' | 'hybrid' | 'hybrid_matched'
    - n_qubits:    qubits used by quantum/hybrid backends

    Flow
    ----
    state --(belief_state)--> belief
    [char, mental, belief] --concat--> policy head --> logits
    """
    def __init__(self, char_dim=22, mental_dim=17, state_dim=17,
                 belief_type: str = "classical", n_qubits: int = 8, device="cpu"):
        super().__init__()
        assert belief_type in {"classical", "quantum", "hybrid", "classical_matched", "hybrid_matched"}
        self.belief_type = belief_type
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
        
        # Belief state representation
        if belief_type == "classical":
            self.belief_state = create_belief_state("classical", state_dim=state_dim)
            belief_dim = 64  # Classical belief state output dimension
        elif belief_type == "classical_matched":
            self.belief_state = create_belief_state("classical_matched", state_dim=state_dim)
            belief_dim = 32  # Parameter-matched classical belief state output dimension
        elif belief_type == "quantum":
            self.belief_state = create_belief_state("quantum", state_dim=state_dim, n_qubits=n_qubits)
            belief_dim = 32  # Quantum belief state output dimension
        elif belief_type == "hybrid":
            self.belief_state = create_belief_state("hybrid", state_dim=state_dim, 
                                                   classical_dim=32, quantum_qubits=n_qubits//2)
            belief_dim = 64  # Hybrid belief state output dimension (32 classical + 32 quantum)
        else:  # hybrid_matched
            self.belief_state = create_belief_state("hybrid_matched", state_dim=state_dim, 
                                                   classical_dim=16, quantum_qubits=n_qubits//2)
            belief_dim = 32  # Parameter-matched hybrid belief state output dimension (16 classical + 16 quantum)

        # Policy head: combines character, mental state, and belief state
        fused_dim = 32 + 32 + belief_dim  # char + mental + belief
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, char, mental, state):
        """Compute action logits from character, mental, and state inputs.

        The belief-state module consumes `state` and produces a belief
        embedding whose dimensionality depends on the configured backend.
        Character and mental encoders are classical MLPs.
        """
        # Inputs: (B, char_dim), (B, mental_dim), (B, state_dim)
        c = self.char_enc(char)
        m = self.mental_enc(mental)
        
        # Encode state into belief representation
        belief = self.belief_state(state)
        
        # Combine all representations
        x = torch.cat([c, m, belief], dim=-1)
        logits = self.head(x)
        return logits
    
    def get_belief_representation(self, state):
        """Return the belief-state embedding for a batch of `state` inputs."""
        return self.belief_state(state)
