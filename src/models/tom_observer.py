"""
Theory of Mind observer models.

This module implements a compact ToM-style observer that predicts an agent's
next action by fusing:
- Character embedding (summary of past episodes; behavioral prior)
- Mental embedding (recent-window context; short-term situational context)
- Encoded state representation derived from current state features

The observer outputs logits over 5 actions (UP/DOWN/LEFT/RIGHT/STAY).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ToMObserver(nn.Module):
    """General ToM-style observer.

    Inputs
    ------
    - char:   (B, char_dim) character summary input
    - mental: (B, mental_dim) mental-state window input
    - state:  (B, state_dim) raw state features to be encoded

    Config
    ------
    - belief_type: kept for backward-compatibility (ignored)
    - n_qubits:    kept for backward-compatibility (ignored)

    Flow
    ----
    state --(state_encoder)--> state_emb
    [char, mental, state_emb] --concat--> policy head --> logits
    """
    def __init__(self, char_dim=22, mental_dim=17, state_dim=17,
                 belief_type: str = "classical", n_qubits: int = 8, device="cpu"):
        super().__init__()
        # `belief_type` and `n_qubits` are accepted but ignored for back-compat
        self.belief_type = belief_type
        self.device = device

        # Character encoder: simple MLP (could be GRU if we kept sequence dims)
        self.char_enc = nn.Sequential(
            nn.Linear(char_dim, char_dim), nn.ReLU(),
            nn.Linear(char_dim, char_dim), nn.ReLU(),
        )
        
        # Mental encoder
        self.mental_enc = nn.Sequential(
            nn.Linear(mental_dim, mental_dim), nn.ReLU(),
            nn.Linear(mental_dim, mental_dim), nn.ReLU(),
        )
        
        # State encoder representation (replaces prior belief-state module)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )

        # Policy head: combines character, mental state, and encoded state
        fused_dim = char_dim + mental_dim + 32  # char + mental + state_emb
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, char, mental, state):
        """Compute action logits from character, mental, and state inputs.

        The state encoder consumes `state` and produces a 32-d embedding.
        Character and mental encoders are classical MLPs.
        """
        # Inputs: (B, char_dim), (B, mental_dim), (B, state_dim)
        c = self.char_enc(char)
        m = self.mental_enc(mental)
        
        # Encode state into representation
        state_emb = self.state_encoder(state)
        
        # Combine all representations
        x = torch.cat([c, m, state_emb], dim=-1)
        logits = self.head(x)
        return logits
    
    def get_belief_representation(self, state):
        """Deprecated: returns the state encoder embedding for compatibility."""
        return self.state_encoder(state)
