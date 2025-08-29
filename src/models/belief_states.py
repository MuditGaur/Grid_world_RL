"""
Belief State Representations for Theory of Mind.

This module implements different belief state representations:
- Classical: Traditional neural network belief state
- Quantum: Quantum circuit belief state
- Hybrid: Combination of classical and quantum belief states
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Check if PennyLane is available
try:
    import pennylane as qml
    _HAS_PENNYLANE = True
except Exception as e:
    qml = None
    _HAS_PENNYLANE = False

class ClassicalBeliefState(nn.Module):
    """
    Classical belief state representation using neural networks.
    
    This represents the agent's beliefs about the world state using
    traditional neural network layers.
    """
    def __init__(self, state_dim: int = 17, belief_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.belief_dim = belief_dim
        
        # Belief state encoder: maps current state to belief representation
        self.belief_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, belief_dim),
            nn.Tanh()  # Normalize belief state to [-1, 1]
        )
        
        # Belief state decoder: maps belief back to state space for comparison
        self.belief_decoder = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Belief update mechanism: combines current belief with new observation
        self.belief_update = nn.Sequential(
            nn.Linear(belief_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, belief_dim),
            nn.Tanh()
        )
    
    def encode_belief(self, state: torch.Tensor) -> torch.Tensor:
        """Encode current state into belief representation."""
        return self.belief_encoder(state)
    
    def decode_belief(self, belief: torch.Tensor) -> torch.Tensor:
        """Decode belief representation back to state space."""
        return self.belief_decoder(belief)
    
    def update_belief(self, current_belief: torch.Tensor, new_observation: torch.Tensor) -> torch.Tensor:
        """Update belief state with new observation."""
        combined = torch.cat([current_belief, new_observation], dim=-1)
        return self.belief_update(combined)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode state to belief representation."""
        return self.encode_belief(state)

class QuantumBeliefState(nn.Module):
    """
    Quantum belief state representation using quantum circuits.
    
    This represents the agent's beliefs about the world state using
    quantum variational circuits.
    """
    def __init__(self, state_dim: int = 17, n_qubits: int = 8, n_layers: int = 2):
        super().__init__()
        assert _HAS_PENNYLANE, "Quantum belief state requires pennylane to be installed."
        
        self.state_dim = state_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Project state to qubit count
        self.state_projection = nn.Linear(state_dim, n_qubits)
        
        # Build quantum circuit for belief encoding
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev, interface="torch")
        def quantum_belief_circuit(inputs, weights):
            # Angle embedding of state
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            
            # Variational layers for belief processing
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            
            # Return expectation values as belief representation
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(quantum_belief_circuit, weight_shapes)
        
        # Classical post-processing of quantum outputs
        self.post_process = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Tanh()
        )
        
        # Belief update mechanism (classical)
        self.belief_update = nn.Sequential(
            nn.Linear(32 + state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Tanh()
        )
    
    def encode_belief(self, state: torch.Tensor) -> torch.Tensor:
        """Encode current state into quantum belief representation."""
        # Project state to qubit count
        projected_state = self.state_projection(state)
        
        # Process through quantum circuit
        quantum_output = self.quantum_layer(projected_state)
        
        # Post-process quantum outputs
        belief = self.post_process(quantum_output)
        return belief
    
    def update_belief(self, current_belief: torch.Tensor, new_observation: torch.Tensor) -> torch.Tensor:
        """Update quantum belief state with new observation."""
        combined = torch.cat([current_belief, new_observation], dim=-1)
        return self.belief_update(combined)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode state to quantum belief representation."""
        return self.encode_belief(state)

class HybridBeliefState(nn.Module):
    """
    Hybrid belief state representation combining classical and quantum components.
    
    This represents the agent's beliefs using both classical neural networks
    and quantum circuits, allowing for richer belief representations.
    """
    def __init__(self, state_dim: int = 17, classical_dim: int = 32, 
                 quantum_qubits: int = 6, n_layers: int = 2):
        super().__init__()
        assert _HAS_PENNYLANE, "Hybrid belief state requires pennylane to be installed."
        
        self.state_dim = state_dim
        self.classical_dim = classical_dim
        self.quantum_qubits = quantum_qubits
        self.n_layers = n_layers
        
        # Classical belief component
        self.classical_belief = ClassicalBeliefState(
            state_dim=state_dim, 
            belief_dim=classical_dim, 
            hidden_dim=64
        )
        
        # Quantum belief component
        self.quantum_belief = QuantumBeliefState(
            state_dim=state_dim,
            n_qubits=quantum_qubits,
            n_layers=n_layers
        )
        
        # Fusion layer: combines classical and quantum beliefs
        self.fusion_layer = nn.Sequential(
            nn.Linear(classical_dim + 32, 64),  # 32 is quantum belief output size
            nn.ReLU(),
            nn.Linear(64, classical_dim + 32),
            nn.Tanh()
        )
        
        # Belief update mechanism
        self.belief_update = nn.Sequential(
            nn.Linear(classical_dim + 32 + state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, classical_dim + 32),
            nn.Tanh()
        )
    
    def encode_belief(self, state: torch.Tensor) -> torch.Tensor:
        """Encode current state into hybrid belief representation."""
        # Get classical belief
        classical_belief = self.classical_belief.encode_belief(state)
        
        # Get quantum belief
        quantum_belief = self.quantum_belief.encode_belief(state)
        
        # Fuse beliefs
        combined_belief = torch.cat([classical_belief, quantum_belief], dim=-1)
        fused_belief = self.fusion_layer(combined_belief)
        
        return fused_belief
    
    def update_belief(self, current_belief: torch.Tensor, new_observation: torch.Tensor) -> torch.Tensor:
        """Update hybrid belief state with new observation."""
        combined = torch.cat([current_belief, new_observation], dim=-1)
        return self.belief_update(combined)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode state to hybrid belief representation."""
        return self.encode_belief(state)

class ParameterMatchedClassicalBeliefState(nn.Module):
    """
    Classical belief state representation with parameter count matched to quantum model.
    
    This version uses smaller hidden dimensions to achieve ~36K parameters,
    similar to the quantum belief state with 8 qubits.
    """
    def __init__(self, state_dim: int = 17, belief_dim: int = 32, hidden_dim: int = 48):
        super().__init__()
        self.state_dim = state_dim
        self.belief_dim = belief_dim
        
        # Belief state encoder: maps current state to belief representation
        self.belief_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, belief_dim),
            nn.Tanh()  # Normalize belief state to [-1, 1]
        )
        
        # Belief state decoder: maps belief back to state space for comparison
        self.belief_decoder = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Belief update mechanism: combines current belief with new observation
        self.belief_update = nn.Sequential(
            nn.Linear(belief_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, belief_dim),
            nn.Tanh()
        )
    
    def encode_belief(self, state: torch.Tensor) -> torch.Tensor:
        """Encode current state into belief representation."""
        return self.belief_encoder(state)
    
    def decode_belief(self, belief: torch.Tensor) -> torch.Tensor:
        """Decode belief representation back to state space."""
        return self.belief_decoder(belief)
    
    def update_belief(self, current_belief: torch.Tensor, new_observation: torch.Tensor) -> torch.Tensor:
        """Update belief state with new observation."""
        combined = torch.cat([current_belief, new_observation], dim=-1)
        return self.belief_update(combined)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode state to belief representation."""
        return self.encode_belief(state)

class ParameterMatchedHybridBeliefState(nn.Module):
    """
    Hybrid belief state representation with parameter count matched to quantum model.
    
    This version uses smaller dimensions to achieve ~36K parameters,
    similar to the quantum belief state with 8 qubits.
    """
    def __init__(self, state_dim: int = 17, classical_dim: int = 16, 
                 quantum_qubits: int = 4, n_layers: int = 2):
        super().__init__()
        assert _HAS_PENNYLANE, "Parameter-matched hybrid belief state requires pennylane to be installed."
        
        self.state_dim = state_dim
        self.classical_dim = classical_dim
        self.quantum_qubits = quantum_qubits
        self.n_layers = n_layers
        
        # Classical belief component (smaller)
        self.classical_belief = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, classical_dim),
            nn.Tanh()
        )
        
        # Quantum belief component (smaller)
        self.state_projection = nn.Linear(state_dim, quantum_qubits)
        
        # Build quantum circuit for belief encoding
        dev = qml.device("default.qubit", wires=quantum_qubits)
        
        @qml.qnode(dev, interface="torch")
        def quantum_belief_circuit(inputs, weights):
            # Angle embedding of state
            qml.AngleEmbedding(inputs, wires=range(quantum_qubits))
            
            # Variational layers for belief processing
            qml.StronglyEntanglingLayers(weights, wires=range(quantum_qubits))
            
            # Return expectation values as belief representation
            return [qml.expval(qml.PauliZ(i)) for i in range(quantum_qubits)]
        
        weight_shapes = {"weights": (n_layers, quantum_qubits, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(quantum_belief_circuit, weight_shapes)
        
        # Classical post-processing of quantum outputs (smaller)
        self.quantum_post_process = nn.Sequential(
            nn.Linear(quantum_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.Tanh()
        )
        
        # Fusion layer: combines classical and quantum beliefs (smaller)
        self.fusion_layer = nn.Sequential(
            nn.Linear(classical_dim + 16, 32),
            nn.ReLU(),
            nn.Linear(32, classical_dim + 16),
            nn.Tanh()
        )
        
        # Belief update mechanism (smaller)
        self.belief_update = nn.Sequential(
            nn.Linear(classical_dim + 16 + state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, classical_dim + 16),
            nn.Tanh()
        )
    
    def encode_belief(self, state: torch.Tensor) -> torch.Tensor:
        """Encode current state into hybrid belief representation."""
        # Get classical belief
        classical_belief = self.classical_belief(state)
        
        # Get quantum belief
        projected_state = self.state_projection(state)
        quantum_output = self.quantum_layer(projected_state)
        quantum_belief = self.quantum_post_process(quantum_output)
        
        # Fuse beliefs
        combined_belief = torch.cat([classical_belief, quantum_belief], dim=-1)
        fused_belief = self.fusion_layer(combined_belief)
        
        return fused_belief
    
    def update_belief(self, current_belief: torch.Tensor, new_observation: torch.Tensor) -> torch.Tensor:
        """Update hybrid belief state with new observation."""
        combined = torch.cat([current_belief, new_observation], dim=-1)
        return self.belief_update(combined)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode state to hybrid belief representation."""
        return self.encode_belief(state)

def create_belief_state(belief_type: str, state_dim: int = 17, **kwargs) -> nn.Module:
    """
    Factory function to create belief state representations.
    
    Args:
        belief_type: "classical", "quantum", "hybrid", "classical_matched", or "hybrid_matched"
        state_dim: Dimension of input state
        **kwargs: Additional arguments for specific belief types
    
    Returns:
        Belief state module
    """
    if belief_type == "classical":
        return ClassicalBeliefState(state_dim=state_dim, **kwargs)
    elif belief_type == "classical_matched":
        return ParameterMatchedClassicalBeliefState(state_dim=state_dim, **kwargs)
    elif belief_type == "quantum":
        if not _HAS_PENNYLANE:
            raise RuntimeError("Quantum belief state requires pennylane to be installed.")
        return QuantumBeliefState(state_dim=state_dim, **kwargs)
    elif belief_type == "hybrid":
        if not _HAS_PENNYLANE:
            raise RuntimeError("Hybrid belief state requires pennylane to be installed.")
        return HybridBeliefState(state_dim=state_dim, **kwargs)
    elif belief_type == "hybrid_matched":
        if not _HAS_PENNYLANE:
            raise RuntimeError("Parameter-matched hybrid belief state requires pennylane to be installed.")
        return ParameterMatchedHybridBeliefState(state_dim=state_dim, **kwargs)
    else:
        raise ValueError(f"Unknown belief type: {belief_type}")
