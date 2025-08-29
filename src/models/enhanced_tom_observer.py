"""
Enhanced Theory of Mind observer models allowing independent variation of
state encoders and belief-state encoders.

- State encoders: Classical | Quantum | Hybrid
- Belief states:  Classical | Quantum | Hybrid

This module is used by experiments that compare (a) belief-state types and
(b) state-encoder types in isolation. The `EnhancedToMObserver` fuses character,
mental, encoded_state, and belief into a policy head that predicts the next
action.
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

class ClassicalStateEncoder(nn.Module):
    """Classical state encoder using neural networks."""
    
    def __init__(self, input_dim: int = 17, output_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.encoder(x)

class QuantumStateEncoder(nn.Module):
    """Quantum state encoder using variational quantum circuits."""
    
    def __init__(self, input_dim: int = 17, output_dim: int = 32, n_qubits: int = 8, n_layers: int = 2):
        super().__init__()
        assert _HAS_PENNYLANE, "Quantum state encoder requires pennylane to be installed."
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Project input to qubit count
        self.input_projection = nn.Linear(input_dim, n_qubits)
        
        # Build quantum circuit
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev, interface="torch")
        def quantum_state_circuit(inputs, weights):
            # Angle embedding of input
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            
            # Variational layers
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            
            # Return expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(quantum_state_circuit, weight_shapes)
        
        # Post-processing to output dimension
        self.post_process = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Project input to qubit count
        projected_input = self.input_projection(x)
        
        # Process through quantum circuit
        quantum_output = self.quantum_layer(projected_input)
        
        # Post-process to final output
        return self.post_process(quantum_output)

class HybridStateEncoder(nn.Module):
    """Hybrid state encoder combining classical and quantum components."""
    
    def __init__(self, input_dim: int = 17, output_dim: int = 32, n_qubits: int = 6, n_layers: int = 2):
        super().__init__()
        assert _HAS_PENNYLANE, "Hybrid state encoder requires pennylane to be installed."
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Classical component
        self.classical_encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, output_dim // 2),
            nn.Tanh()
        )
        
        # Quantum component
        self.input_projection = nn.Linear(input_dim, n_qubits)
        
        # Build quantum circuit
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev, interface="torch")
        def quantum_state_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(quantum_state_circuit, weight_shapes)
        
        # Quantum post-processing
        self.quantum_post_process = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim // 2),
            nn.Tanh()
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Classical encoding
        classical_output = self.classical_encoder(x)
        
        # Quantum encoding
        projected_input = self.input_projection(x)
        quantum_output = self.quantum_layer(projected_input)
        quantum_output = self.quantum_post_process(quantum_output)
        
        # Fuse outputs
        combined = torch.cat([classical_output, quantum_output], dim=-1)
        return self.fusion_layer(combined)

class ClassicalBeliefState(nn.Module):
    """Classical belief state representation."""
    
    def __init__(self, input_dim: int = 32, belief_dim: int = 32):
        super().__init__()
        self.belief_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, belief_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.belief_encoder(x)

class QuantumBeliefState(nn.Module):
    """Quantum belief state representation."""
    
    def __init__(self, input_dim: int = 32, belief_dim: int = 32, n_qubits: int = 8, n_layers: int = 2):
        super().__init__()
        assert _HAS_PENNYLANE, "Quantum belief state requires pennylane to be installed."
        
        self.input_dim = input_dim
        self.belief_dim = belief_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Project input to qubit count
        self.input_projection = nn.Linear(input_dim, n_qubits)
        
        # Build quantum circuit
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev, interface="torch")
        def quantum_belief_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(quantum_belief_circuit, weight_shapes)
        
        # Post-processing
        self.post_process = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, belief_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        projected_input = self.input_projection(x)
        quantum_output = self.quantum_layer(projected_input)
        return self.post_process(quantum_output)

class HybridBeliefState(nn.Module):
    """Hybrid belief state representation."""
    
    def __init__(self, input_dim: int = 32, belief_dim: int = 32, n_qubits: int = 6, n_layers: int = 2):
        super().__init__()
        assert _HAS_PENNYLANE, "Hybrid belief state requires pennylane to be installed."
        
        self.input_dim = input_dim
        self.belief_dim = belief_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Classical component
        self.classical_belief = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, belief_dim // 2),
            nn.Tanh()
        )
        
        # Quantum component
        self.input_projection = nn.Linear(input_dim, n_qubits)
        
        # Build quantum circuit
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev, interface="torch")
        def quantum_belief_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(quantum_belief_circuit, weight_shapes)
        
        # Quantum post-processing
        self.quantum_post_process = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.ReLU(),
            nn.Linear(32, belief_dim // 2),
            nn.Tanh()
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(belief_dim, belief_dim),
            nn.ReLU(),
            nn.Linear(belief_dim, belief_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Classical belief
        classical_belief = self.classical_belief(x)
        
        # Quantum belief
        projected_input = self.input_projection(x)
        quantum_output = self.quantum_layer(projected_input)
        quantum_belief = self.quantum_post_process(quantum_output)
        
        # Fuse beliefs
        combined = torch.cat([classical_belief, quantum_belief], dim=-1)
        return self.fusion_layer(combined)

class EnhancedToMObserver(nn.Module):
    """
    Enhanced ToM-style observer with pluggable state and belief components.

    Inputs
    ------
    - char:   Character summary embedding input (B, char_dim)
    - mental: Mental window embedding input (B, mental_dim)
    - state:  Raw state features for state encoder (B, state_dim)

    Configuration
    -------------
    - state_type:  'classical' | 'quantum' | 'hybrid'
    - belief_type: 'classical' | 'quantum' | 'hybrid'
    - n_qubits:    qubits used by quantum/hybrid modules

    Flow
    ----
    state --(state_encoder)--> encoded_state --(belief_state)--> belief
    [char, mental, encoded_state, belief] --concat--> policy head --> logits
    """
    
    def __init__(self, char_dim: int = 22, mental_dim: int = 17, state_dim: int = 17,
                 state_type: str = "classical", belief_type: str = "classical", 
                 n_qubits: int = 8, device: str = "cpu"):
        super().__init__()
        
        assert state_type in {"classical", "quantum", "hybrid"}
        assert belief_type in {"classical", "quantum", "hybrid"}
        
        self.state_type = state_type
        self.belief_type = belief_type
        self.device = device
        
        # Character encoder
        self.char_enc = nn.Sequential(
            nn.Linear(char_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        
        # Mental encoder
        self.mental_enc = nn.Sequential(
            nn.Linear(mental_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        
        # State encoder
        if state_type == "classical":
            self.state_encoder = ClassicalStateEncoder(state_dim, 32)
        elif state_type == "quantum":
            self.state_encoder = QuantumStateEncoder(state_dim, 32, n_qubits)
        else:  # hybrid
            self.state_encoder = HybridStateEncoder(state_dim, 32, n_qubits // 2)
        
        # Belief state
        if belief_type == "classical":
            self.belief_state = ClassicalBeliefState(32, 32)
        elif belief_type == "quantum":
            self.belief_state = QuantumBeliefState(32, 32, n_qubits)
        else:  # hybrid
            self.belief_state = HybridBeliefState(32, 32, n_qubits // 2)
        
        # Policy head: combines character, mental state, encoded state, and belief state
        fused_dim = 32 + 32 + 32 + 32  # char + mental + encoded_state + belief
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 5)
        )
    
    def forward(self, char, mental, state):
        # Encode character past behavior and recent mental context
        c = self.char_enc(char)
        m = self.mental_enc(mental)

        # Encode current state, then derive belief from the encoded state
        encoded_state = self.state_encoder(state)
        belief = self.belief_state(encoded_state)

        # Fuse and predict next action logits
        x = torch.cat([c, m, encoded_state, belief], dim=-1)
        logits = self.head(x)
        return logits
    
    def get_state_representation(self, state):
        """Return state encoder output for analysis/visualization."""
        return self.state_encoder(state)
    
    def get_belief_representation(self, state):
        """Return belief representation given raw state (encodes state first)."""
        encoded_state = self.state_encoder(state)
        return self.belief_state(encoded_state)

def create_enhanced_tom_observer(state_type: str, belief_type: str, **kwargs):
    """Factory function to create enhanced ToM observers."""
    return EnhancedToMObserver(state_type=state_type, belief_type=belief_type, **kwargs)
