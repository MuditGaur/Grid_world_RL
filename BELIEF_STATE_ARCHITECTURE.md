# Belief State Architecture: Theory of Mind Framework

## Overview

This document describes the architectural changes made to the Theory of Mind (ToM) framework, transitioning from different state space representations to different **belief state representations** while maintaining a consistent state space across all methods.

## Key Architectural Change

### Before: State Space Variation
The original architecture had different state encoders for each method:
- **Classical**: Neural network state encoder
- **Quantum**: Quantum circuit state encoder  
- **Hybrid**: Combination of classical and quantum state encoders

### After: Belief State Variation
The new architecture maintains the same state space but varies the belief state representation:
- **Classical**: Neural network belief state
- **Quantum**: Quantum circuit belief state
- **Hybrid**: Combination of classical and quantum belief states

## Why This Change?

### 1. **Fairer Comparison**
- All methods now use the same input state space (17-dimensional features)
- Differences in performance are purely due to belief state representation
- Eliminates confounding factors from different state encodings

### 2. **Better Theory of Mind Modeling**
- Belief states are more central to ToM than state encodings
- Belief states represent the observer's understanding of the agent's mental state
- More aligned with cognitive science theories of ToM

### 3. **Cleaner Architecture**
- Clear separation between state space and belief representation
- More modular and extensible design
- Easier to add new belief state types

## New Architecture Components

### 1. Belief State Module (`src/models/belief_states.py`)

#### ClassicalBeliefState
```python
class ClassicalBeliefState(nn.Module):
    - belief_encoder: MLP (17→128→64) with Tanh activation
    - belief_decoder: MLP (64→128→17) for belief reconstruction
    - belief_update: MLP (64+17→128→64) for belief evolution
```

#### QuantumBeliefState
```python
class QuantumBeliefState(nn.Module):
    - state_projection: Linear (17→n_qubits)
    - quantum_layer: Variational quantum circuit
    - post_process: MLP (n_qubits→64→32) with Tanh activation
    - belief_update: MLP (32+17→64→32) for belief evolution
```

#### HybridBeliefState
```python
class HybridBeliefState(nn.Module):
    - classical_belief: ClassicalBeliefState component
    - quantum_belief: QuantumBeliefState component
    - fusion_layer: MLP (64→64→64) combining both beliefs
    - belief_update: MLP (64+17→64→64) for belief evolution
```

### 2. Updated ToMObserver (`src/models/tom_observer.py`)

The ToMObserver now uses belief states instead of state encoders:

```python
class ToMObserver(nn.Module):
    def __init__(self, belief_type="classical", n_qubits=8, ...):
        # Character encoder: MLP (24→64→32)
        # Mental encoder: MLP (17→64→32)
        
        # Belief state representation
        if belief_type == "classical":
            self.belief_state = ClassicalBeliefState(state_dim=17)
        elif belief_type == "quantum":
            self.belief_state = QuantumBeliefState(state_dim=17, n_qubits=n_qubits)
        else:  # hybrid
            self.belief_state = HybridBeliefState(state_dim=17, ...)
        
        # Policy head: combines character, mental, and belief representations
```

## Updated Experiments

### 1. Belief State Comparison Experiment (`belief_state_comparison_experiment.py`)
- **Purpose**: Direct comparison of all three belief state types
- **Features**: 
  - Same experimental setup for all methods
  - Comprehensive performance analysis
  - Runtime and parameter comparison
  - Visualization of results

### 2. Updated Individual Experiments
- **Classical Baseline**: Now uses classical belief state
- **Quantum Scaling**: Now uses quantum belief state
- **Hybrid Scaling**: Now uses hybrid belief state

## Key Benefits

### 1. **Consistent State Space**
- All methods receive identical 17-dimensional state features
- No bias from different state encodings
- Pure comparison of belief state representations

### 2. **Rich Belief Representations**
- **Classical**: Traditional neural network beliefs with encoder/decoder
- **Quantum**: Quantum circuit beliefs with superposition and entanglement
- **Hybrid**: Combined classical and quantum beliefs with fusion

### 3. **Belief Evolution**
- All belief states support belief updates with new observations
- Beliefs can evolve over time as new information is received
- More realistic ToM modeling

### 4. **Modular Design**
- Easy to add new belief state types
- Clean separation of concerns
- Factory pattern for belief state creation

## Usage Examples

### Creating Models
```python
from src.models import ToMObserver

# Classical belief state
classical_model = ToMObserver(belief_type="classical")

# Quantum belief state
quantum_model = ToMObserver(belief_type="quantum", n_qubits=8)

# Hybrid belief state
hybrid_model = ToMObserver(belief_type="hybrid", n_qubits=6)
```

### Running Experiments
```bash
# Compare all belief state types
python belief_state_comparison_experiment.py --qubits 8 --episodes 150

# Individual experiments
python classical_baseline_experiment.py --episodes 150
python qubit_scaling_experiment.py --qubits 2,4,6,8
python hybrid_scaling_experiment.py --qubits 2,4,6,8
```

## Testing

Run the comprehensive test suite:
```bash
python test_belief_states.py
```

This tests:
- Classical belief state functionality
- ToMObserver with all belief state types
- Belief state factory function
- Import compatibility

## Migration Guide

### For Existing Code
1. **Update imports**: Use `belief_type` instead of `mode`
2. **Update model creation**: 
   ```python
   # Old
   model = ToMObserver(mode="quantum", n_qubits=8)
   
   # New
   model = ToMObserver(belief_type="quantum", n_qubits=8)
   ```
3. **Update experiment scripts**: Use new parameter names

### For New Code
1. **Choose belief state type**: classical, quantum, or hybrid
2. **Use factory function**: `create_belief_state(belief_type, **kwargs)`
3. **Leverage belief evolution**: Use `update_belief()` method

## Future Extensions

### Potential New Belief State Types
1. **Attention-based**: Belief states with attention mechanisms
2. **Memory-augmented**: Belief states with external memory
3. **Hierarchical**: Multi-level belief state representations
4. **Probabilistic**: Belief states with uncertainty quantification

### Research Directions
1. **Belief State Analysis**: Analyze what different belief states learn
2. **Belief Evolution**: Study how beliefs change over time
3. **Cross-modal Beliefs**: Belief states that integrate multiple modalities
4. **Meta-beliefs**: Beliefs about other agents' beliefs

## Conclusion

The new belief state architecture provides a more principled and fair comparison framework for Theory of Mind models. By maintaining consistent state spaces while varying belief representations, we can better understand the impact of different belief modeling approaches on ToM performance.

This architecture is more aligned with cognitive science theories of Theory of Mind and provides a solid foundation for future research in quantum-enhanced cognitive modeling.
