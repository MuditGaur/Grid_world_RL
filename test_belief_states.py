#!/usr/bin/env python3
"""
Test script for the new belief state architecture.
"""

import torch
import numpy as np
from src.models import ToMObserver
from src.models.belief_states import create_belief_state, ClassicalBeliefState

def test_classical_belief_state():
    """Test classical belief state functionality."""
    print("Testing Classical Belief State...")
    
    # Create belief state
    belief_state = ClassicalBeliefState(state_dim=17, belief_dim=64)
    
    # Test input
    batch_size = 4
    state = torch.randn(batch_size, 17)
    
    # Test forward pass
    belief = belief_state(state)
    assert belief.shape == (batch_size, 64), f"Expected shape (4, 64), got {belief.shape}"
    
    # Test belief update
    new_observation = torch.randn(batch_size, 17)
    updated_belief = belief_state.update_belief(belief, new_observation)
    assert updated_belief.shape == (batch_size, 64), f"Expected shape (4, 64), got {updated_belief.shape}"
    
    # Test belief decoding
    decoded_state = belief_state.decode_belief(belief)
    assert decoded_state.shape == (batch_size, 17), f"Expected shape (4, 17), got {decoded_state.shape}"
    
    print("✓ Classical belief state tests passed!")

def test_tom_observer_with_belief_states():
    """Test ToMObserver with different belief state types."""
    print("Testing ToMObserver with Belief States...")
    
    batch_size = 4
    char_dim, mental_dim, state_dim = 22, 17, 17
    
    # Test classical belief state
    model = ToMObserver(belief_type="classical", device="cpu")
    
    char = torch.randn(batch_size, char_dim)
    mental = torch.randn(batch_size, mental_dim)
    state = torch.randn(batch_size, state_dim)
    
    # Test forward pass
    logits = model(char, mental, state)
    assert logits.shape == (batch_size, 5), f"Expected shape (4, 5), got {logits.shape}"
    
    # Test belief representation extraction
    belief = model.get_belief_representation(state)
    assert belief.shape == (batch_size, 64), f"Expected shape (4, 64), got {belief.shape}"
    
    print("✓ ToMObserver with classical belief state tests passed!")
    
    # Test quantum belief state if available
    try:
        model_quantum = ToMObserver(belief_type="quantum", n_qubits=6, device="cpu")
        logits_q = model_quantum(char, mental, state)
        assert logits_q.shape == (batch_size, 5), f"Expected shape (4, 5), got {logits_q.shape}"
        
        belief_q = model_quantum.get_belief_representation(state)
        assert belief_q.shape == (batch_size, 32), f"Expected shape (4, 32), got {belief_q.shape}"
        
        print("✓ ToMObserver with quantum belief state tests passed!")
        
        # Test hybrid belief state
        model_hybrid = ToMObserver(belief_type="hybrid", n_qubits=6, device="cpu")
        logits_h = model_hybrid(char, mental, state)
        assert logits_h.shape == (batch_size, 5), f"Expected shape (4, 5), got {logits_h.shape}"
        
        belief_h = model_hybrid.get_belief_representation(state)
        assert belief_h.shape == (batch_size, 64), f"Expected shape (4, 64), got {belief_h.shape}"
        
        print("✓ ToMObserver with hybrid belief state tests passed!")
        
    except RuntimeError as e:
        if "pennylane" in str(e).lower():
            print("⚠ Quantum belief state tests skipped (PennyLane not available)")
        else:
            raise e

def test_belief_state_factory():
    """Test the belief state factory function."""
    print("Testing Belief State Factory...")
    
    # Test classical
    classical = create_belief_state("classical", state_dim=17)
    assert isinstance(classical, ClassicalBeliefState)
    
    # Test quantum if available
    try:
        quantum = create_belief_state("quantum", state_dim=17, n_qubits=6)
        assert hasattr(quantum, 'quantum_layer')
        print("✓ Quantum belief state factory test passed!")
    except RuntimeError as e:
        if "pennylane" in str(e).lower():
            print("⚠ Quantum belief state factory test skipped (PennyLane not available)")
        else:
            raise e
    
    # Test hybrid if available
    try:
        hybrid = create_belief_state("hybrid", state_dim=17, classical_dim=32, quantum_qubits=3)
        assert hasattr(hybrid, 'classical_belief')
        assert hasattr(hybrid, 'quantum_belief')
        print("✓ Hybrid belief state factory test passed!")
    except RuntimeError as e:
        if "pennylane" in str(e).lower():
            print("⚠ Hybrid belief state factory test skipped (PennyLane not available)")
        else:
            raise e
    
    print("✓ Belief state factory tests passed!")

def main():
    """Run all tests."""
    print("Running Belief State Architecture Tests...\n")
    
    test_classical_belief_state()
    print()
    
    test_tom_observer_with_belief_states()
    print()
    
    test_belief_state_factory()
    print()
    
    print("🎉 All belief state architecture tests completed successfully!")

if __name__ == '__main__':
    main()
