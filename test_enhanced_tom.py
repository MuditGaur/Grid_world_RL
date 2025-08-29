#!/usr/bin/env python3
"""
Test script for Enhanced ToM Observer

This script tests the enhanced ToM observer with different combinations
of state and belief types to ensure it works correctly.
"""

import torch
import numpy as np
from src.models.enhanced_tom_observer import create_enhanced_tom_observer

def test_enhanced_tom_observer():
    """Test the enhanced ToM observer with different configurations."""
    
    print("Testing Enhanced ToM Observer")
    print("=" * 40)
    
    # Test configurations
    configurations = [
        ("classical", "classical"),
        ("classical", "quantum"),
        ("classical", "hybrid"),
        ("quantum", "classical"),
        ("quantum", "quantum"),
        ("quantum", "hybrid"),
        ("hybrid", "classical"),
        ("hybrid", "quantum"),
        ("hybrid", "hybrid")
    ]
    
    for state_type, belief_type in configurations:
        print(f"\nTesting {state_type} state + {belief_type} belief:")
        
        try:
            # Create model
            model = create_enhanced_tom_observer(
                state_type=state_type,
                belief_type=belief_type,
                n_qubits=8,
                device="cpu"
            )
            
            # Create dummy input data
            batch_size = 4
            char_input = torch.randn(batch_size, 22)
            mental_input = torch.randn(batch_size, 17)
            state_input = torch.randn(batch_size, 17)
            
            # Forward pass
            with torch.no_grad():
                output = model(char_input, mental_input, state_input)
                
                # Get representations
                state_repr = model.get_state_representation(state_input)
                belief_repr = model.get_belief_representation(state_input)
            
            # Check output shapes
            assert output.shape == (batch_size, 5), f"Output shape incorrect: {output.shape}"
            assert state_repr.shape == (batch_size, 32), f"State repr shape incorrect: {state_repr.shape}"
            assert belief_repr.shape == (batch_size, 32), f"Belief repr shape incorrect: {belief_repr.shape}"
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"  ✓ Output shape: {output.shape}")
            print(f"  ✓ State repr shape: {state_repr.shape}")
            print(f"  ✓ Belief repr shape: {belief_repr.shape}")
            print(f"  ✓ Total parameters: {total_params:,}")
            print(f"  ✓ Trainable parameters: {trainable_params:,}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "=" * 40)
    print("Enhanced ToM Observer test completed!")

if __name__ == "__main__":
    test_enhanced_tom_observer()
