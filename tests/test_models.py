"""
Tests for the models module.
"""

import pytest
import torch
import numpy as np
from src.models import ToMObserver, ClassicalStateEncoder

class TestClassicalStateEncoder:
    def test_initialization(self):
        """Test classical state encoder initialization."""
        encoder = ClassicalStateEncoder(in_dim=17, hid=64, out=32)
        assert encoder.net is not None
    
    def test_forward(self):
        """Test classical state encoder forward pass."""
        encoder = ClassicalStateEncoder(in_dim=17, hid=64, out=32)
        x = torch.randn(4, 17)  # batch_size=4, input_dim=17
        
        output = encoder(x)
        assert output.shape == (4, 32)
        assert output.dtype == torch.float32

class TestToMObserver:
    def test_classical_initialization(self):
        """Test classical ToM observer initialization."""
        model = ToMObserver(mode="classical")
        assert model.mode == "classical"
        assert hasattr(model, 'state_enc_c')
        assert not hasattr(model, 'state_enc_q')
    
    def test_classical_forward(self):
        """Test classical ToM observer forward pass."""
        model = ToMObserver(mode="classical")
        
        # Test inputs
        char = torch.randn(4, 22)  # batch_size=4, char_dim=22
        mental = torch.randn(4, 17)  # batch_size=4, mental_dim=17
        state = torch.randn(4, 17)  # batch_size=4, state_dim=17
        
        output = model(char, mental, state)
        assert output.shape == (4, 5)  # batch_size=4, num_actions=5
        assert output.dtype == torch.float32
    
    def test_invalid_mode(self):
        """Test that invalid mode raises error."""
        with pytest.raises(AssertionError):
            ToMObserver(mode="invalid")
    
    def test_quantum_mode_without_pennylane(self):
        """Test quantum mode without PennyLane raises error."""
        # Mock the case where PennyLane is not available
        import src.models.tom_observer
        original_has_pennylane = src.models.tom_observer._HAS_PENNYLANE
        src.models.tom_observer._HAS_PENNYLANE = False
        
        try:
            with pytest.raises(RuntimeError):
                ToMObserver(mode="quantum")
        finally:
            src.models.tom_observer._HAS_PENNYLANE = original_has_pennylane

if __name__ == "__main__":
    pytest.main([__file__])
