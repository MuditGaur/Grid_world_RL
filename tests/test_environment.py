"""
Tests for the gridworld environment module.
"""

import pytest
import numpy as np
from src.environment import Gridworld, UP, DOWN, LEFT, RIGHT, STAY

class TestGridworld:
    def test_initialization(self):
        """Test gridworld initialization."""
        env = Gridworld(n=9, fov=3)
        assert env.n == 9
        assert env.fov == 3
        assert env.fov % 2 == 1  # FOV must be odd
    
    def test_reset(self):
        """Test environment reset."""
        env = Gridworld(n=9, fov=3)
        obs = env.reset()
        
        assert "obs" in obs
        assert "pos" in obs
        assert obs["obs"].shape == (6, 3, 3)  # 6 channels, fov x fov
        assert len(env.objects) == 4
        assert env.t == 0
        assert not env.swapped
    
    def test_step(self):
        """Test environment step function."""
        env = Gridworld(n=9, fov=3)
        env.reset()
        
        # Test valid move
        initial_pos = env.agent_pos
        result = env.step(UP)
        assert "obs" in result
        assert "pos" in result
        assert "done" in result
        assert env.t == 1
    
    def test_invalid_move(self):
        """Test that invalid moves don't change position."""
        env = Gridworld(n=9, fov=3)
        env.reset()
        
        # Place agent at edge and try to move out
        env.agent_pos = (0, 0)
        result = env.step(LEFT)  # Should not move
        assert env.agent_pos == (0, 0)
    
    def test_full_state_features(self):
        """Test full state feature generation."""
        env = Gridworld(n=9, fov=3)
        env.reset()
        
        features = env.full_state_features()
        assert features.shape == (17,)  # 2 + 2 + 4*2 + 5 = 17
        assert features.dtype == np.float32
    
    def test_fov_odd_requirement(self):
        """Test that FOV must be odd."""
        with pytest.raises(AssertionError):
            Gridworld(n=9, fov=4)  # Even FOV should raise error

if __name__ == "__main__":
    pytest.main([__file__])
