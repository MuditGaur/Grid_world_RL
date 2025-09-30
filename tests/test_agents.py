"""
Tests for the agents module.
"""

import pytest
import numpy as np
from src.environment import Gridworld
from src.agents import BeliefAgent, QLearnAgent, RandomAgent

class TestBeliefAgent:
    def test_initialization(self):
        """Test belief agent initialization."""
        agent = BeliefAgent(preferred_kind=0, fov=3, n=9)
        assert agent.pref == 0
        assert agent.fov == 3
        assert agent.n == 9
        assert len(agent.beliefs) == 0
    
    def test_reset(self):
        """Test agent reset."""
        env = Gridworld(n=9, fov=3)
        agent = BeliefAgent(preferred_kind=0, fov=3, n=9)
        
        env.reset()
        agent.reset(env)
        
        assert agent.last_action == 4  # STAY
        # Beliefs should be updated based on initial observation
    
    def test_act(self):
        """Test agent action selection."""
        env = Gridworld(n=9, fov=3)
        agent = BeliefAgent(preferred_kind=0, fov=3, n=9)
        
        env.reset()
        agent.reset(env)
        
        action = agent.act(env)
        assert action in [0, 1, 2, 3, 4]  # Valid actions
        assert agent.last_action == action

class TestQLearnAgent:
    def test_initialization(self):
        """Test Q-learning agent initialization."""
        agent = QLearnAgent(preferred_kind=0, n=9, fov=3)
        assert agent.pref == 0
        assert agent.n == 9
        assert agent.fov == 3
        assert len(agent.Q) == 0
    
    def test_reset(self):
        """Test agent reset."""
        env = Gridworld(n=9, fov=3)
        agent = QLearnAgent(preferred_kind=0, n=9, fov=3)
        
        env.reset()
        agent.reset(env)
        
        assert agent.last_action == 4  # STAY
    
    def test_state_code(self):
        """Test state encoding."""
        env = Gridworld(n=9, fov=3)
        agent = QLearnAgent(preferred_kind=0, n=9, fov=3)
        
        env.reset()
        agent.reset(env)
        
        state = agent._state_code(env)
        assert len(state) == 5  # (dx, dy, wall_front, wall_left, wall_right)
    
    def test_act(self):
        """Test agent action selection."""
        env = Gridworld(n=9, fov=3)
        agent = QLearnAgent(preferred_kind=0, n=9, fov=3)
        
        env.reset()
        agent.reset(env)
        
        action = agent.act(env)
        assert action in [0, 1, 2, 3, 4]  # Valid actions
        assert agent.last_action == action
    
    def test_train_episode(self):
        """Test Q-learning training episode."""
        env = Gridworld(n=9, fov=3)
        agent = QLearnAgent(preferred_kind=0, n=9, fov=3)
        
        total_reward = agent.train_episode(env, iters=10)
        assert isinstance(total_reward, float)


class TestRandomAgent:
    def test_initialization(self):
        agent = RandomAgent(alpha=0.5)
        assert agent.last_action == 4

    def test_reset_and_act(self):
        env = Gridworld(n=9, fov=3)
        env.reset()
        agent = RandomAgent(alpha=[1,1,1,1,1])
        agent.reset(env)
        a = agent.act(env)
        assert a in [0,1,2,3,4]
        assert agent.last_action == a

    def test_alpha_fn(self):
        env = Gridworld(n=9, fov=3)
        env.reset()
        def alpha_policy(e):
            return [2,2,2,2,5]
        agent = RandomAgent(alpha_fn=alpha_policy)
        # sample multiple times to ensure valid
        agent.reset(env)
        for _ in range(10):
            a = agent.act(env)
            assert a in [0,1,2,3,4]

if __name__ == "__main__":
    pytest.main([__file__])
