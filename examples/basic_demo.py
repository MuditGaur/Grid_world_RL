"""
Basic demo script for Quantum Theory of Mind RL.

This script demonstrates how to use the framework components to create
a simple experiment comparing classical and quantum models.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader

from src.environment import Gridworld
from src.agents import BeliefAgent
from src.models import ToMObserver
from src.data import build_rollouts, RolloutDataset
from src.training import train_epoch, eval_model

def main():
    print("=== Quantum Theory of Mind RL - Basic Demo ===\n")
    
    # Create environment and agent
    print("1. Creating environment and agent...")
    env = Gridworld(n=9, fov=3, p_swap=0.25)
    agent = BeliefAgent(preferred_kind=0, fov=3, n=9)
    
    # Run a simple episode
    print("2. Running a test episode...")
    env.reset()
    agent.reset(env)
    
    for step in range(10):
        action = agent.act(env)
        result = env.step(action)
        print(f"   Step {step}: Agent at {env.agent_pos}, Action: {action}")
        if result["done"]:
            break
    
    # Build a small dataset
    print("\n3. Building training dataset...")
    train_samples, val_samples = build_rollouts(
        num_agents=4,
        episodes_per_agent=10,
        k_context=2,
        use_rb_agents=True,
        max_steps=20
    )
    
    print(f"   Dataset sizes: train={len(train_samples)}, val={len(val_samples)}")
    
    # Create datasets and loaders
    train_ds = RolloutDataset(train_samples)
    val_ds = RolloutDataset(val_samples)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    # Train a classical model
    print("\n4. Training classical model...")
    model = ToMObserver(mode="classical")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(3):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_results = eval_model(model, val_loader)
        print(f"   Epoch {epoch+1}: train_loss={train_loss:.4f}, "
              f"val_acc={val_results['acc']:.3f}")
    
    print("\n=== Demo completed successfully! ===")
    print("To run the full experiment, use: python main.py --model all --episodes 300")

if __name__ == "__main__":
    main()
