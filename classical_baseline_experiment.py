#!/usr/bin/env python3
"""
Classical Baseline Experiment

This script runs the classical ToM model as a baseline for comparison
with the quantum scaling experiments.
"""

import time
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import json
from typing import Dict

from src.data import build_rollouts, RolloutDataset
from src.models import ToMObserver
from src.training import train_epoch, eval_model

def set_seed(seed: int = 1234):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_classical_experiment(train_loader, val_loader, 
                           epochs: int = 3, lr: float = 3e-4, device: str = "cpu") -> Dict:
    """Run classical baseline experiment."""
    print("Running classical baseline experiment...")
    
    # Create model
    start_time = time.time()
    model = ToMObserver(mode="classical", device=device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    train_losses = []
    val_accuracies = []
    val_fb_accuracies = []
    val_vis_accuracies = []
    
    best_val_acc = 0.0
    best_results = None
    
    for ep in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Training
        tr_loss = train_epoch(model, train_loader, opt, device=device)
        train_losses.append(tr_loss)
        
        # Validation
        val_results = eval_model(model, val_loader, device=device)
        val_accuracies.append(val_results['acc'])
        val_fb_accuracies.append(val_results['acc_false_belief'])
        val_vis_accuracies.append(val_results['acc_visible'])
        
        epoch_time = time.time() - epoch_start
        
        # Track best results
        if val_results['acc'] > best_val_acc:
            best_val_acc = val_results['acc']
            best_results = val_results.copy()
        
        print(f"  Epoch {ep:02d} | Loss: {tr_loss:.4f} | Acc: {val_results['acc']:.3f} | "
              f"FB: {val_results['acc_false_belief']:.3f} | Time: {epoch_time:.1f}s")
    
    total_time = time.time() - start_time
    
    # Compile results
    results = {
        'model_type': 'classical',
        'total_time': total_time,
        'avg_epoch_time': total_time / epochs,
        'best_overall_acc': best_results['acc'],
        'best_fb_acc': best_results['acc_false_belief'],
        'best_vis_acc': best_results['acc_visible'],
        'best_loss': best_results['loss'],
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'val_fb_accuracies': val_fb_accuracies,
        'val_vis_accuracies': val_vis_accuracies,
        'model_params': sum(p.numel() for p in model.parameters())
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Classical Baseline Experiment")
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--episodes', type=int, default=150, help='Episodes per agent')
    parser.add_argument('--val-episodes', type=int, default=50, help='Validation episodes')
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--use-rb-agents', action='store_true', help='Use rule-based agents')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--save-results', type=str, default='classical_baseline_results.json',
                       help='File to save results')
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # Build datasets (same as quantum experiment)
    print("Building rollout datasets...")
    train_samp, val_samp = build_rollouts(
        num_agents=4,
        episodes_per_agent=args.episodes,
        k_context=3,
        grid=9,
        fov=3,
        use_rb_agents=args.use_rb_agents or True,
        use_qlearn_agents=False,
        max_steps=60,
        seed=args.seed,
    )
    
    train_ds = RolloutDataset(train_samp)
    val_ds = RolloutDataset(val_samp)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    
    print(f"Dataset sizes: train={len(train_samp)}, val={len(val_samp)}")
    
    # Run classical experiment
    results = run_classical_experiment(train_loader, val_loader, args.epochs, args.lr, args.device)
    
    # Display results
    print(f"\nClassical Baseline Results:")
    print(f"Overall Accuracy: {results['best_overall_acc']:.3f}")
    print(f"False-Belief Accuracy: {results['best_fb_acc']:.3f}")
    print(f"Visible Accuracy: {results['best_vis_acc']:.3f}")
    print(f"Total Time: {results['total_time']:.1f}s")
    print(f"Avg Epoch Time: {results['avg_epoch_time']:.1f}s")
    print(f"Model Parameters: {results['model_params']}")
    
    # Save results
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.save_results}")
    
    print("\nClassical baseline experiment completed!")

if __name__ == '__main__':
    main()
