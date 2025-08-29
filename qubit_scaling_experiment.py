#!/usr/bin/env python3
"""
Qubit Scaling Experiment

This script runs experiments with varying numbers of qubits to measure
how performance and runtime scale with quantum circuit size.
"""

import time
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Tuple

from src.data import build_rollouts, RolloutDataset
from src.models import ToMObserver
from src.training import train_epoch, eval_model

# Check if PennyLane is available
try:
    import pennylane as qml
    _HAS_PENNYLANE = True
except Exception as e:
    qml = None
    _HAS_PENNYLANE = False

def set_seed(seed: int = 1234):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_single_experiment(n_qubits: int, train_loader, val_loader, 
                         max_epochs: int = 20, patience: int = 5, lr: float = 3e-4, device: str = "cpu") -> Dict:
    """Run a single experiment with specified number of qubits until convergence."""
    print(f"Running experiment with {n_qubits} qubits...")
    
    # Create model
    start_time = time.time()
    model = ToMObserver(belief_type="quantum", n_qubits=n_qubits, device=device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop with early stopping
    train_losses = []
    val_accuracies = []
    val_fb_accuracies = []
    val_vis_accuracies = []
    
    best_val_acc = 0.0
    best_results = None
    best_epoch = 0
    patience_counter = 0
    
    for ep in range(1, max_epochs + 1):
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
            best_epoch = ep
            patience_counter = 0
        else:
            patience_counter += 1
        
        print(f"  Epoch {ep:02d} | Loss: {tr_loss:.4f} | Acc: {val_results['acc']:.3f} | "
              f"FB: {val_results['acc_false_belief']:.3f} | Time: {epoch_time:.1f}s | "
              f"Best: {best_val_acc:.3f} (Epoch {best_epoch})")
        
        # Early stopping if no improvement for patience epochs
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {ep} (no improvement for {patience} epochs)")
            break
    
    total_time = time.time() - start_time
    
    # Compile results
    actual_epochs = len(train_losses)
    results = {
        'n_qubits': n_qubits,
        'total_time': total_time,
        'avg_epoch_time': total_time / actual_epochs,
        'best_overall_acc': best_results['acc'],
        'best_fb_acc': best_results['acc_false_belief'],
        'best_vis_acc': best_results['acc_visible'],
        'best_loss': best_results['loss'],
        'best_epoch': best_epoch,
        'total_epochs': actual_epochs,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'val_fb_accuracies': val_fb_accuracies,
        'val_vis_accuracies': val_vis_accuracies,
        'model_params': sum(p.numel() for p in model.parameters())
    }
    
    return results

def run_scaling_experiments(qubit_counts: List[int], 
                           train_loader, val_loader,
                           max_epochs: int = 20, patience: int = 5, lr: float = 3e-4, 
                           device: str = "cpu") -> List[Dict]:
    """Run experiments with varying qubit counts."""
    if not _HAS_PENNYLANE:
        print("ERROR: PennyLane not available. Cannot run quantum experiments.")
        return []
    
    results = []
    
    for n_qubits in qubit_counts:
        try:
            result = run_single_experiment(n_qubits, train_loader, val_loader, 
                                         max_epochs, patience, lr, device)
            results.append(result)
            print(f"Completed {n_qubits} qubits: Acc={result['best_overall_acc']:.3f}, "
                  f"Epochs={result['total_epochs']}, Time={result['total_time']:.1f}s\n")
        except Exception as e:
            print(f"ERROR: Failed to run experiment with {n_qubits} qubits: {e}")
            continue
    
    return results

def plot_scaling_results(results: List[Dict], save_path: str = None):
    """Plot scaling results."""
    if not results:
        print("No results to plot")
        return
    
    qubit_counts = [r['n_qubits'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Quantum Belief State ToM Model Scaling with Qubit Count', fontsize=16)
    
    # Performance metrics
    overall_acc = [r['best_overall_acc'] for r in results]
    fb_acc = [r['best_fb_acc'] for r in results]
    vis_acc = [r['best_vis_acc'] for r in results]
    
    # Runtime metrics
    total_times = [r['total_time'] for r in results]
    epoch_times = [r['avg_epoch_time'] for r in results]
    model_params = [r['model_params'] for r in results]
    
    # Plot 1: Accuracy scaling
    ax1 = axes[0, 0]
    ax1.plot(qubit_counts, overall_acc, 'o-', label='Overall Accuracy', linewidth=2, markersize=8)
    ax1.plot(qubit_counts, fb_acc, 's-', label='False-Belief Accuracy', linewidth=2, markersize=8)
    ax1.plot(qubit_counts, vis_acc, '^-', label='Visible Accuracy', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Qubits')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Performance Scaling')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Set y-axis range to show more detail in the performance differences
    min_acc = min(min(overall_acc), min(fb_acc), min(vis_acc))
    max_acc = max(max(overall_acc), max(fb_acc), max(vis_acc))
    margin = (max_acc - min_acc) * 0.1  # 10% margin
    ax1.set_ylim(max(0.0, min_acc - margin), min(1.0, max_acc + margin))
    
    # Plot 2: Runtime scaling
    ax2 = axes[0, 1]
    ax2.plot(qubit_counts, total_times, 'o-', color='red', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Qubits')
    ax2.set_ylabel('Total Training Time (s)')
    ax2.set_title('Runtime Scaling')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Per-epoch time scaling
    ax3 = axes[1, 0]
    ax3.plot(qubit_counts, epoch_times, 's-', color='orange', linewidth=2, markersize=8)
    ax3.set_xlabel('Number of Qubits')
    ax3.set_ylabel('Average Epoch Time (s)')
    ax3.set_title('Per-Epoch Runtime Scaling')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Model parameters scaling
    ax4 = axes[1, 1]
    ax4.plot(qubit_counts, model_params, '^-', color='green', linewidth=2, markersize=8)
    ax4.set_xlabel('Number of Qubits')
    ax4.set_ylabel('Number of Parameters')
    ax4.set_title('Model Size Scaling')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to {save_path}")
    plt.show()

def print_summary_table(results: List[Dict]):
    """Print a summary table of results."""
    if not results:
        print("No results to display")
        return
    
    print("\n" + "="*100)
    print("QUANTUM BELIEF STATE TOM SCALING EXPERIMENT RESULTS")
    print("="*100)
    print(f"{'Qubits':<8} {'Overall Acc':<12} {'FB Acc':<10} {'Vis Acc':<10} "
          f"{'Total Time':<12} {'Epochs':<8} {'Best Epoch':<12} {'Params':<10}")
    print("-"*100)
    
    for r in results:
        print(f"{r['n_qubits']:<8} {r['best_overall_acc']:<12.3f} "
              f"{r['best_fb_acc']:<10.3f} {r['best_vis_acc']:<10.3f} "
              f"{r['total_time']:<12.1f} {r['total_epochs']:<8} {r['best_epoch']:<12} "
              f"{r['model_params']:<10}")
    
    print("="*100)

def main():
    parser = argparse.ArgumentParser(description="Qubit Scaling Experiment")
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--qubits', type=str, default='2,4,6,8,10,12', 
                       help='Comma-separated list of qubit counts to test')
    parser.add_argument('--episodes', type=int, default=200, help='Episodes per agent')
    parser.add_argument('--val-episodes', type=int, default=50, help='Validation episodes')
    parser.add_argument('--max-epochs', type=int, default=20, help='Maximum training epochs')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--use-rb-agents', action='store_true', help='Use rule-based agents')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--save-results', type=str, default='qubit_scaling_results.json',
                       help='File to save results')
    parser.add_argument('--save-plots', type=str, default='qubit_scaling_plots.png',
                       help='File to save plots')
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # Parse qubit counts
    qubit_counts = [int(x.strip()) for x in args.qubits.split(',')]
    print(f"Testing qubit counts: {qubit_counts}")
    
    if not _HAS_PENNYLANE:
        print("ERROR: PennyLane not available. Please install pennylane to run quantum experiments.")
        return
    
    # Build datasets
    print("Building rollout datasets...")
    train_samp, val_samp = build_rollouts(
        num_agents=4,  # Reduced for faster experiments
        episodes_per_agent=args.episodes,
        k_context=3,
        grid=9,
        fov=3,
        use_rb_agents=args.use_rb_agents or True,  # Default to RB agents
        use_qlearn_agents=False,  # Skip Q-learning for faster experiments
        max_steps=60,  # Reduced for faster experiments
        seed=args.seed,
    )
    
    train_ds = RolloutDataset(train_samp)
    val_ds = RolloutDataset(val_samp)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    
    print(f"Dataset sizes: train={len(train_samp)}, val={len(val_samp)}")
    
    # Run experiments
    print(f"\nStarting scaling experiments with {len(qubit_counts)} qubit configurations...")
    results = run_scaling_experiments(qubit_counts, train_loader, val_loader,
                                    args.max_epochs, args.patience, args.lr, args.device)
    
    # Display results
    print_summary_table(results)
    
    # Save results
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.save_results}")
    
    # Plot results
    if results:
        plot_scaling_results(results, args.save_plots)
    
    print("\nExperiment completed!")

if __name__ == '__main__':
    main()
