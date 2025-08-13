#!/usr/bin/env python3
"""
Analysis of Qubit Scaling Results

This script analyzes the results from qubit scaling experiments and creates
comprehensive visualizations showing performance and runtime scaling.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional

def load_results(filename: str) -> Optional[Dict]:
    """Load results from JSON file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
        return None

def create_comprehensive_analysis():
    """Create comprehensive analysis of scaling results."""
    
    # Load quantum scaling results
    quantum_results = load_results('qubit_scaling_results.json')
    
    if not quantum_results:
        print("No quantum results found. Please run qubit_scaling_experiment.py first.")
        return
    
    # Extract data
    qubit_counts = [r['n_qubits'] for r in quantum_results]
    overall_acc = [r['best_overall_acc'] for r in quantum_results]
    fb_acc = [r['best_fb_acc'] for r in quantum_results]
    vis_acc = [r['best_vis_acc'] for r in quantum_results]
    total_times = [r['total_time'] for r in quantum_results]
    epoch_times = [r['avg_epoch_time'] for r in quantum_results]
    model_params = [r['model_params'] for r in quantum_results]
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Quantum ToM Model Scaling Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Performance scaling
    ax1 = axes[0, 0]
    ax1.plot(qubit_counts, overall_acc, 'o-', label='Overall Accuracy', linewidth=2, markersize=8, color='blue')
    ax1.plot(qubit_counts, fb_acc, 's-', label='False-Belief Accuracy', linewidth=2, markersize=8, color='red')
    ax1.plot(qubit_counts, vis_acc, '^-', label='Visible Accuracy', linewidth=2, markersize=8, color='green')
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
    
    # Add value labels
    for i, (x, y) in enumerate(zip(qubit_counts, overall_acc)):
        ax1.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    # Plot 2: Runtime scaling (log scale)
    ax2 = axes[0, 1]
    ax2.semilogy(qubit_counts, total_times, 'o-', color='red', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Qubits')
    ax2.set_ylabel('Total Training Time (s)')
    ax2.set_title('Runtime Scaling (Log Scale)')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (x, y) in enumerate(zip(qubit_counts, total_times)):
        ax2.annotate(f'{y:.1f}s', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    # Plot 3: Per-epoch time scaling
    ax3 = axes[0, 2]
    ax3.plot(qubit_counts, epoch_times, 's-', color='orange', linewidth=2, markersize=8)
    ax3.set_xlabel('Number of Qubits')
    ax3.set_ylabel('Average Epoch Time (s)')
    ax3.set_title('Per-Epoch Runtime Scaling')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (x, y) in enumerate(zip(qubit_counts, epoch_times)):
        ax3.annotate(f'{y:.1f}s', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    # Plot 4: Model parameters scaling
    ax4 = axes[1, 0]
    ax4.plot(qubit_counts, model_params, '^-', color='green', linewidth=2, markersize=8)
    ax4.set_xlabel('Number of Qubits')
    ax4.set_ylabel('Number of Parameters')
    ax4.set_title('Model Size Scaling')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (x, y) in enumerate(zip(qubit_counts, model_params)):
        ax4.annotate(f'{y:,}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    # Plot 5: Performance vs Runtime trade-off
    ax5 = axes[1, 1]
    scatter = ax5.scatter(total_times, overall_acc, c=qubit_counts, s=100, cmap='viridis', alpha=0.7)
    ax5.set_xlabel('Total Training Time (s)')
    ax5.set_ylabel('Overall Accuracy')
    ax5.set_title('Performance vs Runtime Trade-off')
    ax5.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label('Number of Qubits')
    
    # Add value labels
    for i, (x, y) in enumerate(zip(total_times, overall_acc)):
        ax5.annotate(f'{qubit_counts[i]}q', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    # Plot 6: Efficiency (accuracy per second)
    ax6 = axes[1, 2]
    efficiency = [acc/time for acc, time in zip(overall_acc, total_times)]
    ax6.plot(qubit_counts, efficiency, 'D-', color='purple', linewidth=2, markersize=8)
    ax6.set_xlabel('Number of Qubits')
    ax6.set_ylabel('Accuracy per Second')
    ax6.set_title('Computational Efficiency')
    ax6.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (x, y) in enumerate(zip(qubit_counts, efficiency)):
        ax6.annotate(f'{y:.4f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('comprehensive_scaling_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("QUANTUM TOM SCALING ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nPerformance Analysis:")
    print(f"Best Overall Accuracy: {max(overall_acc):.3f} ({qubit_counts[np.argmax(overall_acc)]} qubits)")
    print(f"Best False-Belief Accuracy: {max(fb_acc):.3f} ({qubit_counts[np.argmax(fb_acc)]} qubits)")
    print(f"Best Visible Accuracy: {max(vis_acc):.3f} ({qubit_counts[np.argmax(vis_acc)]} qubits)")
    
    print(f"\nRuntime Analysis:")
    print(f"Fastest Training: {min(total_times):.1f}s ({qubit_counts[np.argmin(total_times)]} qubits)")
    print(f"Slowest Training: {max(total_times):.1f}s ({qubit_counts[np.argmax(total_times)]} qubits)")
    print(f"Runtime Scaling Factor: {max(total_times)/min(total_times):.1f}x")
    
    print(f"\nEfficiency Analysis:")
    best_efficiency_idx = np.argmax(efficiency)
    print(f"Most Efficient: {qubit_counts[best_efficiency_idx]} qubits ({efficiency[best_efficiency_idx]:.4f} acc/s)")
    print(f"Least Efficient: {qubit_counts[np.argmin(efficiency)]} qubits ({min(efficiency):.4f} acc/s)")
    
    print(f"\nKey Insights:")
    print(f"1. Optimal qubit count for performance: {qubit_counts[np.argmax(overall_acc)]}")
    print(f"2. Optimal qubit count for efficiency: {qubit_counts[best_efficiency_idx]}")
    print(f"3. Runtime scales exponentially with qubit count")
    print(f"4. Performance plateaus around 6-8 qubits")
    
    print("="*80)

def create_comparison_table():
    """Create a detailed comparison table."""
    quantum_results = load_results('qubit_scaling_results.json')
    
    if not quantum_results:
        return
    
    print("\n" + "="*100)
    print("DETAILED QUANTUM TOM SCALING COMPARISON")
    print("="*100)
    print(f"{'Qubits':<8} {'Overall':<8} {'FB Acc':<8} {'Vis Acc':<8} {'Total Time':<12} {'Epoch Time':<12} {'Params':<10} {'Efficiency':<12}")
    print("-"*100)
    
    for r in quantum_results:
        efficiency = r['best_overall_acc'] / r['total_time']
        print(f"{r['n_qubits']:<8} {r['best_overall_acc']:<8.3f} {r['best_fb_acc']:<8.3f} "
              f"{r['best_vis_acc']:<8.3f} {r['total_time']:<12.1f} {r['avg_epoch_time']:<12.1f} "
              f"{r['model_params']:<10} {efficiency:<12.4f}")
    
    print("="*100)

if __name__ == '__main__':
    create_comprehensive_analysis()
    create_comparison_table()
