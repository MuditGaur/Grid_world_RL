#!/usr/bin/env python3
"""
Hybrid ToM Scaling Analysis

This script analyzes the results from hybrid scaling experiments and creates
comprehensive visualizations comparing hybrid vs quantum performance.
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

def create_hybrid_analysis():
    """Create comprehensive analysis of hybrid scaling results."""
    
    # Load hybrid scaling results
    hybrid_results = load_results('hybrid_scaling_results.json')
    quantum_results = load_results('qubit_scaling_results.json')
    
    if not hybrid_results:
        print("No hybrid results found. Please run hybrid_scaling_experiment.py first.")
        return
    
    # Extract hybrid data
    qubit_counts = [r['n_qubits'] for r in hybrid_results]
    overall_acc = [r['best_overall_acc'] for r in hybrid_results]
    fb_acc = [r['best_fb_acc'] for r in hybrid_results]
    vis_acc = [r['best_vis_acc'] for r in hybrid_results]
    total_times = [r['total_time'] for r in hybrid_results]
    epoch_times = [r['avg_epoch_time'] for r in hybrid_results]
    model_params = [r['model_params'] for r in hybrid_results]
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Hybrid ToM Model Scaling Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Performance scaling
    ax1 = axes[0, 0]
    ax1.plot(qubit_counts, overall_acc, 'o-', label='Overall Accuracy', linewidth=2, markersize=8, color='blue')
    ax1.plot(qubit_counts, fb_acc, 's-', label='False-Belief Accuracy', linewidth=2, markersize=8, color='red')
    ax1.plot(qubit_counts, vis_acc, '^-', label='Visible Accuracy', linewidth=2, markersize=8, color='green')
    ax1.set_xlabel('Number of Qubits')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Hybrid Performance Scaling')
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
    ax2.set_title('Hybrid Runtime Scaling (Log Scale)')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (x, y) in enumerate(zip(qubit_counts, total_times)):
        ax2.annotate(f'{y:.1f}s', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    # Plot 3: Per-epoch time scaling
    ax3 = axes[0, 2]
    ax3.plot(qubit_counts, epoch_times, 's-', color='orange', linewidth=2, markersize=8)
    ax3.set_xlabel('Number of Qubits')
    ax3.set_ylabel('Average Epoch Time (s)')
    ax3.set_title('Hybrid Per-Epoch Runtime Scaling')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (x, y) in enumerate(zip(qubit_counts, epoch_times)):
        ax3.annotate(f'{y:.1f}s', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    # Plot 4: Model parameters scaling
    ax4 = axes[1, 0]
    ax4.plot(qubit_counts, model_params, '^-', color='green', linewidth=2, markersize=8)
    ax4.set_xlabel('Number of Qubits')
    ax4.set_ylabel('Number of Parameters')
    ax4.set_title('Hybrid Model Size Scaling')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (x, y) in enumerate(zip(qubit_counts, model_params)):
        ax4.annotate(f'{y:,}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    # Plot 5: Performance vs Runtime trade-off
    ax5 = axes[1, 1]
    scatter = ax5.scatter(total_times, overall_acc, c=qubit_counts, s=100, cmap='viridis', alpha=0.7)
    ax5.set_xlabel('Total Training Time (s)')
    ax5.set_ylabel('Overall Accuracy')
    ax5.set_title('Hybrid Performance vs Runtime Trade-off')
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
    ax6.set_title('Hybrid Computational Efficiency')
    ax6.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (x, y) in enumerate(zip(qubit_counts, efficiency)):
        ax6.annotate(f'{y:.4f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('hybrid_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("HYBRID TOM SCALING ANALYSIS SUMMARY")
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

def create_hybrid_vs_quantum_comparison():
    """Create comparison between hybrid and quantum models."""
    
    hybrid_results = load_results('hybrid_scaling_results.json')
    quantum_results = load_results('qubit_scaling_results.json')
    
    if not hybrid_results or not quantum_results:
        print("Missing results files for comparison.")
        return
    
    # Extract data for comparison
    qubit_counts = [r['n_qubits'] for r in hybrid_results]
    
    # Hybrid data
    hybrid_overall = [r['best_overall_acc'] for r in hybrid_results]
    hybrid_fb = [r['best_fb_acc'] for r in hybrid_results]
    hybrid_times = [r['total_time'] for r in hybrid_results]
    hybrid_params = [r['model_params'] for r in hybrid_results]
    
    # Quantum data (match qubit counts)
    quantum_overall = []
    quantum_fb = []
    quantum_times = []
    quantum_params = []
    
    for qc in qubit_counts:
        for r in quantum_results:
            if r['n_qubits'] == qc:
                quantum_overall.append(r['best_overall_acc'])
                quantum_fb.append(r['best_fb_acc'])
                quantum_times.append(r['total_time'])
                quantum_params.append(r['model_params'])
                break
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Hybrid vs Quantum ToM Model Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Overall Accuracy Comparison
    ax1 = axes[0, 0]
    ax1.plot(qubit_counts, hybrid_overall, 'o-', label='Hybrid', linewidth=2, markersize=8, color='blue')
    ax1.plot(qubit_counts, quantum_overall, 's-', label='Quantum', linewidth=2, markersize=8, color='red')
    ax1.set_xlabel('Number of Qubits')
    ax1.set_ylabel('Overall Accuracy')
    ax1.set_title('Overall Accuracy Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (x, y) in enumerate(zip(qubit_counts, hybrid_overall)):
        ax1.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    for i, (x, y) in enumerate(zip(qubit_counts, quantum_overall)):
        ax1.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8)
    
    # Plot 2: False-Belief Accuracy Comparison
    ax2 = axes[0, 1]
    ax2.plot(qubit_counts, hybrid_fb, 'o-', label='Hybrid', linewidth=2, markersize=8, color='blue')
    ax2.plot(qubit_counts, quantum_fb, 's-', label='Quantum', linewidth=2, markersize=8, color='red')
    ax2.set_xlabel('Number of Qubits')
    ax2.set_ylabel('False-Belief Accuracy')
    ax2.set_title('False-Belief Accuracy Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Runtime Comparison
    ax3 = axes[1, 0]
    ax3.semilogy(qubit_counts, hybrid_times, 'o-', label='Hybrid', linewidth=2, markersize=8, color='blue')
    ax3.semilogy(qubit_counts, quantum_times, 's-', label='Quantum', linewidth=2, markersize=8, color='red')
    ax3.set_xlabel('Number of Qubits')
    ax3.set_ylabel('Total Training Time (s)')
    ax3.set_title('Runtime Comparison (Log Scale)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Model Parameters Comparison
    ax4 = axes[1, 1]
    ax4.plot(qubit_counts, hybrid_params, 'o-', label='Hybrid', linewidth=2, markersize=8, color='blue')
    ax4.plot(qubit_counts, quantum_params, 's-', label='Quantum', linewidth=2, markersize=8, color='red')
    ax4.set_xlabel('Number of Qubits')
    ax4.set_ylabel('Number of Parameters')
    ax4.set_title('Model Size Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hybrid_vs_quantum_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print comparison summary
    print("\n" + "="*80)
    print("HYBRID VS QUANTUM COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\nPerformance Comparison:")
    hybrid_best = max(hybrid_overall)
    quantum_best = max(quantum_overall)
    print(f"Best Hybrid Accuracy: {hybrid_best:.3f}")
    print(f"Best Quantum Accuracy: {quantum_best:.3f}")
    print(f"Performance Difference: {hybrid_best - quantum_best:+.3f}")
    
    print(f"\nRuntime Comparison:")
    hybrid_fastest = min(hybrid_times)
    quantum_fastest = min(quantum_times)
    print(f"Fastest Hybrid: {hybrid_fastest:.1f}s")
    print(f"Fastest Quantum: {quantum_fastest:.1f}s")
    print(f"Runtime Ratio (Hybrid/Quantum): {hybrid_fastest/quantum_fastest:.2f}x")
    
    print(f"\nParameter Comparison:")
    hybrid_params_avg = np.mean(hybrid_params)
    quantum_params_avg = np.mean(quantum_params)
    print(f"Average Hybrid Parameters: {hybrid_params_avg:.0f}")
    print(f"Average Quantum Parameters: {quantum_params_avg:.0f}")
    print(f"Parameter Ratio (Hybrid/Quantum): {hybrid_params_avg/quantum_params_avg:.2f}x")
    
    print("="*80)

def create_hybrid_comparison_table():
    """Create a detailed comparison table for hybrid results."""
    hybrid_results = load_results('hybrid_scaling_results.json')
    
    if not hybrid_results:
        return
    
    print("\n" + "="*100)
    print("DETAILED HYBRID TOM SCALING COMPARISON")
    print("="*100)
    print(f"{'Qubits':<8} {'Overall':<8} {'FB Acc':<8} {'Vis Acc':<8} {'Total Time':<12} {'Epoch Time':<12} {'Params':<10} {'Efficiency':<12}")
    print("-"*100)
    
    for r in hybrid_results:
        efficiency = r['best_overall_acc'] / r['total_time']
        print(f"{r['n_qubits']:<8} {r['best_overall_acc']:<8.3f} {r['best_fb_acc']:<8.3f} "
              f"{r['best_vis_acc']:<8.3f} {r['total_time']:<12.1f} {r['avg_epoch_time']:<12.1f} "
              f"{r['model_params']:<10} {efficiency:<12.4f}")
    
    print("="*100)

if __name__ == '__main__':
    create_hybrid_analysis()
    create_hybrid_vs_quantum_comparison()
    create_hybrid_comparison_table()
