"""
Results visualization and comparison script.

This script provides utilities for plotting and comparing the performance
of different ToM models (classical, quantum, hybrid).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from typing import Dict, List

def plot_training_curves(results: Dict[str, List[Dict]], save_path: str = None):
    """Plot training curves for different models."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Training Curves Comparison', fontsize=16)
    
    metrics = ['loss', 'acc', 'acc_false_belief', 'acc_visible']
    titles = ['Training Loss', 'Overall Accuracy', 'False Belief Accuracy', 'Visible Accuracy']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i//2, i%2]
        for model_name, epochs in results.items():
            values = [epoch[metric] for epoch in epochs if metric in epoch]
            if values:
                ax.plot(values, label=model_name, marker='o')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_final_comparison(results: Dict[str, Dict], save_path: str = None):
    """Plot final performance comparison."""
    models = list(results.keys())
    metrics = ['acc', 'acc_false_belief', 'acc_visible']
    metric_names = ['Overall', 'False Belief', 'Visible']
    
    x = np.arange(len(metric_names))
    width = 0.8 / len(models)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, model in enumerate(models):
        values = [results[model][metric] for metric in metrics]
        ax.bar(x + i * width, values, width, label=model, alpha=0.8)
    
    ax.set_xlabel('Evaluation Metric')
    ax.set_ylabel('Accuracy')
    ax.set_title('Final Model Performance Comparison')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, model in enumerate(models):
        for j, metric in enumerate(metrics):
            value = results[model][metric]
            ax.text(j + i * width, value + 0.01, f'{value:.3f}', 
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_results(results: Dict, filename: str = "results.json"):
    """Save results to JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")

def load_results(filename: str = "results.json") -> Dict:
    """Load results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def main():
    """Example usage of visualization functions."""
    print("=== Results Visualization Demo ===\n")
    
    # Example results (replace with actual results from training)
    example_results = {
        'classical': {
            'acc': 0.75,
            'acc_false_belief': 0.68,
            'acc_visible': 0.82,
            'loss': 0.45
        },
        'quantum': {
            'acc': 0.78,
            'acc_false_belief': 0.72,
            'acc_visible': 0.84,
            'loss': 0.42
        },
        'hybrid': {
            'acc': 0.80,
            'acc_false_belief': 0.75,
            'acc_visible': 0.85,
            'loss': 0.40
        }
    }
    
    print("1. Plotting final performance comparison...")
    plot_final_comparison(example_results, "final_comparison.png")
    
    print("2. Saving results to JSON...")
    save_results(example_results, "example_results.json")
    
    print("3. Loading and displaying results...")
    loaded_results = load_results("example_results.json")
    print("Loaded results:")
    for model, metrics in loaded_results.items():
        print(f"  {model}: {metrics}")
    
    print("\n=== Visualization demo completed! ===")
    print("To use with real results, modify the results dictionary in this script.")

if __name__ == "__main__":
    main()
