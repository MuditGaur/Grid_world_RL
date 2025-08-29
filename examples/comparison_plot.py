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
import argparse

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
    """CLI to plot final comparison from pre-saved results only when requested."""
    parser = argparse.ArgumentParser(description="Plot final comparison from pre-saved JSON results")
    parser.add_argument('--use-fixed-results', action='store_true',
                        help='If set, load results from JSON and generate the plot')
    parser.add_argument('--fixed-results-file', type=str, default='final_comparison.json',
                        help='Path to JSON file with results (default: final_comparison.json)')
    parser.add_argument('--output', type=str, default='final_comparison.png',
                        help='Output PNG filename (default: final_comparison.png)')
    args = parser.parse_args()

    if not args.use_fixed_results:
        print("Nothing to do: pass --use-fixed-results to load a JSON and plot.")
        print("Example: python examples/comparison_plot.py --use-fixed-results --fixed-results-file final_comparison.json")
        return

    if not os.path.isfile(args.fixed_results_file):
        print(f"ERROR: Results file not found: {args.fixed_results_file}")
        return

    results = load_results(args.fixed_results_file)
    print(f"Loaded results from {args.fixed_results_file}. Generating {args.output}...")
    plot_final_comparison(results, args.output)
    print(f"Saved plot to {args.output}")

if __name__ == "__main__":
    main()
