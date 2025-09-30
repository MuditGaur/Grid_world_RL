#!/usr/bin/env python3
"""
Belief State Comparison Experiment (with Control Model)

Adds a control model that removes the belief-state pathway entirely.
The control model only consumes the character and mental embeddings,
ignoring the state input. This isolates the contribution of the belief
state embedding.

Provides:
- ControlObserver: model with only character and mental inputs
- run_control_experiment: train/evaluate the control model
- run_comparison_experiments_with_control: includes control + existing belief types
- plot_comparison_results: plots metrics including the control model
- CLI
"""

import time
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
from typing import Dict, List

from src.data import build_rollouts, RolloutDataset
from src.models import ToMObserver
from src.training import train_epoch, eval_model


# Optional: check if PennyLane is available for quantum/hybrid inclusion
try:
    import pennylane as qml  # noqa: F401
    _HAS_PENNYLANE = True
except Exception:
    qml = None
    _HAS_PENNYLANE = False


def set_seed(seed: int = 1234):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ControlObserver(torch.nn.Module):
    """Control ToM-style observer without a belief-state embedding.

    Inputs
    ------
    - char:   (B, 22) character summary input
    - mental: (B, 17) mental-state window input

    Flow
    ----
    [char, mental] --concat--> policy head --> logits
    """

    def __init__(self, char_dim: int = 22, mental_dim: int = 17):
        super().__init__()
        # Character encoder
        self.char_enc = torch.nn.Sequential(
            torch.nn.Linear(char_dim, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 32), torch.nn.ReLU(),
        )
        # Mental encoder
        self.mental_enc = torch.nn.Sequential(
            torch.nn.Linear(mental_dim, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 32), torch.nn.ReLU(),
        )
        # Policy head: combines only character and mental encodings
        fused_dim = 32 + 32
        self.head = torch.nn.Sequential(
            torch.nn.Linear(fused_dim, 185), torch.nn.ReLU(),
            torch.nn.Linear(185, 93), torch.nn.ReLU(),
            torch.nn.Linear(93, 5),
        )

    def forward(self, char: torch.Tensor, mental: torch.Tensor, state: torch.Tensor | None = None):
        # Ignore state entirely (control condition)
        c = self.char_enc(char)
        m = self.mental_enc(mental)
        x = torch.cat([c, m], dim=-1)
        logits = self.head(x)
        return logits


def run_control_experiment(train_loader, val_loader, max_epochs: int = 20, patience: int = 5,
                           lr: float = 3e-4, device: str = "cpu") -> Dict:
    """Train/evaluate the control observer that ignores state entirely."""
    print("Running control (no-belief) experiment...")

    start_time = time.time()
    model = ControlObserver()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses: List[float] = []
    val_accuracies: List[float] = []
    val_fb_accuracies: List[float] = []
    val_vis_accuracies: List[float] = []

    best_val_acc = 0.0
    best_results: Dict | None = None
    best_epoch = 0
    patience_counter = 0

    for ep in range(1, max_epochs + 1):
        epoch_start = time.time()
        tr_loss = train_epoch(model, train_loader, opt, device=device)
        train_losses.append(tr_loss)

        val_results = eval_model(model, val_loader, device=device)
        val_accuracies.append(val_results['acc'])
        val_fb_accuracies.append(val_results['acc_false_belief'])
        val_vis_accuracies.append(val_results['acc_visible'])

        epoch_time = time.time() - epoch_start

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

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {ep} (no improvement for {patience} epochs)")
            break

    total_time = time.time() - start_time
    actual_epochs = len(train_losses)

    assert best_results is not None, "Validation did not run; check data loaders."

    results = {
        'belief_type': 'control',
        'n_qubits': 0,
        'total_time': total_time,
        'avg_epoch_time': total_time / max(1, actual_epochs),
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
        'model_params': sum(p.numel() for p in model.parameters()),
    }
    return results


def run_comparison_experiments_with_control(train_loader, val_loader, n_qubits: int = 8,
                                            max_epochs: int = 20, patience: int = 5,
                                            lr: float = 3e-4, device: str = "cpu",
                                            include_non_control: bool = True) -> List[Dict]:
    """Run experiments including the control model. Optionally include other belief types."""
    results: List[Dict] = []

    # Always run control
    ctrl = run_control_experiment(train_loader, val_loader, max_epochs, patience, lr, device)
    results.append(ctrl)

    if include_non_control:
        # Use parameter-matched variants to keep model sizes comparable
        belief_types = ["classical_matched"]
        if _HAS_PENNYLANE:
            belief_types.extend(["quantum", "hybrid_matched"])
        else:
            print("WARNING: PennyLane not available. Skipping quantum/hybrid.")

        for belief_type in belief_types:
            try:
                print(f"Running {belief_type} belief state experiment...")
                model = ToMObserver(belief_type=belief_type, n_qubits=n_qubits, device=device)
                model.to(device)
                opt = torch.optim.Adam(model.parameters(), lr=lr)

                train_losses: List[float] = []
                val_accuracies: List[float] = []
                val_fb_accuracies: List[float] = []
                val_vis_accuracies: List[float] = []

                best_val_acc = 0.0
                best_results: Dict | None = None
                best_epoch = 0
                patience_counter = 0
                start_time = time.time()

                for ep in range(1, max_epochs + 1):
                    epoch_start = time.time()
                    tr_loss = train_epoch(model, train_loader, opt, device=device)
                    train_losses.append(tr_loss)
                    val_results = eval_model(model, val_loader, device=device)
                    val_accuracies.append(val_results['acc'])
                    val_fb_accuracies.append(val_results['acc_false_belief'])
                    val_vis_accuracies.append(val_results['acc_visible'])
                    epoch_time = time.time() - epoch_start

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

                    if patience_counter >= patience:
                        print(f"  Early stopping at epoch {ep} (no improvement for {patience} epochs)")
                        break

                total_time = time.time() - start_time
                actual_epochs = len(train_losses)
                assert best_results is not None

                res = {
                    'belief_type': belief_type,
                    'n_qubits': n_qubits if belief_type in ['quantum', 'hybrid'] else 0,
                    'total_time': total_time,
                    'avg_epoch_time': total_time / max(1, actual_epochs),
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
                    'model_params': sum(p.numel() for p in model.parameters()),
                }
                results.append(res)
                print(f"Completed {belief_type}: Acc={res['best_overall_acc']:.3f}, "
                      f"Epochs={res['total_epochs']}, Time={res['total_time']:.1f}s\n")
            except Exception as e:
                print(f"ERROR: Failed to run {belief_type} belief state experiment: {e}")
                continue

    return results


def plot_comparison_results(results: List[Dict], save_path: str | None = None):
    """Plot comparison results including control model."""
    if not results:
        print("No results to plot")
        return

    belief_types = [r['belief_type'] for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Belief State Comparison (with Control) - Final Epoch Metrics', fontsize=16)

    # Performance metrics (final epoch)
    overall_acc = [
        (r.get('final_overall_acc') if 'final_overall_acc' in r else (r['val_accuracies'][-1] if r.get('val_accuracies') else float('nan')))
        for r in results
    ]
    fb_acc = [
        (r.get('final_fb_acc') if 'final_fb_acc' in r else (r['val_fb_accuracies'][-1] if r.get('val_fb_accuracies') else float('nan')))
        for r in results
    ]
    vis_acc = [
        (r.get('final_vis_acc') if 'final_vis_acc' in r else (r['val_vis_accuracies'][-1] if r.get('val_vis_accuracies') else float('nan')))
        for r in results
    ]

    # Runtime metrics
    total_times = [r['total_time'] for r in results]
    epoch_times = [r['avg_epoch_time'] for r in results]
    model_params = [r['model_params'] for r in results]

    # Colors for types
    colors = {'control': 'purple', 'classical': 'blue', 'quantum': 'red', 'hybrid': 'green'}
    belief_colors = [colors.get(bt, 'gray') for bt in belief_types]

    # Plot 1: Overall Accuracy
    ax1 = axes[0, 0]
    bars1 = ax1.bar(belief_types, overall_acc, color=belief_colors, alpha=0.7)
    ax1.set_ylabel('Overall Accuracy (Final)')
    ax1.set_title('Final Overall Performance')
    ax1.grid(True, alpha=0.3)
    for bar, acc in zip(bars1, overall_acc):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{acc:.3f}', ha='center', va='bottom')

    # Plot 2: False-Belief Accuracy
    ax2 = axes[0, 1]
    bars2 = ax2.bar(belief_types, fb_acc, color=belief_colors, alpha=0.7)
    ax2.set_ylabel('False-Belief Accuracy (Final)')
    ax2.set_title('Final False-Belief Performance')
    ax2.grid(True, alpha=0.3)
    for bar, acc in zip(bars2, fb_acc):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{acc:.3f}', ha='center', va='bottom')

    # Plot 3: Visible Accuracy
    ax3 = axes[0, 2]
    bars3 = ax3.bar(belief_types, vis_acc, color=belief_colors, alpha=0.7)
    ax3.set_ylabel('Visible Accuracy (Final)')
    ax3.set_title('Final Visible Performance')
    ax3.grid(True, alpha=0.3)
    for bar, acc in zip(bars3, vis_acc):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{acc:.3f}', ha='center', va='bottom')

    # Plot 4: Total Training Time
    ax4 = axes[1, 0]
    bars4 = ax4.bar(belief_types, total_times, color=belief_colors, alpha=0.7)
    ax4.set_ylabel('Total Training Time (s)')
    ax4.set_title('Training Time')
    ax4.grid(True, alpha=0.3)
    for bar, time_val in zip(bars4, total_times):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{time_val:.1f}s', ha='center', va='bottom')

    # Plot 5: Per-Epoch Time
    ax5 = axes[1, 1]
    bars5 = ax5.bar(belief_types, epoch_times, color=belief_colors, alpha=0.7)
    ax5.set_ylabel('Average Epoch Time (s)')
    ax5.set_title('Per-Epoch Runtime')
    ax5.grid(True, alpha=0.3)
    for bar, time_val in zip(bars5, epoch_times):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{time_val:.1f}s', ha='center', va='bottom')

    # Plot 6: Model Parameters
    ax6 = axes[1, 2]
    bars6 = ax6.bar(belief_types, model_params, color=belief_colors, alpha=0.7)
    ax6.set_ylabel('Number of Parameters')
    ax6.set_title('Model Size')
    ax6.grid(True, alpha=0.3)
    for bar, params in zip(bars6, model_params):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, f'{params:,}', ha='center', va='bottom')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plots saved to {save_path}")
    plt.show()


def print_comparison_table(results: List[Dict]):
    if not results:
        print("No results to display")
        return
    print("\n" + "="*120)
    print("BELIEF STATE COMPARISON (WITH CONTROL) RESULTS")
    print("="*120)
    print(f"{'Belief Type':<12} {'Qubits':<8} {'Overall Acc':<12} {'FB Acc':<10} {'Vis Acc':<10} "
          f"{'Total Time':<12} {'Epochs':<8} {'Best Epoch':<12} {'Params':<10}")
    print("-"*120)
    for r in results:
        print(f"{r['belief_type']:<12} {r['n_qubits']:<8} {r['best_overall_acc']:<12.3f} "
              f"{r['best_fb_acc']:<10.3f} {r['best_vis_acc']:<10.3f} "
              f"{r['total_time']:<12.1f} {r['total_epochs']:<8} {r['best_epoch']:<12} "
              f"{r['model_params']:<10}")
    print("="*120)


def main():
    parser = argparse.ArgumentParser(description="Belief State Comparison with Control Model")
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--qubits', type=int, default=8, help='Number of qubits for quantum/hybrid')
    parser.add_argument('--episodes', type=int, default=150, help='Episodes per agent')
    parser.add_argument('--val-episodes', type=int, default=50, help='Validation episodes (unused here)')
    parser.add_argument('--max-epochs', type=int, default=20, help='Maximum training epochs')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--use-rb-agents', action='store_true', help='Use rule-based agents')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--save-results', type=str, default='belief_state_comparison_results_with_control.json',
                        help='File to save results')
    parser.add_argument('--save-plots', type=str, default='belief_state_comparison_plots_with_control.png',
                        help='File to save plots')
    parser.add_argument('--control-only', action='store_true', help='Run only the control model')

    args = parser.parse_args()
    set_seed(args.seed)

    print(f"Running comparison with control model (qubits={args.qubits} for quantum/hybrid)")

    # Build datasets
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

    # Run experiments
    results = run_comparison_experiments_with_control(
        train_loader, val_loader, args.qubits, args.max_epochs, args.patience, args.lr, args.device,
        include_non_control=(not args.control_only)
    )

    # Display results
    print_comparison_table(results)

    # Save results
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.save_results}")

    # Plot results
    if results:
        plot_comparison_results(results, args.save_plots)

    print("\nBelief state comparison with control completed!")


if __name__ == '__main__':
    main()


