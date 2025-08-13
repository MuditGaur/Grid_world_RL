"""
Main entry point for Quantum Theory of Mind Reinforcement Learning.

This script provides the command-line interface for training and evaluating
ToM observer models with different state representations.
"""

import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

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

def main():
    parser = argparse.ArgumentParser(description="Quantum Theory of Mind RL Training")
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--grid', type=int, default=9, help='Grid size')
    parser.add_argument('--fov', type=int, default=3, help='Field of view')
    parser.add_argument('--episodes', type=int, default=400, help='Episodes per agent used to build dataset')
    parser.add_argument('--val-episodes', type=int, default=120, help='Validation episodes')
    parser.add_argument('--k-context', type=int, default=3, help='Number of context episodes')
    parser.add_argument('--num-agents', type=int, default=8, help='Number of agents')
    parser.add_argument('--max-steps', type=int, default=80, help='Maximum steps per episode')
    parser.add_argument('--use-rb-agents', action='store_true', help='Use rule-based agents')
    parser.add_argument('--use-qlearn-agents', action='store_true', help='Use Q-learning agents')
    parser.add_argument('--qlearn-iters', type=int, default=8000, help='Q-learning training iterations')
    parser.add_argument('--batch', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=6, help='Training epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--model', type=str, default='all', 
                       choices=['classical','quantum','hybrid','all'], help='Model type')
    parser.add_argument('--n-qubits', type=int, default=8, help='Number of qubits for quantum models')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')

    args = parser.parse_args()
    set_seed(args.seed)

    if args.model in ('quantum','hybrid','all') and not _HAS_PENNYLANE:
        print("[WARN] PennyLane not found. Quantum/Hybrid models will be skipped.")
        if args.model != 'classical':
            args.model = 'classical'

    # Build datasets
    print("Building rollout datasets...")
    train_samp, val_samp = build_rollouts(
        num_agents=args.num_agents,
        episodes_per_agent=args.k_context + 1 + max(1, (args.episodes//args.num_agents) - (args.k_context + 1)),
        k_context=args.k_context,
        grid=args.grid,
        fov=args.fov,
        use_rb_agents=args.use_rb_agents or not args.use_qlearn_agents,  # default to RB if none selected
        use_qlearn_agents=args.use_qlearn_agents,
        qlearn_iters=args.qlearn_iters,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    train_ds = RolloutDataset(train_samp)
    val_ds = RolloutDataset(val_samp)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    print(f"Dataset sizes: train={len(train_samp)}, val={len(val_samp)}")

    modes = [args.model] if args.model != 'all' else ['classical'] + (["quantum","hybrid"] if _HAS_PENNYLANE else [])

    results = {}
    for mode in modes:
        print(f"\n=== Training {mode.upper()} model ===")
        model = ToMObserver(mode=mode, n_qubits=args.n_qubits, device=args.device)
        model.to(args.device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        best = None
        for ep in range(1, args.epochs+1):
            tr_loss = train_epoch(model, train_loader, opt, device=args.device)
            val = eval_model(model, val_loader, device=args.device)
            if best is None or val['acc'] > best['acc']:
                best = val
            print(f"Epoch {ep:02d} | train loss {tr_loss:.4f} | val acc {val['acc']:.3f} | FB {val['acc_false_belief']:.3f} | VIS {val['acc_visible']:.3f}")
        results[mode] = best

    print("\n=== Best validation results ===")
    for k,v in results.items():
        print(f"{k:>9s}: acc={v['acc']:.3f}, FB={v['acc_false_belief']:.3f}, VIS={v['acc_visible']:.3f}")

if __name__ == '__main__':
    main()
