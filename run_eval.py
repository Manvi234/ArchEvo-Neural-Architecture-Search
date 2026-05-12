#!/usr/bin/env python3
"""
run_eval.py
-----------
Full training from scratch using a saved genotype, then evaluation.

Usage:
    python run_eval.py \
        --genotype_path results/cifar10_darts_none/genotype.json \
        --dataset cifar10 \
        --epochs 100 \
        --device cuda \
        --output_dir results/eval/
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Any

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from archevo.data.datamodule import ArchEvoDataModule, DATASET_STATS
from archevo.search_space import build_from_genotype, str_to_genotype
from archevo.pressure import count_params

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('run_eval')


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader,
    optimizer: SGD,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 5.0,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_samples += x.size(0)

    return {
        'train_loss': total_loss / max(total_samples, 1),
        'train_acc': total_correct / max(total_samples, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_samples += x.size(0)

    return {
        'val_loss': total_loss / max(total_samples, 1),
        'val_acc': total_correct / max(total_samples, 1),
    }


def train_from_scratch(
    model: nn.Module,
    data_module: ArchEvoDataModule,
    num_epochs: int,
    device: torch.device,
    lr: float = 0.025,
    weight_decay: float = 3e-4,
    momentum: float = 0.9,
) -> List[Dict[str, Any]]:
    """Train model from scratch on full dataset. Returns per-epoch logs."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)

    train_loader = data_module.get_train_loader()
    val_loader   = data_module.get_val_loader()

    logs = []
    best_val_acc = 0.0
    t_start = time.time()

    for epoch in range(1, num_epochs + 1):
        t_epoch = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t_epoch
        wall_clock = time.time() - t_start

        log_entry = {
            'epoch': epoch,
            **train_metrics,
            **val_metrics,
            'lr': scheduler.get_last_lr()[0],
            'elapsed_sec': elapsed,
            'wall_clock_sec': wall_clock,
        }
        logs.append(log_entry)

        if val_metrics['val_acc'] > best_val_acc:
            best_val_acc = val_metrics['val_acc']

        if epoch % 10 == 0 or epoch == num_epochs:
            logger.info(
                f"Epoch {epoch}/{num_epochs} | "
                f"train_loss={train_metrics['train_loss']:.4f} | "
                f"train_acc={train_metrics['train_acc']:.4f} | "
                f"val_loss={val_metrics['val_loss']:.4f} | "
                f"val_acc={val_metrics['val_acc']:.4f} | "
                f"best={best_val_acc:.4f} | "
                f"time={elapsed:.1f}s"
            )

    return logs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="ArchEvo: Full training evaluation from genotype",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--genotype_path', type=str, required=True,
                   help='Path to saved genotype JSON file')
    p.add_argument('--dataset', type=str, default='cifar10',
                   choices=['cifar10', 'eurosat', 'isic', 'cub200'],
                   help='Dataset name')
    p.add_argument('--epochs', type=int, default=100,
                   help='Training epochs')
    p.add_argument('--device', type=str, default='cpu',
                   help='PyTorch device (cpu/cuda/mps)')
    p.add_argument('--output_dir', type=str, default='results/eval',
                   help='Directory to save evaluation results')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=0.025)
    p.add_argument('--weight_decay', type=float, default=3e-4)
    p.add_argument('--C_init', type=int, default=16,
                   help='Initial channel count (must match searched network)')
    p.add_argument('--data_root', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load genotype
    logger.info(f"Loading genotype from: {args.genotype_path}")
    with open(args.genotype_path) as f:
        genotype_raw = json.load(f)

    # Normalise: genotype may be a list-of-lists (per-node) or nested
    if isinstance(genotype_raw[0], list) and isinstance(genotype_raw[0][0], list):
        # Multiple cell genotypes; use first
        genotype = genotype_raw[0]
    elif isinstance(genotype_raw[0], list):
        genotype = genotype_raw
    else:
        # Flat list of strings — shouldn't happen but handle gracefully
        genotype = genotype_raw

    stats = DATASET_STATS.get(args.dataset, {})
    num_classes = stats.get('num_classes', 10)

    # Build network
    logger.info("Building network from genotype...")
    network = build_from_genotype(
        genotype=[genotype],
        C_init=args.C_init,
        num_classes=num_classes,
    )
    n_params = count_params(network)
    logger.info(f"Network built. Parameters: {n_params:,}")

    # Data module
    data_root = args.data_root or os.path.join('data', args.dataset)
    data_module = ArchEvoDataModule(
        dataset_name=args.dataset,
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data_module.setup()
    logger.info(f"Data module: {data_module}")

    # Train
    logger.info(f"Training from scratch for {args.epochs} epochs on {args.device}...")
    t0 = time.time()
    logs = train_from_scratch(
        model=network,
        data_module=data_module,
        num_epochs=args.epochs,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_time = time.time() - t0

    # Best metrics
    best_epoch = max(logs, key=lambda e: e['val_acc'])
    final_epoch = logs[-1]

    # Save logs
    logs_path = os.path.join(args.output_dir, 'training_logs.json')
    with open(logs_path, 'w') as f:
        json.dump(logs, f, indent=2)

    # Save summary
    summary = {
        'genotype_path': args.genotype_path,
        'dataset': args.dataset,
        'num_epochs': args.epochs,
        'device': args.device,
        'param_count': n_params,
        'best_val_acc': best_epoch['val_acc'],
        'best_epoch': best_epoch['epoch'],
        'final_val_acc': final_epoch['val_acc'],
        'final_train_acc': final_epoch['train_acc'],
        'total_training_sec': total_time,
    }
    summary_path = os.path.join(args.output_dir, 'eval_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"  Dataset:           {args.dataset}")
    print(f"  Epochs:            {args.epochs}")
    print(f"  Parameters:        {n_params:,}")
    print(f"  Best val accuracy: {best_epoch['val_acc']:.4f} (epoch {best_epoch['epoch']})")
    print(f"  Final val acc:     {final_epoch['val_acc']:.4f}")
    print(f"  Training time:     {total_time:.1f}s")
    print(f"  Logs saved to:     {logs_path}")
    print(f"  Summary saved to:  {summary_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
