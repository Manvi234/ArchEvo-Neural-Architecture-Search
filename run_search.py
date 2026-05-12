#!/usr/bin/env python3
"""
run_search.py
-------------
CLI entry point for neural architecture search.

Usage:
    python run_search.py \
        --dataset cifar10 \
        --algorithm darts \
        --pressure memory \
        --lambda 0.1 \
        --search_epochs 50 \
        --device cuda \
        --output_dir results/

    python run_search.py \
        --dataset cifar10 \
        --algorithm evolutionary \
        --pressure none \
        --device cpu
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime
from typing import Optional

import torch
import yaml

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from archevo.data.datamodule import ArchEvoDataModule
from archevo.pressure import make_pressure_fn
from archevo.search_space import (
    Network,
    build_from_genotype,
    genotype_to_str,
    str_to_genotype,
)
from archevo.pressure import count_params

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('run_search')


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(dataset: str, algorithm: str, pressure: str) -> dict:
    """Merge base config with dataset/algorithm/pressure overrides."""
    config_dir = os.path.join(os.path.dirname(__file__), 'configs')
    base = load_yaml(os.path.join(config_dir, 'base.yaml'))
    ds_cfg = load_yaml(os.path.join(config_dir, 'datasets', f'{dataset}.yaml'))
    algo_cfg = load_yaml(os.path.join(config_dir, 'algorithms', f'{algorithm}.yaml'))

    pressure_cfg = {}
    if pressure != 'none':
        pname = pressure.replace('-', '_')
        pressure_cfg = load_yaml(os.path.join(config_dir, 'pressure', f'{pname}.yaml'))

    # Merge: base < algo_cfg < ds_cfg < pressure_cfg
    config = {**base}
    config['dataset'] = ds_cfg
    config['algorithm'] = algo_cfg
    config['pressure_config'] = pressure_cfg
    return config


# ---------------------------------------------------------------------------
# SQLite logging
# ---------------------------------------------------------------------------

def ensure_db(db_path: str):
    """Create the experiments table if it doesn't exist."""
    os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset          TEXT NOT NULL,
            algorithm        TEXT NOT NULL,
            pressure_mode    TEXT,
            lambda_          REAL,
            genotype_json    TEXT,
            val_accuracy     REAL,
            param_count      INTEGER,
            training_time_sec REAL,
            timestamp        TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_experiment(
    db_path: str,
    dataset: str,
    algorithm: str,
    pressure_mode: str,
    lambda_: float,
    genotype_json: str,
    val_accuracy: float,
    param_count: int,
    training_time_sec: float,
) -> int:
    """Insert an experiment result into SQLite. Returns the inserted row ID."""
    ensure_db(db_path)
    conn = sqlite3.connect(db_path)
    timestamp = datetime.utcnow().isoformat()
    cursor = conn.execute(
        """
        INSERT INTO experiments
            (dataset, algorithm, pressure_mode, lambda_, genotype_json,
             val_accuracy, param_count, training_time_sec, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (dataset, algorithm, pressure_mode, lambda_, genotype_json,
         val_accuracy, param_count, training_time_sec, timestamp),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


# ---------------------------------------------------------------------------
# Search runners
# ---------------------------------------------------------------------------

def run_darts(args, config, data_module, pressure_fn, output_dir):
    from archevo.search.darts import DARTSSearcher

    algo_cfg = config.get('algorithm', {})
    search_cfg = config.get('search', {})
    ds_cfg = config.get('dataset', {})

    network = Network(
        C_init=search_cfg.get('C_init', 16),
        num_classes=ds_cfg.get('num_classes', 10),
        num_cells=search_cfg.get('num_cells', 10),
        use_mixed_ops=True,
    )

    searcher = DARTSSearcher(
        network=network,
        data_module=data_module,
        pressure_fn=pressure_fn,
        order=algo_cfg.get('order', 'first'),
        lr_w=algo_cfg.get('lr_w', 0.025),
        lr_alpha=algo_cfg.get('lr_alpha', 3e-4),
        momentum=algo_cfg.get('momentum', 0.9),
        weight_decay=algo_cfg.get('weight_decay', 3e-4),
        grad_clip=algo_cfg.get('grad_clip', 5.0),
        epsilon=algo_cfg.get('epsilon', 0.01),
        device=args.device,
    )

    num_epochs = args.search_epochs or search_cfg.get('search_epochs', 50)
    logger.info(f"Running DARTS for {num_epochs} epochs")
    t0 = time.time()
    genotype = searcher.search(num_epochs=num_epochs)
    elapsed = time.time() - t0

    logs = searcher.get_logs()
    val_acc = logs[-1]['val_acc'] if logs else 0.0

    # Save logs
    logs_path = os.path.join(output_dir, 'search_logs.json')
    with open(logs_path, 'w') as f:
        json.dump(logs, f, indent=2)

    return genotype, val_acc, elapsed


def run_evolutionary(args, config, data_module, pressure_fn, output_dir):
    from archevo.search.evolutionary import EvolutionarySearcher

    algo_cfg = config.get('algorithm', {})
    search_cfg = config.get('search', {})
    ds_cfg = config.get('dataset', {})

    searcher = EvolutionarySearcher(
        data_module=data_module,
        pressure_fn=pressure_fn,
        pop_size=algo_cfg.get('pop_size', 30),
        num_generations=algo_cfg.get('num_generations', 20),
        tournament_k=algo_cfg.get('tournament_k', 5),
        elite_k=algo_cfg.get('elite_k', 3),
        proxy_epochs=args.proxy_epochs or search_cfg.get('proxy_epochs', 7),
        C_init=search_cfg.get('C_init', 16),
        device=args.device,
    )

    logger.info("Running evolutionary search")
    t0 = time.time()
    genotype = searcher.search()
    elapsed = time.time() - t0

    # Save lineage log
    lineage_path = os.path.join(output_dir, 'lineage_log.json')
    searcher.save_lineage_log(lineage_path)
    logger.info(f"Lineage log saved to: {lineage_path}")

    # Determine best fitness as proxy for val_acc
    lineage = searcher.get_lineage_log()
    val_acc = max((e['fitness'] for e in lineage), default=0.0)

    return genotype, val_acc, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="ArchEvo Neural Architecture Search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--dataset', type=str, default='cifar10',
                   choices=['cifar10', 'eurosat', 'isic', 'cub200'],
                   help='Dataset name')
    p.add_argument('--algorithm', type=str, default='darts',
                   choices=['darts', 'evolutionary'],
                   help='Search algorithm')
    p.add_argument('--pressure', type=str, default='none',
                   choices=['memory', 'latency', 'data_scarce', 'distribution_shift', 'none'],
                   help='Pressure mode')
    p.add_argument('--lambda', dest='lambda_', type=float, default=0.1,
                   help='Pressure penalty weight')
    p.add_argument('--output_dir', type=str, default='results',
                   help='Directory to save outputs')
    p.add_argument('--device', type=str, default='cpu',
                   help='PyTorch device (cpu/cuda/mps)')
    p.add_argument('--search_epochs', type=int, default=None,
                   help='Override number of search epochs')
    p.add_argument('--proxy_epochs', type=int, default=None,
                   help='Override proxy training epochs (evolutionary only)')
    p.add_argument('--batch_size', type=int, default=64,
                   help='DataLoader batch size')
    p.add_argument('--num_workers', type=int, default=4,
                   help='DataLoader workers')
    p.add_argument('--data_root', type=str, default=None,
                   help='Root directory for dataset (default: data/<dataset>)')
    p.add_argument('--db_path', type=str, default='results/experiments.db',
                   help='SQLite database path')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    return p.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)

    # Load configs
    config = load_config(args.dataset, args.algorithm, args.pressure)

    # Output directory
    run_name = f"{args.dataset}_{args.algorithm}_{args.pressure}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Save run config
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'algorithm': args.algorithm,
            'pressure': args.pressure,
            'lambda_': args.lambda_,
            'device': args.device,
            'config': config,
        }, f, indent=2)

    # Data module
    data_root = args.data_root or os.path.join('data', args.dataset)
    ds_cfg = config.get('dataset', {})

    data_module = ArchEvoDataModule(
        dataset_name=args.dataset,
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        proxy_train_size=ds_cfg.get('proxy_train_size', 10000),
        proxy_val_size=ds_cfg.get('proxy_val_size', 2000),
        seed=args.seed,
    )
    data_module.setup()
    logger.info(f"Data module ready: {data_module}")

    # Pressure function
    search_cfg = config.get('search', {})
    image_size = ds_cfg.get('image_size', 32)
    pressure_fn = make_pressure_fn(
        mode_str=args.pressure,
        lambda_=args.lambda_,
        budget_params=1e6,
        budget_flops=100.0,
        image_size=image_size,
    )
    if pressure_fn is not None:
        logger.info(f"Pressure function: {pressure_fn}")

    # Run search
    if args.algorithm == 'darts':
        genotype, val_acc, elapsed = run_darts(args, config, data_module, pressure_fn, output_dir)
    elif args.algorithm == 'evolutionary':
        genotype, val_acc, elapsed = run_evolutionary(args, config, data_module, pressure_fn, output_dir)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    # Save genotype
    genotype_path = os.path.join(output_dir, 'genotype.json')
    with open(genotype_path, 'w') as f:
        json.dump(genotype, f, indent=2)
    logger.info(f"Genotype saved to: {genotype_path}")

    # Count parameters of best network
    try:
        final_network = build_from_genotype(
            genotype=[genotype] if not isinstance(genotype[0], list) else genotype,
            C_init=search_cfg.get('C_init', 16),
            num_classes=ds_cfg.get('num_classes', 10),
        )
        n_params = count_params(final_network)
    except Exception as e:
        logger.warning(f"Could not count params: {e}")
        n_params = 0

    # Log to SQLite
    ensure_db(args.db_path)
    row_id = log_experiment(
        db_path=args.db_path,
        dataset=args.dataset,
        algorithm=args.algorithm,
        pressure_mode=args.pressure,
        lambda_=args.lambda_,
        genotype_json=genotype_to_str(genotype),
        val_accuracy=val_acc,
        param_count=n_params,
        training_time_sec=elapsed,
    )
    logger.info(f"Experiment logged to DB (id={row_id}): val_acc={val_acc:.4f}, params={n_params:,}")

    print("\n" + "=" * 60)
    print(f"Search complete!")
    print(f"  Dataset:      {args.dataset}")
    print(f"  Algorithm:    {args.algorithm}")
    print(f"  Pressure:     {args.pressure} (lambda={args.lambda_})")
    print(f"  Val accuracy: {val_acc:.4f}")
    print(f"  Param count:  {n_params:,}")
    print(f"  Time:         {elapsed:.1f}s")
    print(f"  Genotype:     {genotype_path}")
    print(f"  DB row id:    {row_id}")
    print("=" * 60)


if __name__ == '__main__':
    main()
