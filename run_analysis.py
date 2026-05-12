#!/usr/bin/env python3
"""
run_analysis.py
---------------
Generate all analysis plots from experiment results stored in SQLite DB.

Generates:
  1. Lineage trees (one per dataset that has evolutionary results)
  2. Op-preference heatmaps (per dataset)
  3. Convergence plots (val_acc vs wall-clock time)
  4. Pressure response curves

Usage:
    python run_analysis.py \
        --results_db results/experiments.db \
        --output_dir results/analysis/
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from collections import defaultdict
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('run_analysis')


# ---------------------------------------------------------------------------
# DB loading
# ---------------------------------------------------------------------------

def load_experiments(db_path: str) -> List[Dict[str, Any]]:
    """Load all experiments from the SQLite database."""
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        return []

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT id, dataset, algorithm, pressure_mode, lambda_,
                   genotype_json, val_accuracy, param_count, training_time_sec, timestamp
            FROM experiments
            ORDER BY dataset, algorithm, val_accuracy DESC
            """
        )
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
        experiments = [dict(zip(cols, row)) for row in rows]
        logger.info(f"Loaded {len(experiments)} experiments from {db_path}")
        return experiments
    except sqlite3.OperationalError as e:
        logger.error(f"DB read error: {e}")
        return []
    finally:
        conn.close()


def load_lineage_logs(results_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load lineage logs from run directories.
    Returns {dataset_algorithm_pressure: lineage_log}.
    """
    lineage_logs = {}
    if not os.path.isdir(results_dir):
        return lineage_logs

    for run_dir in os.listdir(results_dir):
        full_path = os.path.join(results_dir, run_dir)
        if not os.path.isdir(full_path):
            continue
        lineage_path = os.path.join(full_path, 'lineage_log.json')
        if os.path.exists(lineage_path):
            try:
                with open(lineage_path) as f:
                    lineage_logs[run_dir] = json.load(f)
                logger.info(f"Loaded lineage log: {lineage_path}")
            except Exception as e:
                logger.warning(f"Failed to load lineage log {lineage_path}: {e}")
    return lineage_logs


def load_search_logs(results_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load DARTS search logs from run directories.
    Returns {run_name: search_logs}.
    """
    search_logs = {}
    if not os.path.isdir(results_dir):
        return search_logs

    for run_dir in os.listdir(results_dir):
        full_path = os.path.join(results_dir, run_dir)
        if not os.path.isdir(full_path):
            continue
        log_path = os.path.join(full_path, 'search_logs.json')
        if os.path.exists(log_path):
            try:
                with open(log_path) as f:
                    search_logs[run_dir] = json.load(f)
                logger.info(f"Loaded search logs: {log_path}")
            except Exception as e:
                logger.warning(f"Failed to load search log {log_path}: {e}")
    return search_logs


# ---------------------------------------------------------------------------
# Build analysis structures
# ---------------------------------------------------------------------------

def build_results_list(experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert DB rows to results_list format for heatmap analysis."""
    results = []
    for exp in experiments:
        genotype_json = exp.get('genotype_json', '[]')
        try:
            genotype = json.loads(genotype_json) if genotype_json else []
        except (json.JSONDecodeError, TypeError):
            genotype = []

        results.append({
            'dataset': exp.get('dataset', ''),
            'algorithm': exp.get('algorithm', ''),
            'pressure_mode': exp.get('pressure_mode', 'none'),
            'lambda_': exp.get('lambda_', 0.0),
            'genotype': genotype,
            'val_accuracy': exp.get('val_accuracy', 0.0),
            'param_count': exp.get('param_count', 0),
        })
    return results


def build_convergence_logs(
    search_logs: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, Dict[str, List]]]:
    """
    Build logs_dict for convergence plotting.
    Parses run_dir names as {dataset}_{algorithm}_{pressure}.
    """
    logs_dict: Dict[str, Dict[str, Dict[str, List]]] = defaultdict(lambda: defaultdict(dict))

    for run_name, logs in search_logs.items():
        # Parse run_name
        parts = run_name.split('_', 2)
        if len(parts) < 2:
            continue
        dataset = parts[0]
        algorithm = parts[1]

        times = []
        val_accs = []
        cumulative = 0.0

        for entry in logs:
            elapsed = entry.get('elapsed_sec', 0.0)
            cumulative += elapsed
            val_acc = entry.get('val_acc', 0.0) or entry.get('fitness', 0.0)
            times.append(cumulative)
            val_accs.append(val_acc)

        if times:
            logs_dict[dataset][algorithm] = {
                'time': times,
                'val_acc': val_accs,
            }

    # Convert defaultdicts to plain dicts
    return {k: dict(v) for k, v in logs_dict.items()}


def build_pressure_response(
    experiments: List[Dict[str, Any]],
) -> Dict[str, Dict[float, float]]:
    """
    Build pressure_results for pressure response plotting.
    Groups by pressure_mode, maps lambda -> best accuracy.
    """
    pressure_map: Dict[str, Dict[float, float]] = defaultdict(dict)

    for exp in experiments:
        mode = exp.get('pressure_mode', 'none')
        if not mode or mode == 'none':
            continue
        lambda_ = exp.get('lambda_', 0.0) or 0.0
        val_acc = exp.get('val_accuracy', 0.0) or 0.0

        key = (mode, lambda_)
        if key not in pressure_map[mode] or val_acc > pressure_map[mode].get(lambda_, -1):
            pressure_map[mode][lambda_] = val_acc

    return dict(pressure_map)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="ArchEvo analysis: generate all plots from experiment DB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--results_db', type=str, default='results/experiments.db',
                   help='Path to SQLite experiments database')
    p.add_argument('--results_dir', type=str, default='results',
                   help='Results directory containing run subdirectories')
    p.add_argument('--output_dir', type=str, default='results/analysis',
                   help='Directory to save analysis plots')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Import analysis modules
    from analysis.lineage_tree import plot_lineage_tree, lineage_stats
    from analysis.heatmaps import plot_op_preference_heatmap, plot_cross_dataset_heatmap
    from analysis.convergence import plot_convergence, plot_pressure_response

    # Load data
    experiments = load_experiments(args.results_db)
    lineage_logs = load_lineage_logs(args.results_dir)
    search_logs  = load_search_logs(args.results_dir)

    if not experiments and not lineage_logs and not search_logs:
        logger.warning("No data found. Run some searches first with run_search.py")
        return

    results_list = build_results_list(experiments)
    datasets = sorted(set(e.get('dataset', '') for e in experiments if e.get('dataset')))

    # -------------------------------------------------------------------
    # 1. Lineage trees (one per lineage log / run)
    # -------------------------------------------------------------------
    logger.info("Generating lineage trees...")
    for run_name, lineage_log in lineage_logs.items():
        if not lineage_log:
            continue
        stats = lineage_stats(lineage_log)
        logger.info(
            f"  {run_name}: {stats.get('total_architectures', 0)} archs, "
            f"best fitness={stats.get('best_fitness', 0):.4f}"
        )
        save_path = os.path.join(args.output_dir, f'lineage_{run_name}.png')
        plot_lineage_tree(
            lineage_log=lineage_log,
            save_path=save_path,
            title=f"Lineage Tree: {run_name}",
        )

    if not lineage_logs:
        logger.info("  No lineage logs found (run evolutionary search first)")

    # -------------------------------------------------------------------
    # 2. Op-preference heatmaps
    # -------------------------------------------------------------------
    if results_list and datasets:
        logger.info("Generating op-preference heatmaps...")

        # Per-algorithm heatmaps
        for algo in ['darts', 'evolutionary']:
            algo_results = [r for r in results_list if r.get('algorithm') == algo]
            if algo_results:
                hmap_path = os.path.join(args.output_dir, f'op_preference_{algo}.png')
                plot_op_preference_heatmap(
                    results_list=algo_results,
                    datasets=datasets,
                    save_path=hmap_path,
                    algorithm=algo,
                )

        # Cross-dataset heatmap (all algorithms combined)
        cross_path = os.path.join(args.output_dir, 'op_preference_cross_dataset.png')
        plot_cross_dataset_heatmap(
            results_list=results_list,
            datasets=datasets,
            save_path=cross_path,
        )
    else:
        logger.info("  No experiment results for heatmaps")

    # -------------------------------------------------------------------
    # 3. Convergence plots
    # -------------------------------------------------------------------
    logs_dict = build_convergence_logs(search_logs)
    if logs_dict:
        logger.info("Generating convergence plots...")
        conv_path = os.path.join(args.output_dir, 'convergence.png')
        plot_convergence(logs_dict=logs_dict, save_path=conv_path)
    else:
        logger.info("  No search logs for convergence plot")
        # Create a placeholder with empty data
        if datasets:
            empty_logs = {ds: {} for ds in datasets}
            conv_path = os.path.join(args.output_dir, 'convergence.png')
            logger.info("  Creating placeholder convergence plot")
            plot_convergence(logs_dict=empty_logs, save_path=conv_path)

    # -------------------------------------------------------------------
    # 4. Pressure response curves
    # -------------------------------------------------------------------
    pressure_results = build_pressure_response(experiments)
    if pressure_results:
        logger.info("Generating pressure response plots...")
        press_path = os.path.join(args.output_dir, 'pressure_response.png')
        plot_pressure_response(
            pressure_results=pressure_results,
            save_path=press_path,
        )
    else:
        logger.info("  No pressure experiments found for pressure response plot")

    # -------------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"  Experiments:     {len(experiments)}")
    print(f"  Datasets:        {datasets}")
    print(f"  Lineage logs:    {len(lineage_logs)}")
    print(f"  Search log runs: {len(search_logs)}")
    print(f"  Pressure modes:  {list(pressure_results.keys())}")
    print(f"  Output dir:      {args.output_dir}")
    print("=" * 60)

    if experiments:
        best = max(experiments, key=lambda e: e.get('val_accuracy', 0))
        print(f"\nBest experiment:")
        print(f"  Dataset:    {best.get('dataset')}")
        print(f"  Algorithm:  {best.get('algorithm')}")
        print(f"  Pressure:   {best.get('pressure_mode')} (lambda={best.get('lambda_')})")
        print(f"  Val acc:    {best.get('val_accuracy', 0):.4f}")
        print(f"  Params:     {best.get('param_count', 0):,}")


if __name__ == '__main__':
    main()
