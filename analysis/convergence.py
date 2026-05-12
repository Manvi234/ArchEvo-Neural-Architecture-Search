"""
analysis/convergence.py
-----------------------
Convergence analysis plots for ArchEvo experiments.

Provides:
  - plot_convergence: val_acc vs wall-clock time per dataset
  - plot_pressure_response: best accuracy vs lambda per pressure mode
"""

import os
from typing import Dict, List, Any, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Colour palette for algorithms
ALGO_COLORS = {
    'darts':        '#1f77b4',
    'evolutionary': '#ff7f0e',
    'random':       '#2ca02c',
}

PRESSURE_COLORS = {
    'memory':             '#d62728',
    'latency':            '#9467bd',
    'data_scarce':        '#8c564b',
    'distribution_shift': '#e377c2',
}


# ---------------------------------------------------------------------------
# plot_convergence
# ---------------------------------------------------------------------------

def plot_convergence(
    logs_dict: Dict[str, Dict[str, Dict[str, List]]],
    save_path: str,
    figsize_per_dataset: tuple = (6, 4),
    smooth_window: int = 1,
):
    """
    Plot validation accuracy vs wall-clock time.

    Args:
        logs_dict: nested dict structured as:
            {
              dataset_name: {
                algorithm_name: {
                  'time': [t0, t1, ...],      # cumulative wall-clock seconds
                  'val_acc': [acc0, acc1, ...]  # validation accuracy at each step
                }
              }
            }
        save_path: where to save the figure
        figsize_per_dataset: (width, height) per dataset subplot
        smooth_window: moving average window for smoothing curves (1 = no smoothing)
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    datasets = list(logs_dict.keys())
    n = len(datasets)
    if n == 0:
        print("Warning: empty logs_dict, nothing to plot.")
        return

    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_dataset[0] * ncols, figsize_per_dataset[1] * nrows),
        squeeze=False,
    )

    for plot_idx, dataset in enumerate(datasets):
        row = plot_idx // ncols
        col = plot_idx % ncols
        ax = axes[row][col]

        algo_logs = logs_dict[dataset]
        lines_plotted = False

        for algo_name, data in algo_logs.items():
            times = data.get('time', [])
            accs  = data.get('val_acc', [])

            if len(times) == 0 or len(accs) == 0:
                continue

            times = np.array(times, dtype=float)
            accs  = np.array(accs, dtype=float)

            # Make times cumulative if not already
            if times[0] < 0 or (len(times) > 1 and times[1] < times[0]):
                times = np.cumsum(np.abs(times))

            # Optional smoothing
            if smooth_window > 1 and len(accs) >= smooth_window:
                kernel = np.ones(smooth_window) / smooth_window
                accs_smooth = np.convolve(accs, kernel, mode='valid')
                times_smooth = times[:len(accs_smooth)]
            else:
                accs_smooth = accs
                times_smooth = times

            color = ALGO_COLORS.get(algo_name.lower(), None)
            ax.plot(times_smooth, accs_smooth, label=algo_name, color=color, linewidth=2)
            ax.scatter(times_smooth, accs_smooth, s=20, color=color, alpha=0.5)
            lines_plotted = True

        if not lines_plotted:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

        ax.set_title(dataset, fontsize=11, fontweight='bold')
        ax.set_xlabel("Wall-clock time (s)", fontsize=9)
        ax.set_ylabel("Validation accuracy", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylim([0, 1])

    # Hide unused subplots
    for plot_idx in range(n, nrows * ncols):
        row = plot_idx // ncols
        col = plot_idx % ncols
        axes[row][col].set_visible(False)

    plt.suptitle("Convergence: Val Accuracy vs Wall-clock Time", fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Convergence plot saved to: {save_path}")


# ---------------------------------------------------------------------------
# plot_pressure_response
# ---------------------------------------------------------------------------

def plot_pressure_response(
    pressure_results: Dict[str, Dict[float, float]],
    save_path: str,
    figsize: tuple = (10, 5),
    log_scale_x: bool = True,
):
    """
    Plot best accuracy vs lambda for each pressure mode.

    Args:
        pressure_results: dict structured as:
            {
              pressure_mode_str: {
                lambda_value: best_accuracy,
                ...
              }
            }
        save_path: where to save the figure
        figsize: overall figure size
        log_scale_x: if True, use log scale on x-axis (lambda)
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    if not pressure_results:
        print("Warning: empty pressure_results, nothing to plot.")
        return

    fig, ax = plt.subplots(figsize=figsize)

    for mode_str, lambda_acc_map in pressure_results.items():
        if not lambda_acc_map:
            continue

        lambdas = sorted(lambda_acc_map.keys())
        accs = [lambda_acc_map[l] for l in lambdas]

        color = PRESSURE_COLORS.get(mode_str.lower(), None)
        ax.plot(lambdas, accs, marker='o', label=mode_str, color=color, linewidth=2)

    if log_scale_x:
        ax.set_xscale('log')

    ax.set_xlabel("Penalty weight (λ)", fontsize=11)
    ax.set_ylabel("Best validation accuracy", fontsize=11)
    ax.set_title("Pressure Mode Response: Accuracy vs λ", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Pressure response plot saved to: {save_path}")


# ---------------------------------------------------------------------------
# plot_training_curves (bonus: full training curves from run_eval logs)
# ---------------------------------------------------------------------------

def plot_training_curves(
    training_logs: Dict[str, List[Dict[str, Any]]],
    save_path: str,
    metric: str = 'val_acc',
    figsize: tuple = (12, 5),
):
    """
    Plot training metric curves for multiple experiment runs.

    Args:
        training_logs: {experiment_name: [{'epoch': int, metric: float, ...}]}
        save_path: where to save the figure
        metric: which metric to plot (e.g. 'val_acc', 'train_loss')
        figsize: figure size
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax_loss = axes[0]
    ax_acc  = axes[1]

    for exp_name, logs in training_logs.items():
        epochs = [e.get('epoch', i) for i, e in enumerate(logs)]
        train_loss = [e.get('train_loss', float('nan')) for e in logs]
        val_acc    = [e.get('val_acc', float('nan')) for e in logs]

        ax_loss.plot(epochs, train_loss, label=exp_name, linewidth=1.5)
        ax_acc.plot(epochs, val_acc, label=exp_name, linewidth=1.5)

    ax_loss.set_title("Training Loss", fontsize=11)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, linestyle='--', alpha=0.5)

    ax_acc.set_title("Validation Accuracy", fontsize=11)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend(fontsize=8)
    ax_acc.grid(True, linestyle='--', alpha=0.5)
    ax_acc.set_ylim([0, 1])

    plt.suptitle("Training Curves", fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Training curves saved to: {save_path}")
