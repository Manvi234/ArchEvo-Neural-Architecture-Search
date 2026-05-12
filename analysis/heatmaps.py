"""
analysis/heatmaps.py
--------------------
Operation preference heatmaps.

For each dataset, count operation frequencies per cell/edge position,
then plot a heatmap: rows = ops, cols = cell positions.
"""

import os
from typing import List, Dict, Any, Optional
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from archevo.primitives import OP_NAMES
from archevo.search_space import NUM_INTERMEDIATE, _num_edges_for_node


# ---------------------------------------------------------------------------
# Edge position labels
# ---------------------------------------------------------------------------

def _get_edge_labels() -> List[str]:
    """Generate human-readable labels for each edge in a cell DAG."""
    labels = []
    for node_idx in range(NUM_INTERMEDIATE):
        for edge_idx in range(_num_edges_for_node(node_idx)):
            labels.append(f"N{node_idx}_E{edge_idx}")
    return labels


EDGE_LABELS = _get_edge_labels()
NUM_EDGES = len(EDGE_LABELS)


# ---------------------------------------------------------------------------
# Frequency computation
# ---------------------------------------------------------------------------

def _compute_op_frequencies(
    results_list: List[Dict[str, Any]],
    dataset: str,
    algorithm: Optional[str] = None,
) -> np.ndarray:
    """
    Count operation frequencies per edge position for a given dataset.

    Args:
        results_list: list of dicts with keys 'dataset', 'algorithm', 'genotype'
        dataset: filter by this dataset name
        algorithm: optionally filter by algorithm name (None = all)

    Returns:
        freq_matrix: (num_ops, num_edges) frequency matrix (counts)
    """
    freq = np.zeros((len(OP_NAMES), NUM_EDGES), dtype=float)

    for result in results_list:
        if result.get('dataset') != dataset:
            continue
        if algorithm is not None and result.get('algorithm') != algorithm:
            continue

        genotype = result.get('genotype')
        if genotype is None:
            continue

        # Flatten genotype into edge list
        edge_idx = 0
        for node_idx, node_ops in enumerate(genotype):
            for op_name in node_ops:
                if op_name in OP_NAMES and edge_idx < NUM_EDGES:
                    op_idx = OP_NAMES.index(op_name)
                    freq[op_idx, edge_idx] += 1
                edge_idx += 1

    return freq


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------

def plot_op_preference_heatmap(
    results_list: List[Dict[str, Any]],
    datasets: List[str],
    save_path: str,
    algorithm: Optional[str] = None,
    figsize_per_dataset: tuple = (10, 4),
    cmap: str = 'YlOrRd',
):
    """
    Plot operation preference heatmaps, one subplot per dataset.

    Args:
        results_list: list of dicts, each with:
            - 'dataset' (str): dataset name
            - 'algorithm' (str): search algorithm name
            - 'genotype' (Genotype): list of lists of op names
        datasets: list of dataset names to include
        save_path: path to save the figure
        algorithm: if provided, filter to this algorithm only
        figsize_per_dataset: (width, height) per dataset subplot
        cmap: seaborn/matplotlib colormap name
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    n_datasets = len(datasets)
    fig_w = figsize_per_dataset[0] * min(n_datasets, 2)
    fig_h = figsize_per_dataset[1] * ((n_datasets + 1) // 2)

    ncols = min(n_datasets, 2)
    nrows = (n_datasets + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    for plot_idx, dataset in enumerate(datasets):
        row = plot_idx // ncols
        col = plot_idx % ncols
        ax = axes[row][col]

        freq = _compute_op_frequencies(results_list, dataset, algorithm)

        # Normalise to [0, 1] per column (edge position)
        col_sums = freq.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        freq_norm = freq / col_sums

        # Trim edge labels to actual data width (in case of short genotypes)
        edge_labels_used = EDGE_LABELS[:freq_norm.shape[1]]

        if HAS_SEABORN:
            sns.heatmap(
                freq_norm,
                ax=ax,
                xticklabels=edge_labels_used,
                yticklabels=OP_NAMES,
                cmap=cmap,
                vmin=0.0, vmax=1.0,
                annot=True,
                fmt='.2f',
                linewidths=0.5,
                cbar_kws={'label': 'Relative frequency'},
            )
        else:
            im = ax.imshow(freq_norm, aspect='auto', cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks(range(len(edge_labels_used)))
            ax.set_xticklabels(edge_labels_used, rotation=90, fontsize=7)
            ax.set_yticks(range(len(OP_NAMES)))
            ax.set_yticklabels(OP_NAMES, fontsize=9)
            plt.colorbar(im, ax=ax, label='Relative frequency')

            # Annotate
            for yi in range(freq_norm.shape[0]):
                for xi in range(freq_norm.shape[1]):
                    ax.text(xi, yi, f"{freq_norm[yi, xi]:.2f}",
                            ha='center', va='center', fontsize=6, color='black')

        algo_str = f" ({algorithm})" if algorithm else ""
        ax.set_title(f"{dataset}{algo_str}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Edge position", fontsize=9)
        ax.set_ylabel("Operation", fontsize=9)
        ax.tick_params(axis='x', labelsize=7, rotation=90)

    # Hide any unused subplots
    for plot_idx in range(n_datasets, nrows * ncols):
        row = plot_idx // ncols
        col = plot_idx % ncols
        axes[row][col].set_visible(False)

    plt.suptitle("Operation Preference by Dataset and Cell Position", fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Op preference heatmap saved to: {save_path}")


# ---------------------------------------------------------------------------
# Cross-dataset comparison heatmap
# ---------------------------------------------------------------------------

def plot_cross_dataset_heatmap(
    results_list: List[Dict[str, Any]],
    datasets: List[str],
    save_path: str,
    cmap: str = 'Blues',
):
    """
    Plot a single heatmap: rows = datasets, cols = ops,
    showing overall op preference (avg across all edges).
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    data = np.zeros((len(datasets), len(OP_NAMES)))

    for di, dataset in enumerate(datasets):
        freq = _compute_op_frequencies(results_list, dataset)
        # Average across edge positions
        avg = freq.mean(axis=1)
        total = avg.sum()
        if total > 0:
            avg /= total
        data[di] = avg

    fig, ax = plt.subplots(figsize=(len(OP_NAMES) * 1.5 + 2, len(datasets) * 1.2 + 2))

    if HAS_SEABORN:
        sns.heatmap(
            data, ax=ax,
            xticklabels=OP_NAMES,
            yticklabels=datasets,
            cmap=cmap,
            vmin=0, vmax=1,
            annot=True, fmt='.3f',
            linewidths=0.5,
            cbar_kws={'label': 'Mean op frequency'},
        )
    else:
        im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(OP_NAMES)))
        ax.set_xticklabels(OP_NAMES, rotation=45, ha='right')
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets)
        plt.colorbar(im, ax=ax, label='Mean op frequency')

    ax.set_title("Cross-Dataset Operation Preference", fontsize=13, fontweight='bold')
    ax.set_xlabel("Operation", fontsize=10)
    ax.set_ylabel("Dataset", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Cross-dataset heatmap saved to: {save_path}")
