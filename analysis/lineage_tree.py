"""
analysis/lineage_tree.py
------------------------
Visualise evolutionary search lineage as a directed graph.

Nodes represent individual architectures (by ID).
Edges represent parent→child relationships.
Node colour encodes fitness.
"""

import os
import json
from typing import List, Dict, Any, Optional

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_lineage_tree(
    lineage_log: List[Dict[str, Any]],
    save_path: str,
    title: str = "Architecture Lineage Tree",
    figsize: tuple = (16, 10),
    node_size: int = 300,
    colormap: str = 'viridis',
    font_size: int = 7,
):
    """
    Build and plot the lineage tree from a lineage log.

    Args:
        lineage_log: list of dicts with keys:
            - child_id (int): unique architecture ID
            - parent_ids (list[int]): IDs of parents (empty for gen-0)
            - fitness (float): fitness score
            - generation (int): generation number
        save_path: path to save the figure (PNG/PDF/SVG)
        title: plot title
        figsize: figure size (width, height)
        node_size: matplotlib scatter node size
        colormap: matplotlib colormap name for fitness colouring
        font_size: node label font size
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    # --- Build directed graph ---
    G = nx.DiGraph()
    fitnesses: Dict[int, float] = {}
    generations: Dict[int, int] = {}

    for entry in lineage_log:
        child_id = entry['child_id']
        fitness = entry.get('fitness', 0.0)
        gen = entry.get('generation', 0)
        G.add_node(child_id, fitness=fitness, generation=gen)
        fitnesses[child_id] = fitness
        generations[child_id] = gen

        for parent_id in entry.get('parent_ids', []):
            if parent_id in fitnesses or parent_id in G.nodes:
                G.add_edge(parent_id, child_id)

    if len(G.nodes) == 0:
        print("Warning: empty lineage log, skipping plot.")
        return

    # --- Layout ---
    # Use multipartite layout if generation info available
    for node in G.nodes:
        G.nodes[node]['layer'] = generations.get(node, 0)

    try:
        pos = nx.multipartite_layout(G, subset_key='layer', align='vertical')
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    # --- Fitness colour mapping ---
    all_fitnesses = [fitnesses.get(n, 0.0) for n in G.nodes]
    min_fit = min(all_fitnesses) if all_fitnesses else 0.0
    max_fit = max(all_fitnesses) if all_fitnesses else 1.0
    if max_fit == min_fit:
        max_fit = min_fit + 1.0

    cmap = cm.get_cmap(colormap)
    norm = mcolors.Normalize(vmin=min_fit, vmax=max_fit)
    node_colors = [cmap(norm(fitnesses.get(n, 0.0))) for n in G.nodes]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color='#aaaaaa',
        arrows=True,
        arrowsize=10,
        alpha=0.6,
        connectionstyle='arc3,rad=0.1',
    )

    nodes_drawn = nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_size,
        alpha=0.9,
    )

    # Labels: show id + fitness
    labels = {n: f"{n}\n{fitnesses.get(n, 0.0):.3f}" for n in G.nodes}
    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax,
        font_size=font_size,
        font_color='black',
    )

    # Colourbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Fitness', shrink=0.8)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Lineage tree saved to: {save_path}")


# ---------------------------------------------------------------------------
# Convenience: load from JSON file
# ---------------------------------------------------------------------------

def plot_lineage_from_file(json_path: str, save_path: str, **kwargs):
    """Load a lineage log from a JSON file and plot it."""
    with open(json_path) as f:
        lineage_log = json.load(f)
    plot_lineage_tree(lineage_log, save_path, **kwargs)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def lineage_stats(lineage_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics from a lineage log."""
    if not lineage_log:
        return {}

    fitnesses = [e['fitness'] for e in lineage_log]
    generations = [e['generation'] for e in lineage_log]

    best_entry = max(lineage_log, key=lambda e: e['fitness'])
    worst_entry = min(lineage_log, key=lambda e: e['fitness'])

    # Fitness improvement per generation
    gen_best: Dict[int, float] = {}
    for e in lineage_log:
        g = e['generation']
        gen_best[g] = max(gen_best.get(g, -1e9), e['fitness'])

    return {
        'total_architectures': len(lineage_log),
        'num_generations': max(generations) + 1,
        'best_fitness': best_entry['fitness'],
        'best_id': best_entry['child_id'],
        'worst_fitness': worst_entry['fitness'],
        'mean_fitness': sum(fitnesses) / len(fitnesses),
        'gen_best_fitness': gen_best,
    }
