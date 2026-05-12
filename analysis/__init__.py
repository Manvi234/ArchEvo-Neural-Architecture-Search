"""
analysis: experiment analysis and visualisation utilities.
"""

from analysis.lineage_tree import plot_lineage_tree, plot_lineage_from_file, lineage_stats
from analysis.heatmaps import plot_op_preference_heatmap, plot_cross_dataset_heatmap
from analysis.convergence import plot_convergence, plot_pressure_response, plot_training_curves

__all__ = [
    'plot_lineage_tree', 'plot_lineage_from_file', 'lineage_stats',
    'plot_op_preference_heatmap', 'plot_cross_dataset_heatmap',
    'plot_convergence', 'plot_pressure_response', 'plot_training_curves',
]
