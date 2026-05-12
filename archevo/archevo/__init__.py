"""
ArchEvo: Neural Architecture Search Framework
"""

__version__ = "1.0.0"
__author__ = "ArchEvo Team"

from archevo.primitives import OPS_REGISTRY, OP_NAMES, NUM_OPS
from archevo.search_space import (
    Network,
    Cell,
    MixedOp,
    arch_parameters,
    extract_genotype,
    build_from_genotype,
    random_genotype,
    Genotype,
)
from archevo.pressure import PressureMode, PressureFn, count_params, estimate_flops, make_pressure_fn

__all__ = [
    'OPS_REGISTRY', 'OP_NAMES', 'NUM_OPS',
    'Network', 'Cell', 'MixedOp',
    'arch_parameters', 'extract_genotype', 'build_from_genotype',
    'random_genotype', 'Genotype',
    'PressureMode', 'PressureFn', 'count_params', 'estimate_flops', 'make_pressure_fn',
]
