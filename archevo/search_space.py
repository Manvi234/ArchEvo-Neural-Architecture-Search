"""
archevo/search_space.py
-----------------------
Cell-based search space for ArchEvo.

Architecture:
  - 2 input nodes (prev_prev, prev cell outputs)
  - N=4 intermediate nodes
  - Each intermediate node i receives edges from all i+2 predecessors
  - Total edges = sum(i+2 for i in range(4)) = 14
  - A Cell is a DAG combining these nodes.
  - Network stacks 10 cells (8 normal + 2 reduction).
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple, Any

from archevo.primitives import OPS_REGISTRY, OP_NAMES, NUM_OPS


# ---------------------------------------------------------------------------
# Genotype type
# ---------------------------------------------------------------------------

# A Genotype is a list (one entry per intermediate node) of lists of op names,
# one op name per incoming edge of that node.
# E.g. for 4 nodes: [[op, op], [op, op, op], [op, op, op, op], [op, op, op, op, op]]
# Node 0 gets 2 edges, node 1 gets 3, node 2 gets 4, node 3 gets 5 -> total 14.

Genotype = List[List[str]]

NUM_INTERMEDIATE = 4
NUM_INPUT_NODES = 2


def _num_edges_for_node(node_idx: int) -> int:
    """Number of incoming edges for intermediate node node_idx (0-indexed)."""
    return node_idx + NUM_INPUT_NODES


def _total_edges() -> int:
    return sum(_num_edges_for_node(i) for i in range(NUM_INTERMEDIATE))


TOTAL_EDGES = _total_edges()  # 14


# ---------------------------------------------------------------------------
# MixedOp
# ---------------------------------------------------------------------------

class MixedOp(nn.Module):
    """
    Computes sum(softmax(alpha_i) * op_i(x)) over candidate ops.
    Used in DARTS-mode search.
    """

    def __init__(self, C_in: int, C_out: int):
        super().__init__()
        self.ops = nn.ModuleList([
            OPS_REGISTRY[name](C_in, C_out) for name in OP_NAMES
        ])
        # Architecture weights (raw logits; softmax applied at forward)
        self.alpha = nn.Parameter(torch.zeros(NUM_OPS))

    def forward(self, x: torch.Tensor, stride: int = 1) -> torch.Tensor:
        weights = F.softmax(self.alpha, dim=0)
        out = None
        for w, op in zip(weights, self.ops):
            op_out = op(x, stride=stride)
            if out is None:
                out = w * op_out
            else:
                out = out + w * op_out
        return out

    def entropy(self) -> torch.Tensor:
        """Shannon entropy of the softmax distribution (diversity measure)."""
        probs = F.softmax(self.alpha, dim=0)
        return -(probs * torch.log(probs + 1e-8)).sum()

    def best_op_name(self) -> str:
        return OP_NAMES[self.alpha.argmax().item()]


# ---------------------------------------------------------------------------
# DiscreteOp: wraps a single op for use after architecture is finalised
# ---------------------------------------------------------------------------

class DiscreteOp(nn.Module):
    """Single discrete operation (used after genotype is fixed)."""

    def __init__(self, C_in: int, C_out: int, op_name: str):
        super().__init__()
        self.op = OPS_REGISTRY[op_name](C_in, C_out)
        self.op_name = op_name

    def forward(self, x: torch.Tensor, stride: int = 1) -> torch.Tensor:
        return self.op(x, stride=stride)


# ---------------------------------------------------------------------------
# Cell
# ---------------------------------------------------------------------------

class Cell(nn.Module):
    """
    A DAG cell with NUM_INPUT_NODES=2 inputs and NUM_INTERMEDIATE=4 intermediate nodes.

    Connectivity: intermediate node i receives edges from nodes 0..i+1 (i.e. i+2 predecessors).
    The output of the cell is the concatenation of all intermediate node outputs.

    Args:
        C: number of channels per node (input and output)
        stride: applied on the edge from input nodes (0,1) to the first intermediate node
        cell_type: 'normal' or 'reduction'
        use_mixed_ops: if True, edges are MixedOps (DARTS); else DiscreteOps from genotype
        genotype: required when use_mixed_ops=False; list of list of op names per node
    """

    def __init__(
        self,
        C_in_prev_prev: int,
        C_in_prev: int,
        C: int,
        stride: int = 1,
        cell_type: str = 'normal',
        use_mixed_ops: bool = True,
        genotype: Optional[Genotype] = None,
    ):
        super().__init__()
        self.stride = stride
        self.cell_type = cell_type
        self.use_mixed_ops = use_mixed_ops
        self.C = C
        self.num_intermediate = NUM_INTERMEDIATE

        # Preprocessing: project prev_prev and prev to C channels
        self.preprocess0 = nn.Sequential(
            nn.Conv2d(C_in_prev_prev, C, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )
        self.preprocess1 = nn.Sequential(
            nn.Conv2d(C_in_prev, C, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

        # Build edges: self._ops[node_idx] = ModuleList of ops for each incoming edge
        self._ops = nn.ModuleList()
        node_outputs: List[int] = [C, C]  # channels of input nodes after preprocessing

        for i in range(NUM_INTERMEDIATE):
            num_pred = _num_edges_for_node(i)  # i+2 predecessors
            ops_for_node = nn.ModuleList()
            for j in range(num_pred):
                # Stride is applied only on edges from input nodes (j < NUM_INPUT_NODES)
                # in reduction cells
                edge_stride = stride if (cell_type == 'reduction' and j < NUM_INPUT_NODES) else 1
                _ = edge_stride  # stored conceptually; applied at forward time
                if use_mixed_ops:
                    ops_for_node.append(MixedOp(C, C))
                else:
                    assert genotype is not None, "Genotype required for discrete cell"
                    op_name = genotype[i][j] if j < len(genotype[i]) else 'skip'
                    ops_for_node.append(DiscreteOp(C, C, op_name))
            self._ops.append(ops_for_node)
            node_outputs.append(C)

        # Output channels = C * num_intermediate
        self.out_channels = C * NUM_INTERMEDIATE

    def forward(self, s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s0: output of cell at position i-2 (prev_prev)
            s1: output of cell at position i-1 (prev)
        Returns:
            concatenated intermediate node outputs
        """
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        node_outputs = [s0, s1]

        for i, ops_for_node in enumerate(self._ops):
            num_pred = _num_edges_for_node(i)
            agg = None
            for j, op in enumerate(ops_for_node):
                pred = node_outputs[j]
                # Apply stride on input-node edges in reduction cells
                edge_stride = (
                    self.stride
                    if (self.cell_type == 'reduction' and j < NUM_INPUT_NODES)
                    else 1
                )
                out = op(pred, stride=edge_stride)
                if agg is None:
                    agg = out
                else:
                    # Handle potential shape mismatch (can occur in reduction with mixed strides)
                    if agg.shape != out.shape:
                        # Interpolate to match
                        out = F.interpolate(out, size=agg.shape[2:], mode='bilinear', align_corners=False)
                    agg = agg + out
            node_outputs.append(agg)

        # Concatenate all intermediate node outputs
        return torch.cat(node_outputs[NUM_INPUT_NODES:], dim=1)

    def arch_parameters(self) -> List[nn.Parameter]:
        """Return all alpha parameters from MixedOps."""
        params = []
        if self.use_mixed_ops:
            for ops_for_node in self._ops:
                for op in ops_for_node:
                    if isinstance(op, MixedOp):
                        params.append(op.alpha)
        return params

    def alpha_entropy(self) -> torch.Tensor:
        """Mean entropy across all MixedOps."""
        entropies = []
        for ops_for_node in self._ops:
            for op in ops_for_node:
                if isinstance(op, MixedOp):
                    entropies.append(op.entropy())
        if not entropies:
            return torch.tensor(0.0)
        return torch.stack(entropies).mean()


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class Network(nn.Module):
    """
    Full network stacking 8 normal + 2 reduction cells.

    Reduction cells are placed at positions ~1/3 and ~2/3 of total depth.
    Architecture:
      stem -> [cells] -> global avg pool -> classifier
    """

    def __init__(
        self,
        C_init: int = 16,
        num_classes: int = 10,
        num_cells: int = 10,
        use_mixed_ops: bool = True,
        genotype: Optional[Genotype] = None,
    ):
        super().__init__()
        self.C_init = C_init
        self.num_classes = num_classes
        self.num_cells = num_cells
        self.use_mixed_ops = use_mixed_ops

        # Reduction cell positions (0-indexed)
        self.reduction_positions = {num_cells // 3, 2 * num_cells // 3}

        # Stem: single 3x3 conv, stride=1
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_init, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C_init),
            nn.ReLU(inplace=True),
        )

        # Build cells
        self.cells = nn.ModuleList()
        C_prev_prev = C_init
        C_prev = C_init
        C_curr = C_init

        for i in range(num_cells):
            if i in self.reduction_positions:
                cell_type = 'reduction'
                stride = 2
                C_curr = C_curr * 2  # double channels at reduction
            else:
                cell_type = 'normal'
                stride = 1

            cell = Cell(
                C_in_prev_prev=C_prev_prev,
                C_in_prev=C_prev,
                C=C_curr,
                stride=stride,
                cell_type=cell_type,
                use_mixed_ops=use_mixed_ops,
                genotype=genotype,
            )
            self.cells.append(cell)

            # Output of cell is C_curr * NUM_INTERMEDIATE channels
            C_prev_prev = C_prev
            C_prev = cell.out_channels

        # Classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        out = self.global_pool(s1)
        out = out.view(out.size(0), -1)
        return self.classifier(out)

    def arch_parameters(self) -> List[nn.Parameter]:
        """Return all alpha parameters from all cells."""
        params = []
        for cell in self.cells:
            params.extend(cell.arch_parameters())
        return params

    def weight_parameters(self) -> List[nn.Parameter]:
        """Return all non-arch parameters."""
        arch_param_ids = {id(p) for p in self.arch_parameters()}
        return [p for p in self.parameters() if id(p) not in arch_param_ids]

    def alpha_entropy(self) -> torch.Tensor:
        """Mean alpha entropy across all cells (diversity measure)."""
        entropies = [cell.alpha_entropy() for cell in self.cells]
        return torch.stack(entropies).mean()


# ---------------------------------------------------------------------------
# arch_parameters
# ---------------------------------------------------------------------------

def arch_parameters(network: Network) -> List[nn.Parameter]:
    """Return all alpha parameters from MixedOps in the network."""
    return network.arch_parameters()


# ---------------------------------------------------------------------------
# extract_genotype
# ---------------------------------------------------------------------------

def extract_genotype(network: Network) -> Genotype:
    """
    For each edge, argmax over alphas -> op name.
    Returns a Genotype (list of lists of op names), one list per intermediate node.
    """
    genotype: Genotype = []
    for cell in network.cells:
        if not cell.use_mixed_ops:
            # Already discrete — reconstruct from ops
            cell_geno: List[List[str]] = []
            for ops_for_node in cell._ops:
                node_ops = []
                for op in ops_for_node:
                    if isinstance(op, DiscreteOp):
                        node_ops.append(op.op_name)
                    else:
                        node_ops.append('skip')
                cell_geno.append(node_ops)
            genotype.append(cell_geno)
        else:
            cell_geno = []
            for ops_for_node in cell._ops:
                node_ops = []
                for op in ops_for_node:
                    if isinstance(op, MixedOp):
                        node_ops.append(op.best_op_name())
                    else:
                        node_ops.append('skip')
                cell_geno.append(node_ops)
            genotype.append(cell_geno)
    return genotype


# ---------------------------------------------------------------------------
# build_from_genotype
# ---------------------------------------------------------------------------

def build_from_genotype(
    genotype: Genotype,
    C_init: int,
    num_classes: int,
    num_cells: int = 10,
) -> Network:
    """
    Build a clean Network with discrete ops from a genotype.
    If genotype has one entry (shared across all cells), it is broadcast.
    Otherwise it must have num_cells entries.
    """
    if len(genotype) == 1:
        # Broadcast single cell genotype to all cells
        shared = genotype[0]
        full_genotype = [shared] * num_cells
    elif len(genotype) == num_cells:
        full_genotype = genotype
    else:
        # Try to use first entry as template
        full_genotype = [genotype[0]] * num_cells

    return Network(
        C_init=C_init,
        num_classes=num_classes,
        num_cells=num_cells,
        use_mixed_ops=False,
        genotype=full_genotype[0],  # Cell accepts per-node genotype
    )


# ---------------------------------------------------------------------------
# Utility: random genotype
# ---------------------------------------------------------------------------

def random_genotype() -> Genotype:
    """Generate a random genotype (one cell's worth of op assignments)."""
    geno: Genotype = []
    for i in range(NUM_INTERMEDIATE):
        num_edges = _num_edges_for_node(i)
        node_ops = [random.choice(OP_NAMES) for _ in range(num_edges)]
        geno.append(node_ops)
    return geno


def genotype_to_str(genotype: Genotype) -> str:
    """Serialise genotype to a compact string."""
    import json
    return json.dumps(genotype)


def str_to_genotype(s: str) -> Genotype:
    """Deserialise genotype from string."""
    import json
    return json.loads(s)
