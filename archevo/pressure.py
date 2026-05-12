"""
archevo/pressure.py
-------------------
Pressure functions for resource-aware neural architecture search.

PressureMode enum defines the four pressure types:
  MEMORY           - penalise large parameter count
  LATENCY          - penalise slow forward pass (measured as wall-clock ms)
  DATA_SCARCE      - no penalty (data module handles scarce split)
  DISTRIBUTION_SHIFT - no penalty (handled in evaluation)
"""

import time
from enum import Enum, auto
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# PressureMode
# ---------------------------------------------------------------------------

class PressureMode(Enum):
    MEMORY = auto()
    LATENCY = auto()
    DATA_SCARCE = auto()
    DISTRIBUTION_SHIFT = auto()

    @classmethod
    def from_str(cls, s: str) -> 'PressureMode':
        mapping = {
            'memory':             cls.MEMORY,
            'latency':            cls.LATENCY,
            'data_scarce':        cls.DATA_SCARCE,
            'distribution_shift': cls.DISTRIBUTION_SHIFT,
        }
        key = s.lower().replace('-', '_')
        if key not in mapping:
            raise ValueError(
                f"Unknown pressure mode '{s}'. "
                f"Choose from: {list(mapping.keys())}"
            )
        return mapping[key]


# ---------------------------------------------------------------------------
# Utility: count parameters
# ---------------------------------------------------------------------------

def count_params(network: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in network.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Utility: estimate FLOPs via timing
# ---------------------------------------------------------------------------

def estimate_flops(
    network: nn.Module,
    x: Optional[torch.Tensor] = None,
    image_size: int = 32,
    warmup: int = 3,
    repeats: int = 10,
) -> float:
    """
    Run the forward pass and return mean wall-clock time in milliseconds
    as a proxy for FLOPs (avoids heavyweight FLOP-counting libraries).

    Args:
        network: the nn.Module to time
        x: optional input tensor; if None, a random (1, 3, image_size, image_size) is used
        image_size: spatial size when constructing dummy input
        warmup: number of warmup forward passes
        repeats: number of timed passes

    Returns:
        Mean forward pass time in milliseconds
    """
    device = next(network.parameters()).device
    if x is None:
        x = torch.randn(1, 3, image_size, image_size, device=device)
    else:
        x = x[:1].to(device)  # use single sample

    network.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = network(x)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = network(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)  # ms

    return sum(times) / len(times)


# ---------------------------------------------------------------------------
# PressureFn
# ---------------------------------------------------------------------------

class PressureFn:
    """
    Callable pressure penalty.

    Args:
        mode: PressureMode enum value
        lambda_: weight of the penalty
        budget_params: parameter budget for MEMORY mode (default 1M)
        budget_flops: FLOP / latency budget for LATENCY mode (in ms; default 100ms)
        scarce_samples: ignored here (handled by DataModule)
        image_size: used to create dummy input for LATENCY mode
    """

    def __init__(
        self,
        mode: PressureMode,
        lambda_: float = 0.1,
        budget_params: float = 1e6,
        budget_flops: float = 100.0,  # ms
        scarce_samples: int = 500,
        image_size: int = 32,
    ):
        self.mode = mode
        self.lambda_ = lambda_
        self.budget_params = budget_params
        self.budget_flops = budget_flops
        self.scarce_samples = scarce_samples
        self.image_size = image_size

    def __call__(
        self,
        network: nn.Module,
        x_sample: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute and return a scalar penalty tensor.

        Args:
            network: the network being evaluated
            x_sample: optional input sample for LATENCY mode

        Returns:
            scalar torch.Tensor with the penalty value
        """
        if self.mode == PressureMode.MEMORY:
            n_params = count_params(network)
            penalty = self.lambda_ * (n_params / self.budget_params)
            return torch.tensor(penalty, dtype=torch.float32)

        elif self.mode == PressureMode.LATENCY:
            latency_ms = estimate_flops(
                network, x=x_sample, image_size=self.image_size
            )
            penalty = self.lambda_ * (latency_ms / self.budget_flops)
            return torch.tensor(penalty, dtype=torch.float32)

        elif self.mode == PressureMode.DATA_SCARCE:
            # Penalty handled by DataModule (scarce proxy split); return 0 here
            return torch.tensor(0.0)

        elif self.mode == PressureMode.DISTRIBUTION_SHIFT:
            # Penalty handled during evaluation; return 0 here
            return torch.tensor(0.0)

        else:
            raise ValueError(f"Unknown pressure mode: {self.mode}")

    def __repr__(self) -> str:
        return (
            f"PressureFn(mode={self.mode.name}, lambda_={self.lambda_}, "
            f"budget_params={self.budget_params:.0e}, budget_flops={self.budget_flops}ms)"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_pressure_fn(
    mode_str: str,
    lambda_: float = 0.1,
    budget_params: float = 1e6,
    budget_flops: float = 100.0,
    image_size: int = 32,
) -> Optional[PressureFn]:
    """
    Convenience factory.

    Args:
        mode_str: one of 'memory', 'latency', 'data_scarce', 'distribution_shift', 'none'
        lambda_: penalty weight
        budget_params: param budget for MEMORY mode
        budget_flops: latency budget (ms) for LATENCY mode
        image_size: spatial size for dummy forward pass

    Returns:
        PressureFn instance, or None if mode_str == 'none'
    """
    if mode_str.lower() == 'none':
        return None
    mode = PressureMode.from_str(mode_str)
    return PressureFn(
        mode=mode,
        lambda_=lambda_,
        budget_params=budget_params,
        budget_flops=budget_flops,
        image_size=image_size,
    )
