"""
archevo/search/darts.py
-----------------------
DARTS (Differentiable Architecture Search) searcher.

Implements bilevel optimisation:
  - Inner loop: update network weights on train data (SGD)
  - Outer loop: update architecture weights (alpha) on val data (Adam)
  - Supports first-order and second-order (Hessian-vector product via finite diff)
"""

import copy
import time
import logging
from typing import Optional, List, Dict, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from archevo.search_space import Network, arch_parameters, extract_genotype, Genotype

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        return (pred == targets).float().mean().item()


def _clip_grad_norm(params, max_norm: float = 5.0):
    nn.utils.clip_grad_norm_(params, max_norm)


# ---------------------------------------------------------------------------
# DARTSSearcher
# ---------------------------------------------------------------------------

class DARTSSearcher:
    """
    Bilevel optimisation for differentiable architecture search.

    Args:
        network: Network with use_mixed_ops=True
        data_module: ArchEvoDataModule (already set up)
        pressure_fn: optional callable(network) -> scalar penalty added to arch loss
        order: 'first' or 'second' (Hessian-vector product approximation)
        lr_w: learning rate for network weights (SGD)
        lr_alpha: learning rate for arch weights (Adam)
        momentum: SGD momentum
        weight_decay: L2 regularisation on weights
        grad_clip: gradient clipping norm for weights
        epsilon: finite-difference epsilon for second-order approximation
        device: torch device string
    """

    def __init__(
        self,
        network: Network,
        data_module,
        pressure_fn: Optional[Callable] = None,
        order: str = 'first',
        lr_w: float = 0.025,
        lr_alpha: float = 3e-4,
        momentum: float = 0.9,
        weight_decay: float = 3e-4,
        grad_clip: float = 5.0,
        epsilon: float = 0.01,
        device: str = 'cpu',
    ):
        self.network = network.to(device)
        self.data_module = data_module
        self.pressure_fn = pressure_fn
        self.order = order
        self.lr_w = lr_w
        self.lr_alpha = lr_alpha
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.epsilon = epsilon
        self.device = torch.device(device)

        # Separate weight and arch params
        self._arch_params = arch_parameters(self.network)
        self._weight_params = self.network.weight_parameters()

        # Optimisers
        self.w_optimizer = SGD(
            self._weight_params,
            lr=lr_w,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        self.alpha_optimizer = Adam(
            self._arch_params,
            lr=lr_alpha,
            betas=(0.5, 0.999),
            weight_decay=1e-3,
        )

        self.criterion = nn.CrossEntropyLoss()

        # Logging
        self.logs: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Main search loop
    # ------------------------------------------------------------------

    def search(self, num_epochs: int = 50) -> Genotype:
        """
        Run DARTS bilevel optimisation for num_epochs.
        Returns the best genotype found.
        """
        logger.info(f"Starting DARTS search ({self.order}-order) for {num_epochs} epochs")

        train_loader = self.data_module.get_proxy_train_loader()
        val_loader   = self.data_module.get_proxy_val_loader()

        scheduler = CosineAnnealingLR(self.w_optimizer, T_max=num_epochs, eta_min=1e-3)

        best_val_acc = 0.0
        best_genotype = None

        for epoch in range(num_epochs):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader, val_loader)
            val_loss, val_acc = self._evaluate(val_loader)
            entropy = self.network.alpha_entropy().item()
            scheduler.step()
            elapsed = time.time() - t0

            log_entry = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'alpha_entropy': entropy,
                'elapsed_sec': elapsed,
            }
            self.logs.append(log_entry)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_genotype = extract_genotype(self.network)

            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f} | entropy={entropy:.4f} | "
                f"time={elapsed:.1f}s"
            )

        if best_genotype is None:
            best_genotype = extract_genotype(self.network)

        return best_genotype

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, train_loader, val_loader) -> float:
        self.network.train()
        total_loss = 0.0
        n_batches = 0

        val_iter = iter(val_loader)

        for batch_idx, (x_train, y_train) in enumerate(train_loader):
            x_train = x_train.to(self.device)
            y_train = y_train.to(self.device)

            # Get val batch for arch update
            try:
                x_val, y_val = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                x_val, y_val = next(val_iter)

            x_val = x_val.to(self.device)
            y_val = y_val.to(self.device)

            # ---- Arch weight update (outer) ----
            self.alpha_optimizer.zero_grad()

            if self.order == 'second':
                self._second_order_arch_step(x_train, y_train, x_val, y_val)
            else:
                # First-order: simple gradient step on val loss
                logits_val = self.network(x_val)
                loss_val = self.criterion(logits_val, y_val)
                if self.pressure_fn is not None:
                    loss_val = loss_val + self.pressure_fn(self.network)
                loss_val.backward()

            self.alpha_optimizer.step()

            # ---- Weight update (inner) ----
            self.w_optimizer.zero_grad()
            logits_train = self.network(x_train)
            loss_train = self.criterion(logits_train, y_train)
            loss_train.backward()
            _clip_grad_norm(self._weight_params, self.grad_clip)
            self.w_optimizer.step()

            total_loss += loss_train.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Second-order arch step (Hessian-vector product via finite difference)
    # ------------------------------------------------------------------

    def _second_order_arch_step(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
    ):
        """
        Finite difference approximation of the implicit gradient.

        dL_val/d_alpha ≈ dL_val/d_alpha (direct)
                        - xi * (dL_train_plus/d_alpha - dL_train_minus/d_alpha) / (2*eps)

        where w_plus  = w + eps * dL_val/dw
              w_minus = w - eps * dL_val/dw
        """
        # Step 1: compute dL_val/dw
        logits_val = self.network(x_val)
        loss_val = self.criterion(logits_val, y_val)
        if self.pressure_fn is not None:
            loss_val = loss_val + self.pressure_fn(self.network)

        # grad wrt val loss for both weights and alphas
        grads_val_alpha = torch.autograd.grad(
            loss_val,
            self._arch_params,
            allow_unused=True,
            retain_graph=True,
        )

        grads_val_w = torch.autograd.grad(
            loss_val,
            self._weight_params,
            allow_unused=True,
        )

        # Step 2: perturb weights +/- eps * grad_w
        eps = self.epsilon
        # Save original weights
        orig_weights = [p.data.clone() for p in self._weight_params]

        # Compute perturbation norm for normalisation
        grad_norm = 0.0
        for g in grads_val_w:
            if g is not None:
                grad_norm += g.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        if grad_norm < 1e-8:
            # Fallback to first-order
            for g, p in zip(grads_val_alpha, self._arch_params):
                if p.grad is None:
                    p.grad = g.clone() if g is not None else torch.zeros_like(p)
                else:
                    if g is not None:
                        p.grad += g
            return

        # w+ = w + eps/||g|| * g
        for p, g in zip(self._weight_params, grads_val_w):
            if g is not None:
                p.data.add_(g, alpha=eps / grad_norm)

        logits_train_plus = self.network(x_train)
        loss_plus = self.criterion(logits_train_plus, y_train)
        grads_plus = torch.autograd.grad(loss_plus, self._arch_params, allow_unused=True)

        # w- = w - eps/||g|| * g (restore first, then subtract)
        for p, orig, g in zip(self._weight_params, orig_weights, grads_val_w):
            p.data.copy_(orig)
            if g is not None:
                p.data.sub_(g, alpha=eps / grad_norm)

        logits_train_minus = self.network(x_train)
        loss_minus = self.criterion(logits_train_minus, y_train)
        grads_minus = torch.autograd.grad(loss_minus, self._arch_params, allow_unused=True)

        # Restore original weights
        for p, orig in zip(self._weight_params, orig_weights):
            p.data.copy_(orig)

        # Step 3: implicit gradient = dL_val/d_alpha - lr_w * (g+ - g-) / (2*eps/||g||)
        # = dL_val/d_alpha - lr_w * ||g|| / (2*eps) * (g+ - g-)
        scale = self.lr_w * grad_norm / (2.0 * eps)

        for p, g_alpha, g_plus, g_minus in zip(
            self._arch_params, grads_val_alpha, grads_plus, grads_minus
        ):
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            if g_alpha is not None:
                p.grad.add_(g_alpha)
            if g_plus is not None and g_minus is not None:
                p.grad.sub_((g_plus - g_minus) * scale)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self, val_loader) -> tuple:
        self.network.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x, y in val_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.network(x)
            loss = self.criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_samples += x.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)
        return avg_loss, accuracy

    # ------------------------------------------------------------------
    # Genotype extraction
    # ------------------------------------------------------------------

    def get_genotype(self) -> Genotype:
        return extract_genotype(self.network)

    def get_logs(self) -> List[Dict[str, Any]]:
        return self.logs
