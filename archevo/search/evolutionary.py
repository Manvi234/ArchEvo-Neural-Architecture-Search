"""
archevo/search/evolutionary.py
------------------------------
Evolutionary architecture search.

Uses a tournament-selection evolutionary algorithm with:
  - Random initialisation
  - Proxy training for fitness evaluation
  - Tournament selection, mutation, crossover, elitism
  - Lineage logging for lineage tree analysis
"""

import copy
import json
import logging
import random
import time
from typing import Optional, List, Dict, Any, Callable, Tuple

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from archevo.search_space import (
    Genotype,
    Network,
    build_from_genotype,
    random_genotype,
    NUM_INTERMEDIATE,
    _num_edges_for_node,
)
from archevo.primitives import OP_NAMES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fitness evaluation helper
# ---------------------------------------------------------------------------

def _proxy_train_and_eval(
    network: nn.Module,
    train_loader,
    val_loader,
    num_epochs: int,
    device: torch.device,
    lr: float = 0.025,
) -> float:
    """Train network for num_epochs on proxy data and return val accuracy."""
    network = network.to(device)
    network.train()

    optimizer = SGD(network.parameters(), lr=lr, momentum=0.9, weight_decay=3e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = network(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            optimizer.step()
        scheduler.step()

    # Evaluate
    network.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = network(x).argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)

    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# EvolutionarySearcher
# ---------------------------------------------------------------------------

class EvolutionarySearcher:
    """
    Evolutionary architecture search using tournament selection.

    Args:
        data_module: ArchEvoDataModule (already set up)
        pressure_fn: optional callable(network) -> scalar penalty
        pop_size: population size
        num_generations: number of evolution generations
        tournament_k: tournament size for parent selection
        elite_k: number of elite individuals carried over each generation
        proxy_epochs: epochs for proxy training during fitness evaluation
        C_init: initial channel count for network
        device: torch device string
    """

    def __init__(
        self,
        data_module,
        pressure_fn: Optional[Callable] = None,
        pop_size: int = 30,
        num_generations: int = 20,
        tournament_k: int = 5,
        elite_k: int = 3,
        proxy_epochs: int = 7,
        C_init: int = 16,
        device: str = 'cpu',
    ):
        self.data_module = data_module
        self.pressure_fn = pressure_fn
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.tournament_k = min(tournament_k, pop_size)
        self.elite_k = min(elite_k, pop_size)
        self.proxy_epochs = proxy_epochs
        self.C_init = C_init
        self.device = torch.device(device)

        self.num_classes = data_module.num_classes

        # Lineage log: list of dicts tracking evolutionary history
        self.lineage_log: List[Dict[str, Any]] = []
        self._next_id = 0

    # ------------------------------------------------------------------
    # Genotype utilities
    # ------------------------------------------------------------------

    def _new_id(self) -> int:
        uid = self._next_id
        self._next_id += 1
        return uid

    def random_genotype(self) -> Genotype:
        """Generate a random genotype."""
        return random_genotype()

    def mutate(self, genotype: Genotype) -> Genotype:
        """Randomly pick one edge and swap its op to a different random op."""
        new_geno = copy.deepcopy(genotype)
        # Pick random node and edge
        node_idx = random.randint(0, NUM_INTERMEDIATE - 1)
        num_edges = _num_edges_for_node(node_idx)
        edge_idx = random.randint(0, num_edges - 1)

        current_op = new_geno[node_idx][edge_idx]
        other_ops = [op for op in OP_NAMES if op != current_op]
        new_geno[node_idx][edge_idx] = random.choice(other_ops)
        return new_geno

    def crossover(self, g1: Genotype, g2: Genotype) -> Genotype:
        """
        Split at a random intermediate node boundary and combine.
        g1 supplies nodes [0, split_point), g2 supplies [split_point, NUM_INTERMEDIATE).
        """
        split = random.randint(1, NUM_INTERMEDIATE - 1)
        child = copy.deepcopy(g1[:split]) + copy.deepcopy(g2[split:])
        return child

    # ------------------------------------------------------------------
    # Fitness evaluation
    # ------------------------------------------------------------------

    def evaluate_fitness(
        self,
        genotype: Genotype,
        train_loader=None,
        val_loader=None,
    ) -> float:
        """
        Build network from genotype, proxy-train, and return:
            val_accuracy - pressure_penalty
        """
        if train_loader is None:
            train_loader = self.data_module.get_proxy_train_loader()
        if val_loader is None:
            val_loader = self.data_module.get_proxy_val_loader()

        try:
            network = build_from_genotype(
                genotype=[genotype],
                C_init=self.C_init,
                num_classes=self.num_classes,
            )
            acc = _proxy_train_and_eval(
                network, train_loader, val_loader,
                num_epochs=self.proxy_epochs,
                device=self.device,
            )

            penalty = 0.0
            if self.pressure_fn is not None:
                try:
                    pen = self.pressure_fn(network)
                    penalty = pen.item() if hasattr(pen, 'item') else float(pen)
                except Exception as e:
                    logger.warning(f"Pressure fn error: {e}")

            fitness = acc - penalty
            del network
            torch.cuda.empty_cache()
            return fitness

        except Exception as e:
            logger.warning(f"Fitness evaluation failed for genotype: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def tournament_select(
        self,
        population: List[Tuple[int, Genotype]],
        fitnesses: Dict[int, float],
    ) -> Tuple[int, Genotype]:
        """
        Sample tournament_k individuals uniformly, return the one with highest fitness.
        population: list of (id, genotype) tuples
        fitnesses: mapping from id to fitness score
        """
        contestants = random.sample(population, self.tournament_k)
        return max(contestants, key=lambda item: fitnesses[item[0]])

    # ------------------------------------------------------------------
    # Main search loop
    # ------------------------------------------------------------------

    def search(self) -> Genotype:
        """
        Main evolutionary loop.

        1. Initialise random population
        2. Evaluate all fitness scores
        3. For each generation:
            a. Carry over elite individuals
            b. Tournament select parents
            c. Crossover + mutate to produce offspring
            d. Evaluate offspring fitness
            e. Form next generation
        4. Return best genotype found
        """
        logger.info(
            f"Starting evolutionary search: pop_size={self.pop_size}, "
            f"generations={self.num_generations}, proxy_epochs={self.proxy_epochs}"
        )

        train_loader = self.data_module.get_proxy_train_loader()
        val_loader   = self.data_module.get_proxy_val_loader()

        # --- Initialise population ---
        population: List[Tuple[int, Genotype]] = []
        for _ in range(self.pop_size):
            uid = self._new_id()
            geno = self.random_genotype()
            population.append((uid, geno))

        # --- Evaluate initial population ---
        fitnesses: Dict[int, float] = {}
        logger.info("Evaluating initial population...")
        for uid, geno in tqdm(population, desc="Init eval"):
            t0 = time.time()
            fit = self.evaluate_fitness(geno, train_loader, val_loader)
            fitnesses[uid] = fit
            self.lineage_log.append({
                'generation': 0,
                'parent_ids': [],
                'child_id': uid,
                'genotype': geno,
                'fitness': fit,
                'elapsed_sec': time.time() - t0,
            })
            logger.debug(f"  id={uid} fitness={fit:.4f}")

        best_uid, best_geno = self._best(population, fitnesses)
        logger.info(f"Gen 0 best fitness: {fitnesses[best_uid]:.4f}")

        # --- Evolution generations ---
        for gen in range(1, self.num_generations + 1):
            logger.info(f"Generation {gen}/{self.num_generations}")

            # Sort current population by fitness (descending)
            sorted_pop = sorted(population, key=lambda item: fitnesses[item[0]], reverse=True)

            # Elite carry-over
            next_pop: List[Tuple[int, Genotype]] = sorted_pop[:self.elite_k]

            # Fill rest of next generation via selection + crossover + mutation
            while len(next_pop) < self.pop_size:
                parent1_id, parent1_geno = self.tournament_select(population, fitnesses)
                parent2_id, parent2_geno = self.tournament_select(population, fitnesses)

                # Crossover
                child_geno = self.crossover(parent1_geno, parent2_geno)

                # Mutate with probability 0.5
                if random.random() < 0.5:
                    child_geno = self.mutate(child_geno)

                child_uid = self._new_id()
                next_pop.append((child_uid, child_geno))

                # Evaluate child
                t0 = time.time()
                fit = self.evaluate_fitness(child_geno, train_loader, val_loader)
                fitnesses[child_uid] = fit
                self.lineage_log.append({
                    'generation': gen,
                    'parent_ids': [parent1_id, parent2_id],
                    'child_id': child_uid,
                    'genotype': child_geno,
                    'fitness': fit,
                    'elapsed_sec': time.time() - t0,
                })

            population = next_pop

            gen_best_id, gen_best_geno = self._best(population, fitnesses)
            gen_best_fit = fitnesses[gen_best_id]

            if gen_best_fit > fitnesses[best_uid]:
                best_uid = gen_best_id
                best_geno = gen_best_geno

            logger.info(
                f"  Gen {gen} best fitness: {gen_best_fit:.4f} | "
                f"all-time best: {fitnesses[best_uid]:.4f}"
            )

        logger.info(f"Search complete. Best fitness: {fitnesses[best_uid]:.4f}")
        return best_geno

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _best(
        self,
        population: List[Tuple[int, Genotype]],
        fitnesses: Dict[int, float],
    ) -> Tuple[int, Genotype]:
        return max(population, key=lambda item: fitnesses[item[0]])

    def get_lineage_log(self) -> List[Dict[str, Any]]:
        """Return lineage log (JSON-serialisable)."""
        return self.lineage_log

    def save_lineage_log(self, path: str):
        import json
        with open(path, 'w') as f:
            json.dump(self.lineage_log, f, indent=2)
        logger.info(f"Lineage log saved to {path}")
