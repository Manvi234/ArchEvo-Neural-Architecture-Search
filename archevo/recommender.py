"""
archevo/recommender.py
----------------------
Dataset profiling and architecture recommendation.

DatasetProfiler: computes statistical features of a dataset
ArchitectureRecommender: maps dataset stats -> recommended genotype + pressure mode
"""

import os
import json
import pickle
import sqlite3
import logging
import warnings
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DatasetProfiler
# ---------------------------------------------------------------------------

class DatasetProfiler:
    """
    Computes statistical features of a dataset for architecture recommendation.

    Features computed:
      - image_variance: mean pixel variance across the dataset
      - high_freq_energy: mean magnitude of high-frequency FFT components (texture/edge measure)
      - num_classes: number of unique classes in the loader
      - num_samples: total sample count
      - mean_aspect_ratio: mean H/W ratio of images
      - channel_mean: per-channel mean (3 values)
      - channel_std: per-channel std (3 values)
    """

    def __init__(self, max_batches: int = 50):
        """
        Args:
            max_batches: maximum batches to process (for speed on large datasets)
        """
        self.max_batches = max_batches

    def compute_stats(self, dataset_loader: DataLoader) -> Dict[str, Any]:
        """
        Compute dataset statistics from a DataLoader.

        Args:
            dataset_loader: a DataLoader yielding (images, labels) tuples
                            images: (B, C, H, W) float tensor

        Returns:
            dict of statistics
        """
        variances = []
        high_freq_energies = []
        aspect_ratios = []
        labels_seen = set()
        total_samples = 0

        ch_sums = None
        ch_sq_sums = None
        n_pixels = 0

        for batch_idx, batch in enumerate(dataset_loader):
            if batch_idx >= self.max_batches:
                break

            if isinstance(batch, (list, tuple)):
                images, labels = batch[0], batch[1]
            else:
                images = batch
                labels = torch.zeros(images.size(0), dtype=torch.long)

            if not isinstance(images, torch.Tensor):
                continue

            # Ensure float
            images = images.float()
            B, C, H, W = images.shape
            total_samples += B

            # Aspect ratio
            aspect_ratios.extend([H / W] * B)

            # Unique labels
            if hasattr(labels, 'tolist'):
                labels_seen.update(labels.tolist())

            # Per-image variance (mean across channels)
            img_flat = images.view(B, C, -1)  # (B, C, H*W)
            var_per_image = img_flat.var(dim=-1).mean(dim=-1)  # (B,)
            variances.extend(var_per_image.tolist())

            # Channel statistics
            if ch_sums is None:
                ch_sums = torch.zeros(C)
                ch_sq_sums = torch.zeros(C)

            img_spatial = images.view(B, C, -1)  # (B, C, H*W)
            ch_sums += img_spatial.sum(dim=[0, 2])
            ch_sq_sums += (img_spatial ** 2).sum(dim=[0, 2])
            n_pixels += B * H * W

            # High-frequency FFT energy
            hf_energy = self._compute_high_freq_energy(images)
            high_freq_energies.extend([hf_energy] * B)

        # Aggregate
        if n_pixels == 0:
            return self._empty_stats()

        ch_mean = (ch_sums / n_pixels).tolist()
        ch_var = (ch_sq_sums / n_pixels - (ch_sums / n_pixels) ** 2).clamp(min=0)
        ch_std = ch_var.sqrt().tolist()

        stats = {
            'image_variance': float(np.mean(variances)) if variances else 0.0,
            'high_freq_energy': float(np.mean(high_freq_energies)) if high_freq_energies else 0.0,
            'num_classes': len(labels_seen) if labels_seen else 0,
            'num_samples': total_samples,
            'mean_aspect_ratio': float(np.mean(aspect_ratios)) if aspect_ratios else 1.0,
            'channel_mean': ch_mean,
            'channel_std': ch_std,
        }
        return stats

    def _compute_high_freq_energy(self, images: torch.Tensor) -> float:
        """
        Compute mean magnitude of high-frequency FFT components.
        High-frequency = components with frequency > 50% of Nyquist.
        """
        B, C, H, W = images.shape
        # Use first channel for efficiency
        imgs_gray = images[:, 0, :, :]  # (B, H, W)

        try:
            fft = torch.fft.fft2(imgs_gray)
            magnitude = torch.abs(fft)

            # Create high-frequency mask (outer 50% of freq domain)
            freq_h = torch.fft.fftfreq(H)
            freq_w = torch.fft.fftfreq(W)
            fh, fw = torch.meshgrid(freq_h, freq_w, indexing='ij')
            radius = torch.sqrt(fh ** 2 + fw ** 2)
            high_freq_mask = (radius > 0.25).float()  # outer ~70% of freq radius

            hf_energy = (magnitude * high_freq_mask.unsqueeze(0)).mean().item()
        except Exception:
            hf_energy = 0.0

        return hf_energy

    def _empty_stats(self) -> Dict[str, Any]:
        return {
            'image_variance': 0.0,
            'high_freq_energy': 0.0,
            'num_classes': 0,
            'num_samples': 0,
            'mean_aspect_ratio': 1.0,
            'channel_mean': [0.5, 0.5, 0.5],
            'channel_std': [0.25, 0.25, 0.25],
        }

    def stats_to_feature_vector(self, stats: Dict[str, Any]) -> np.ndarray:
        """Convert stats dict to a flat numpy feature vector for model input."""
        return np.array([
            stats.get('image_variance', 0.0),
            stats.get('high_freq_energy', 0.0),
            float(stats.get('num_classes', 0)),
            float(stats.get('num_samples', 0)),
            stats.get('mean_aspect_ratio', 1.0),
        ] + list(stats.get('channel_mean', [0.5, 0.5, 0.5]))
          + list(stats.get('channel_std', [0.25, 0.25, 0.25])),
        dtype=float)


# ---------------------------------------------------------------------------
# ArchitectureRecommender
# ---------------------------------------------------------------------------

class ArchitectureRecommender:
    """
    Nearest-neighbor recommender that maps dataset statistics to
    a recommended genotype and pressure mode based on past experiments.

    Args:
        results_db_path: path to the SQLite experiments database
    """

    FEATURE_DIM = 11  # matches stats_to_feature_vector output

    def __init__(self, results_db_path: str):
        self.results_db_path = results_db_path
        self.profiler = DatasetProfiler()

        # Will be populated by fit()
        self._feature_matrix: Optional[np.ndarray] = None  # (N, FEATURE_DIM)
        self._genotypes: List[Any] = []
        self._pressure_modes: List[str] = []
        self._accuracies: List[float] = []
        self._dataset_names: List[str] = []
        self._is_fitted = False

        # Dataset-level feature cache (computed from stored stats or averages)
        self._dataset_features: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Database interaction
    # ------------------------------------------------------------------

    def _load_experiments(self) -> List[Dict[str, Any]]:
        """Load all experiments from SQLite DB."""
        if not os.path.exists(self.results_db_path):
            logger.warning(f"DB not found: {self.results_db_path}")
            return []

        conn = sqlite3.connect(self.results_db_path)
        try:
            cursor = conn.execute(
                """
                SELECT id, dataset, algorithm, pressure_mode, lambda_,
                       genotype_json, val_accuracy, param_count, training_time_sec, timestamp
                FROM experiments
                ORDER BY val_accuracy DESC
                """
            )
            rows = cursor.fetchall()
            cols = [desc[0] for desc in cursor.description]
            return [dict(zip(cols, row)) for row in rows]
        except sqlite3.OperationalError as e:
            logger.warning(f"DB query failed: {e}")
            return []
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Dataset feature proxies (when real loader not available)
    # ------------------------------------------------------------------

    # Hardcoded approximate feature vectors for known datasets
    # [image_variance, high_freq_energy, num_classes, num_samples, aspect_ratio,
    #  ch_mean_R, ch_mean_G, ch_mean_B, ch_std_R, ch_std_G, ch_std_B]
    KNOWN_DATASET_FEATURES = {
        'cifar10':  np.array([0.08, 0.15, 10, 50000, 1.0, 0.491, 0.482, 0.447, 0.247, 0.243, 0.262]),
        'eurosat':  np.array([0.05, 0.10, 10, 27000, 1.0, 0.344, 0.380, 0.408, 0.203, 0.137, 0.116]),
        'isic':     np.array([0.10, 0.20, 7,  10015, 1.0, 0.701, 0.552, 0.488, 0.141, 0.153, 0.170]),
        'cub200':   np.array([0.09, 0.22, 200, 11788, 1.33, 0.486, 0.499, 0.432, 0.227, 0.223, 0.261]),
    }

    def _get_dataset_feature(self, dataset_name: str) -> np.ndarray:
        """Return feature vector for a dataset (from cache or known defaults)."""
        if dataset_name in self._dataset_features:
            return self._dataset_features[dataset_name]
        if dataset_name in self.KNOWN_DATASET_FEATURES:
            return self.KNOWN_DATASET_FEATURES[dataset_name]
        # Unknown dataset: return zero vector
        return np.zeros(self.FEATURE_DIM)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self):
        """
        Train recommender on all existing experiment results.

        For each experiment entry, uses the known dataset feature vector
        (or a cached computed feature if available).
        Groups by dataset and takes the best-accuracy entry per dataset.
        """
        experiments = self._load_experiments()

        if not experiments:
            logger.warning("No experiments found in DB. Recommender cannot be fitted.")
            self._is_fitted = True
            return

        features = []
        genotypes = []
        pressure_modes = []
        accuracies = []
        dataset_names = []

        for exp in experiments:
            dataset = exp.get('dataset', '')
            genotype_json = exp.get('genotype_json', '[]')
            val_accuracy = exp.get('val_accuracy', 0.0) or 0.0
            pressure_mode = exp.get('pressure_mode', 'none') or 'none'

            try:
                genotype = json.loads(genotype_json) if genotype_json else []
            except (json.JSONDecodeError, TypeError):
                genotype = []

            feat = self._get_dataset_feature(dataset)
            features.append(feat)
            genotypes.append(genotype)
            pressure_modes.append(pressure_mode)
            accuracies.append(val_accuracy)
            dataset_names.append(dataset)

        self._feature_matrix = np.stack(features) if features else np.zeros((0, self.FEATURE_DIM))
        self._genotypes = genotypes
        self._pressure_modes = pressure_modes
        self._accuracies = accuracies
        self._dataset_names = dataset_names
        self._is_fitted = True

        logger.info(f"Recommender fitted on {len(experiments)} experiments.")

    # ------------------------------------------------------------------
    # Recommend
    # ------------------------------------------------------------------

    def recommend(
        self,
        dataset_loader: Optional[DataLoader] = None,
        dataset_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute dataset stats, find closest match in DB, return recommendation.

        Args:
            dataset_loader: DataLoader to profile (optional; use dataset_name as fallback)
            dataset_name: known dataset name for feature lookup fallback

        Returns:
            dict with keys:
              - 'genotype': recommended Genotype
              - 'pressure_mode': recommended pressure mode string
              - 'matched_dataset': name of matched dataset in DB
              - 'expected_accuracy': expected val accuracy of matched experiment
              - 'stats': computed dataset stats (if loader provided)
        """
        if not self._is_fitted:
            self.fit()

        if self._feature_matrix is None or len(self._feature_matrix) == 0:
            logger.warning("No experiments to match against. Returning random genotype.")
            from archevo.search_space import random_genotype
            return {
                'genotype': random_genotype(),
                'pressure_mode': 'none',
                'matched_dataset': None,
                'expected_accuracy': 0.0,
                'stats': {},
            }

        # Compute query feature
        stats = {}
        if dataset_loader is not None:
            stats = self.profiler.compute_stats(dataset_loader)
            query_feat = self.profiler.stats_to_feature_vector(stats)
        elif dataset_name is not None:
            query_feat = self._get_dataset_feature(dataset_name)
        else:
            raise ValueError("Provide either dataset_loader or dataset_name")

        # Cache if dataset_name provided
        if dataset_name and len(stats) > 0:
            self._dataset_features[dataset_name] = query_feat

        # L2 nearest-neighbor search
        diffs = self._feature_matrix - query_feat[np.newaxis, :]
        distances = np.linalg.norm(diffs, axis=1)

        # Weight by accuracy: among close matches, prefer high accuracy
        # combined_score = distance / (accuracy + epsilon)
        eps = 1e-6
        combined_scores = distances / (np.array(self._accuracies) + eps)
        best_idx = int(np.argmin(combined_scores))

        best_genotype = self._genotypes[best_idx]
        best_pressure = self._pressure_modes[best_idx]
        best_accuracy = self._accuracies[best_idx]
        best_dataset = self._dataset_names[best_idx]

        return {
            'genotype': best_genotype,
            'pressure_mode': best_pressure,
            'matched_dataset': best_dataset,
            'expected_accuracy': best_accuracy,
            'stats': stats,
            'nearest_distance': float(distances[best_idx]),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save recommender state to a pickle file."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        state = {
            'feature_matrix': self._feature_matrix,
            'genotypes': self._genotypes,
            'pressure_modes': self._pressure_modes,
            'accuracies': self._accuracies,
            'dataset_names': self._dataset_names,
            'dataset_features': self._dataset_features,
            'is_fitted': self._is_fitted,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Recommender saved to {path}")

    def load(self, path: str):
        """Load recommender state from a pickle file."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self._feature_matrix = state.get('feature_matrix')
        self._genotypes = state.get('genotypes', [])
        self._pressure_modes = state.get('pressure_modes', [])
        self._accuracies = state.get('accuracies', [])
        self._dataset_names = state.get('dataset_names', [])
        self._dataset_features = state.get('dataset_features', {})
        self._is_fitted = state.get('is_fitted', False)
        logger.info(f"Recommender loaded from {path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        if not self._is_fitted:
            return "ArchitectureRecommender (not fitted)"
        n = len(self._genotypes)
        datasets = set(self._dataset_names)
        return (
            f"ArchitectureRecommender: {n} experiments, "
            f"{len(datasets)} datasets: {sorted(datasets)}"
        )
