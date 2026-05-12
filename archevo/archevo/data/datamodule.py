"""
archevo/data/datamodule.py
--------------------------
ArchEvoDataModule: unified data loading for CIFAR-10, EuroSAT, ISIC, CUB-200.
"""

import os
import warnings
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
import torchvision.transforms as T
from torchvision import datasets


# ---------------------------------------------------------------------------
# Per-dataset statistics
# ---------------------------------------------------------------------------

DATASET_STATS = {
    'cifar10': {
        'mean': [0.4914, 0.4822, 0.4465],
        'std':  [0.2470, 0.2435, 0.2616],
        'num_classes': 10,
        'image_size': 32,
    },
    'eurosat': {
        'mean': [0.3444, 0.3803, 0.4078],
        'std':  [0.2026, 0.1365, 0.1155],
        'num_classes': 10,
        'image_size': 64,
    },
    'isic': {
        'mean': [0.7012, 0.5517, 0.4875],
        'std':  [0.1411, 0.1526, 0.1699],
        'num_classes': 7,
        'image_size': 224,
    },
    'cub200': {
        'mean': [0.4856, 0.4994, 0.4324],
        'std':  [0.2272, 0.2226, 0.2613],
        'num_classes': 200,
        'image_size': 224,
    },
}

# Proxy split sizes
PROXY_TRAIN_SIZE = 10_000
PROXY_VAL_SIZE = 2_000


# ---------------------------------------------------------------------------
# Augmentation factories
# ---------------------------------------------------------------------------

def _make_train_transform(dataset_name: str):
    stats = DATASET_STATS[dataset_name]
    mean, std = stats['mean'], stats['std']
    sz = stats['image_size']

    if dataset_name == 'cifar10':
        return T.Compose([
            T.RandomCrop(sz, padding=4),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    elif dataset_name == 'eurosat':
        return T.Compose([
            T.Resize(sz + 8),
            T.RandomCrop(sz),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    elif dataset_name == 'isic':
        return T.Compose([
            T.Resize(sz + 32),
            T.RandomResizedCrop(sz, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.RandomRotation(30),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    elif dataset_name == 'cub200':
        return T.Compose([
            T.Resize(sz + 32),
            T.RandomResizedCrop(sz, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _make_val_transform(dataset_name: str):
    stats = DATASET_STATS[dataset_name]
    mean, std = stats['mean'], stats['std']
    sz = stats['image_size']

    if dataset_name == 'cifar10':
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    elif dataset_name in ('eurosat', 'isic', 'cub200'):
        return T.Compose([
            T.Resize(int(sz * 1.14)),
            T.CenterCrop(sz),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ---------------------------------------------------------------------------
# Dataset loaders per source
# ---------------------------------------------------------------------------

def _load_cifar10(data_root: str, train: bool, transform):
    return datasets.CIFAR10(
        root=data_root,
        train=train,
        download=True,
        transform=transform,
    )


def _load_imagefolder(data_root: str, split: str, transform):
    """
    Loads from data_root/{split}/ using ImageFolder convention.
    Prints a helpful message if the directory is missing.
    """
    split_dir = os.path.join(data_root, split)
    if not os.path.isdir(split_dir):
        warnings.warn(
            f"Dataset directory not found: {split_dir}\n"
            f"Please download the dataset and organise it as:\n"
            f"  {data_root}/\n"
            f"    train/<class_name>/<image>.jpg\n"
            f"    val/<class_name>/<image>.jpg\n"
            f"A dummy placeholder dataset is returned."
        )
        # Return a stub dataset so the code doesn't crash in CI / import time
        return _DummyDataset(length=100)
    return datasets.ImageFolder(root=split_dir, transform=transform)


class _DummyDataset(torch.utils.data.Dataset):
    """Placeholder when real data is unavailable."""

    def __init__(self, length: int = 100, num_classes: int = 10, image_size: int = 32):
        self.length = length
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = torch.randn(3, self.image_size, self.image_size)
        label = idx % self.num_classes
        return img, label


# ---------------------------------------------------------------------------
# ArchEvoDataModule
# ---------------------------------------------------------------------------

class ArchEvoDataModule:
    """
    Unified data module for ArchEvo experiments.

    Args:
        dataset_name: one of 'cifar10', 'eurosat', 'isic', 'cub200'
        data_root: root directory for datasets (default: 'data/<dataset_name>')
        batch_size: default batch size for train/val loaders
        num_workers: dataloader workers
        proxy_train_size: number of samples in proxy train set
        proxy_val_size: number of samples in proxy val set
        seed: random seed for subset splitting
    """

    def __init__(
        self,
        dataset_name: str,
        data_root: Optional[str] = None,
        batch_size: int = 64,
        num_workers: int = 4,
        proxy_train_size: int = PROXY_TRAIN_SIZE,
        proxy_val_size: int = PROXY_VAL_SIZE,
        seed: int = 42,
    ):
        if dataset_name not in DATASET_STATS:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Choose from: {list(DATASET_STATS.keys())}"
            )
        self.dataset_name = dataset_name
        self.data_root = data_root or os.path.join('data', dataset_name)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.proxy_train_size = proxy_train_size
        self.proxy_val_size = proxy_val_size
        self.seed = seed
        self.stats = DATASET_STATS[dataset_name]

        self._train_dataset = None
        self._val_dataset = None
        self._proxy_train_dataset = None
        self._proxy_val_dataset = None

    # ------------------------------------------------------------------
    # Setup (call before accessing loaders)
    # ------------------------------------------------------------------

    def setup(self):
        """Load full train/val datasets and create proxy subsets."""
        train_transform = _make_train_transform(self.dataset_name)
        val_transform = _make_val_transform(self.dataset_name)

        if self.dataset_name == 'cifar10':
            self._train_dataset = _load_cifar10(self.data_root, train=True,  transform=train_transform)
            self._val_dataset   = _load_cifar10(self.data_root, train=False, transform=val_transform)
        elif self.dataset_name == 'eurosat':
            # EuroSAT: downloaded manually or via HuggingFace; expected in data/eurosat/{train,val}/
            self._train_dataset = _load_imagefolder(self.data_root, 'train', train_transform)
            self._val_dataset   = _load_imagefolder(self.data_root, 'val',   val_transform)
        elif self.dataset_name == 'isic':
            self._train_dataset = _load_imagefolder(self.data_root, 'train', train_transform)
            self._val_dataset   = _load_imagefolder(self.data_root, 'val',   val_transform)
        elif self.dataset_name == 'cub200':
            self._train_dataset = _load_imagefolder(self.data_root, 'train', train_transform)
            self._val_dataset   = _load_imagefolder(self.data_root, 'val',   val_transform)

        # Create proxy subsets (for architecture search phase)
        self._proxy_train_dataset, self._proxy_val_dataset = self._make_proxy_split(
            self._train_dataset
        )

    # ------------------------------------------------------------------
    # Proxy split
    # ------------------------------------------------------------------

    def _make_proxy_split(
        self,
        dataset,
    ) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """
        Creates proxy_train (10k) and proxy_val (2k) subsets from the training set.
        Falls back to smaller sizes if the dataset is smaller.
        """
        n = len(dataset)
        train_sz = min(self.proxy_train_size, n - self.proxy_val_size)
        val_sz = min(self.proxy_val_size, n - train_sz)
        remainder = n - train_sz - val_sz

        generator = torch.Generator().manual_seed(self.seed)
        proxy_train, proxy_val, _ = random_split(
            dataset,
            [train_sz, val_sz, remainder],
            generator=generator,
        )
        return proxy_train, proxy_val

    # ------------------------------------------------------------------
    # DataLoader factories
    # ------------------------------------------------------------------

    def _ensure_setup(self):
        if self._train_dataset is None:
            self.setup()

    def get_train_loader(self, batch_size: Optional[int] = None, shuffle: bool = True) -> DataLoader:
        self._ensure_setup()
        return DataLoader(
            self._train_dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def get_val_loader(self, batch_size: Optional[int] = None) -> DataLoader:
        self._ensure_setup()
        return DataLoader(
            self._val_dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_proxy_train_loader(self, batch_size: Optional[int] = None, shuffle: bool = True) -> DataLoader:
        self._ensure_setup()
        return DataLoader(
            self._proxy_train_dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def get_proxy_val_loader(self, batch_size: Optional[int] = None) -> DataLoader:
        self._ensure_setup()
        return DataLoader(
            self._proxy_val_dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_classes(self) -> int:
        return self.stats['num_classes']

    @property
    def image_size(self) -> int:
        return self.stats['image_size']

    @property
    def mean(self):
        return self.stats['mean']

    @property
    def std(self):
        return self.stats['std']

    def __repr__(self) -> str:
        return (
            f"ArchEvoDataModule(dataset={self.dataset_name}, "
            f"num_classes={self.num_classes}, "
            f"batch_size={self.batch_size})"
        )


# ---------------------------------------------------------------------------
# EuroSAT download helper (HuggingFace)
# ---------------------------------------------------------------------------

def download_eurosat_from_huggingface(target_dir: str):
    """
    Download EuroSAT from HuggingFace datasets and organise into ImageFolder structure.
    Requires: pip install datasets Pillow
    """
    try:
        from datasets import load_dataset
        from PIL import Image
    except ImportError:
        raise RuntimeError(
            "Install 'datasets' and 'Pillow' to download EuroSAT: "
            "pip install datasets Pillow"
        )

    print("Downloading EuroSAT from HuggingFace...")
    ds = load_dataset("timm/eurosat-rgb", trust_remote_code=True)

    label_names = ds['train'].features['label'].names

    for split_name, hf_split in [('train', 'train'), ('val', 'validation')]:
        split_data = ds[hf_split]
        for idx, sample in enumerate(split_data):
            img: Image.Image = sample['image']
            label_idx: int = sample['label']
            label_str = label_names[label_idx]
            out_dir = os.path.join(target_dir, split_name, label_str)
            os.makedirs(out_dir, exist_ok=True)
            img_path = os.path.join(out_dir, f"{idx:06d}.jpg")
            if not os.path.exists(img_path):
                img.save(img_path)
        print(f"  Saved {split_name} split.")
    print(f"EuroSAT saved to {target_dir}")
