# ArchEvo — Dataset-Aware Neural Architecture Search

ArchEvo is a research framework for discovering efficient neural architectures under real-world deployment constraints. It supports two search algorithms (DARTS and evolutionary), four benchmark datasets, and four pressure modes that steer the search toward architectures that fit memory, latency, or data budgets.

A Streamlit dashboard ties everything together: run searches, visualise evolution, inspect genotypes, and get architecture recommendations — all from a browser.

---

## Features

| Feature | Details |
|---|---|
| **Search algorithms** | DARTS (gradient-based), Evolutionary (tournament selection) |
| **Datasets** | CIFAR-10, EuroSAT, ISIC, CUB-200 |
| **Pressure modes** | `memory`, `latency`, `data_scarce`, `distribution_shift` |
| **Cell operations** | `conv3`, `conv5`, `attention`, `mlp_mixer`, `skip`, `zero` |
| **Search space** | Cell-DAG — 4 intermediate nodes, 14 edges per cell, 10 stacked cells |
| **Logging** | SQLite experiment database, JSON genotype & lineage logs |
| **UI** | 5-page Streamlit dashboard |
| **Recommender** | Dataset profiler + trained surrogate model for architecture recommendation |

---

## Project Structure

```
Archevo/
├── app.py                   # Streamlit dashboard (5 pages)
├── run_search.py            # CLI entry point for architecture search
├── run_eval.py              # Evaluate a saved genotype
├── run_analysis.py          # Generate analysis plots from CLI
├── run_recommender.py       # CLI for the architecture recommender
├── requirements.txt
├── configs/
│   ├── base.yaml            # Default search/training config
│   ├── datasets/            # Per-dataset overrides (cifar10, eurosat, isic, cub200)
│   ├── algorithms/          # Per-algorithm overrides (darts, evolutionary)
│   └── pressure/            # Per-pressure-mode config (memory, latency, …)
├── archevo/
│   ├── search_space.py      # Cell-DAG definition, Network, genotype helpers
│   ├── primitives.py        # Operation registry (conv, attention, mlp_mixer, …)
│   ├── pressure.py          # PressureFn, PressureMode, count_params, estimate_flops
│   ├── recommender.py       # Surrogate recommender model
│   ├── search/
│   │   ├── darts.py         # DARTSSearcher
│   │   └── evolutionary.py  # EvolutionarySearcher
│   └── data/
│       └── datamodule.py    # ArchEvoDataModule (proxy splits, augmentation)
├── analysis/
│   ├── convergence.py       # Convergence curve plots
│   ├── heatmaps.py          # Operation preference heatmaps
│   └── lineage_tree.py      # Evolutionary lineage DAG
├── data/                    # Dataset files (download separately)
└── results/                 # Search outputs, SQLite DB, saved genotypes
```

---

## Installation

```bash
# 1. Clone the repo
git clone <repo-url> && cd Archevo

# 2. Create a virtual environment (Python 3.10+)
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) install NetworkX for lineage-tree and genotype DAG views
pip install networkx
```

### Dataset setup

Place datasets under `data/<dataset_name>/`. CIFAR-10 is downloaded automatically by torchvision on first use. For EuroSAT, ISIC, and CUB-200, organise each as:

```
data/<dataset_name>/
    train/
        <class_name>/
            img1.jpg …
    val/
        <class_name>/
            …
```

---

## Quick Start

### Launch the Streamlit UI

```bash
streamlit run app.py
```

Open `http://localhost:8501` in a browser. The sidebar lets you navigate between the five pages described below.

If you have no experiment data yet, go to **Analysis → Seed demo data** to populate 80 synthetic runs and preview all charts immediately.

### Run a search from the CLI

```bash
# DARTS search on CIFAR-10 with memory pressure
python run_search.py \
    --dataset cifar10 \
    --algorithm darts \
    --pressure memory \
    --lambda 0.1 \
    --search_epochs 50 \
    --device cuda \
    --output_dir results/

# Evolutionary search, no pressure, CPU
python run_search.py \
    --dataset eurosat \
    --algorithm evolutionary \
    --pressure none \
    --device cpu
```

Results are written to `results/<dataset>_<algorithm>_<pressure>/` and logged to `results/experiments.db`.

### Evaluate a saved genotype

```bash
python run_eval.py --genotype results/cifar10_darts_memory/genotype.json --dataset cifar10
```

### Get an architecture recommendation

```bash
python run_recommender.py --dataset_name cifar10 --results_db results/experiments.db
```

---

## CLI Reference — `run_search.py`

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `cifar10` | `cifar10`, `eurosat`, `isic`, `cub200` |
| `--algorithm` | `darts` | `darts`, `evolutionary` |
| `--pressure` | `none` | `none`, `memory`, `latency`, `data_scarce`, `distribution_shift` |
| `--lambda` | `0.1` | Penalty weight λ — higher = stronger pressure |
| `--search_epochs` | config default (50) | Number of search epochs |
| `--proxy_epochs` | config default (7) | Proxy training epochs (evolutionary only) |
| `--device` | `cpu` | `cpu`, `cuda`, `mps` |
| `--batch_size` | `64` | DataLoader batch size |
| `--data_root` | `data/<dataset>` | Override dataset directory |
| `--db_path` | `results/experiments.db` | SQLite database path |
| `--seed` | `42` | Random seed |

---

## Pressure Modes

| Mode | Effect |
|---|---|
| `memory` | Penalises parameter count relative to a 1 M-param budget |
| `latency` | Penalises wall-clock forward-pass time relative to a 100 ms budget |
| `data_scarce` | Trains on a small proxy subset; no explicit loss penalty |
| `distribution_shift` | Evaluated on a shifted validation split; no explicit loss penalty |
| `none` | Unconstrained search |

λ (lambda) controls how strongly the penalty is weighted against validation accuracy. λ = 0 is equivalent to `none`.

---

## Streamlit Dashboard Pages

| Page | What it does |
|---|---|
| **Dashboard** | KPI summary (total runs, best accuracy), filterable experiment table, bar charts and pressure-response curves from the SQLite DB |
| **Run Search** | Configure and launch a search as a subprocess; streams live log output; browse and download saved genotypes |
| **Analysis** | Op preference heatmaps, convergence curves (DARTS vs evolutionary), evolutionary lineage tree, pressure-response curves — all downloadable as PNG |
| **Recommender** | Profile a known or custom dataset (visual statistics + FFT frequency ratio) and query the surrogate recommender |
| **Genotype Viewer** | Inspect any saved genotype: raw JSON, edge→operation table, op frequency bar chart, cell DAG visualisation, and estimated parameter count + latency |

---

## Search Space

Each cell is a directed acyclic graph (DAG) with:
- 2 fixed input nodes (outputs of the previous two cells)
- 4 intermediate nodes; node *i* receives edges from all *i* + 2 predecessors
- 14 edges total per cell

The 6 candidate operations per edge are:

| Operation | Description |
|---|---|
| `conv3` | 3×3 separable convolution |
| `conv5` | 5×5 separable convolution |
| `attention` | Multi-head self-attention |
| `mlp_mixer` | MLP-Mixer token/channel mixing |
| `skip` | Identity connection |
| `zero` | No connection |

A **genotype** is a list of lists — one inner list per intermediate node, each containing the selected operation name for every incoming edge.

---

## Outputs

Each search run produces a directory `results/<dataset>_<algorithm>_<pressure>/` containing:

```
config.json         # Full run configuration
genotype.json       # Best discovered architecture
search_logs.json    # Per-epoch val accuracy log (DARTS)
lineage_log.json    # Mutation history (evolutionary)
```

All runs are also appended to the SQLite database at `results/experiments.db`.

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- See [requirements.txt](requirements.txt) for the full list

---

## Authors

Built by **Manvi Gawande** and **Sanskar Srivastava**.
