"""
app.py — ArchEvo Streamlit UI
------------------------------
Multi-page dashboard for the ArchEvo neural architecture search framework.

Pages:
  1. Dashboard   — experiment overview from SQLite DB
  2. Run Search  — configure & launch a search (subprocess)
  3. Analysis    — lineage trees, op heatmaps, convergence plots
  4. Recommender — dataset profiling + architecture recommendation
  5. Genotype    — inspect a saved genotype visually
"""

import io
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "results" / "experiments.db"
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"

OP_NAMES = ["conv3", "conv5", "attention", "mlp_mixer", "skip", "zero"]
DATASETS = ["cifar10", "eurosat", "isic", "cub200"]
ALGORITHMS = ["darts", "evolutionary"]
PRESSURES = ["none", "memory", "latency", "data_scarce", "distribution_shift"]


# ── helpers ─────────────────────────────────────────────────────────────────

def load_db() -> Optional[pd.DataFrame]:
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM experiments ORDER BY id DESC", conn)
    except Exception:
        df = None
    conn.close()
    return df


def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return buf.read()


def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(ROOT)
    )
    return result.returncode, result.stdout, result.stderr


# ── page: Dashboard ─────────────────────────────────────────────────────────

def page_dashboard():
    st.header("Experiment Dashboard")

    df = load_db()
    if df is None or df.empty:
        st.info("No experiments found. Run a search from the **Run Search** page.")
        return

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total runs", len(df))
    c2.metric("Best accuracy", f"{df['val_accuracy'].max():.3f}" if "val_accuracy" in df else "—")
    c3.metric("Datasets", df["dataset"].nunique() if "dataset" in df else "—")
    c4.metric("Algorithms", df["algorithm"].nunique() if "algorithm" in df else "—")

    st.divider()

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    ds_filter = col_f1.multiselect("Dataset", DATASETS, default=list(df["dataset"].unique()) if "dataset" in df else [])
    algo_filter = col_f2.multiselect("Algorithm", ALGORITHMS, default=list(df["algorithm"].unique()) if "algorithm" in df else [])
    pressure_filter = col_f3.multiselect("Pressure", PRESSURES, default=list(df["pressure_mode"].unique()) if "pressure_mode" in df else [])

    mask = pd.Series(True, index=df.index)
    if ds_filter:
        mask &= df["dataset"].isin(ds_filter)
    if algo_filter:
        mask &= df["algorithm"].isin(algo_filter)
    if pressure_filter:
        mask &= df["pressure_mode"].isin(pressure_filter)

    filtered = df[mask].copy()

    # Table
    display_cols = [c for c in ["id", "dataset", "algorithm", "pressure_mode", "lambda_",
                                 "val_accuracy", "param_count", "training_time_sec", "timestamp"]
                    if c in filtered.columns]
    st.dataframe(filtered[display_cols], use_container_width=True, height=320)

    if filtered.empty:
        return

    st.divider()

    # Charts
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Val Accuracy by Dataset & Algorithm")
        pivot = filtered.groupby(["dataset", "algorithm"])["val_accuracy"].mean().reset_index()
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(5, 3))
            for algo in pivot["algorithm"].unique():
                sub = pivot[pivot["algorithm"] == algo]
                ax.bar([f"{d}\n({algo})" for d in sub["dataset"]], sub["val_accuracy"], label=algo, alpha=0.8)
            ax.set_ylabel("Mean Val Accuracy")
            ax.set_ylim(0, 1)
            ax.legend()
            ax.set_title("Accuracy by Dataset & Algorithm")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with col_b:
        st.subheader("Param Count Distribution")
        if "param_count" in filtered.columns and filtered["param_count"].notna().any():
            fig, ax = plt.subplots(figsize=(5, 3))
            filtered["param_count"].dropna().astype(float).hist(ax=ax, bins=15, color="#4c72b0", edgecolor="white")
            ax.set_xlabel("Parameter Count")
            ax.set_ylabel("Frequency")
            ax.set_title("Param Count Distribution")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No parameter count data available yet.")

    # Pressure response
    if "lambda_" in filtered.columns and "pressure_mode" in filtered.columns:
        st.subheader("Pressure Response — Best Accuracy vs Lambda")
        pr_df = filtered[filtered["pressure_mode"] != "none"].copy()
        if not pr_df.empty:
            fig, ax = plt.subplots(figsize=(8, 3))
            for pm in pr_df["pressure_mode"].unique():
                sub = pr_df[pr_df["pressure_mode"] == pm].sort_values("lambda_")
                ax.plot(sub["lambda_"], sub["val_accuracy"], marker="o", label=pm)
            ax.set_xlabel("Lambda (pressure strength)")
            ax.set_ylabel("Val Accuracy")
            ax.set_title("Accuracy vs Pressure Strength")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Run experiments with different pressure modes to see response curves.")


# ── page: Run Search ────────────────────────────────────────────────────────

def page_run_search():
    st.header("Run Architecture Search")
    st.markdown("Configure a search run. It will execute `run_search.py` as a subprocess and stream the output.")

    col1, col2 = st.columns(2)
    with col1:
        dataset = st.selectbox("Dataset", DATASETS)
        algorithm = st.selectbox("Algorithm", ALGORITHMS)
        pressure = st.selectbox("Pressure mode", PRESSURES)
    with col2:
        lambda_ = st.slider("Pressure lambda (λ)", 0.0, 1.0, 0.1, step=0.05)
        search_epochs = st.number_input("Search epochs", 1, 200, 10, step=1)
        proxy_epochs = st.number_input("Proxy epochs (evolutionary only)", 1, 50, 3, step=1)
        device = st.selectbox("Device", ["cpu", "cuda", "mps"], index=0)
        batch_size = st.number_input("Batch size", 8, 512, 32, step=8)

    st.divider()
    data_root = st.text_input("Data root (leave blank to use default)", "")
    db_path = st.text_input("SQLite DB path", str(DB_PATH))

    if st.button("Launch Search", type="primary"):
        cmd = [
            sys.executable, str(ROOT / "run_search.py"),
            "--dataset", dataset,
            "--algorithm", algorithm,
            "--pressure", pressure,
            "--lambda", str(lambda_),
            "--search_epochs", str(search_epochs),
            "--proxy_epochs", str(proxy_epochs),
            "--device", device,
            "--batch_size", str(batch_size),
            "--db_path", db_path,
            "--output_dir", str(RESULTS_DIR),
        ]
        if data_root:
            cmd += ["--data_root", data_root]

        st.info(f"Running: `{' '.join(cmd)}`")
        log_box = st.empty()
        log_lines: list[str] = []

        with st.spinner("Searching…"):
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=str(ROOT)
            )
            for line in proc.stdout:
                log_lines.append(line.rstrip())
                log_box.code("\n".join(log_lines[-60:]), language="")
            proc.wait()

        if proc.returncode == 0:
            st.success("Search completed successfully!")
        else:
            st.error(f"Search failed (exit code {proc.returncode}). See log above.")

    # Show existing result directories
    st.divider()
    st.subheader("Saved Search Runs")
    result_dirs = sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir()], reverse=True) if RESULTS_DIR.exists() else []
    if result_dirs:
        for d in result_dirs[:10]:
            geno_path = d / "genotype.json"
            cfg_path = d / "config.json"
            with st.expander(d.name):
                if cfg_path.exists():
                    cfg = json.loads(cfg_path.read_text())
                    st.json(cfg)
                if geno_path.exists():
                    st.download_button(
                        "Download genotype.json",
                        data=geno_path.read_bytes(),
                        file_name=f"{d.name}_genotype.json",
                        mime="application/json",
                        key=f"dl_{d.name}",
                    )
    else:
        st.info("No result directories yet.")


# ── demo data ────────────────────────────────────────────────────────────────

DEMO_GENOTYPES = {
    "cifar10": [
        ["conv3", "skip"],
        ["attention", "conv5", "skip"],
        ["mlp_mixer", "conv3", "attention", "skip"],
        ["conv5", "skip", "mlp_mixer", "conv3", "attention"],
    ],
    "eurosat": [
        ["conv5", "conv3"],
        ["conv5", "conv3", "skip"],
        ["conv3", "conv5", "conv3", "skip"],
        ["conv5", "conv3", "conv5", "conv3", "skip"],
    ],
    "isic": [
        ["attention", "mlp_mixer"],
        ["attention", "mlp_mixer", "skip"],
        ["attention", "conv3", "mlp_mixer", "skip"],
        ["mlp_mixer", "attention", "conv3", "skip", "attention"],
    ],
    "cub200": [
        ["attention", "skip"],
        ["attention", "conv3", "mlp_mixer"],
        ["attention", "mlp_mixer", "conv5", "skip"],
        ["attention", "skip", "attention", "mlp_mixer", "conv3"],
    ],
}

DEMO_LINEAGE = [
    {"generation": 0, "child_id": "arch_0", "parent_ids": [], "fitness": 0.42},
    {"generation": 0, "child_id": "arch_1", "parent_ids": [], "fitness": 0.38},
    {"generation": 1, "child_id": "arch_2", "parent_ids": ["arch_0"], "fitness": 0.51},
    {"generation": 1, "child_id": "arch_3", "parent_ids": ["arch_1"], "fitness": 0.45},
    {"generation": 1, "child_id": "arch_4", "parent_ids": ["arch_0", "arch_1"], "fitness": 0.48},
    {"generation": 2, "child_id": "arch_5", "parent_ids": ["arch_2"], "fitness": 0.59},
    {"generation": 2, "child_id": "arch_6", "parent_ids": ["arch_2", "arch_3"], "fitness": 0.55},
    {"generation": 3, "child_id": "arch_7", "parent_ids": ["arch_5"], "fitness": 0.63},
    {"generation": 3, "child_id": "arch_8", "parent_ids": ["arch_5", "arch_6"], "fitness": 0.67},
    {"generation": 4, "child_id": "arch_9", "parent_ids": ["arch_8"], "fitness": 0.71},
]

DEMO_CONVERGENCE = {
    "cifar10": {
        "darts":        {"time": list(range(1, 11)), "val_acc": [0.31, 0.41, 0.50, 0.56, 0.60, 0.63, 0.65, 0.67, 0.68, 0.69]},
        "evolutionary": {"time": list(range(1, 11)), "val_acc": [0.28, 0.36, 0.44, 0.52, 0.57, 0.61, 0.63, 0.65, 0.66, 0.67]},
    },
    "eurosat": {
        "darts":        {"time": list(range(1, 11)), "val_acc": [0.40, 0.52, 0.60, 0.66, 0.70, 0.73, 0.75, 0.77, 0.78, 0.79]},
        "evolutionary": {"time": list(range(1, 11)), "val_acc": [0.35, 0.47, 0.55, 0.61, 0.66, 0.70, 0.72, 0.74, 0.75, 0.76]},
    },
}

DEMO_PRESSURE = {
    "memory":             {0.0: 0.69, 0.1: 0.67, 0.2: 0.64, 0.3: 0.60, 0.5: 0.54, 0.8: 0.46},
    "latency":            {0.0: 0.69, 0.1: 0.66, 0.2: 0.62, 0.3: 0.57, 0.5: 0.50, 0.8: 0.41},
    "data_scarce":        {0.0: 0.69, 0.1: 0.65, 0.2: 0.60, 0.3: 0.55, 0.5: 0.48, 0.8: 0.39},
    "distribution_shift": {0.0: 0.69, 0.1: 0.68, 0.2: 0.66, 0.3: 0.63, 0.5: 0.59, 0.8: 0.52},
}


def seed_demo_db():
    """Insert demo rows into SQLite so all DB-driven charts work."""
    import random, math
    random.seed(0)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset TEXT, algorithm TEXT, pressure_mode TEXT,
            lambda_ REAL, genotype_json TEXT,
            val_accuracy REAL, param_count INTEGER,
            training_time_sec REAL, timestamp TEXT
        )
    """)
    base_acc = {"cifar10": 0.69, "eurosat": 0.78, "isic": 0.72, "cub200": 0.65}
    from datetime import datetime
    rows = []
    for ds in DATASETS:
        for algo in ALGORITHMS:
            for pm in PRESSURES:
                for lam in ([0.0, 0.1, 0.2, 0.3, 0.5, 0.8] if pm != "none" else [0.0]):
                    penalty = lam * 0.25 if pm != "none" else 0.0
                    acc = max(0.1, base_acc[ds] - penalty + random.gauss(0, 0.02))
                    rows.append((
                        ds, algo, pm, lam,
                        json.dumps(DEMO_GENOTYPES.get(ds, DEMO_GENOTYPES["cifar10"])),
                        round(acc, 4),
                        random.randint(200_000, 2_000_000),
                        round(random.uniform(30, 300), 1),
                        datetime.utcnow().isoformat(),
                    ))
    conn.executemany(
        "INSERT INTO experiments (dataset,algorithm,pressure_mode,lambda_,genotype_json,"
        "val_accuracy,param_count,training_time_sec,timestamp) VALUES (?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    return len(rows)


# ── page: Analysis ──────────────────────────────────────────────────────────

def page_analysis():
    st.header("Analysis & Visualisation")

    df = load_db()
    no_data = df is None or df.empty

    if no_data:
        st.warning("No experiments in DB yet.")
        st.markdown("You can **seed demo data** to preview all charts, or run real searches from the **Run Search** page.")
        if st.button("🌱 Seed demo data (80 synthetic runs)", type="primary"):
            n = seed_demo_db()
            st.success(f"Inserted {n} demo rows. Reloading…")
            st.rerun()
        return

    tab1, tab2, tab3, tab4 = st.tabs(["Op Heatmap", "Convergence", "Lineage Tree", "Pressure Response"])

    # ── Op Preference Heatmap ───────────────────────────────────────────────
    with tab1:
        st.subheader("Operation Preference Heatmap")
        st.markdown("Fraction of edges that chose each op, per dataset.")

        if "genotype_json" not in df.columns:
            st.info("No genotype data available.")
        else:
            valid = df[df["genotype_json"].notna()].copy()
            if valid.empty:
                st.info("No genotype data stored.")
            else:
                ds_sel = st.multiselect("Datasets to plot", DATASETS,
                                        default=list(valid["dataset"].unique()))
                filtered = valid[valid["dataset"].isin(ds_sel)]
                if filtered.empty:
                    st.warning("No matching rows.")
                else:
                    n_ds = len(ds_sel)
                    fig, axes = plt.subplots(1, max(n_ds, 1), figsize=(5 * max(n_ds, 1), 4), squeeze=False)
                    import numpy as np
                    for col_idx, ds in enumerate(ds_sel):
                        ax = axes[0][col_idx]
                        sub = filtered[filtered["dataset"] == ds]
                        # Count op frequencies across all edges
                        counts = {op: 0 for op in OP_NAMES}
                        total = 0
                        for geno_str in sub["genotype_json"]:
                            try:
                                geno = json.loads(geno_str)
                                # geno can be list of lists of strings
                                if isinstance(geno, list):
                                    for item in geno:
                                        if isinstance(item, list):
                                            for op in item:
                                                if isinstance(op, str) and op in counts:
                                                    counts[op] += 1
                                                    total += 1
                                        elif isinstance(item, str) and item in counts:
                                            counts[item] += 1
                                            total += 1
                            except Exception:
                                pass
                        if total == 0:
                            ax.set_title(f"{ds}\n(no data)")
                            continue
                        fracs = np.array([counts[op] / total for op in OP_NAMES])
                        bars = ax.barh(OP_NAMES, fracs, color=plt.cm.tab10(np.linspace(0, 1, len(OP_NAMES))))
                        ax.set_xlim(0, 1)
                        ax.set_xlabel("Fraction of edges")
                        ax.set_title(ds)
                        for bar, frac in zip(bars, fracs):
                            if frac > 0.05:
                                ax.text(frac + 0.01, bar.get_y() + bar.get_height() / 2,
                                        f"{frac:.2f}", va="center", fontsize=8)
                    plt.suptitle("Operation Preference per Dataset", fontsize=13, y=1.02)
                    plt.tight_layout()
                    st.pyplot(fig)
                    st.download_button("Download plot", fig_to_bytes(fig), "op_heatmap.png", "image/png")
                    plt.close(fig)

    # ── Convergence ──────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Convergence Comparison")
        st.markdown("Validation accuracy vs epoch, DARTS vs evolutionary per dataset.")

        # Prefer real log files; fall back to demo curves
        result_dirs = sorted(RESULTS_DIR.iterdir()) if RESULTS_DIR.exists() else []
        log_data: dict = {}
        for d in result_dirs:
            if not d.is_dir():
                continue
            cfg_path = d / "config.json"
            logs_path = d / "search_logs.json"
            if not cfg_path.exists() or not logs_path.exists():
                continue
            cfg = json.loads(cfg_path.read_text())
            ds, algo = cfg.get("dataset", "?"), cfg.get("algorithm", "?")
            logs = json.loads(logs_path.read_text())
            log_data.setdefault(ds, {})[algo] = {
                "time": [e.get("epoch", i) for i, e in enumerate(logs)],
                "val_acc": [e.get("val_acc", 0.0) for e in logs],
            }

        using_demo = False
        if not log_data:
            log_data = DEMO_CONVERGENCE
            using_demo = True

        if using_demo:
            st.info("Showing **demo convergence curves**. Run real DARTS searches to replace with live data.")

        datasets_with_logs = list(log_data.keys())
        fig, axes = plt.subplots(1, len(datasets_with_logs),
                                 figsize=(5 * len(datasets_with_logs), 4),
                                 squeeze=False)
        for col_idx, ds in enumerate(datasets_with_logs):
            ax = axes[0][col_idx]
            for algo, data in log_data[ds].items():
                ax.plot(data["time"], data["val_acc"], marker=".", label=algo, linewidth=2)
            ax.set_title(ds)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Val Accuracy")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend()
        plt.suptitle("Convergence: DARTS vs Evolutionary", y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        st.download_button("Download plot", fig_to_bytes(fig), "convergence.png", "image/png")
        plt.close(fig)

    # ── Lineage Tree ─────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Evolutionary Lineage Tree")
        st.markdown("Parent→child mutation graph (node colour = fitness).")

        lineage_files = list(RESULTS_DIR.rglob("lineage_log.json")) if RESULTS_DIR.exists() else []

        if lineage_files:
            selected_file = st.selectbox("Select lineage log", lineage_files,
                                          format_func=lambda p: str(p.relative_to(ROOT)))
            lineage = json.loads(selected_file.read_text())
            demo_label = ""
        else:
            st.info("No lineage_log.json found. Showing **demo lineage tree** — run an evolutionary search for a real one.")
            lineage = DEMO_LINEAGE
            demo_label = " (demo)"

        try:
            import networkx as nx
            G = nx.DiGraph()
            for entry in lineage:
                child_id = entry.get("child_id", entry.get("id", "?"))
                fitness = entry.get("fitness", 0.0)
                G.add_node(child_id, fitness=fitness, generation=entry.get("generation", 0))
                for pid in entry.get("parent_ids", []):
                    if pid:
                        G.add_edge(pid, child_id)

            fitnesses = nx.get_node_attributes(G, "fitness")
            if nx.is_directed_acyclic_graph(G):
                pos = nx.multipartite_layout(G, subset_key="generation")
            else:
                pos = nx.spring_layout(G, seed=42)

            node_colors = [fitnesses.get(n, 0.0) for n in G.nodes()]
            vmin, vmax = min(node_colors, default=0), max(node_colors, default=1)

            fig, ax = plt.subplots(figsize=(10, 5))
            nx.draw(G, pos, ax=ax, with_labels=True,
                    labels={n: n for n in G.nodes()},
                    node_color=node_colors, cmap=plt.cm.RdYlGn,
                    vmin=vmin, vmax=vmax,
                    node_size=600, arrows=True, arrowsize=12,
                    edge_color="#888888", font_size=7, alpha=0.9)
            sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn,
                                        norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label="Fitness", shrink=0.7)
            ax.set_title(f"Lineage Tree{demo_label} — {len(G.nodes())} architectures, {len(G.edges())} mutations")
            plt.tight_layout()
            st.pyplot(fig)
            st.download_button("Download plot", fig_to_bytes(fig), "lineage.png", "image/png")
            plt.close(fig)

            with st.expander("Raw lineage data"):
                st.json(lineage)
        except ImportError:
            st.error("NetworkX not installed. Run `pip install networkx`.")

    # ── Pressure Response ────────────────────────────────────────────────────
    with tab4:
        st.subheader("Pressure Response Curves")
        st.markdown("How does best accuracy change as λ increases per pressure mode?")

        if df is None or df.empty or "lambda_" not in df.columns:
            st.info("No data.")
        else:
            pr_df = df[df["pressure_mode"].notna() & (df["pressure_mode"] != "none")].copy()
            if pr_df.empty:
                st.info("Run experiments with different lambda values and pressure modes to see curves.")
            else:
                fig, ax = plt.subplots(figsize=(8, 4))
                for pm in sorted(pr_df["pressure_mode"].unique()):
                    sub = pr_df[pr_df["pressure_mode"] == pm].sort_values("lambda_")
                    best = sub.groupby("lambda_")["val_accuracy"].max().reset_index()
                    ax.plot(best["lambda_"], best["val_accuracy"], marker="o", label=pm, linewidth=2)
                ax.set_xlabel("Lambda (pressure strength λ)")
                ax.set_ylabel("Best Val Accuracy")
                ax.set_title("Architecture Quality vs Pressure Strength")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                st.download_button("Download plot", fig_to_bytes(fig), "pressure_response.png", "image/png")
                plt.close(fig)


# ── page: Recommender ────────────────────────────────────────────────────────

def page_recommender():
    st.header("Architecture Recommender")
    st.markdown(
        "Upload a dataset (as a folder of images) or select an existing dataset "
        "to get a predicted best architecture based on its visual statistics."
    )

    col1, col2 = st.columns(2)
    with col1:
        dataset_name = st.selectbox("Known dataset", DATASETS + ["custom"])
    with col2:
        db_path = st.text_input("Results DB path", str(DB_PATH))

    if st.button("Get Recommendation", type="primary"):
        cmd = [
            sys.executable, str(ROOT / "run_recommender.py"),
            "--dataset_name", dataset_name,
            "--results_db", db_path,
            "--output_dir", str(RESULTS_DIR),
        ]
        data_root = str(DATA_DIR / dataset_name)
        if Path(data_root).exists():
            cmd += ["--data_dir", data_root]

        with st.spinner("Profiling dataset and querying recommender…"):
            rc, stdout, stderr = run_cmd(cmd)

        if rc == 0:
            st.success("Recommendation ready!")
            st.code(stdout, language="")
        else:
            st.error("Recommender failed.")
            st.code(stderr, language="")

    # Manual dataset profiler
    st.divider()
    st.subheader("Manual Dataset Statistics")
    st.markdown("Select a dataset and compute its visual statistics in-browser.")

    sel_ds = st.selectbox("Dataset to profile", DATASETS, key="prof_ds")
    if st.button("Compute Stats"):
        data_root = DATA_DIR / sel_ds
        if not data_root.exists():
            st.warning(f"Data directory not found: {data_root}")
        else:
            try:
                import numpy as np
                import torchvision
                import torch

                with st.spinner("Loading dataset…"):
                    if sel_ds == "cifar10":
                        ds = torchvision.datasets.CIFAR10(
                            root=str(data_root), train=True, download=False,
                            transform=torchvision.transforms.ToTensor()
                        )
                        loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False)
                    else:
                        ds = torchvision.datasets.ImageFolder(
                            root=str(data_root / "train"),
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.Resize(64),
                                torchvision.transforms.ToTensor(),
                            ])
                        )
                        loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)

                    all_imgs = []
                    for batch, _ in loader:
                        all_imgs.append(batch)
                        if sum(b.shape[0] for b in all_imgs) >= 2000:
                            break
                    imgs = torch.cat(all_imgs, dim=0)[:2000]

                variance = imgs.var(dim=[0, 2, 3]).mean().item()
                mean_px = imgs.mean().item()
                # FFT high-freq energy
                gray = imgs.mean(dim=1)
                fft_mag = torch.fft.fft2(gray).abs()
                h, w = gray.shape[2], gray.shape[3]
                hf_mask = torch.zeros(h, w, dtype=torch.bool)
                hf_mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True
                lf_energy = fft_mag[:, hf_mask].mean().item()
                hf_energy = fft_mag[:, ~hf_mask].mean().item()
                freq_ratio = hf_energy / (lf_energy + 1e-8)

                st.json({
                    "dataset": sel_ds,
                    "sample_count": len(ds),
                    "class_count": len(ds.classes) if hasattr(ds, "classes") else "?",
                    "mean_pixel_value": round(mean_px, 4),
                    "pixel_variance": round(variance, 4),
                    "high_freq_energy_ratio": round(freq_ratio, 4),
                })

                # Visualize a few samples
                import torchvision.transforms.functional as TF
                grid_imgs = [TF.to_pil_image(imgs[i].clamp(0, 1)) for i in range(min(8, len(imgs)))]
                fig, axes = plt.subplots(1, len(grid_imgs), figsize=(12, 2))
                for ax, img in zip(axes, grid_imgs):
                    ax.imshow(img)
                    ax.axis("off")
                plt.suptitle(f"Sample images — {sel_ds}")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            except Exception as e:
                st.error(f"Failed to profile dataset: {e}")


# ── page: Genotype Viewer ────────────────────────────────────────────────────

def page_genotype():
    st.header("Genotype Inspector")
    st.markdown("Inspect and visualise a saved architecture genotype.")

    # Load from file or paste
    source = st.radio("Input source", ["Select from results/", "Paste JSON", "From DB (by ID)"])

    genotype = None

    if source == "Select from results/":
        geno_files = list(RESULTS_DIR.rglob("genotype.json")) if RESULTS_DIR.exists() else []
        if not geno_files:
            st.info("No genotype.json files found.")
        else:
            sel = st.selectbox("Select genotype", geno_files,
                                format_func=lambda p: str(p.relative_to(ROOT)))
            genotype = json.loads(sel.read_text())

    elif source == "Paste JSON":
        raw = st.text_area("Paste genotype JSON here", height=200,
                           placeholder='[["conv3","skip"],["attention","mlp_mixer","conv5"],...]')
        if raw.strip():
            try:
                genotype = json.loads(raw)
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

    elif source == "From DB (by ID)":
        df = load_db()
        if df is None or df.empty:
            st.info("No experiments in DB.")
        else:
            row_id = st.number_input("Experiment ID", min_value=int(df["id"].min()),
                                      max_value=int(df["id"].max()),
                                      value=int(df["id"].iloc[0]))
            row = df[df["id"] == row_id]
            if not row.empty and "genotype_json" in row.columns:
                geno_str = row.iloc[0]["genotype_json"]
                if geno_str:
                    try:
                        genotype = json.loads(geno_str)
                    except Exception:
                        st.error("Could not parse genotype from DB.")

    if genotype is None:
        return

    st.divider()
    st.subheader("Raw Genotype")
    st.json(genotype)

    # Summary table
    import numpy as np
    rows = []
    for node_idx, node_ops in enumerate(genotype):
        if not isinstance(node_ops, list):
            continue
        for edge_idx, op in enumerate(node_ops):
            rows.append({"Node": node_idx, "Edge (from)": edge_idx, "Operation": op})
    if rows:
        edge_df = pd.DataFrame(rows)
        st.subheader("Edge → Operation Table")
        st.dataframe(edge_df, use_container_width=True)

        # Op frequency bar chart
        st.subheader("Operation Frequency")
        op_counts = edge_df["Operation"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = plt.cm.tab10(np.linspace(0, 1, len(op_counts)))
        op_counts.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
        ax.set_ylabel("Count")
        ax.set_title("Op Usage in This Genotype")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # DAG visualisation
    st.subheader("Cell DAG Visualisation")
    st.markdown("Nodes = cell nodes (0,1 = inputs; 2–5 = intermediate). Edges labelled with chosen op.")
    try:
        import networkx as nx
        G = nx.DiGraph()
        node_labels = {0: "in₀", 1: "in₁"}
        for i in range(len(genotype)):
            node_labels[i + 2] = f"n{i}"

        edge_labels = {}
        for node_idx, node_ops in enumerate(genotype):
            if not isinstance(node_ops, list):
                continue
            target = node_idx + 2
            for edge_idx, op in enumerate(node_ops):
                src = edge_idx
                G.add_edge(src, target)
                edge_labels[(src, target)] = op

        pos = {}
        n_nodes = len(genotype) + 2
        for node_id in range(n_nodes):
            pos[node_id] = (node_id, 0)

        fig, ax = plt.subplots(figsize=(10, 3))
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#4c72b0", node_size=800, alpha=0.9)
        nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax, font_color="white", font_size=9)
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=15,
                               edge_color="#aaaaaa", connectionstyle="arc3,rad=0.2")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                     font_size=7, label_pos=0.4)
        ax.set_title("Cell DAG — edge labels = chosen operation")
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)
        st.download_button("Download DAG", fig_to_bytes(fig), "genotype_dag.png", "image/png")
        plt.close(fig)
    except ImportError:
        st.warning("Install networkx for DAG visualisation: `pip install networkx`")

    # Network stats
    st.divider()
    st.subheader("Network Statistics (estimated)")
    try:
        import torch
        from archevo.search_space import build_from_genotype, Network
        from archevo.pressure import count_params
        net = build_from_genotype(genotype, C_init=16, num_classes=10)
        n_params = count_params(net)
        dummy = torch.zeros(1, 3, 32, 32)
        t0 = time.time()
        with torch.no_grad():
            _ = net(dummy)
        latency_ms = (time.time() - t0) * 1000
        st.metric("Parameters", f"{n_params:,}")
        st.metric("Latency (32×32 input, CPU)", f"{latency_ms:.1f} ms")
    except Exception as e:
        st.warning(f"Could not build network for stats: {e}")


# ── main layout ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="ArchEvo",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("🧬 ArchEvo")
    st.sidebar.markdown("Dataset-Aware Neural Architecture Discovery")
    st.sidebar.divider()

    pages = {
        "📊 Dashboard": page_dashboard,
        "🔍 Run Search": page_run_search,
        "📈 Analysis": page_analysis,
        "🤖 Recommender": page_recommender,
        "🔬 Genotype Viewer": page_genotype,
    }

    selected = st.sidebar.radio("Navigate", list(pages.keys()), label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption("Built by **Manvi Gawande** and **Sanskar Srivastava**")
    st.sidebar.divider()
    st.sidebar.caption(f"DB: `{DB_PATH.relative_to(ROOT) if DB_PATH.exists() else 'not created yet'}`")
    st.sidebar.caption(f"Results: `{RESULTS_DIR.relative_to(ROOT)}`")

    pages[selected]()


if __name__ == "__main__":
    main()
