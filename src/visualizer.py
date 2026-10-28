"""
visualizer.py
=============
All plotting functions for the echo chamber temporal analysis project.

Functions:
  - plot_temporal_metrics()   — Line charts of modularity / assortativity over time
  - plot_event_comparison()   — Before/after bar charts for each event
  - plot_network_snapshot()   — Network graph colored by community
  - plot_bridge_users()       — Bar chart of top bridge users
  - plot_community_heatmap()  — Heatmap of community sizes over time
  - plot_correlation_matrix() — Correlation between all metrics

Usage:
    from src.visualizer import Visualizer
    viz = Visualizer(output_dir="figures")
    viz.plot_temporal_metrics(temporal_df)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
import logging
from pathlib import Path
from typing import Dict, Optional, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Global style settings
# ------------------------------------------------------------------
PALETTE = sns.color_palette("tab20", 20)
plt.rcParams.update(
    {
        "figure.facecolor": "#0f1117",
        "axes.facecolor": "#1a1d27",
        "axes.edgecolor": "#444",
        "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#aaa",
        "ytick.color": "#aaa",
        "text.color": "#e0e0e0",
        "grid.color": "#2d2d3a",
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "lines.linewidth": 2.2,
        "font.family": "DejaVu Sans",
        "figure.autolayout": True,
    }
)

HIGHLIGHT = "#00e5ff"  # cyan accent
ACCENT1 = "#ff6b6b"   # red accent
ACCENT2 = "#51cf66"   # green accent


class Visualizer:
    """Creates and saves all project visualizations."""

    def __init__(self, output_dir: str = "figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name: str):
        path = self.output_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info(f"Saved figure → {path}")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # 1. Temporal metrics line charts
    # ------------------------------------------------------------------

    def plot_temporal_metrics(
        self,
        temporal_df: pd.DataFrame,
        filename: str = "temporal_metrics.png",
    ):
        """
        4-panel line chart of key metrics over time:
          modularity, assortativity, num_communities, avg_clustering
        """
        metrics = [
            ("modularity", "Modularity", HIGHLIGHT),
            ("degree_assortativity", "Degree Assortativity", ACCENT1),
            ("num_communities", "Number of Communities", ACCENT2),
            ("avg_clustering", "Avg. Clustering Coefficient", "#ffd43b"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(
            "Echo Chamber Temporal Evolution – Climate Subreddits",
            fontsize=18, fontweight="bold", color="#ffffff", y=1.02,
        )

        for ax, (col, title, color) in zip(axes.flatten(), metrics):
            if col not in temporal_df.columns:
                ax.set_visible(False)
                continue

            x = range(len(temporal_df))
            y = temporal_df[col].values

            ax.plot(x, y, color=color, linewidth=2.5, marker="o", markersize=4, alpha=0.9)
            ax.fill_between(x, y, alpha=0.15, color=color)

            # Add rolling mean trend line
            if len(y) >= 6:
                rolling = pd.Series(y).rolling(3, center=True).mean()
                ax.plot(x, rolling, color="#ffffff", linewidth=1.5, linestyle="--",
                        alpha=0.5, label="3-month rolling avg")
                ax.legend(fontsize=8, loc="upper right")

            # X-axis labels (every 3 months)
            labels = list(temporal_df.index)
            step = max(1, len(labels) // 12)
            ax.set_xticks(range(0, len(labels), step))
            ax.set_xticklabels(labels[::step], rotation=45, ha="right", fontsize=8)

            ax.set_title(title, fontsize=13, color="#ffffff", pad=8)
            ax.set_xlabel("Time Period", fontsize=10)
            ax.set_ylabel(title, fontsize=10)
            ax.grid(True)

        self._save(fig, filename)
        return fig

    # ------------------------------------------------------------------
    # 2. Event comparison bar charts
    # ------------------------------------------------------------------

    def plot_event_comparison(
        self,
        event_df: pd.DataFrame,
        metric: str = "modularity",
        filename: str = "event_comparison.png",
    ):
        """
        Before/after grouped bar chart for a given metric across all events.
        """
        before_col = f"before_{metric}"
        after_col = f"after_{metric}"

        if before_col not in event_df.columns or after_col not in event_df.columns:
            logger.warning(f"Columns {before_col} / {after_col} not found in event_df.")
            return

        events = event_df.index.tolist()
        before_vals = event_df[before_col].values
        after_vals = event_df[after_col].values
        deltas = after_vals - before_vals

        x = np.arange(len(events))
        width = 0.35

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(
            f"Effect of Climate Events on {metric.replace('_', ' ').title()}",
            fontsize=16, fontweight="bold", color="#ffffff",
        )

        # Left: grouped bars
        bars1 = ax1.bar(x - width / 2, before_vals, width, label="Before", color=ACCENT1, alpha=0.85)
        bars2 = ax1.bar(x + width / 2, after_vals, width, label="After", color=ACCENT2, alpha=0.85)
        ax1.set_xticks(x)
        ax1.set_xticklabels(events, rotation=40, ha="right", fontsize=9)
        ax1.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
        ax1.set_title("Before vs After Each Event", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, axis="y")

        # Value labels on bars
        for bar in bars1:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.002, f"{h:.3f}",
                     ha="center", va="bottom", fontsize=7, color="#ddd")
        for bar in bars2:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.002, f"{h:.3f}",
                     ha="center", va="bottom", fontsize=7, color="#ddd")

        # Right: delta bars
        colors = [ACCENT2 if d >= 0 else ACCENT1 for d in deltas]
        ax2.bar(x, deltas, color=colors, alpha=0.85)
        ax2.axhline(0, color="#aaa", linewidth=1)
        ax2.set_xticks(x)
        ax2.set_xticklabels(events, rotation=40, ha="right", fontsize=9)
        ax2.set_ylabel(f"Δ {metric.replace('_', ' ').title()} (After − Before)", fontsize=11)
        ax2.set_title("Change (Δ) After Each Event", fontsize=12)
        ax2.grid(True, axis="y")

        self._save(fig, filename)
        return fig

    # ------------------------------------------------------------------
    # 3. Network snapshot (community-colored)
    # ------------------------------------------------------------------

    def plot_network_snapshot(
        self,
        G: nx.Graph,
        partition: dict,
        label: str = "",
        max_nodes: int = 500,
        filename: Optional[str] = None,
    ):
        """
        Visualise a network snapshot with nodes colored by community.
        For readability, only the top `max_nodes` degree nodes are shown.
        """
        if filename is None:
            safe_label = label.replace(" ", "_").replace("/", "-")
            filename = f"network_{safe_label}.png"

        G_und = G.to_undirected() if G.is_directed() else G

        # Prune to top-degree nodes for readability
        if G_und.number_of_nodes() > max_nodes:
            top = sorted(G_und.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
            nodes_to_keep = {n for n, _ in top}
            G_und = G_und.subgraph(nodes_to_keep).copy()

        fig, ax = plt.subplots(figsize=(14, 12))
        ax.set_title(
            f"Network Snapshot – {label}\n"
            f"({G_und.number_of_nodes()} nodes, {G_und.number_of_edges()} edges)",
            fontsize=14, color="#ffffff",
        )
        ax.axis("off")

        # Layout
        pos = nx.spring_layout(G_und, k=0.5, seed=42)

        # Node colours by community
        communities = sorted(set(partition.get(n, -1) for n in G_und.nodes()))
        cmap = plt.cm.get_cmap("tab20", max(1, len(communities)))
        node_colors = [cmap(communities.index(partition.get(n, -1))) for n in G_und.nodes()]

        # Node sizes proportional to degree
        degrees = dict(G_und.degree())
        sizes = [max(20, degrees.get(n, 1) * 8) for n in G_und.nodes()]

        nx.draw_networkx_nodes(
            G_und, pos, ax=ax,
            node_color=node_colors, node_size=sizes, alpha=0.85,
        )
        nx.draw_networkx_edges(
            G_und, pos, ax=ax,
            edge_color="#444", alpha=0.3, width=0.5,
        )

        # Draw labels only for high-degree nodes
        degree_threshold = sorted(degrees.values(), reverse=True)[min(20, len(degrees) - 1)]
        labels = {n: n for n, d in degrees.items() if d >= degree_threshold}
        nx.draw_networkx_labels(
            G_und, pos, labels=labels, ax=ax,
            font_size=7, font_color="#ffffff",
        )

        # Legend for top 10 communities
        patches = []
        for i, cid in enumerate(communities[:10]):
            size = sum(1 for n in G_und.nodes() if partition.get(n, -1) == cid)
            patches.append(mpatches.Patch(color=cmap(i), label=f"Community {cid} (n={size})"))
        ax.legend(handles=patches, loc="lower left", fontsize=8,
                  facecolor="#1a1d27", edgecolor="#444", labelcolor="#ddd")

        self._save(fig, filename)
        return fig

    # ------------------------------------------------------------------
    # 4. Bridge users bar chart
    # ------------------------------------------------------------------

    def plot_bridge_users(
        self,
        bridge_df: pd.DataFrame,
        top_n: int = 20,
        filename: str = "bridge_users.png",
        title: str = "Top Bridge Users by Betweenness Centrality",
    ):
        """
        Horizontal bar chart of top bridge users.
        """
        if bridge_df.empty:
            logger.warning("bridge_df is empty — skipping bridge user plot.")
            return

        plot_df = bridge_df.head(top_n).sort_values("betweenness", ascending=True)

        fig, ax = plt.subplots(figsize=(12, max(6, len(plot_df) * 0.45)))
        bars = ax.barh(
            plot_df["user"], plot_df["betweenness"],
            color=HIGHLIGHT, alpha=0.85, edgecolor="#333",
        )

        # Annotate with community count
        if "num_communities_bridged" in plot_df.columns:
            for bar, (_, row) in zip(bars, plot_df.iterrows()):
                ax.text(
                    bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                    f"bridges {int(row['num_communities_bridged'])} communities",
                    va="center", ha="left", fontsize=7, color="#aaa",
                )

        ax.set_xlabel("Betweenness Centrality (normalised)", fontsize=11)
        ax.set_title(title, fontsize=14, color="#ffffff")
        ax.grid(True, axis="x")

        self._save(fig, filename)
        return fig

    # ------------------------------------------------------------------
    # 5. Community size heatmap over time
    # ------------------------------------------------------------------

    def plot_community_heatmap(
        self,
        graphs: Dict[str, nx.Graph],
        partitions: Dict[str, dict],
        top_communities: int = 10,
        filename: str = "community_heatmap.png",
    ):
        """
        Heatmap where rows = top communities, cols = time periods,
        values = number of users in that community.
        """
        # Collect per-period sizes
        period_sizes = {}
        for label, partition in partitions.items():
            sizes = {}
            if partition:
                for node, cid in partition.items():
                    sizes[cid] = sizes.get(cid, 0) + 1
            period_sizes[label] = sizes

        # Build matrix: we'll track the global top communities by total size
        all_sizes = {}
        for sizes in period_sizes.values():
            for cid, sz in sizes.items():
                all_sizes[cid] = all_sizes.get(cid, 0) + sz

        top_cids = sorted(all_sizes, key=lambda c: all_sizes[c], reverse=True)[:top_communities]
        periods = sorted(period_sizes.keys())

        matrix = []
        for cid in top_cids:
            row = [period_sizes.get(p, {}).get(cid, 0) for p in periods]
            matrix.append(row)

        if not matrix:
            logger.warning("No data to plot community heatmap.")
            return

        df_heat = pd.DataFrame(
            matrix,
            index=[f"Community {c}" for c in top_cids],
            columns=periods,
        )

        fig, ax = plt.subplots(figsize=(max(14, len(periods) * 0.8), 7))
        sns.heatmap(
            df_heat, ax=ax, cmap="viridis", linewidths=0.3,
            linecolor="#2d2d3a", annot=False,
            cbar_kws={"label": "Number of Users"},
        )
        ax.set_title(
            "Community Size Over Time (top communities)",
            fontsize=14, color="#ffffff",
        )
        ax.set_xlabel("Time Period", fontsize=11)
        ax.set_ylabel("Community", fontsize=11)
        plt.xticks(rotation=45, ha="right", fontsize=8)

        self._save(fig, filename)
        return fig

    # ------------------------------------------------------------------
    # 6. Metric correlation matrix
    # ------------------------------------------------------------------

    def plot_correlation_matrix(
        self,
        temporal_df: pd.DataFrame,
        filename: str = "metric_correlation.png",
    ):
        """
        Correlation heatmap between all numeric metrics in the temporal DataFrame.
        """
        numeric_cols = temporal_df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")

        fig, ax = plt.subplots(figsize=(10, 8))
        corr = numeric_cols.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        sns.heatmap(
            corr, mask=mask, ax=ax, cmap="coolwarm", center=0,
            annot=True, fmt=".2f", linewidths=0.5, linecolor="#2d2d3a",
            annot_kws={"size": 9},
        )
        ax.set_title("Metric Correlation Matrix", fontsize=14, color="#ffffff")
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(fontsize=9)

        self._save(fig, filename)
        return fig
