"""
metrics.py
==========
Calculates all SNA metrics for a single graph or a dict of temporal graphs.

Metrics computed:
  - Modularity          (via Louvain community detection)
  - Number of communities
  - Average community size
  - Degree assortativity coefficient
  - Degree centrality   (mean / top-k users)
  - Betweenness centrality (approximate, mean / top-k users)
  - Network density
  - Average clustering coefficient
  - Bridge user identification

Usage:
    from src.metrics import GraphMetrics
    gm = GraphMetrics()
    stats = gm.compute_all(G, label="2021-03")
    temporal_df = gm.compute_temporal(monthly_graphs)
"""

import networkx as nx
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

try:
    import community as community_louvain  # python-louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    logging.warning("python-louvain not found. Install with: pip install python-louvain")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class GraphMetrics:
    """Computes and stores SNA metrics for temporal network analysis."""

    def __init__(self, betweenness_k: int = 100, top_users: int = 50):
        """
        Parameters
        ----------
        betweenness_k : int
            Number of pivot nodes for approximate betweenness centrality.
            Lower = faster but less accurate. Use None for exact (slow!).
        top_users : int
            Number of top users to track for centrality rankings.
        """
        self.betweenness_k = betweenness_k
        self.top_users = top_users

    # ------------------------------------------------------------------
    # Community Detection (Louvain)
    # ------------------------------------------------------------------

    def detect_communities(self, G: nx.Graph) -> Tuple[dict, float]:
        """
        Run Louvain community detection on graph G.

        Parameters
        ----------
        G : undirected NetworkX Graph

        Returns
        -------
        partition : dict { node → community_id }
        modularity : float (0–1; higher = stronger communities)
        """
        if G.is_directed():
            G = G.to_undirected()

        # Remove isolated nodes for detection quality
        G_pruned = G.copy()
        isolates = list(nx.isolates(G_pruned))
        G_pruned.remove_nodes_from(isolates)

        if G_pruned.number_of_nodes() < 2:
            return {}, 0.0

        if LOUVAIN_AVAILABLE:
            partition = community_louvain.best_partition(G_pruned, weight="weight", random_state=42)
            modularity = community_louvain.modularity(partition, G_pruned, weight="weight")
        else:
            # Fallback to NetworkX greedy modularity (less accurate but no extra dependency)
            from networkx.algorithms.community import greedy_modularity_communities
            comms = greedy_modularity_communities(G_pruned, weight="weight")
            partition = {}
            for cid, comm in enumerate(comms):
                for node in comm:
                    partition[node] = cid
            from networkx.algorithms.community.quality import modularity as nx_mod
            modularity = nx_mod(G_pruned, comms, weight="weight")

        return partition, modularity

    # ------------------------------------------------------------------
    # Individual metric calculations
    # ------------------------------------------------------------------

    def _community_stats(self, partition: dict) -> dict:
        """Return community count and size statistics."""
        if not partition:
            return {"num_communities": 0, "avg_community_size": 0, "max_community_size": 0}
        sizes = pd.Series(partition).value_counts()
        return {
            "num_communities": len(sizes),
            "avg_community_size": float(sizes.mean()),
            "max_community_size": int(sizes.max()),
        }

    def _centrality_stats(self, G: nx.Graph) -> dict:
        """Compute degree and betweenness centrality summaries."""
        stats = {}

        # Degree centrality
        deg_cent = nx.degree_centrality(G)
        stats["mean_degree_centrality"] = float(np.mean(list(deg_cent.values())))
        stats["max_degree_centrality"] = float(max(deg_cent.values())) if deg_cent else 0.0

        # Betweenness centrality (approximate for large graphs)
        k = min(self.betweenness_k, G.number_of_nodes())
        btw = nx.betweenness_centrality(G, k=k, normalized=True, weight="weight")
        stats["mean_betweenness"] = float(np.mean(list(btw.values())))
        stats["max_betweenness"] = float(max(btw.values())) if btw else 0.0

        # Store top users for later analysis
        stats["_degree_centrality"] = deg_cent
        stats["_betweenness"] = btw

        return stats

    # ------------------------------------------------------------------
    # Single-graph full computation
    # ------------------------------------------------------------------

    def compute_all(self, G: nx.Graph, label: str = "") -> dict:
        """
        Compute ALL metrics for a single graph.

        Returns a flat dict with all metrics plus internal '_' keys
        for the raw centrality dicts (used for bridge detection).
        """
        stats = {"label": label}

        if G.number_of_nodes() == 0:
            logger.warning(f"[{label}] Empty graph — returning zeros.")
            return stats

        G_und = G.to_undirected() if G.is_directed() else G

        # Basic network properties
        stats["num_nodes"] = G.number_of_nodes()
        stats["num_edges"] = G.number_of_edges()
        stats["density"] = nx.density(G_und)

        # Assortativity
        try:
            stats["degree_assortativity"] = nx.degree_assortativity_coefficient(G_und)
        except Exception:
            stats["degree_assortativity"] = np.nan

        # Clustering
        try:
            stats["avg_clustering"] = nx.average_clustering(G_und, weight="weight")
        except Exception:
            stats["avg_clustering"] = np.nan

        # Community detection
        partition, modularity = self.detect_communities(G_und)
        stats["modularity"] = modularity
        stats["_partition"] = partition
        stats.update(self._community_stats(partition))

        # Centrality
        cent_stats = self._centrality_stats(G_und)
        stats.update(cent_stats)

        logger.info(
            f"[{label}] nodes={stats['num_nodes']:,}, edges={stats['num_edges']:,}, "
            f"modularity={modularity:.4f}, communities={stats.get('num_communities', 0)}, "
            f"assortativity={stats.get('degree_assortativity', 0):.4f}"
        )
        return stats

    # ------------------------------------------------------------------
    # Temporal analysis
    # ------------------------------------------------------------------

    def compute_temporal(self, graphs: Dict[str, nx.Graph]) -> pd.DataFrame:
        """
        Compute metrics for every graph in a time-ordered dict.

        Parameters
        ----------
        graphs : { label → nx.Graph }  (e.g., output of NetworkBuilder.build_monthly_graphs())

        Returns
        -------
        pd.DataFrame with one row per time period and all metric columns.
        """
        rows = []
        for label in sorted(graphs.keys()):
            G = graphs[label]
            stats = self.compute_all(G, label=label)
            # Drop private keys before storing in DataFrame
            public_stats = {k: v for k, v in stats.items() if not k.startswith("_")}
            rows.append(public_stats)

        df = pd.DataFrame(rows)
        if "label" in df.columns:
            df = df.set_index("label").sort_index()
        return df

    # ------------------------------------------------------------------
    # Event comparison
    # ------------------------------------------------------------------

    def compare_events(
        self, event_graphs: Dict[str, Dict[str, nx.Graph]]
    ) -> pd.DataFrame:
        """
        Compare before/after metrics for each event.

        Parameters
        ----------
        event_graphs : output of NetworkBuilder.build_event_graphs()

        Returns
        -------
        pd.DataFrame with before/after metrics and delta columns.
        """
        rows = []
        for event_name, windows in event_graphs.items():
            before_stats = self.compute_all(windows["before"], label=f"{event_name}_before")
            after_stats = self.compute_all(windows["after"], label=f"{event_name}_after")

            row = {"event": event_name}
            metric_cols = [
                "num_nodes", "num_edges", "density", "modularity",
                "num_communities", "degree_assortativity", "avg_clustering",
                "mean_degree_centrality", "mean_betweenness",
            ]
            for col in metric_cols:
                before_val = before_stats.get(col, np.nan)
                after_val = after_stats.get(col, np.nan)
                row[f"before_{col}"] = before_val
                row[f"after_{col}"] = after_val
                # Delta = after − before
                try:
                    row[f"delta_{col}"] = after_val - before_val
                except TypeError:
                    row[f"delta_{col}"] = np.nan

            rows.append(row)

        return pd.DataFrame(rows).set_index("event")

    # ------------------------------------------------------------------
    # Bridge User Detection
    # ------------------------------------------------------------------

    def find_bridge_users(
        self,
        G: nx.Graph,
        partition: dict,
        top_n: int = 50,
    ) -> pd.DataFrame:
        """
        Identify bridge users — nodes with high betweenness that connect
        different communities.

        A user is a bridge candidate if:
          - They have high betweenness centrality (top quantile)
          - Their neighbours span ≥ 2 different communities

        Parameters
        ----------
        G         : undirected NetworkX graph
        partition : { node → community_id }  (from Louvain)
        top_n     : return top N bridge users

        Returns
        -------
        pd.DataFrame sorted by betweenness descending.
        """
        if not partition:
            logger.warning("Empty partition — cannot find bridge users.")
            return pd.DataFrame()

        G_und = G.to_undirected() if G.is_directed() else G

        k = min(self.betweenness_k, G_und.number_of_nodes())
        btw = nx.betweenness_centrality(G_und, k=k, normalized=True, weight="weight")
        deg_cent = nx.degree_centrality(G_und)

        records = []
        for node in G_und.nodes():
            if node not in partition:
                continue
            neighbours = list(G_und.neighbors(node))
            neighbour_communities = {
                partition[nb] for nb in neighbours if nb in partition
            }
            num_bridges = len(neighbour_communities)
            records.append(
                {
                    "user": node,
                    "community": partition[node],
                    "betweenness": btw.get(node, 0.0),
                    "degree_centrality": deg_cent.get(node, 0.0),
                    "degree": G_und.degree(node),
                    "num_communities_bridged": num_bridges,
                    "is_bridge": num_bridges >= 2,
                }
            )

        df = pd.DataFrame(records)
        if df.empty:
            return df

        # Bridge users = those connecting ≥ 2 communities
        bridges = df[df["is_bridge"]].sort_values("betweenness", ascending=False)
        return bridges.head(top_n).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Persistent bridge users (users who appear across multiple periods)
    # ------------------------------------------------------------------

    def find_persistent_bridges(
        self,
        graphs: Dict[str, nx.Graph],
        min_periods: int = 3,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Find users who are bridge users in at least `min_periods` time slices.

        Returns
        -------
        pd.DataFrame with user, number of periods as bridge, avg betweenness.
        """
        user_bridge_periods = {}
        user_betweenness = {}

        for label, G in sorted(graphs.items()):
            G_und = G.to_undirected() if G.is_directed() else G
            partition, _ = self.detect_communities(G_und)
            bridges = self.find_bridge_users(G_und, partition, top_n=200)
            for _, row in bridges.iterrows():
                u = row["user"]
                user_bridge_periods[u] = user_bridge_periods.get(u, 0) + 1
                user_betweenness.setdefault(u, []).append(row["betweenness"])

        records = []
        for user, count in user_bridge_periods.items():
            if count >= min_periods:
                records.append(
                    {
                        "user": user,
                        "bridge_periods": count,
                        "avg_betweenness": float(np.mean(user_betweenness[user])),
                    }
                )

        df = pd.DataFrame(records).sort_values("bridge_periods", ascending=False)
        return df.head(top_n).reset_index(drop=True)
