"""
network_builder.py
==================
Builds NetworkX graphs from the cleaned comment edge DataFrame.
Supports:
  - Monthly temporal slices
  - Custom event-window slices (N days before/after a date)
  - Single full-period graph

Usage:
    from src.network_builder import NetworkBuilder
    builder = NetworkBuilder(df)
    graphs = builder.build_monthly_graphs()
    event_graphs = builder.build_event_graphs(events=EVENT_WINDOWS)
"""

import pandas as pd
import networkx as nx
import logging
from datetime import timedelta
from typing import Dict, Tuple, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Major Climate Events (extend / adjust as needed)
# -----------------------------------------------------------------------
CLIMATE_EVENTS = {
    "COP26_2021": "2021-11-01",
    "IPCC_AR6_WG1_2021": "2021-08-09",
    "COP27_2022": "2022-11-07",
    "IPCC_AR6_WG3_2022": "2022-04-04",
    "COP28_2023": "2023-11-30",
    "IPCC_AR6_SYR_2023": "2023-03-20",
    "USInflationReductionAct_2022": "2022-08-16",
}

# How many days before and after an event to include
EVENT_WINDOW_DAYS = 60


class NetworkBuilder:
    """
    Builds NetworkX graphs from a cleaned, edge-ready DataFrame.

    Expected DataFrame columns:
        author         – source node (the replier)
        parent_author  – target node (the user being replied to)
        created_utc    – Unix timestamp (float/int)
        datetime       – pd.Timestamp with UTC timezone
        year_month     – pandas Period('M')
        subreddit      – string
    """

    def __init__(self, df: pd.DataFrame, directed: bool = False):
        """
        Parameters
        ----------
        df       : cleaned comment DataFrame with 'author' and 'parent_author' columns
        directed : if True → DiGraph; if False → Graph (default, better for Louvain)
        """
        self.df = df.copy()
        self.directed = directed
        self.GraphClass = nx.DiGraph if directed else nx.Graph

    # ------------------------------------------------------------------
    # Core builder
    # ------------------------------------------------------------------

    def _build_from_slice(self, df_slice: pd.DataFrame, label: str = "") -> nx.Graph:
        """
        Build a single graph from a DataFrame slice.
        Edges carry weight (number of interactions) and subreddit metadata.
        """
        G = self.GraphClass()

        # Aggregate edges: count interactions between each pair
        edge_counts = (
            df_slice.groupby(["author", "parent_author"])
            .size()
            .reset_index(name="weight")
        )

        for _, row in edge_counts.iterrows():
            u, v, w = row["author"], row["parent_author"], row["weight"]
            if G.has_edge(u, v):
                G[u][v]["weight"] += w
            else:
                G.add_edge(u, v, weight=w)

        # Node attribute: subreddits they were active in
        for user, group in df_slice.groupby("author"):
            subs = list(group["subreddit"].unique())
            G.nodes[user]["subreddits"] = subs

        logger.debug(
            f"Graph [{label}]: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )
        return G

    # ------------------------------------------------------------------
    # Monthly temporal slices
    # ------------------------------------------------------------------

    def build_monthly_graphs(self) -> Dict[str, nx.Graph]:
        """
        Build one graph per calendar month.

        Returns
        -------
        dict of { 'YYYY-MM' : nx.Graph }
        """
        if "year_month" not in self.df.columns:
            raise ValueError("DataFrame must have a 'year_month' column (pandas Period 'M').")

        graphs = {}
        periods = sorted(self.df["year_month"].dropna().unique())

        for period in periods:
            label = str(period)
            slice_df = self.df[self.df["year_month"] == period]

            if len(slice_df) < 10:
                logger.warning(f"Skipping {label}: only {len(slice_df)} edges.")
                continue

            graphs[label] = self._build_from_slice(slice_df, label)
            logger.info(
                f"[{label}] nodes={graphs[label].number_of_nodes():,}, "
                f"edges={graphs[label].number_of_edges():,}"
            )

        logger.info(f"Built {len(graphs)} monthly graphs.")
        return graphs

    # ------------------------------------------------------------------
    # Quarterly slices
    # ------------------------------------------------------------------

    def build_quarterly_graphs(self) -> Dict[str, nx.Graph]:
        """Build one graph per calendar quarter (Q1, Q2, Q3, Q4)."""
        if "quarter" not in self.df.columns:
            self.df["quarter"] = self.df["datetime"].dt.to_period("Q")

        graphs = {}
        for qtr in sorted(self.df["quarter"].dropna().unique()):
            label = str(qtr)
            slice_df = self.df[self.df["quarter"] == qtr]
            if len(slice_df) < 10:
                continue
            graphs[label] = self._build_from_slice(slice_df, label)
            logger.info(f"[{label}] nodes={graphs[label].number_of_nodes():,}, "
                        f"edges={graphs[label].number_of_edges():,}")
        return graphs

    # ------------------------------------------------------------------
    # Event-based comparison windows
    # ------------------------------------------------------------------

    def build_event_graphs(
        self,
        events: Optional[Dict[str, str]] = None,
        window_days: int = EVENT_WINDOW_DAYS,
    ) -> Dict[str, Dict[str, nx.Graph]]:
        """
        For each event, build a 'before' and 'after' graph.

        Parameters
        ----------
        events      : dict of { event_name : 'YYYY-MM-DD' }. Uses CLIMATE_EVENTS if None.
        window_days : how many days before/after the event date.

        Returns
        -------
        dict of { event_name : { 'before': Graph, 'after': Graph } }
        """
        if events is None:
            events = CLIMATE_EVENTS

        event_graphs = {}
        self.df["datetime_naive"] = self.df["datetime"].dt.tz_localize(None)

        for name, date_str in events.items():
            event_dt = pd.to_datetime(date_str)
            before_start = event_dt - timedelta(days=window_days)
            before_end = event_dt
            after_start = event_dt
            after_end = event_dt + timedelta(days=window_days)

            before_df = self.df[
                (self.df["datetime_naive"] >= before_start)
                & (self.df["datetime_naive"] < before_end)
            ]
            after_df = self.df[
                (self.df["datetime_naive"] >= after_start)
                & (self.df["datetime_naive"] < after_end)
            ]

            if len(before_df) < 5 or len(after_df) < 5:
                logger.warning(f"Not enough data for event '{name}'. Skipping.")
                continue

            event_graphs[name] = {
                "before": self._build_from_slice(before_df, f"{name}_before"),
                "after": self._build_from_slice(after_df, f"{name}_after"),
            }
            logger.info(
                f"[{name}] before: {event_graphs[name]['before'].number_of_nodes()} nodes | "
                f"after: {event_graphs[name]['after'].number_of_nodes()} nodes"
            )

        return event_graphs

    # ------------------------------------------------------------------
    # Full-period graph
    # ------------------------------------------------------------------

    def build_full_graph(self) -> nx.Graph:
        """Build a single graph from the entire dataset (useful for EDA)."""
        logger.info("Building full-period graph …")
        G = self._build_from_slice(self.df, label="full")
        logger.info(f"Full graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        return G

    # ------------------------------------------------------------------
    # Utility: get the largest connected component
    # ------------------------------------------------------------------

    @staticmethod
    def largest_component(G: nx.Graph) -> nx.Graph:
        """Return the subgraph induced by the largest connected component."""
        if G.is_directed():
            comps = nx.weakly_connected_components(G)
        else:
            comps = nx.connected_components(G)
        lcc = max(comps, key=len)
        return G.subgraph(lcc).copy()
