"""
generate_demo_data.py
=====================
Generates realistic synthetic SNA data so you can demo ALL visualizations
BEFORE collecting real Reddit data. Run this once, then open the dashboard.

Usage:
    python3 generate_demo_data.py
"""

import numpy as np
import pandas as pd
import networkx as nx
import json
import os
import pickle
from pathlib import Path

np.random.seed(42)
rng = np.random.default_rng(42)

OUT = Path("data/processed")
GRAPH_OUT = OUT / "graphs"
REPORT_OUT = Path("report")
FIGURES_OUT = Path("figures")
for d in [OUT, GRAPH_OUT, REPORT_OUT, FIGURES_OUT]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# 1. Generate temporal metrics CSV (48 months: Jan 2021 – Dec 2024)
# ──────────────────────────────────────────────────────────────────────
periods = pd.period_range("2021-01", "2024-12", freq="M")
n = len(periods)

# Modularity: slight upward trend with noise + event dips
t = np.linspace(0, 1, n)
modularity = (
    0.38 + 0.10 * t
    + 0.04 * np.sin(2 * np.pi * t * 4)
    + rng.normal(0, 0.018, n)
)
# Dip at COP events (roughly months 10, 22, 34, 46)
for m in [9, 21, 33, 45]:
    if m < n:
        modularity[m] -= 0.035

# Assortativity: slightly negative, oscillating
assortativity = (
    -0.05 + 0.03 * np.sin(2 * np.pi * t * 3)
    + rng.normal(0, 0.012, n)
)

# Number of communities: grows slowly
num_communities = (
    12 + (t * 8).astype(int)
    + rng.integers(-2, 3, n)
)
num_communities = np.clip(num_communities, 8, 25)

nodes = rng.integers(800, 4500, n)
edges = (nodes * rng.uniform(1.5, 2.5, n)).astype(int)
density = edges / (nodes * (nodes - 1) / 2)
avg_clustering = 0.18 + 0.06 * t + rng.normal(0, 0.015, n)
mean_betweenness = 0.0008 + rng.normal(0, 0.0002, n)
mean_degree_centrality = 0.002 + rng.normal(0, 0.0003, n)

temporal_df = pd.DataFrame({
    "label": [str(p) for p in periods],
    "num_nodes": nodes,
    "num_edges": edges,
    "density": density,
    "modularity": np.clip(modularity, 0.2, 0.7),
    "num_communities": num_communities,
    "avg_community_size": (nodes / num_communities).astype(int),
    "degree_assortativity": assortativity,
    "avg_clustering": np.clip(avg_clustering, 0.05, 0.45),
    "mean_betweenness": np.abs(mean_betweenness),
    "mean_degree_centrality": mean_degree_centrality,
    "max_degree_centrality": mean_degree_centrality * rng.uniform(8, 14, n),
}).set_index("label")

temporal_df.to_csv(REPORT_OUT / "temporal_metrics.csv")
print(f"✅ Saved temporal_metrics.csv  ({len(temporal_df)} rows)")

# ──────────────────────────────────────────────────────────────────────
# 2. Generate event comparison CSV
# ──────────────────────────────────────────────────────────────────────
events = [
    "COP26_2021",
    "IPCC_AR6_WG1_2021",
    "COP27_2022",
    "IPCC_AR6_WG3_2022",
    "COP28_2023",
    "IPCC_AR6_SYR_2023",
    "USInflationReductionAct_2022",
]

rows = []
for ev in events:
    bm = rng.uniform(0.35, 0.55)
    am = bm + rng.uniform(-0.06, 0.04)
    ba = rng.uniform(-0.08, 0.02)
    aa = ba + rng.uniform(-0.03, 0.03)
    rows.append({
        "event": ev,
        "before_modularity": bm,
        "after_modularity": am,
        "delta_modularity": am - bm,
        "before_degree_assortativity": ba,
        "after_degree_assortativity": aa,
        "delta_degree_assortativity": aa - ba,
        "before_num_communities": int(rng.uniform(10, 18)),
        "after_num_communities": int(rng.uniform(10, 20)),
        "before_num_nodes": int(rng.uniform(1000, 3000)),
        "after_num_nodes": int(rng.uniform(1000, 3500)),
    })

event_df = pd.DataFrame(rows).set_index("event")
event_df["delta_num_communities"] = event_df["after_num_communities"] - event_df["before_num_communities"]
event_df.to_csv(REPORT_OUT / "event_comparison.csv")
print(f"✅ Saved event_comparison.csv  ({len(event_df)} events)")

# ──────────────────────────────────────────────────────────────────────
# 3. Generate bridge users CSV
# ──────────────────────────────────────────────────────────────────────
usernames = [
    "climate_hawk42", "greenplanet_r", "arctic_watcher", "co2tracker",
    "renewables_now", "ipcc_reader", "copXX_live", "climatedebate",
    "skeptic_bridge", "data_planet", "carbonwatch", "scicomm_r",
    "policy_watcher", "methane_alert", "netzerohope", "energytransit",
    "fossilfuels_x", "solarwind99", "tipping_pt", "bluemarble_r",
]
bridge_df = pd.DataFrame({
    "user": usernames,
    "community": rng.integers(0, 8, len(usernames)),
    "betweenness": np.sort(rng.uniform(0.001, 0.045, len(usernames)))[::-1],
    "degree_centrality": rng.uniform(0.005, 0.06, len(usernames)),
    "degree": rng.integers(20, 300, len(usernames)),
    "num_communities_bridged": rng.integers(2, 7, len(usernames)),
    "is_bridge": [True] * len(usernames),
})
bridge_df.to_csv(REPORT_OUT / "bridge_users_2024-12.csv", index=False)
print(f"✅ Saved bridge_users CSV  ({len(bridge_df)} users)")

# Persistent bridges
persistent_df = pd.DataFrame({
    "user": usernames[:10],
    "bridge_periods": rng.integers(6, 36, 10),
    "avg_betweenness": rng.uniform(0.005, 0.04, 10),
}).sort_values("bridge_periods", ascending=False)
persistent_df.to_csv(REPORT_OUT / "persistent_bridge_users.csv", index=False)
print(f"✅ Saved persistent_bridge_users.csv")

# ──────────────────────────────────────────────────────────────────────
# 4. Generate synthetic monthly graphs (for network snapshots)
# ──────────────────────────────────────────────────────────────────────
print("\nGenerating synthetic NetworkX graphs...")
monthly_graphs = {}
sample_periods = [str(p) for p in periods[::6]]  # every 6 months (smaller set for speed)

for label in sample_periods:
    n_nodes = rng.integers(200, 600)
    # Stochastic block model: 4 communities
    sizes = [n_nodes // 4] * 4
    sizes[-1] = n_nodes - sum(sizes[:-1])
    p_in  = 0.05   # within-community connection probability
    p_out = 0.003  # between-community connection probability
    probs = [[p_in if i == j else p_out for j in range(4)] for i in range(4)]
    G = nx.stochastic_block_model(sizes, probs, seed=42)
    G = nx.Graph(G)  # remove self-loops
    monthly_graphs[label] = G
    print(f"  {label}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

with open(GRAPH_OUT / "monthly_graphs.pkl", "wb") as f:
    pickle.dump(monthly_graphs, f)
print(f"\n✅ Saved monthly_graphs.pkl  ({len(monthly_graphs)} graphs)")

# ──────────────────────────────────────────────────────────────────────
# 5. Generate interpretation text
# ──────────────────────────────────────────────────────────────────────
interp = """
=== INTERPRETATION & FINDINGS ===

MODULARITY TREND : 📈 increasing
  • Mean: 0.4312  |  Slope: +0.00083/month  |  R²: 0.712
  • Echo chambers appear to be growing stronger over time.
  • The climate discourse on Reddit is becoming more siloed from 2021 → 2024.

ASSORTATIVITY TREND : slightly negative (disassortative)
  • Mean: -0.042  |  Range: -0.09 to +0.03
  • Hub users (high-degree) tend to connect to low-degree nodes.
  • This is typical for discussion forums — influential users bridge many communities.

CLIMATE EVENTS EFFECT :
  • Average Δ Modularity (after − before) = -0.021
  • Major events (COP26, COP27, COP28) slightly WEAKEN community separation.
  • Interpretation: Climate events draw new participants who cross community lines.

COMMUNITIES :
  • Average communities per month: 15.3
  • Range: 8 – 25 communities detected
  • Communities grow in number and diversity over the 4-year period.

BRIDGE USERS :
  • 10 users consistently bridge communities across 6+ months.
  • These users are likely science communicators, journalists, or moderators.

COMPARISON WITH TWITTER (Treen et al., 2022):
  • Reddit shows LOWER polarization than Twitter climate debates.
  • Modularity on Reddit (~0.43) vs Twitter estimates (~0.55–0.65).
  • Longer-form Reddit discussions appear to encourage more cross-community dialogue.
"""
with open(REPORT_OUT / "interpretation.txt", "w") as f:
    f.write(interp)
print("✅ Saved interpretation.txt")

print("\n🎉 All demo data generated! You can now open the dashboard.")
print("   Run: python3 -m http.server 8080  then visit http://localhost:8080/dashboard.html")
