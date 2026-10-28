# Echo Chambers in Climate Change Discussions on Reddit
### CSE 356 – Social Network Analysis Project

> **Research Question:** Are echo chambers in Reddit climate discussions becoming stronger or weaker over time, and how do major climate events affect community structure?

---

## 📁 Project Structure

```
SNA/
├── data/
│   ├── raw/                    # Downloaded JSONL files (not in git)
│   └── processed/
│       ├── comments_clean.parquet   # Cleaned edge table
│       └── graphs/
│           ├── monthly_graphs.pkl   # Dict of monthly NetworkX graphs
│           └── event_graphs.pkl     # Before/after event graphs
├── notebooks/
│   ├── 01_data_collection.ipynb    # Download from Arctic Shift API
│   ├── 02_cleaning.ipynb           # Clean, filter bots, resolve parents
│   ├── 03_network_building.ipynb   # Build NetworkX graphs
│   ├── 04_temporal_analysis.ipynb  # Louvain + all SNA metrics
│   └── 05_visualization.ipynb      # Final figures + interpretation
├── src/
│   ├── __init__.py
│   ├── data_collector.py           # Arctic Shift API client
│   ├── cleaner.py                  # Data cleaning pipeline
│   ├── network_builder.py          # Graph construction
│   ├── metrics.py                  # SNA metrics (Louvain, centrality, etc.)
│   └── visualizer.py               # All plotting functions
├── figures/                         # Generated PNG files
│   └── gephi_exports/               # GraphML files for Gephi
├── report/                          # CSV tables + interpretation text
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
python -m venv sna_env
source sna_env/bin/activate       # Windows: sna_env\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch Jupyter
```bash
cd notebooks/
jupyter notebook
```

### 3. Run notebooks in order
| Notebook | Purpose | Est. Time |
|----------|---------|-----------|
| `01_data_collection.ipynb` | Download Reddit data from Arctic Shift | 30 min – several hours |
| `02_cleaning.ipynb` | Clean & process data | 5–15 min |
| `03_network_building.ipynb` | Build NetworkX graphs | 10–30 min |
| `04_temporal_analysis.ipynb` | Louvain + metrics (main analysis) | 20–60 min |
| `05_visualization.ipynb` | Final plots + interpretation | 10–20 min |

---

## 📊 Metrics Computed

| Metric | Description |
|--------|-------------|
| **Modularity** | How separated communities are (0–1; higher = stronger echo chambers) |
| **Assortativity** | How much users prefer similar-degree users |
| **Betweenness Centrality** | Who are the key bridge users |
| **Degree Centrality** | Who has the most connections |
| **Avg. Clustering** | How tightly-knit local groups are |
| **Number of Communities** | Count of distinct user clusters per period |

---

## 🌍 Climate Events Tracked

| Event | Date |
|-------|------|
| IPCC AR6 Working Group 1 | 2021-08-09 |
| COP26 | 2021-11-01 |
| IPCC AR6 Working Group 3 | 2022-04-04 |
| US Inflation Reduction Act | 2022-08-16 |
| COP27 | 2022-11-07 |
| IPCC AR6 Synthesis Report | 2023-03-20 |
| COP28 | 2023-11-30 |

---

## 🔧 Common Issues

| Problem | Solution |
|---------|----------|
| API returns empty data | Check date range; Arctic Shift may have gaps for some subreddits |
| `python-louvain` import error | `pip install python-louvain` (the package installs as `community`) |
| Memory error on large graphs | Process month-by-month; never load all at once |
| Betweenness too slow | Reduce `betweenness_k` in `GraphMetrics(betweenness_k=50)` |
| `parquet` format error | `pip install pyarrow` |

---

## 📚 References

- Treen et al. (2022). *Online misinformation about climate change*. WIREs Climate Change.
- Blondel et al. (2008). *Fast unfolding of communities in large networks.* J. Stat. Mech.
- Arctic Shift: https://arctic-shift.photon-reddit.com
- NetworkX docs: https://networkx.org
- Gephi: https://gephi.org

---

*Single-member project by a BTech CSE student, 2024–2025*
