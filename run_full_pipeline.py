"""
run_full_pipeline.py
====================
ONE-CLICK script: Collects REAL Reddit data → Cleans → Builds networks
→ Computes all SNA metrics → Saves CSVs → Launches dashboard in browser.

Timeline estimate:
  - Data collection  : ~20–40 min (depends on internet)
  - Cleaning         : ~2–5 min
  - Network building : ~3–8 min
  - Metrics (Louvain): ~10–20 min
  - Total            : ~35–75 min

Usage:
    python3 run_full_pipeline.py

Progress is printed at every step. Safe to Ctrl+C and re-run — already
downloaded files are skipped automatically.
"""

import sys, os, time, pickle, logging
from pathlib import Path

# ── ensure we can import src/ ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import networkx as nx

from src.data_collector import RedditCollector
from src.cleaner       import DataCleaner
from src.network_builder import NetworkBuilder, CLIMATE_EVENTS
from src.metrics       import GraphMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION  ← Edit here if needed
# ══════════════════════════════════════════════════════════════════════

# ── Scope reduced for evaluation speed ──────────────────────────────
# Using 6 months around COP26/COP27 — real data, finishes in ~15–20 min
# Expand SUBREDDITS / dates after evaluation if you want more data
SUBREDDITS = [
    "climate",          # Main climate subreddit (~2–4k comments/month)
    "climatechange",    # Activist-focused      (~1–2k comments/month)
]

# 4 months around COP27 (Nov 2022) — REAL data, done in ~20–30 min
START_DATE = "2022-09-01"
END_DATE   = "2022-12-31"

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
REPORT_DIR    = "report"
FIGURES_DIR   = "figures"
GRAPH_DIR     = "data/processed/graphs"

# Louvain betweenness approximation (50 is fine for eval)
BETWEENNESS_K = 50

# ══════════════════════════════════════════════════════════════════════

def banner(msg):
    print(f"\n{'═'*60}")
    print(f"  {msg}")
    print(f"{'═'*60}")

# ── Create dirs ───────────────────────────────────────────────────
for d in [RAW_DIR, PROCESSED_DIR, REPORT_DIR, FIGURES_DIR, GRAPH_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# STEP 1: DATA COLLECTION
# ══════════════════════════════════════════════════════════════════════
banner("STEP 1/5 — Collecting real Reddit data from Arctic Shift")

collector = RedditCollector(output_dir=RAW_DIR, delay=1.2)

print(f"Subreddits : {SUBREDDITS}")
print(f"Period     : {START_DATE}  →  {END_DATE}")
print(f"Output dir : {RAW_DIR}/")
print()

for sub in SUBREDDITS:
    log.info(f"Fetching r/{sub} comments …")
    collector.collect_comments(sub, START_DATE, END_DATE)
    log.info(f"Fetching r/{sub} posts …")
    collector.collect_posts(sub, START_DATE, END_DATE)

# ══════════════════════════════════════════════════════════════════════
# STEP 2: CLEANING
# ══════════════════════════════════════════════════════════════════════
banner("STEP 2/5 — Cleaning data")

cleaner = DataCleaner(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR)

raw_comments = cleaner.load_comments(SUBREDDITS)
raw_posts    = cleaner.load_posts(SUBREDDITS)

log.info(f"Raw comments: {len(raw_comments):,}")

clean_comments = cleaner.clean(raw_comments)

posts_for_map = pd.DataFrame()
if not raw_posts.empty and "id" in raw_posts.columns:
    posts_for_map = raw_posts[["id", "author"]].dropna()

id_to_author = cleaner.build_id_to_author(clean_comments, posts_for_map)
df = cleaner.add_parent_author(clean_comments, id_to_author)

log.info(f"Clean edge table: {len(df):,} rows, {df['author'].nunique():,} unique users")

# Save parquet
parquet_path = Path(PROCESSED_DIR) / "comments_clean.parquet"
df.to_parquet(parquet_path, index=False)
log.info(f"Saved → {parquet_path}")

# ══════════════════════════════════════════════════════════════════════
# STEP 3: BUILD NETWORKS
# ══════════════════════════════════════════════════════════════════════
banner("STEP 3/5 — Building monthly & event networks")

builder = NetworkBuilder(df, directed=False)

log.info("Building monthly graphs …")
monthly_graphs = builder.build_monthly_graphs()
log.info(f"Monthly graphs: {len(monthly_graphs)}")

log.info("Building event graphs (before/after 60 days) …")
event_graphs = builder.build_event_graphs(window_days=60)
log.info(f"Event graphs: {len(event_graphs)}")

with open(Path(GRAPH_DIR) / "monthly_graphs.pkl", "wb") as f:
    pickle.dump(monthly_graphs, f)
with open(Path(GRAPH_DIR) / "event_graphs.pkl", "wb") as f:
    pickle.dump(event_graphs, f)
log.info("Graphs saved.")

# ══════════════════════════════════════════════════════════════════════
# STEP 4: COMPUTE METRICS (Louvain + SNA metrics)
# ══════════════════════════════════════════════════════════════════════
banner("STEP 4/5 — Computing SNA metrics (Louvain community detection)")

gm = GraphMetrics(betweenness_k=BETWEENNESS_K, top_users=50)

# 4a. Temporal metrics
log.info("Computing metrics for all monthly graphs (this takes the most time) …")
temporal_df = gm.compute_temporal(monthly_graphs)
temporal_df.to_csv(Path(REPORT_DIR) / "temporal_metrics.csv")
log.info(f"Temporal metrics saved → {REPORT_DIR}/temporal_metrics.csv")
print(temporal_df[["num_nodes","num_edges","modularity","num_communities","degree_assortativity"]].to_string())

# 4b. Event comparison
log.info("Computing event comparison metrics …")
event_df = gm.compare_events(event_graphs)
event_df.to_csv(Path(REPORT_DIR) / "event_comparison.csv")
log.info(f"Event comparison saved → {REPORT_DIR}/event_comparison.csv")

# 4c. Bridge users (latest period)
latest_label = sorted(monthly_graphs.keys())[-1]
G_latest     = monthly_graphs[latest_label]
stats_latest = gm.compute_all(G_latest, label=latest_label)
partition    = stats_latest.get("_partition", {})

bridge_df = gm.find_bridge_users(G_latest, partition, top_n=50)
bridge_df.to_csv(Path(REPORT_DIR) / f"bridge_users_{latest_label}.csv", index=False)
log.info(f"Bridge users saved → {REPORT_DIR}/bridge_users_{latest_label}.csv")

# 4d. Persistent bridge users
log.info("Finding persistent bridge users across all periods …")
persistent_df = gm.find_persistent_bridges(monthly_graphs, min_periods=3, top_n=20)
persistent_df.to_csv(Path(REPORT_DIR) / "persistent_bridge_users.csv", index=False)
log.info(f"Persistent bridges saved → {REPORT_DIR}/persistent_bridge_users.csv")

# 4e. Summary statistics
summary = temporal_df[[
    "modularity","num_communities","degree_assortativity",
    "avg_clustering","density","mean_betweenness"
]].describe()
summary.to_csv(Path(REPORT_DIR) / "summary_statistics.csv")
log.info(f"Summary statistics saved → {REPORT_DIR}/summary_statistics.csv")

# ══════════════════════════════════════════════════════════════════════
# STEP 5: UPDATE DASHBOARD WITH REAL DATA & LAUNCH
# ══════════════════════════════════════════════════════════════════════
banner("STEP 5/5 — Injecting real data into dashboard & launching browser")

def df_to_js_array(series):
    """Convert a pandas Series to a compact JS array string."""
    vals = series.tolist()
    clean = []
    for v in vals:
        if isinstance(v, (int, np.integer)):
            clean.append(int(v))
        elif isinstance(v, (float, np.floating)):
            clean.append(round(float(v), 6) if not np.isnan(v) else 0)
        else:
            clean.append(str(v))
    return str(clean).replace("'", '"')

def series_to_js(series):
    return df_to_js_array(series)

# ── Build the JavaScript data block from real CSVs ────────────────
temp  = pd.read_csv(Path(REPORT_DIR) / "temporal_metrics.csv", index_col=0)
ev    = pd.read_csv(Path(REPORT_DIR) / "event_comparison.csv", index_col=0)
br_path = sorted(Path(REPORT_DIR).glob("bridge_users_*.csv"))[-1]
br    = pd.read_csv(br_path)
pers  = pd.read_csv(Path(REPORT_DIR) / "persistent_bridge_users.csv")

periods_js   = str(list(temp.index)).replace("'",'"')
mod_js       = series_to_js(temp["modularity"])
ass_js       = series_to_js(temp["degree_assortativity"])
com_js       = series_to_js(temp["num_communities"].astype(int))
nod_js       = series_to_js(temp["num_nodes"].astype(int))
den_js       = series_to_js(temp["density"])
clu_js       = series_to_js(temp["avg_clustering"])
bet_js       = series_to_js(temp["mean_betweenness"])

ev_labels_js = str(list(ev.index)).replace("'",'"')

def ev_col(col):
    if col in ev.columns:
        return series_to_js(ev[col].fillna(0))
    return "[]"

ev_bmod_js  = ev_col("before_modularity")
ev_amod_js  = ev_col("after_modularity")
ev_dmod_js  = ev_col("delta_modularity")
ev_bcom_js  = ev_col("before_num_communities")
ev_acom_js  = ev_col("after_num_communities")
ev_bass_js  = ev_col("before_degree_assortativity")
ev_aass_js  = ev_col("after_degree_assortativity")

# Bridge user arrays
br_names_js = str(list(br["user"].head(20))).replace("'",'"') if not br.empty else "[]"
br_bet_js   = str([round(float(x),5) for x in br["betweenness"].head(20)]) if not br.empty else "[]"
br_deg_js   = str([int(x) for x in br["degree"].head(20)]) if not br.empty else "[]"
br_com_js   = str([int(x) for x in br["num_communities_bridged"].head(20)]) if not br.empty else "[]"
br_cid_js   = str([int(x % 12) for x in br["community"].head(20)]) if not br.empty else "[]"

pers_names_js = str(list(pers["user"].head(10))).replace("'",'"') if not pers.empty else "[]"
pers_mon_js   = str([int(x) for x in pers["bridge_periods"].head(10)]) if not pers.empty else "[]"
pers_bet_js   = str([round(float(x),5) for x in pers["avg_betweenness"].head(10)]) if not pers.empty else "[]"

# Network snapshots: use up to 8 actual period labels
net_periods = sorted(monthly_graphs.keys())
step_n = max(1, len(net_periods)//8)
avail_periods = net_periods[::step_n][:8]
avail_js = str(avail_periods).replace("'",'"')

# Key stats for header badges
n_periods = len(temp)
n_subs    = len(SUBREDDITS)

real_data_js = f"""
// ═══════════════════════════════════════════════════
//  REAL DATA — injected by run_full_pipeline.py
//  Subreddits: {SUBREDDITS}   Period: {START_DATE}–{END_DATE}
// ═══════════════════════════════════════════════════
const PERIODS = {periods_js};
const N = PERIODS.length;
const MOD = {mod_js};
const ASS = {ass_js};
const COM = {com_js};
const NOD = {nod_js};
const DEN = {den_js};
const CLU = {clu_js};
const BET = {bet_js};

const EV_LABELS = {ev_labels_js};
const EV_BMOD = {ev_bmod_js};
const EV_AMOD = {ev_amod_js};
const EV_DMOD = {ev_dmod_js};
const EV_BCOM = {ev_bcom_js};
const EV_ACOM = {ev_acom_js};
const EV_BASS = {ev_bass_js};
const EV_AASS = {ev_aass_js};

const BRIDGE_NAMES = {br_names_js};
const BR_BET = {br_bet_js};
const BR_DEG = {br_deg_js};
const BR_COM = {br_com_js};
const BR_CID = {br_cid_js};

const PERS_USERS = {pers_names_js};
const PERS_MON   = {pers_mon_js};
const PERS_BET   = {pers_bet_js};

const AVAIL_PERIODS = {avail_js};
const DATA_SOURCE   = "Real Reddit Data ({START_DATE} to {END_DATE})";
const N_SUBREDDITS  = "{n_subs} subreddits: {', '.join(['r/'+s for s in SUBREDDITS])}";
const N_PERIODS_LBL = "{n_periods} monthly snapshots";
"""

# ── Read dashboard template, inject real data ────────────────────
dash_path = Path(__file__).parent / "dashboard.html"
if not dash_path.exists():
    log.error("dashboard.html not found! Run the setup again.")
    sys.exit(1)

html_content = dash_path.read_text(encoding="utf-8")

# Replace the synthetic data section with real data
# We look for the SEEDED PRNG section and replace up to CHART.JS DEFAULTS
INJECT_MARKER_START = "// ─── SEEDED PRNG ──────────────────────────────────────────────────"
INJECT_MARKER_END   = "// ─── CHART.JS DEFAULTS ────────────────────────────────────────────"

if INJECT_MARKER_START in html_content and INJECT_MARKER_END in html_content:
    before = html_content.split(INJECT_MARKER_START)[0]
    after  = html_content.split(INJECT_MARKER_END, 1)[1]

    # Remove the helper functions that are no longer needed (mkRng, normal, etc.)
    # We keep everything after CHART.JS DEFAULTS
    # Also define the arrays referenced in chart code (BRIDGE_NAMES → already in real_data_js)
    new_html = before + real_data_js + "\n" + INJECT_MARKER_END + after
    log.info("Injecting real data into dashboard …")
else:
    # Fallback: inject as a <script> block at the top of existing script
    log.warning("Marker not found — prepending real data to script block.")
    new_html = html_content.replace(
        "<script>\n",
        f"<script>\n{real_data_js}\n",
        1
    )

# Update header badge to say "Real Data"
new_html = new_html.replace(
    'class="badge badge-amber">48 Monthly Snapshots',
    f'class="badge badge-amber">{n_periods} Real Snapshots'
)
new_html = new_html.replace(
    'class="badge badge-green">4 Subreddits',
    f'class="badge badge-green">{n_subs} Subreddits'
)

out_html = Path(__file__).parent / "dashboard_real.html"
out_html.write_text(new_html, encoding="utf-8")
log.info(f"Real-data dashboard saved → {out_html}")

# ── Open in browser ────────────────────────────────────────────────
import subprocess, platform
if platform.system() == "Darwin":
    subprocess.Popen(["open", str(out_html)])
elif platform.system() == "Windows":
    os.startfile(str(out_html))
else:
    subprocess.Popen(["xdg-open", str(out_html)])

# ══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
banner("✅ PIPELINE COMPLETE")

print(f"""
  Subreddits collected : {SUBREDDITS}
  Date range           : {START_DATE}  →  {END_DATE}
  Monthly graphs       : {len(monthly_graphs)}
  Event graphs         : {len(event_graphs)}
  Avg modularity       : {temp['modularity'].mean():.4f}
  Avg assortativity    : {temp['degree_assortativity'].mean():.4f}
  Avg communities/mo   : {temp['num_communities'].mean():.1f}

  Dashboard (real data): dashboard_real.html   ← opened in browser
  Raw files            : data/raw/
  Processed data       : data/processed/
  CSVs for report      : report/
  Figures              : figures/

  To re-open dashboard:
    open dashboard_real.html
""")
