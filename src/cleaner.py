"""
cleaner.py
==========
Loads raw JSONL comment/post files, cleans them, and saves a processed
Parquet (or CSV) file ready for network construction.

Usage:
    from src.cleaner import DataCleaner
    cleaner = DataCleaner(raw_dir="data/raw", processed_dir="data/processed")
    df = cleaner.clean_and_merge(subreddits=["climate","climatechange"])
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Bot / deleted usernames to remove (extend as needed)
# ------------------------------------------------------------------
KNOWN_BOTS = {
    "[deleted]",
    "[removed]",
    "AutoModerator",
    "reddit",
    "BotTerminator",
    "anti-gif-bot",
    "imgurtranscriber",
    "WikiTextBot",
    "SnapshillBot",
    "TotesMessenger",
    "video_descriptionbot",
    "RepostSleuthBot",
    "RemindMeBot",
    "url-bot",
    "bot",
    "Bot",
}

# Users who post suspiciously many times (likely bots/spam accounts)
BOT_ACTIVITY_THRESHOLD = 5000  # max comments per user across entire dataset


class DataCleaner:
    """Loads raw JSONL files, cleans, and persists a unified DataFrame."""

    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_jsonl(self, filepath: Path) -> pd.DataFrame:
        """Load a JSONL file into a DataFrame."""
        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Reading {filepath.name}", unit=" lines"):
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return pd.DataFrame(records)

    def load_comments(self, subreddits: list, start: str = None, end: str = None) -> pd.DataFrame:
        """
        Load all comment JSONL files for the given subreddits.
        File naming convention: {subreddit}_comments_{start}_{end}.jsonl
        """
        dfs = []
        for path in self.raw_dir.glob("*_comments_*.jsonl"):
            sub = path.stem.split("_comments_")[0]
            if subreddits and sub not in subreddits:
                continue
            logger.info(f"Loading {path.name} …")
            dfs.append(self._load_jsonl(path))
        if not dfs:
            raise FileNotFoundError(
                f"No comment JSONL files found in {self.raw_dir} for subreddits {subreddits}"
            )
        return pd.concat(dfs, ignore_index=True)

    def load_posts(self, subreddits: list) -> pd.DataFrame:
        """Load all post JSONL files for the given subreddits."""
        dfs = []
        for path in self.raw_dir.glob("*_posts_*.jsonl"):
            sub = path.stem.split("_posts_")[0]
            if subreddits and sub not in subreddits:
                continue
            logger.info(f"Loading {path.name} …")
            dfs.append(self._load_jsonl(path))
        if not dfs:
            logger.warning("No post JSONL files found. Continuing without post data.")
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    def _standardise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only the columns we need and rename for consistency."""
        needed = {
            "id": "id",
            "author": "author",
            "parent_id": "parent_id",
            "link_id": "link_id",
            "created_utc": "created_utc",
            "subreddit": "subreddit",
            "body": "body",
            "score": "score",
        }
        # Rename if present
        df = df.rename(columns={k: v for k, v in needed.items() if k in df.columns})
        # Drop extra columns
        keep = [c for c in needed.values() if c in df.columns]
        return df[keep].copy()

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full cleaning pipeline:
        1. Standardise columns
        2. Drop known bots / deleted users
        3. Remove high-activity suspected bots
        4. Parse timestamps
        5. Build time features (year_month, year, quarter)
        6. Keep only reply comments (parent_id starts with t1_)
        7. Deduplicate by comment id
        """
        logger.info(f"Starting cleaning — {len(df):,} raw rows")

        df = self._standardise_columns(df)

        # 1. Drop rows without essential fields
        df = df.dropna(subset=["id", "author", "created_utc"])

        # 2. Remove known bots / deleted
        df = df[~df["author"].isin(KNOWN_BOTS)]
        df = df[~df["author"].str.lower().str.contains("bot$", na=False)]

        logger.info(f"After bot/deleted removal: {len(df):,} rows")

        # 3. Remove high-activity users (suspected bots)
        if "author" in df.columns:
            activity = df["author"].value_counts()
            legit = activity[activity <= BOT_ACTIVITY_THRESHOLD].index
            df = df[df["author"].isin(legit)]
            logger.info(f"After high-activity filter: {len(df):,} rows")

        # 4. Parse timestamp
        df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")
        df = df.dropna(subset=["created_utc"])
        df["datetime"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)

        # 5. Time features
        df["year_month"] = df["datetime"].dt.to_period("M")
        df["year"] = df["datetime"].dt.year
        df["quarter"] = df["datetime"].dt.to_period("Q")

        # 6. Keep only reply comments (parent is another comment, not a post)
        if "parent_id" in df.columns:
            df = df[df["parent_id"].str.startswith("t1_", na=False)]
            logger.info(f"After keeping only reply comments (t1_): {len(df):,} rows")

        # 7. Deduplicate
        df = df.drop_duplicates(subset=["id"])
        logger.info(f"After deduplication: {len(df):,} rows")

        df = df.reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # ID → Author mapping (needed to resolve parent author)
    # ------------------------------------------------------------------

    def build_id_to_author(self, comments_df: pd.DataFrame, posts_df: pd.DataFrame = None) -> dict:
        """
        Build a global {comment_id → author} dictionary.
        Also includes post authors so top-level comments can resolve their parent.
        """
        mapping = {}
        if "id" in comments_df.columns and "author" in comments_df.columns:
            mapping.update(dict(zip(comments_df["id"], comments_df["author"])))
        if posts_df is not None and not posts_df.empty:
            if "id" in posts_df.columns and "author" in posts_df.columns:
                mapping.update(dict(zip(posts_df["id"], posts_df["author"])))
        logger.info(f"ID→Author map built: {len(mapping):,} entries")
        return mapping

    def add_parent_author(self, df: pd.DataFrame, id_to_author: dict) -> pd.DataFrame:
        """
        Add a 'parent_author' column by looking up parent_id in id_to_author.
        parent_id format: 't1_XXXXX' → strip 't1_' prefix.
        """
        df = df.copy()
        df["parent_comment_id"] = df["parent_id"].str.replace("t1_", "", regex=False)
        df["parent_author"] = df["parent_comment_id"].map(id_to_author)
        # Drop rows where parent author is unknown or same as author (self-reply)
        df = df[df["parent_author"].notna()]
        df = df[df["parent_author"] != df["author"]]
        # Also filter parent bots
        df = df[~df["parent_author"].isin(KNOWN_BOTS)]
        logger.info(f"After parent-author resolution: {len(df):,} rows with valid edges")
        return df

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def clean_and_merge(self, subreddits: list, save: bool = True) -> pd.DataFrame:
        """
        End-to-end: load → clean → resolve parent authors → save.

        Returns the cleaned edge-ready DataFrame.
        """
        raw_comments = self.load_comments(subreddits)
        raw_posts = self.load_posts(subreddits)

        # Clean comments
        clean_comments = self.clean(raw_comments)

        # Clean posts (minimal — just need id & author)
        if not raw_posts.empty:
            posts_df = raw_posts[["id", "author"]].dropna() if "id" in raw_posts else pd.DataFrame()
        else:
            posts_df = pd.DataFrame()

        # Build mapping and resolve parent authors
        id_to_author = self.build_id_to_author(clean_comments, posts_df)
        df = self.add_parent_author(clean_comments, id_to_author)

        if save:
            out_path = self.processed_dir / "comments_clean.parquet"
            df.to_parquet(out_path, index=False)
            logger.info(f"Saved cleaned data → {out_path}")

            # Also save a lightweight CSV for quick inspection
            csv_path = self.processed_dir / "comments_clean_sample.csv"
            df.sample(min(5000, len(df))).to_csv(csv_path, index=False)
            logger.info(f"Saved 5k sample CSV → {csv_path}")

        return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    SUBREDDITS = ["climate", "climatechange", "environment", "Collapse"]
    cleaner = DataCleaner(raw_dir="data/raw", processed_dir="data/processed")
    df = cleaner.clean_and_merge(subreddits=SUBREDDITS)
    print(df.head())
    print(df.dtypes)
    print(f"\nFinal dataset: {len(df):,} reply edges, {df['author'].nunique():,} unique users")
