"""
data_collector.py  (final working version)
==========================================
Arctic Shift API confirmed behaviour:
  - max limit = 100 (250/500/1000 → 400 Bad Request)
  - after/before = date string 'YYYY-MM-DD' ✅ OR datetime 'YYYY-MM-DDTHH:MM:SS' ✅
  - Mixed formats (one date, one datetime) → 422 Unprocessable Entity ❌
  - sort param → 400 Bad Request ❌
  - API returns newest-first within the window

Pagination strategy: 7-day sliding windows.
  For each 7-day chunk, slide 'before' backward until < 100 records returned.
"""

import requests
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_URL   = "https://arctic-shift.photon-reddit.com/api"
BATCH_SIZE = 100   # confirmed max


class RedditCollector:

    def __init__(self, output_dir: str = "data/raw", delay: float = 0.6):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay   = delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "SNA-EchoChamber-Study/1.0"})

    # ── HTTP ──────────────────────────────────────────────────────────
    def _get(self, endpoint: str, params: dict) -> list:
        """GET one page; returns list of records (empty on failure)."""
        url = f"{BASE_URL}/{endpoint}"
        for attempt in range(4):
            try:
                r = self.session.get(url, params=params, timeout=30)
                r.raise_for_status()
                return r.json().get("data", [])
            except requests.RequestException as e:
                wait = 2 ** attempt
                logger.debug(f"  Attempt {attempt+1} failed: {e}. Retry in {wait}s …")
                time.sleep(wait)
        return []

    # ── Date helpers ──────────────────────────────────────────────────
    @staticmethod
    def _date_range(start: str, end: str):
        """Yield (window_after, window_before) pairs of 7-day chunks."""
        cur = datetime.strptime(start, "%Y-%m-%d")
        fin = datetime.strptime(end,   "%Y-%m-%d")
        while cur <= fin:
            nxt = min(cur + timedelta(days=6), fin)
            yield cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")
            cur = nxt + timedelta(days=1)

    @staticmethod
    def _ts_to_date(ts: int) -> str:
        return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")

    # ── Core fetcher (one 7-day window, sliding before) ───────────────
    def _fetch_window(self, endpoint: str, subreddit: str,
                      after: str, before: str) -> list:
        """
        Collect ALL records in [after, before] by sliding 'before' backwards.
        Both after/before are always date strings (no mixing).
        """
        all_records = []
        seen_ids    = set()
        current_before = before

        while True:
            params = {
                "subreddit": subreddit,
                "after":     after,
                "before":    current_before,
                "limit":     BATCH_SIZE,
            }
            recs = self._get(endpoint, params)

            if not recs:
                break

            new = [r for r in recs if r.get("id") not in seen_ids]
            for r in new:
                seen_ids.add(r["id"])
            all_records.extend(new)

            if len(recs) < BATCH_SIZE:
                break   # last page

            # Slide: API returns newest-first, so oldest = recs[-1]
            oldest_ts   = int(recs[-1].get("created_utc", 0))
            oldest_date = self._ts_to_date(oldest_ts)

            if oldest_date <= after:
                break   # can't go further back

            # Move before to the day of the oldest record (exclusive via -1 day)
            new_before = (datetime.strptime(oldest_date, "%Y-%m-%d")
                          ).strftime("%Y-%m-%d")
            if new_before == current_before:
                break
            current_before = new_before
            time.sleep(self.delay)

        return all_records

    # ── Public: collect comments ──────────────────────────────────────
    def collect_comments(self, subreddit: str, start: str, end: str) -> Path:
        filepath = self.output_dir / f"{subreddit}_comments_{start}_{end}.jsonl"
        if filepath.exists() and filepath.stat().st_size > 500:
            logger.info(f"Already exists: {filepath.name} — skipping.")
            return filepath
        filepath.unlink(missing_ok=True)

        windows = list(self._date_range(start, end))
        total   = 0
        logger.info(f"Collecting r/{subreddit} comments ({start}→{end}), {len(windows)} windows …")

        for w_after, w_before in tqdm(windows, desc=f"r/{subreddit}", unit="wk"):
            recs = self._fetch_window("comments/search", subreddit, w_after, w_before)
            if recs:
                with open(filepath, "a", encoding="utf-8") as f:
                    for r in recs:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                total += len(recs)
            time.sleep(self.delay)

        logger.info(f"✅ r/{subreddit} comments: {total:,} → {filepath.name}")
        return filepath

    # ── Public: collect posts ─────────────────────────────────────────
    def collect_posts(self, subreddit: str, start: str, end: str) -> Path:
        filepath = self.output_dir / f"{subreddit}_posts_{start}_{end}.jsonl"
        if filepath.exists() and filepath.stat().st_size > 500:
            logger.info(f"Already exists: {filepath.name} — skipping.")
            return filepath
        filepath.unlink(missing_ok=True)

        windows = list(self._date_range(start, end))
        total   = 0
        logger.info(f"Collecting r/{subreddit} posts ({start}→{end}), {len(windows)} windows …")

        for w_after, w_before in tqdm(windows, desc=f"r/{subreddit} posts", unit="wk"):
            recs = self._fetch_window("posts/search", subreddit, w_after, w_before)
            if recs:
                with open(filepath, "a", encoding="utf-8") as f:
                    for r in recs:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                total += len(recs)
            time.sleep(self.delay)

        logger.info(f"✅ r/{subreddit} posts: {total:,} → {filepath.name}")
        return filepath

    def collect_all(self, subreddits, start, end):
        results = {}
        for sub in subreddits:
            logger.info(f"\n{'='*50}\nr/{sub}\n{'='*50}")
            results[sub] = {
                "comments": self.collect_comments(sub, start, end),
                "posts":    self.collect_posts(sub, start, end),
            }
        return results
