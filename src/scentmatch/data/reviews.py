"""Review texts for the k-core (Data Engineer, Week 3).

The rating-only file tells us *how many stars* a user gave, not *why*. To check whether a
1-3 star rating really means "I dislike this scent" (or "it leaked", "it does not last",
"it is a fake"...), we need the review text. The full review file is 11 GB, so we stream it
from Hugging Face and keep only the reviews written by our k-core users on our k-core
perfumes (about 106k reviews). Nothing is written to disk except the filtered result.

Output: data/raw/kcore_reviews.parquet
        (user_id, parent_asin, rating, timestamp, title, text)

Run:  python -m scentmatch.data.reviews            (about 8 min, network bound)
      python -m scentmatch.data.reviews --file X   (read a local copy of the jsonl instead)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time

import pandas as pd

from scentmatch.config import load_config

URL = ("https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
       "raw/review_categories/Beauty_and_Personal_Care.jsonl")


def _lines(local_file: str | None):
    if local_file:
        with open(local_file, "rb") as f:
            yield from f
        return
    proc = subprocess.Popen(["curl", "-sL", "--retry", "5", URL], stdout=subprocess.PIPE, bufsize=1 << 20)
    assert proc.stdout is not None
    yield from proc.stdout
    if proc.wait() != 0:
        raise RuntimeError("download of the review file failed")


def fetch_kcore_reviews(cfg: dict, local_file: str | None = None) -> pd.DataFrame:
    proc_dir = cfg["paths"]["processed_dir"]
    asins = set(pd.read_parquet(proc_dir / "items.parquet", columns=["parent_asin"])["parent_asin"])
    users = set(pd.read_parquet(proc_dir / "users.parquet", columns=["user_id"])["user_id"])
    keep, n, t0 = [], 0, time.time()
    for raw in _lines(local_file):
        n += 1
        if n % 2_000_000 == 0:
            print(f"  {n/1e6:.0f}M lines read, {len(keep):,} kept, {time.time() - t0:.0f}s", flush=True)
        # cheap pre-filter on the raw bytes before parsing the JSON
        i = raw.find(b'"parent_asin": "')
        if i < 0 or raw[i + 16:i + 26].decode("ascii", "ignore") not in asins:
            continue
        rec = json.loads(raw)
        if rec["user_id"] in users and rec["parent_asin"] in asins:
            keep.append({k: rec.get(k) for k in ("user_id", "parent_asin", "rating", "timestamp", "title", "text")})
    df = pd.DataFrame(keep)
    # one review per (user, perfume): the most recent, as in the rating pipeline
    df = df.sort_values("timestamp").drop_duplicates(["user_id", "parent_asin"], keep="last").reset_index(drop=True)
    print(f"read {n:,} lines in {time.time() - t0:.0f}s, kept {len(df):,} reviews")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=None, help="local copy of Beauty_and_Personal_Care.jsonl")
    args = ap.parse_args()
    cfg = load_config()
    df = fetch_kcore_reviews(cfg, args.file)
    out = cfg["paths"]["raw_dir"] / "kcore_reviews.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
