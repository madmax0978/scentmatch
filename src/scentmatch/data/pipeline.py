"""V1 data pipeline (Data Engineer).

    raw Amazon Reviews 2023 (Beauty_and_Personal_Care)
      -> 1. fragrance items (categories contain "Fragrance"), merged by parent_asin
      -> 2. note extraction from title / description / features / Scent field
      -> 3. ratings >= 4 as positives, iterative k-core (k = 3)
      -> 4. leave-last-out temporal split (train / val / test)
      -> 5. parquet files in data/processed + catalogue loaded into SQLite

Run:  python -m scentmatch.data.pipeline
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scentmatch.config import load_config
from scentmatch.data.vocab import NOT_A_NOTE_AFTER, build_vocabulary, load_vocabulary, save_vocabulary

META_FILE = "meta_Beauty_and_Personal_Care.jsonl"
RATINGS_FILE = "Beauty_and_Personal_Care.csv"
PARFUMO_FILE = "parfumo_data_clean.csv"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- 1
def filter_fragrance_items(meta_path: Path, category: str = "Fragrance") -> pd.DataFrame:
    """Stream the 2.8 GB metadata file and keep items whose categories contain `category`."""
    rows = []
    needle = f'"{category}'
    with open(meta_path) as f:
        for line in f:
            if needle not in line:          # cheap pre-filter before json parsing
                continue
            rec = json.loads(line)
            cats = rec.get("categories") or []
            if not any(category in c for c in cats):
                continue
            details = rec.get("details") or {}
            scent = details.get("Scent") or ""
            text = " ".join([
                rec.get("title") or "",
                " ".join(rec.get("description") or []),
                " ".join(rec.get("features") or []),
                scent,
            ])
            rows.append({
                "parent_asin": rec["parent_asin"],
                "title": rec.get("title") or "",
                "store": rec.get("store") or "",
                "categories": " > ".join(cats),
                "scent": scent,
                "average_rating": rec.get("average_rating"),
                "rating_number": rec.get("rating_number"),
                "image": ((rec.get("images") or [{}])[0] or {}).get("large"),
                "text": text,
            })
    items = pd.DataFrame(rows)
    # variants share a parent_asin: keep one row per parent (the one with most ratings)
    items = (items.sort_values("rating_number", ascending=False, na_position="last")
                  .drop_duplicates("parent_asin").reset_index(drop=True))
    return items


# --------------------------------------------------------------------------- 2
def compile_note_matcher(surface_forms: dict[str, str]) -> re.Pattern:
    forms = sorted(surface_forms, key=len, reverse=True)       # longest match first
    # optional plural ("blossoms", "roses"): without it, "orange blossoms" fell back to "orange"
    pattern = r"(?<![a-z])(" + "|".join(re.escape(f) for f in forms) + r")(?:e?s)?(?![a-z])"
    return re.compile(pattern, flags=re.IGNORECASE)


def extract_notes(items: pd.DataFrame, vocab: dict) -> pd.DataFrame:
    """Return a long table (parent_asin, note) of notes found in each item's text.

    Week 3 audit fixes: plurals are matched ("orange blossoms" -> orange blossom, it used to
    fall back to "orange"), and a note followed by a non-scent word is dropped
    ("rose gold", "amber glass", "coconut oil")."""
    surface = vocab["surface_forms"]
    matcher = compile_note_matcher(surface)
    pairs = []
    for asin, text in zip(items["parent_asin"], items["text"]):
        found = set()
        for m in matcher.finditer(text):
            form = m.group(1).lower()
            if form not in surface:
                continue
            note = surface[form]
            stop = NOT_A_NOTE_AFTER.get(note) or NOT_A_NOTE_AFTER.get(form)
            if stop and re.match(rf"\s*-?\s*({stop})\b", text[m.end():m.end() + 25], flags=re.I):
                continue
            found.add(note)
        pairs.extend((asin, n) for n in found)
    return pd.DataFrame(pairs, columns=["parent_asin", "note"])


# Week 3: product type, so that the app can show perfumes first. Sets, minis and body
# products stay in the data (people rated them) but are flagged.
_SET = r"gift set|\bset\b|\d\s*-?\s*pc\b|\d\s*-?\s*pieces?\b|coffret|sampler|samples?\b|discovery|variety pack|\bkit\b"
_BODY = r"body mist|body spray|hair mist|hair & body|lotion|shower gel|body wash|deodorant|dusting powder|candle|wax melt"
_MINI = r"\bmini\b|miniature|travel size|rollerball|roll-on|(?<![\d.])0?\.\d+\s*(fl\.?\s*)?(oz|ounce)\b|(?<![\d.])([1-9]|1[0-5])\s*ml\b"


def product_type(title: str) -> str:
    t = str(title).lower()
    if re.search(_SET, t):
        return "set"
    if re.search(_BODY, t):
        return "body"
    if re.search(_MINI, t):
        return "mini"
    return "perfume"


# --------------------------------------------------------------------------- 3
def load_fragrance_ratings(ratings_path: Path, asins: set[str]) -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(ratings_path, chunksize=2_000_000,
                             dtype={"user_id": str, "parent_asin": str,
                                    "rating": np.float32, "timestamp": np.int64}):
        chunks.append(chunk[chunk["parent_asin"].isin(asins)])
    ratings = pd.concat(chunks, ignore_index=True)
    # one rating per (user, item): keep the most recent
    ratings = (ratings.sort_values("timestamp")
                      .drop_duplicates(["user_id", "parent_asin"], keep="last")
                      .reset_index(drop=True))
    return ratings


def k_core(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Iteratively drop users and items with fewer than k interactions."""
    while True:
        u_ok = df.groupby("user_id")["parent_asin"].transform("size") >= k
        i_ok = df.groupby("parent_asin")["user_id"].transform("size") >= k
        keep = u_ok & i_ok
        if keep.all():
            return df.reset_index(drop=True)
        df = df[keep]


# --------------------------------------------------------------------------- 4
def leave_last_out(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per user, sorted by time: last -> test, second to last -> val, rest -> train."""
    df = df.sort_values(["user_idx", "timestamp", "item_idx"]).reset_index(drop=True)
    rank_from_end = df.groupby("user_idx").cumcount(ascending=False)
    test = df[rank_from_end == 0]
    val = df[rank_from_end == 1]
    train = df[rank_from_end >= 2]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# --------------------------------------------------------------------------- main
def run(cfg: dict, k: int | None = None) -> dict:
    p = cfg["paths"]
    raw, out = p["raw_dir"], p["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)
    k = k or cfg["data"]["k_core"]
    stats: dict = {"k_core": k, "positive_threshold": cfg["data"]["positive_threshold"]}

    log("building note vocabulary")
    vocab = build_vocabulary(p["vocab_dir"] / PARFUMO_FILE, cfg["data"]["min_note_freq"])
    save_vocabulary(vocab, p["vocab_dir"] / "note_vocabulary.json")
    stats["vocab_notes"] = vocab["n_notes"]

    log("filtering fragrance items from metadata")
    items = filter_fragrance_items(raw / META_FILE, cfg["data"]["fragrance_category"])
    stats["fragrance_items"] = len(items)

    log(f"extracting notes for {len(items):,} items")
    item_notes = extract_notes(items, vocab)
    stats["items_with_notes_all"] = int(item_notes["parent_asin"].nunique())

    log("loading ratings for fragrance items")
    ratings = load_fragrance_ratings(raw / RATINGS_FILE, set(items["parent_asin"]))
    stats["ratings"] = len(ratings)
    stats["rating_users"] = int(ratings["user_id"].nunique())
    stats["rating_items"] = int(ratings["parent_asin"].nunique())
    stats["date_min"] = str(pd.to_datetime(ratings["timestamp"].min(), unit="ms").date())
    stats["date_max"] = str(pd.to_datetime(ratings["timestamp"].max(), unit="ms").date())

    pos = ratings[ratings["rating"] >= cfg["data"]["positive_threshold"]]
    stats["positives"] = len(pos)

    log(f"k-core (k={k})")
    core = k_core(pos, k)
    users = np.sort(core["user_id"].unique())
    asins = np.sort(core["parent_asin"].unique())
    u_map = pd.Series(np.arange(len(users)), index=users)
    i_map = pd.Series(np.arange(len(asins)), index=asins)
    core = core.assign(user_idx=core["user_id"].map(u_map).values,
                       item_idx=core["parent_asin"].map(i_map).values)

    # notes restricted to k-core items; note ids over notes actually present
    kn = item_notes[item_notes["parent_asin"].isin(i_map.index)].copy()
    note_names = np.sort(kn["note"].unique())
    n_map = pd.Series(np.arange(len(note_names)), index=note_names)
    kn["item_idx"] = kn["parent_asin"].map(i_map).values
    kn["note_idx"] = kn["note"].map(n_map).values

    train, val, test = leave_last_out(core[["user_idx", "item_idx", "timestamp", "rating"]])
    stats.update({
        "users": len(users), "items": len(asins), "interactions": len(core),
        "train": len(train), "val": len(val), "test": len(test),
        "density": len(core) / (len(users) * len(asins)),
        "positives_per_user": len(core) / len(users),
        "notes_in_graph": len(note_names),
        "item_note_edges": len(kn),
        "items_with_notes": int(kn["item_idx"].nunique()),
        "items_with_notes_pct": kn["item_idx"].nunique() / len(asins),
    })

    log("writing parquet files")
    cat = items[items["parent_asin"].isin(i_map.index)].copy()
    cat["item_idx"] = cat["parent_asin"].map(i_map).values
    notes_per_item = kn.groupby("item_idx")["note"].apply(lambda s: ", ".join(sorted(s)))
    cat["notes"] = cat["item_idx"].map(notes_per_item).fillna("")
    cat["train_popularity"] = cat["item_idx"].map(train["item_idx"].value_counts()).fillna(0).astype(int)
    cat["product_type"] = cat["title"].map(product_type)
    stats["product_types"] = cat["product_type"].value_counts().to_dict()
    cat.drop(columns=["text"]).sort_values("item_idx").to_parquet(out / "items.parquet", index=False)
    train.to_parquet(out / "train.parquet", index=False)
    val.to_parquet(out / "val.parquet", index=False)
    test.to_parquet(out / "test.parquet", index=False)
    kn[["item_idx", "note_idx"]].to_parquet(out / "item_notes.parquet", index=False)
    (out / "notes.json").write_text(json.dumps(list(note_names)))
    pd.DataFrame({"user_id": users, "user_idx": np.arange(len(users))}).to_parquet(out / "users.parquet", index=False)
    (out / "stats.json").write_text(json.dumps(stats, indent=1))

    log("loading catalogue into SQLite")
    from scentmatch.api.db import init_db, load_catalogue
    init_db(p["db_path"])
    load_catalogue(p["db_path"], out / "items.parquet")

    log("done: " + json.dumps(stats))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--k", type=int, default=None, help="override k-core (fallbacks: 2 or 5)")
    args = ap.parse_args()
    run(load_config(args.config), args.k)


if __name__ == "__main__":
    main()
