"""Cold-start evaluation sets (Data Engineer, Week 3).

Three things the accuracy number over all users hides:

1. User segments. 56% of our users have a single training perfume, 20% have two. A model can
   look good on average and be useless for exactly the users the quiz targets. We write the
   number of training perfumes per user so that every metric can be reported per segment
   (1 | 2 | 3-5 | 6+).

2. Cold perfumes. 243 perfumes have no training interaction at all (their ratings fell in
   validation/test). A pure user-perfume graph cannot place them; the note graph can.

3. Brand-new users (the quiz scenario). We hold out 2,000 users completely: all their
   interactions are removed from the training file `cold/train.parquet`, so a model trained on
   it has never seen them. Offline we cannot make them fill the quiz, so - as planned in Week 1 -
   the notes of their *first* liked perfume stand in for their quiz answers ("seed notes"),
   and we predict all their other liked perfumes. Half of them (`half = "val"`) is used to tune
   the quiz scoring weights, the other half (`half = "test"`) only to report the result.

Output: data/processed/user_segments.parquet, data/processed/cold/{train,users}.parquet,
        data/processed/cold/stats.json
Run:    python -m scentmatch.data.coldstart
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scentmatch.config import load_config

N_COLD_USERS = 2000
SEED = 2026


def segment(n: int) -> str:
    return "1" if n == 1 else "2" if n == 2 else "3-5" if n <= 5 else "6+"


def build(cfg: dict) -> dict:
    proc = cfg["paths"]["processed_dir"]
    train = pd.read_parquet(proc / "train.parquet")
    val = pd.read_parquet(proc / "val.parquet")
    test = pd.read_parquet(proc / "test.parquet")
    item_notes = pd.read_parquet(proc / "item_notes.parquet")
    items = pd.read_parquet(proc / "items.parquet", columns=["item_idx", "train_popularity"])

    # 1. segments ---------------------------------------------------------------------------
    n_train = train.groupby("user_idx").size().rename("n_train").reset_index()
    n_train["segment"] = n_train["n_train"].map(segment)
    n_train.to_parquet(proc / "user_segments.parquet", index=False)

    # 2. cold perfumes -----------------------------------------------------------------------
    cold_items = items.loc[items["train_popularity"] == 0, "item_idx"].tolist()

    # 3. brand-new users ---------------------------------------------------------------------
    first = train.sort_values(["user_idx", "timestamp", "item_idx"]).groupby("user_idx").head(1)
    has_notes = set(item_notes["item_idx"])
    eligible = first[first["item_idx"].isin(has_notes)]
    rng = np.random.default_rng(SEED)
    chosen = eligible.sample(n=N_COLD_USERS, random_state=SEED).sort_values("user_idx")
    chosen = chosen.assign(half=np.where(rng.random(len(chosen)) < 0.5, "val", "test"))
    every = pd.concat([train, val, test])
    targets = (every[every["user_idx"].isin(chosen["user_idx"])]
               .merge(chosen[["user_idx", "item_idx"]].rename(columns={"item_idx": "seed_item"}), on="user_idx"))
    targets = targets[targets["item_idx"] != targets["seed_item"]]
    tgt = targets.groupby("user_idx")["item_idx"].apply(lambda s: sorted(set(s))).rename("targets")
    cold_users = chosen.rename(columns={"item_idx": "seed_item"})[["user_idx", "seed_item", "half"]].merge(
        tgt, on="user_idx")
    cold_dir = proc / "cold"
    cold_dir.mkdir(exist_ok=True)
    cold_train = train[~train["user_idx"].isin(cold_users["user_idx"])]
    cold_train.to_parquet(cold_dir / "train.parquet", index=False)
    cold_users.to_parquet(cold_dir / "users.parquet", index=False)

    stats = {
        "segments": n_train["segment"].value_counts().reindex(["1", "2", "3-5", "6+"]).to_dict(),
        "cold_items": len(cold_items),
        "test_targets_on_cold_items": int(test["item_idx"].isin(cold_items).sum()),
        "cold_users": len(cold_users),
        "cold_users_by_half": cold_users["half"].value_counts().to_dict(),
        "cold_targets_per_user": float(cold_users["targets"].map(len).mean()),
        "cold_train_interactions": len(cold_train),
        "seed": SEED,
    }
    (cold_dir / "stats.json").write_text(json.dumps({**stats, "cold_item_ids": cold_items}, indent=1))
    return stats


def main() -> None:
    print(json.dumps(build(load_config()), indent=1))


if __name__ == "__main__":
    main()
