"""Evaluation harness (Project Lead).

Protocol (fixed for the whole project, so every model is scored identically):
  * leave-last-out split produced by scentmatch.data.pipeline;
  * full ranking over all perfumes (no sampled negatives);
  * items already seen are excluded: train items when scoring validation,
    train + validation items when scoring test;
  * metrics: NDCG@10, Recall@20 (accuracy), coverage@10 and average popularity@10
    (popularity bias), popularity counted on training interactions only.

Two entry points:
  * evaluate_scores(score_fn)   - score_fn(user_idx array) -> [batch, n_items] scores
  * evaluate_rankings(rankings) - {user_idx: [item_idx, ...]} precomputed (e.g. mock) lists

Run:  python -m scentmatch.eval.harness      (scores random + most-popular baselines)
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from scentmatch.config import load_config
from scentmatch.eval.metrics import average_popularity, catalogue_coverage, ndcg_at_k, recall_at_k

ScoreFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class EvalData:
    n_users: int
    n_items: int
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame | None          # None while training: the test split is never loaded
    popularity: np.ndarray = field(init=False)

    def __post_init__(self):
        self.popularity = np.bincount(self.train["item_idx"], minlength=self.n_items).astype(float)

    @classmethod
    def load(cls, processed_dir: Path, with_test: bool = True, train_file: str = "train.parquet") -> "EvalData":
        """with_test=False is what training uses: the test file is not even read, so no
        training or tuning decision can depend on it (leakage impossible by construction)."""
        tr = pd.read_parquet(processed_dir / train_file)
        va = pd.read_parquet(processed_dir / "val.parquet")
        te = pd.read_parquet(processed_dir / "test.parquet") if with_test else None
        stats = json.loads((processed_dir / "stats.json").read_text())
        return cls(stats["users"], stats["items"], tr, va, te)

    def seen_items(self, split: str) -> dict[int, np.ndarray]:
        frames = [self.train] if split == "val" else [self.train, self.val]
        seen = pd.concat(frames)
        return {u: g.values for u, g in seen.groupby("user_idx")["item_idx"]}

    def targets(self, split: str) -> dict[int, set[int]]:
        if split == "test" and self.test is None:
            raise RuntimeError("the test split was not loaded (EvalData.load(with_test=False))")
        df = self.val if split == "val" else self.test
        return {u: set(g.values) for u, g in df.groupby("user_idx")["item_idx"]}


def _summarise(rankings: dict[int, list[int]], targets: dict[int, set[int]],
               data: EvalData, k_ndcg: int = 10, k_recall: int = 20) -> dict[str, float]:
    users = list(targets)
    return {
        f"NDCG@{k_ndcg}": float(np.mean([ndcg_at_k(rankings[u], targets[u], k_ndcg) for u in users])),
        f"Recall@{k_recall}": float(np.mean([recall_at_k(rankings[u], targets[u], k_recall) for u in users])),
        "Coverage@10": catalogue_coverage((rankings[u] for u in users), data.n_items, 10),
        "AvgPop@10": average_popularity((rankings[u] for u in users), data.popularity, 10),
        "n_users": len(users),
    }


def rank_scores(score_fn: ScoreFn, data: EvalData, split: str = "test",
                batch_size: int = 2048, top: int = 20, users: np.ndarray | None = None) -> dict[int, list[int]]:
    """Top-`top` ranked perfumes per evaluation user, already-seen perfumes masked."""
    seen = data.seen_items(split)
    users = np.array(sorted(data.targets(split))) if users is None else np.asarray(users)
    rankings: dict[int, list[int]] = {}
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        scores = np.asarray(score_fn(batch), dtype=np.float32).copy()
        for row, u in enumerate(batch):
            if u in seen:
                scores[row, seen[u]] = -np.inf
        part = np.argpartition(-scores, top, axis=1)[:, :top]
        order = np.take_along_axis(scores, part, axis=1).argsort(axis=1)[:, ::-1]
        top_items = np.take_along_axis(part, order, axis=1)
        for row, u in enumerate(batch):
            rankings[int(u)] = top_items[row].tolist()
    return rankings


def evaluate_scores(score_fn: ScoreFn, data: EvalData, split: str = "test",
                    batch_size: int = 2048, top: int = 20) -> dict[str, float]:
    return _summarise(rank_scores(score_fn, data, split, batch_size, top), data.targets(split), data)


def evaluate_rankings(rankings: dict[int, list[int]], data: EvalData, split: str = "test") -> dict[str, float]:
    """Score precomputed ranked lists (used to test the harness on mock predictions)."""
    targets = data.targets(split)
    missing = set(targets) - set(rankings)
    if missing:
        raise ValueError(f"{len(missing)} evaluation users have no ranking")
    return _summarise(rankings, targets, data)


def main() -> None:
    from scentmatch.eval.baselines import most_popular_scores, random_scores

    cfg = load_config()
    data = EvalData.load(cfg["paths"]["processed_dir"])
    results = {}
    for name, fn in [("Random", random_scores(data.n_items, seed=0)),
                     ("Most-popular", most_popular_scores(data.popularity))]:
        for split in ("val", "test"):
            results[f"{name}/{split}"] = evaluate_scores(fn, data, split)
            print(name, split, {k: round(v, 4) for k, v in results[f"{name}/{split}"].items()})
    out = cfg["paths"]["processed_dir"].parent.parent / "reports" / "baselines.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
