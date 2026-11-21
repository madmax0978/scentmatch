"""Full evaluation of every ranker on one split (Project Lead / MLE-Ops, Week 3).

Beyond NDCG@10 / Recall@20, it answers the questions the professor asked: *how* does the
model rank, is it personal, is it just popularity, does it know what users dislike?

    accuracy        NDCG@10, Recall@20 (full ranking, seen perfumes masked)
    popularity bias Coverage@10 (share of the catalogue ever recommended), AvgPop@10,
                    TailShare@10 (share of recommended perfumes outside the 20% most popular)
    personalisation Personalisation@10 = 1 - mean Jaccard overlap between two users' top-10
                    (1 = everybody gets a different list, 0 = everybody gets the same list)
    segments        NDCG@10 for users with 1 / 2 / 3-5 / 6+ training perfumes
    cold perfumes   Recall@20 for users whose target perfume has no training interaction
    dislikes        for users who later wrote a review saying they dislike a perfume's smell:
                    LikeVsDislike = P(score(perfume they liked) > score(perfume they disliked))
                    (0.5 = cannot tell them apart), DislikeInTop20 = share of those users whose
                    disliked perfume is in their top-20 (lower is better)

Rankers: Random, Most-popular, Notes-only (IDF note profile of the user's training perfumes,
no learning at all) and every checkpoint given.

Run:  python -m scentmatch.eval.evaluate --split val  --ckpt name=path ...
      python -m scentmatch.eval.evaluate --split test --ckpt ...      (ONCE, after tuning)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scentmatch.config import load_config
from scentmatch.eval.baselines import most_popular_scores, random_scores
from scentmatch.eval.harness import EvalData, _summarise, rank_scores
from scentmatch.eval.metrics import ndcg_at_k, recall_at_k

ROOT = Path(__file__).resolve().parents[3]


def personalisation(rankings: dict[int, list[int]], k: int = 10, n_pairs: int = 20000, seed: int = 0) -> float:
    users = np.array(list(rankings))
    rng = np.random.default_rng(seed)
    a, b = rng.choice(users, n_pairs), rng.choice(users, n_pairs)
    sets = {u: set(rankings[u][:k]) for u in users}
    jac = [len(sets[x] & sets[y]) / len(sets[x] | sets[y]) for x, y in zip(a, b) if x != y]
    return 1.0 - float(np.mean(jac))


def tail_share(rankings: dict[int, list[int]], popularity: np.ndarray, k: int = 10) -> float:
    head = set(np.argsort(-popularity)[: int(0.2 * len(popularity))])
    recs = [i for r in rankings.values() for i in r[:k]]
    return float(np.mean([i not in head for i in recs]))


def notes_only_scores(data: EvalData, proc: Path, split: str):
    """Content baseline: IDF-weighted notes of the perfumes the user liked before the split."""
    inotes = pd.read_parquet(proc / "item_notes.parquet")
    n_notes = int(inotes["note_idx"].max()) + 1
    x = np.zeros((data.n_items, n_notes), dtype=np.float32)
    x[inotes["item_idx"], inotes["note_idx"]] = 1
    idf = np.log((1 + data.n_items) / (1 + x.sum(0))) + 1
    xw = x * idf
    xn = xw / np.maximum(np.linalg.norm(xw, axis=1, keepdims=True), 1e-9)
    hist = pd.concat([data.train] + ([data.val] if split == "test" else []))
    prof = np.zeros((data.n_users, n_notes), dtype=np.float32)
    np.add.at(prof, hist["user_idx"].to_numpy(), x[hist["item_idx"].to_numpy()])
    prof *= idf
    prof /= np.maximum(np.linalg.norm(prof, axis=1, keepdims=True), 1e-9)
    pop = data.popularity / data.popularity.max()

    def score(users: np.ndarray) -> np.ndarray:
        return prof[users] @ xn.T + 1e-3 * pop          # popularity only breaks ties
    return score


def embedding_scores(path: Path, proc: Path, cosine: bool = False, beta: float | None = None):
    """<e_u, e_i> (default), cosine, or cos(e_u, e_i) * |e_i|^beta (the popularity knob)."""
    from scentmatch.train.predict import load_embeddings
    e = load_embeddings(path, proc)
    u, i = e["users"], e["items"]
    if beta is not None:
        n = np.linalg.norm(i, axis=1)
        u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
        i = i / np.maximum(n[:, None], 1e-9) * (n / n.max())[:, None] ** beta
    elif cosine:
        u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
        i = i / np.maximum(np.linalg.norm(i, axis=1, keepdims=True), 1e-9)

    def score(users: np.ndarray) -> np.ndarray:
        return u[users] @ i.T
    return score


def full_report(name: str, score_fn, data: EvalData, split: str, proc: Path, feedback: pd.DataFrame | None) -> dict:
    targets = data.targets(split)
    rankings = rank_scores(score_fn, data, split)
    out = {"ranker": name, **_summarise(rankings, targets, data)}
    out["Personalisation@10"] = personalisation(rankings)
    out["TailShare@10"] = tail_share(rankings, data.popularity)
    seg = pd.read_parquet(proc / "user_segments.parquet").set_index("user_idx")["segment"]
    for s in ("1", "2", "3-5", "6+"):
        us = [u for u in targets if seg.get(u) == s]
        out[f"NDCG@10[{s}]"] = float(np.mean([ndcg_at_k(rankings[u], targets[u], 10) for u in us]))
    cold = set(np.flatnonzero(data.popularity == 0))
    cu = [u for u in targets if targets[u] & cold]
    out["Recall@20[cold perfumes]"] = float(np.mean([recall_at_k(rankings[u], targets[u], 20) for u in cu])) if cu else None
    out["n_cold_perfume_users"] = len(cu)
    if feedback is not None:
        d = feedback[(feedback["label"] == "dislike") & (feedback["period"] == "eval")]
        d = d[d["user_idx"].isin(targets)]
        if len(d):
            tgt = {u: next(iter(t)) for u, t in targets.items()}
            users = d["user_idx"].to_numpy()
            s = score_fn(users)
            like_s = s[np.arange(len(users)), [tgt[u] for u in users]]
            dis_s = s[np.arange(len(users)), d["item_idx"].to_numpy()]
            out["LikeVsDislike"] = float(np.mean((like_s > dis_s) + 0.5 * (like_s == dis_s)))
            out["DislikeInTop20"] = float(np.mean([i in rankings[u][:20] for u, i in zip(users, d["item_idx"])]))
            out["n_dislike_pairs"] = int(len(d))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--ckpt", nargs="*", default=[], help="name=checkpoint path (relative to checkpoints/)")
    ap.add_argument("--cosine", action="store_true", help="also score every checkpoint with cosine similarity")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config()
    proc = cfg["paths"]["processed_dir"]
    data = EvalData.load(proc, with_test=args.split == "test")
    fb_path = proc / "feedback.parquet"
    feedback = pd.read_parquet(fb_path) if fb_path.exists() else None
    rankers = []
    if not args.no_baselines:
        rankers += [("Random", random_scores(data.n_items, seed=0)),
                    ("Most-popular", most_popular_scores(data.popularity)),
                    ("Notes-only (content)", notes_only_scores(data, proc, args.split))]
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        p = cfg["paths"]["checkpoints_dir"] / path
        rankers.append((name, embedding_scores(p, proc)))
        if args.cosine:
            rankers.append((name + " (cosine)", embedding_scores(p, proc, cosine=True)))
    rows = []
    for name, fn in rankers:
        row = full_report(name, fn, data, args.split, proc, feedback)
        rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    out = Path(args.out) if args.out else ROOT / "reports" / f"week3_{args.split}_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
