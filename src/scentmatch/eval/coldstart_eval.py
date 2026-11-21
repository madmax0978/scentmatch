"""Cold-start evaluation: the quiz and the like/dislike loop for brand-new visitors (Week 3).

Protocol (scentmatch.data.coldstart): 2,000 users were removed from training; the model used
here was trained on cold/train.parquet and has never seen them. For each of them:

  * "quiz"         the notes of their first liked perfume stand in for their quiz answers;
  * "quiz + like"  they also clicked "Love it" on that first perfume (its embedding is known);
  * targets        every other perfume they liked. The first perfume itself is excluded.

The scoring is exactly the served one (scentmatch.train.predict.Scorer, without the shop-window
adjustments), so what is measured here is what the app does. Weights are chosen on the
"val" half of the cold users and reported on the "test" half.

Variants: Most-popular | Notes only (w_collab = 0) | Collaborative only (w_notes = 0) |
          Hybrid (w_notes, w_collab) grid | Note-embedding fold-in (mean of the learned note
          embeddings, only for a model with note nodes).

Run:  python -m scentmatch.eval.coldstart_eval --ckpt cold_nlgcl.pt [--ckpt-notes cold_nlgcl_notes.pt]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scentmatch.config import load_config
from scentmatch.eval.metrics import catalogue_coverage, ndcg_at_k, recall_at_k
from scentmatch.train.predict import Catalogue, Profile, Scorer, load_embeddings

ROOT = Path(__file__).resolve().parents[3]


def evaluate(rank_fn, users: pd.DataFrame, n_items: int) -> dict:
    ndcg, rec, lists = [], [], []
    for u in users.itertuples():
        ranked = rank_fn(u)
        t = set(u.targets)
        ndcg.append(ndcg_at_k(ranked, t, 10))
        rec.append(recall_at_k(ranked, t, 20))
        lists.append(ranked)
    return {"NDCG@10": float(np.mean(ndcg)), "Recall@20": float(np.mean(rec)),
            "Coverage@10": catalogue_coverage(lists, n_items, 10), "n_users": len(users)}


def scorer_rank(scorer: Scorer, cat: Catalogue, with_like: bool):
    def rank(u) -> list[int]:
        prof = Profile(seed_notes=sorted(cat.notes_of(u.seed_item)), liked=[u.seed_item] if with_like else [])
        return scorer.rank(prof, 20, exclude=[u.seed_item])
    return rank


def run(ckpt: str, ckpt_notes: str | None = None) -> dict:
    cfg = load_config()
    proc = cfg["paths"]["processed_dir"]
    cat = Catalogue(proc)
    users = pd.read_parquet(proc / "cold" / "users.parquet")
    halves = {h: users[users["half"] == h] for h in ("val", "test")}
    cold_train = pd.read_parquet(proc / "cold" / "train.parquet")
    pop = np.bincount(cold_train["item_idx"], minlength=cat.n_items).astype(float)
    e = load_embeddings(cfg["paths"]["checkpoints_dir"] / ckpt, proc)
    results: dict = {"checkpoint": ckpt, "protocol": "notes of the first liked perfume = quiz; "
                     "targets = all other liked perfumes; model never saw these users"}

    def pop_rank(u) -> list[int]:
        s = pop.copy()
        s[u.seed_item] = -np.inf
        return np.argsort(-s)[:20].tolist()

    # 1. choose w_notes and beta on the validation half, separately for "quiz" and "quiz + like"
    W, B = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0), (0.0, 0.25, 0.5, 0.75, 1.0)
    chosen = {}
    for scenario, with_like in (("quiz", False), ("feedback", True)):
        grid = {}
        for wn in W:
            for beta in B:
                if wn == 1.0 and beta != B[0]:
                    continue                                  # beta has no effect without COLLAB
                sc = Scorer(cat, e["items"], w_notes=wn, beta=beta)
                grid[(wn, beta)] = evaluate(scorer_rank(sc, cat, with_like), halves["val"], cat.n_items)["NDCG@10"]
        # selection rule: among the settings within 2% of the best validation NDCG, prefer the
        # most weight on notes (explainable), then the least popularity (more diverse)
        top = max(grid.values())
        ok = [k for k, v in grid.items() if v >= 0.98 * top]
        best = sorted(ok, key=lambda k: (-k[0], k[1]))[0]
        chosen[scenario] = {"w_notes": best[0], "beta": best[1]}
        results[f"val_grid_{scenario}"] = {f"w_notes={k[0]},beta={k[1]}": round(v, 5) for k, v in grid.items()}
    results["chosen"] = chosen
    best_wn = chosen["quiz"]["w_notes"]

    # 2. report everything on the test half --------------------------------------------------
    test = halves["test"]
    rows = {"Most-popular": evaluate(pop_rank, test, cat.n_items)}
    served = Scorer(cat, e["items"], settings=chosen)
    rows["quiz | ScentMatch (tuned)"] = evaluate(scorer_rank(served, cat, False), test, cat.n_items)
    rows["quiz + 1 like | ScentMatch (tuned)"] = evaluate(scorer_rank(served, cat, True), test, cat.n_items)
    for name, wn, beta in (("Notes only", 1.0, 0.0), ("Collaborative only, pure taste", 0.0, 0.0),
                           ("Collaborative only, model popularity", 0.0, 1.0)):
        sc = Scorer(cat, e["items"], w_notes=wn, beta=beta)
        rows[f"quiz | {name}"] = evaluate(scorer_rank(sc, cat, False), test, cat.n_items)
        rows[f"quiz + 1 like | {name}"] = evaluate(scorer_rank(sc, cat, True), test, cat.n_items)
    if ckpt_notes:
        en = load_embeddings(cfg["paths"]["checkpoints_dir"] / ckpt_notes, proc)
        if en["notes"] is not None:
            def note_fold(u) -> list[int]:
                ids = sorted(cat.notes_of(u.seed_item))
                s = en["items"] @ en["notes"][ids].mean(0)
                s[u.seed_item] = -np.inf
                return np.argsort(-s)[:20].tolist()
            rows["quiz | Note-embedding fold-in (notes model)"] = evaluate(note_fold, test, cat.n_items)
            sc = Scorer(cat, en["items"], settings=chosen)
            rows["quiz | ScentMatch on notes model"] = evaluate(scorer_rank(sc, cat, False), test, cat.n_items)
            rows["quiz + 1 like | ScentMatch on notes model"] = evaluate(scorer_rank(sc, cat, True), test, cat.n_items)
    results["test"] = rows
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-notes", default=None)
    args = ap.parse_args()
    res = run(args.ckpt, args.ckpt_notes)
    out = ROOT / "reports" / "week3_coldstart.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
