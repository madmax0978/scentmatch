"""Does telling us "Not for me" help? Offline test of the AVOID term (Week 3).

Users who wrote a review saying they dislike a perfume's smell *during the training period*
(1,553 users) are exactly the people who would have clicked "Not for me". For each of them we
rank with the served formula (scentmatch.train.predict.Scorer, known-user mode: the learned
embedding + their training perfumes as anchors) twice:

    without  Profile(user, liked = training perfumes)
    with     Profile(user, liked = training perfumes, disliked = their past dislikes)

and measure on the evaluation split:
  * NDCG@10 of the perfume they liked next (must not drop: the penalty must not hide good picks);
  * closeness of the top-10 to the rejected perfumes (mean of the max cosine; should drop);
  * for the 261 users who also disliked a perfume *later*: LikeVsDislike = P(the perfume they
    liked next is ranked above the perfume they disliked later). This is the real test:
    does one rejection teach us about the next one?

Run:  python -m scentmatch.eval.feedback_eval --ckpt final_nlgcl_s1.pt --split val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scentmatch.config import load_config
from scentmatch.eval.metrics import ndcg_at_k
from scentmatch.train.predict import Catalogue, Profile, Scorer, load_embeddings

ROOT = Path(__file__).resolve().parents[3]


def run(ckpt: str, split: str, w_avoid: float) -> dict:
    cfg = load_config()
    proc = cfg["paths"]["processed_dir"]
    cat = Catalogue(proc)
    e = load_embeddings(cfg["paths"]["checkpoints_dir"] / ckpt, proc)
    scorer = Scorer(cat, e["items"], e["users"], w_avoid=w_avoid)
    fb = pd.read_parquet(proc / "feedback.parquet")
    dis = fb[fb["label"] == "dislike"]
    past = dis[dis["period"] == "train"].groupby("user_idx")["item_idx"].apply(list)
    future = dis[dis["period"] == "eval"].groupby("user_idx")["item_idx"].apply(list)
    train = pd.read_parquet(proc / "train.parquet").groupby("user_idx")["item_idx"].apply(list)
    val = pd.read_parquet(proc / "val.parquet").set_index("user_idx")["item_idx"]
    target = val if split == "val" else pd.read_parquet(proc / "test.parquet").set_index("user_idx")["item_idx"]
    rows = {"without": {"ndcg": [], "close": [], "lvd": []}, "with": {"ndcg": [], "close": [], "lvd": []}}
    for u, rejected in past.items():
        liked = train.get(u, [])
        seen = list(liked) + ([int(val[u])] if split == "test" else [])
        for name, prof in (("without", Profile(user_idx=u, liked=list(liked))),
                           ("with", Profile(user_idx=u, liked=list(liked), disliked=list(rejected)))):
            parts = scorer.score(prof)
            s = parts["total"].copy()
            s[seen + list(rejected)] = -np.inf
            top = np.argsort(-s)[:10]
            rows[name]["ndcg"].append(ndcg_at_k(top.tolist(), {int(target[u])}, 10))
            rows[name]["close"].append(float((scorer.item_n[top] @ scorer.item_n[rejected].T).max(1).mean()))
            if u in future.index:
                for d in future[u]:
                    rows[name]["lvd"].append(float(parts["total"][int(target[u])] > parts["total"][d]))
    out = {"checkpoint": ckpt, "split": split, "w_avoid": w_avoid, "n_users": len(past),
           "n_future_dislike_pairs": len(rows["with"]["lvd"])}
    for name, r in rows.items():
        out[name] = {"NDCG@10": float(np.mean(r["ndcg"])), "closeness_to_rejected@10": float(np.mean(r["close"])),
                     "LikeVsDislike(future)": float(np.mean(r["lvd"])) if r["lvd"] else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--w-avoid", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.6])
    args = ap.parse_args()
    runs = [run(args.ckpt, args.split, w) for w in args.w_avoid]
    base = runs[0]["without"]["NDCG@10"]
    # selection rule (on validation): the strongest penalty that costs at most 5% NDCG
    ok = [r["w_avoid"] for r in runs if r["with"]["NDCG@10"] >= 0.95 * base]
    res = {"runs": runs, "baseline_NDCG@10": base, "chosen_w_avoid": max(ok) if ok else min(args.w_avoid)}
    out = ROOT / "reports" / f"week3_feedback_{args.split}.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
