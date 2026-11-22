"""Measuring the innovation: with notes vs without notes (Week 3, notes v2 - Mina).

The question is not only "do note nodes raise NDCG?" but "do they make recommendations that a
perfume buyer can trust and understand?". Three families of metrics, the same for every model:

1. Cold-start (the quiz). Brand-new visitors (scentmatch.data.coldstart: 2,000 users removed from
   training, "test" half reported). The notes of their first liked perfume stand in for their quiz
   answers; targets = every other perfume they liked. The served Scorer is used, with the item
   embeddings of a model trained without those users, with or without note nodes.
2. Coverage and diversity. Coverage@10 (share of the catalogue recommended to someone),
   ILD@10 (intra-list diversity: mean pairwise distance between the note profiles of a top-10;
   the note vectors are the same for every model, so the measure is comparable), brand diversity.
3. Explanation fidelity. An explanation is faithful if what it cites is true AND is what drove
   the ranking:
     NotePrecision    cited notes really are notes of the perfume and of the visitor (should be 1);
     Explained@10     share of the top-10 with a concrete reason (a note or a liked perfume),
                      not the generic "A discovery outside your usual notes";
     DeletionDrop@10  quiz path: remove the cited notes from the quiz answers and re-rank; share
                      of the cited perfumes that leave the top-10. Compared with removing the
                      same number of *uncited* quiz notes at random (RandomDrop@10): if cited
                      notes are the real reason, DeletionDrop >> RandomDrop;
     NoteGrounded@10  known users: share of the top-10 sharing at least one note with the
                      perfumes the user liked, i.e. that can get a truthful note explanation.

Run:  scripts/notes_ablation.sh     (the exact commands used for reports/week3_notes_ablation_*.md)
      python -m scentmatch.eval.innovation --split val|test \
          --warm no_notes=final_nlgcl_rescued_s1.pt,final_nlgcl_rescued_s2.pt,final_nlgcl_rescued_s3.pt \
          --warm main_notes=final_nlgcl_mainnotes_s1.pt,... --cold no_notes=cold_nlgcl_rescued.pt ...
Out:  reports/week3_notes_ablation_<split>.json and .md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scentmatch.config import load_config
from scentmatch.eval.evaluate import embedding_scores
from scentmatch.eval.harness import EvalData, _summarise, rank_scores
from scentmatch.eval.metrics import (brand_diversity, catalogue_coverage, intra_list_diversity,
                                     ndcg_at_k, recall_at_k)
from scentmatch.train.predict import Catalogue, Profile, Scorer, load_embeddings

ROOT = Path(__file__).resolve().parents[3]
GENERIC = "A discovery outside your usual notes"
N_CITED = 3                                      # the sentence names at most 3 notes (_listing)


def diversity_block(lists: list[list[int]], cat: Catalogue, k: int = 10) -> dict:
    brands = cat.items["store"].astype(str).str.lower().to_numpy()
    return {"Coverage@10": catalogue_coverage(lists, cat.n_items, k),
            "ILD@10": intra_list_diversity(lists, cat.note_matrix, k),
            "BrandDiversity@10": brand_diversity(lists, brands, k)}


def note_grounded(rankings: dict[int, list[int]], history: dict[int, np.ndarray], cat: Catalogue, k: int = 10) -> float:
    vals = []
    for u, ranked in rankings.items():
        h = history.get(u)
        if h is None or not len(h):
            continue
        liked_notes = cat.has_note[h].sum(0) > 0
        vals.append(float(np.mean([bool((cat.has_note[i] > 0)[liked_notes].any()) for i in ranked[:k]])))
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------- warm users
def warm_report(ckpt: Path, data: EvalData, split: str, cat: Catalogue, proc: Path, beta: float | None = None) -> dict:
    rankings = rank_scores(embedding_scores(ckpt, proc, beta=beta), data, split)
    targets = data.targets(split)
    out = _summarise(rankings, targets, data)
    lists = [rankings[u] for u in targets]
    out.update(diversity_block(lists, cat))
    hist_df = pd.concat([data.train] + ([data.val] if split == "test" else []))
    history = {u: g.to_numpy() for u, g in hist_df.groupby("user_idx")["item_idx"]}
    out["NoteGrounded@10"] = note_grounded({u: rankings[u] for u in targets}, history, cat)
    return out


# ---------------------------------------------------------------------------- cold users
def cited_notes(scorer: Scorer, i: int, parts: dict, profile: Profile, quiz: bool) -> tuple[list[int], str]:
    text, br = scorer.explain(i, parts, profile, quiz)
    names = [d["note"] for d in br["notes"][:N_CITED]] if br["notes_part"] > 0 else []
    return [scorer.cat.note_idx[n] for n in names], text


def cold_report(ckpt: Path, users: pd.DataFrame, cat: Catalogue, proc: Path, with_like: bool,
                k: int = 10, seed: int = 0) -> dict:
    e = load_embeddings(ckpt, proc)
    scorer = Scorer(cat, e["items"])
    rng = np.random.default_rng(seed)
    ndcg, rec, lists = [], [], []
    precision, explained, drop, rand_drop = [], [], [], []
    for u in users.itertuples():
        seed_notes = sorted(cat.notes_of(u.seed_item))
        prof = Profile(seed_notes=seed_notes, liked=[u.seed_item] if with_like else [])
        ranked = scorer.rank(prof, 20, exclude=[u.seed_item])
        t = set(u.targets)
        ndcg.append(ndcg_at_k(ranked, t, k))
        rec.append(recall_at_k(ranked, t, 20))
        lists.append(ranked)
        parts = scorer.score(prof)
        q_notes = set(np.flatnonzero(cat.note_profile(prof)))
        for i in ranked[:k]:
            cited, text = cited_notes(scorer, i, parts, prof, quiz=not with_like)
            explained.append(text != GENERIC)
            if cited:
                precision.append(np.mean([n in cat.notes_of(i) and n in q_notes for n in cited]))
            if with_like or not cited:
                continue
            # deletion test on the quiz answers (only for perfumes whose explanation cites notes)
            kept = [n for n in seed_notes if n not in cited]
            after = scorer.rank(Profile(seed_notes=kept), k, exclude=[u.seed_item]) if kept else []
            drop.append(i not in after)
            others = [n for n in seed_notes if n not in cited]
            if len(others) >= len(cited):
                removed = set(rng.choice(others, size=len(cited), replace=False).tolist())
                kept_r = [n for n in seed_notes if n not in removed]
                after_r = scorer.rank(Profile(seed_notes=kept_r), k, exclude=[u.seed_item])
                rand_drop.append(i not in after_r)
    out = {"NDCG@10": float(np.mean(ndcg)), "Recall@20": float(np.mean(rec)), **diversity_block(lists, cat, k),
           "NotePrecision": float(np.mean(precision)) if precision else None,
           "Explained@10": float(np.mean(explained)), "n_users": len(users)}
    if not with_like:
        out.update({"DeletionDrop@10": float(np.mean(drop)) if drop else None,
                    "RandomDrop@10": float(np.mean(rand_drop)) if rand_drop else None,
                    "n_deletion_tests": len(drop), "n_random_tests": len(rand_drop)})
    return out


def _mean_std(rows: list[dict]) -> dict:
    out = {}
    for key in rows[0]:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        if vals:
            out[key] = float(np.mean(vals))
            if len(vals) > 1:
                out[key + "_std"] = float(np.std(vals, ddof=1))
    return out


def run(split: str, warm: dict[str, list[str]], cold: dict[str, str], max_cold_users: int | None = None) -> dict:
    """warm: name -> checkpoints (one per seed, averaged); cold: name -> one cold-start checkpoint."""
    cfg = load_config()
    proc, ck = cfg["paths"]["processed_dir"], cfg["paths"]["checkpoints_dir"]
    cat = Catalogue(proc)
    data = EvalData.load(proc, with_test=split == "test")
    res: dict = {"split": split, "checkpoints": {"warm": warm, "cold": cold},
                 "warm": {}, "warm_beta0.5": {}, "cold_quiz": {}, "cold_quiz_like": {}}
    for name, paths in warm.items():
        for key, beta in (("warm", None), ("warm_beta0.5", 0.5)):
            rows = [warm_report(ck / p, data, split, cat, proc, beta=beta) for p in paths]
            res[key][name] = {**_mean_std(rows), "n_seeds": len(rows)}
        print(name, {k: round(v, 4) for k, v in res["warm"][name].items()}, flush=True)
    users = pd.read_parquet(proc / "cold" / "users.parquet")
    users = users[users["half"] == split]
    if max_cold_users:
        users = users.head(max_cold_users)
    for name, path in cold.items():
        res["cold_quiz"][name] = cold_report(ck / path, users, cat, proc, with_like=False)
        res["cold_quiz_like"][name] = cold_report(ck / path, users, cat, proc, with_like=True)
        print(name, "quiz", {k: round(v, 4) if isinstance(v, float) else v for k, v in res["cold_quiz"][name].items()}, flush=True)
    return res


def _fmt(d: dict, key: str, pct: bool = False) -> str:
    v = d.get(key)
    if v is None:
        return "–"
    sd = d.get(key + "_std")
    if pct:
        return f"{100 * v:.1f}%" + (f" ± {100 * sd:.1f}" if sd is not None else "")
    return f"{v:.4f}" + (f" ± {sd:.4f}" if sd is not None else "")


def markdown(res: dict) -> str:
    """One table per question; columns = models."""
    names = list(res["warm"]) or list(res["cold_quiz"])
    cold_names = list(res["cold_quiz"])
    head = lambda cols: ["| Metric | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    md = [f"# With notes vs without notes ({res['split']} split, generated by scentmatch.eval.innovation)\n"]
    if names and res["warm"]:
        md += ["\n## Known users (mean ± std over seeds, plain NLGCL score)\n"] + head(names)
        for key, label, pct in (("NDCG@10", "NDCG@10", False), ("Recall@20", "Recall@20", False),
                                ("Coverage@10", "Coverage@10", True), ("AvgPop@10", "AvgPop@10", False),
                                ("ILD@10", "Note diversity ILD@10", False), ("BrandDiversity@10", "Brand diversity@10", False),
                                ("NoteGrounded@10", "Note-grounded@10 (fidelity)", True)):
            fmt = (lambda d: f"{d[key]:.0f}") if key == "AvgPop@10" else (lambda d: _fmt(d, key, pct))
            md.append(f"| {label} | " + " | ".join(fmt(res["warm"][n]) for n in names) + " |")
        md += ["\n## Known users, served ranking (beta = 0.5)\n"] + head(names)
        for key, pct in (("NDCG@10", False), ("Coverage@10", True), ("ILD@10", False), ("NoteGrounded@10", True)):
            md.append(f"| {key} | " + " | ".join(_fmt(res["warm_beta0.5"][n], key, pct) for n in names) + " |")
    if cold_names:
        for block, title in (("cold_quiz", "Brand-new visitors, quiz only"), ("cold_quiz_like", "Brand-new visitors, quiz + 1 like")):
            md += [f"\n## {title} (served Scorer, model never saw these users)\n"] + head(cold_names)
            keys = [("NDCG@10", False), ("Recall@20", False), ("Coverage@10", True), ("ILD@10", False),
                    ("NotePrecision", True), ("Explained@10", True)]
            if block == "cold_quiz":
                keys += [("DeletionDrop@10", True), ("RandomDrop@10", True)]
            for key, pct in keys:
                md.append(f"| {key} | " + " | ".join(_fmt(res[block][n], key, pct) for n in cold_names) + " |")
    return "\n".join(md) + "\n"


def _kv(specs: list[str], many: bool = False) -> dict:
    out = dict(s.split("=", 1) for s in specs)
    return {k: v.split(",") for k, v in out.items()} if many else out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--warm", action="append", default=[], help="name=ckpt_s1.pt,ckpt_s2.pt,... trained on train.parquet (seeds are averaged)")
    ap.add_argument("--cold", action="append", default=[], help="name=checkpoint trained on cold/train.parquet")
    ap.add_argument("--max-cold-users", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = run(args.split, _kv(args.warm, many=True), _kv(args.cold), args.max_cold_users)
    out = Path(args.out) if args.out else ROOT / "reports" / f"week3_notes_ablation_{args.split}.json"
    out.write_text(json.dumps(res, indent=1))
    out.with_suffix(".md").write_text(markdown(res))
    print(markdown(res))


if __name__ == "__main__":
    main()
