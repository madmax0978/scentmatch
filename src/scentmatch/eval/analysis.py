"""How does the model rank? Figures and numbers for the Week 3 report (validation split only).

1. Loss curves of the final runs (train BPR vs validation BPR, validation NDCG@10).
2. Popularity inside the model: the length |e_i| of a perfume's embedding vs. how many
   training likes it has. The NLGCL score <e_u, e_i> = |e_u| |e_i| cos(e_u, e_i), so a long
   vector means "recommended to everybody". We measure the correlation.
3. The popularity knob beta: rank by cos(e_u, e_i) * |e_i|^beta for beta in [0, 1.25] and
   plot accuracy (NDCG@10) against discovery (share of recommendations outside the 20% most
   popular perfumes). beta = 1 is the plain dot product the model was trained with.
4. What the 1-3 star reviews really say (scentmatch.data.feedback).

Run:  python -m scentmatch.eval.analysis --ckpt final_nlgcl_s1.pt --runs final_nlgcl_s1 final_lightgcn_s1
Out:  reports/figures/week3_*.png, reports/week3_analysis.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scentmatch.config import load_config  # noqa: E402
from scentmatch.eval.evaluate import tail_share  # noqa: E402
from scentmatch.eval.harness import EvalData, _summarise, rank_scores  # noqa: E402
from scentmatch.train.predict import load_embeddings  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
FIG = ROOT / "reports" / "figures"
# reference categorical palette (validated for colour-vision deficiencies), fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.6)


def loss_curves(runs: list[str], labels: list[str]) -> dict:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=SURFACE)
    out = {}
    for n, (run, label) in enumerate(zip(runs, labels)):
        h = pd.read_csv(ROOT / "logs" / "runs" / f"{run}.csv")
        best = int(h["val_NDCG@10"].idxmax())
        a1.plot(h["epoch"], h["bpr"], color=C[n], lw=2, label=f"{label} · train")
        a1.plot(h["epoch"], h["val_bpr"], color=C[n], lw=2, ls="--", label=f"{label} · validation")
        a2.plot(h["epoch"], h["val_NDCG@10"], color=C[n], lw=2, label=label)
        a2.plot(h["epoch"].iloc[best], h["val_NDCG@10"].iloc[best], "o", color=C[n], ms=8,
                mec=SURFACE, mew=2)
        out[label] = {"best_epoch": int(h["epoch"].iloc[best]), "epochs_run": len(h),
                      "train_bpr_at_best": float(h["bpr"].iloc[best]), "val_bpr_at_best": float(h["val_bpr"].iloc[best]),
                      "val_bpr_min": float(h["val_bpr"].min()), "val_bpr_start": float(h["val_bpr"].iloc[0])}
    style(a1, "BPR loss (solid: train, dashed: validation)", "epoch", "BPR loss")
    style(a2, "Validation NDCG@10 (dot = kept checkpoint)", "epoch", "NDCG@10")
    a1.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    a2.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.set_size_inches(10, 4.2)
    fig.tight_layout()
    fig.savefig(FIG / "week3_loss_curves.png", dpi=160)
    plt.close(fig)
    return out


def popularity_inside(ckpt: Path, proc: Path, data: EvalData) -> dict:
    e = load_embeddings(ckpt, proc)
    norm = np.linalg.norm(e["items"], axis=1)
    pop = data.popularity
    rank = lambda x: pd.Series(x).rank().to_numpy()  # noqa: E731
    spearman = float(np.corrcoef(rank(norm), rank(pop))[0, 1])
    fig, ax = plt.subplots(figsize=(5, 3.6), facecolor=SURFACE)
    ax.scatter(np.log1p(pop), norm, s=8, color=C[0], alpha=0.35, linewidths=0)
    style(ax, f"Embedding length grows with popularity (Spearman {spearman:.2f})",
          "log(1 + training likes of the perfume)", "|e_i|")
    fig.tight_layout()
    fig.savefig(FIG / "week3_popularity_in_embeddings.png", dpi=160)
    plt.close(fig)

    # the beta knob
    u, i = e["users"], e["items"]
    un = u / np.linalg.norm(u, axis=1, keepdims=True)
    inn = i / norm[:, None]
    targets = data.targets("val")
    rows = []
    for beta in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25):
        w = norm ** beta
        ranks = rank_scores(lambda users: (un[users] @ inn.T) * w, data, "val")
        m = _summarise(ranks, targets, data)
        rows.append({"beta": beta, "NDCG@10": m["NDCG@10"], "Coverage@10": m["Coverage@10"],
                     "AvgPop@10": m["AvgPop@10"], "TailShare@10": tail_share(ranks, pop)})
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5, 3.6), facecolor=SURFACE)
    ax.plot(d["TailShare@10"], d["NDCG@10"], color=C[0], lw=2, marker="o", ms=8, mec=SURFACE, mew=2)
    offsets = {0.0: (-34, -3), 0.25: (8, 4), 0.5: (4, 8), 1.0: (-58, 4), 1.25: (-8, -16)}
    text = {0.5: "β=0.5 (served)", 1.0: "β=1 (as trained)"}
    for r in d.itertuples():
        if r.beta in offsets:
            ax.annotate(text.get(r.beta, f"β={r.beta:g}"), (r._5, r._2), textcoords="offset points",
                        xytext=offsets[r.beta], fontsize=8, color=INK2)
    style(ax, "The popularity knob β: accuracy vs discovery", "share of niche perfumes in the top-10",
          "validation NDCG@10")
    ax.margins(x=0.08, y=0.15)
    fig.tight_layout()
    fig.savefig(FIG / "week3_beta_tradeoff.png", dpi=160)
    plt.close(fig)
    return {"spearman_norm_vs_popularity": spearman, "beta_tradeoff": rows}


def review_labels(proc: Path) -> dict:
    fb = pd.read_parquet(proc / "feedback.parquet")
    order = ["dislike", "like", "not_scent", "unclear"]
    names = {"dislike": "dislike (smell criticised, or 1-2★ with no reason)", "like": "likes the smell", "not_scent": "not about the smell",
             "unclear": "no reason given"}
    t = fb.groupby(["rating", "label"]).size().unstack(fill_value=0).reindex(columns=order, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=SURFACE)
    left = np.zeros(len(t))
    for n, lab in enumerate(order):
        vals = t[lab].to_numpy()
        ax.barh([f"{int(r)} star{'s' if r > 1 else ''}" for r in t.index], vals, left=left, color=C[n],
                edgecolor=SURFACE, linewidth=2, label=names[lab])
        for y, (l0, v) in enumerate(zip(left, vals)):
            if v > 250:
                ax.text(l0 + v / 2, y, f"{v:,}", ha="center", va="center", fontsize=8, color="white")
        left += vals
    style(ax, "What the 1-3 star reviews say about the scent", "number of reviews", "")
    ax.grid(False)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    fig.savefig(FIG / "week3_review_labels.png", dpi=160)
    plt.close(fig)
    return {str(int(r)): t.loc[r].to_dict() for r in t.index}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint analysed for popularity / beta")
    ap.add_argument("--runs", nargs="+", required=True, help="run names in logs/runs for the loss curves")
    ap.add_argument("--labels", nargs="+", default=None)
    args = ap.parse_args()
    cfg = load_config()
    proc = cfg["paths"]["processed_dir"]
    FIG.mkdir(parents=True, exist_ok=True)
    data = EvalData.load(proc, with_test=False)
    res = {"loss_curves": loss_curves(args.runs, args.labels or args.runs),
           "popularity": popularity_inside(cfg["paths"]["checkpoints_dir"] / args.ckpt, proc, data),
           "review_labels": review_labels(proc)}
    (ROOT / "reports" / "week3_analysis.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
