"""Training / validation loop (ML Engineer - Ops).

* BPR with one sampled negative per training positive, plus NLGCL neighbour-layer
  contrastive loss and L2 regularisation;
* validation NDCG@10 after every epoch through the shared evaluation harness;
* early stopping on validation NDCG@10, best checkpoint saved to checkpoints/;
* per-epoch losses and metrics written to logs/runs/<run>.csv (used for the loss curves).

Week 3
* the test split is never loaded here (EvalData.load(with_test=False));
* every hyper-parameter is a command-line flag, so the search and the 3-seed runs are scripted;
* `--cl-weight 0` trains LightGCN, the baseline that NLGCL extends;
* notes: distinct notes inside a batch, own loss weight, tunable perfume-note edge weight;
* explicit dislikes (`--dislike-neg-prob p`): with probability p the negative of a user who
  disliked a perfume *during the training period* is that perfume instead of a random one,
  so the model learns "not this one" and not only "not a random one";
* the checkpoint stores the final embeddings, so serving never rebuilds the graph.

Run:  python -m scentmatch.train.train [--use-notes] [--seed 1] [--cl-weight 0] ...
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scentmatch.config import load_config
from scentmatch.eval.harness import EvalData, evaluate_scores
from scentmatch.model.graph import build_graph
from scentmatch.model.nlgcl import NLGCL

ROOT = Path(__file__).resolve().parents[3]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_negatives(users: np.ndarray, n_items: int, seen_keys: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    neg = rng.integers(0, n_items, size=len(users))
    for _ in range(10):                                   # re-draw negatives that were seen
        bad = np.isin(users.astype(np.int64) * n_items + neg, seen_keys)
        if not bad.any():
            break
        neg[bad] = rng.integers(0, n_items, size=int(bad.sum()))
    return neg


class Csr:
    """Compact {row: [values]} lookup used to sample a random value per row quickly."""

    def __init__(self, rows: np.ndarray, values: np.ndarray, n_rows: int):
        order = np.argsort(rows, kind="stable")
        self.values = values[order]
        self.counts = np.bincount(rows, minlength=n_rows)
        self.offsets = np.concatenate([[0], np.cumsum(self.counts)[:-1]])

    def sample(self, rows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One random value for each row (rows must have count > 0)."""
        return self.values[self.offsets[rows] + rng.integers(0, self.counts[rows])]


def load_item_notes(proc: Path) -> tuple[pd.DataFrame, int]:
    return pd.read_parquet(proc / "item_notes.parquet"), len(json.loads((proc / "notes.json").read_text()))


def load_feedback(proc: Path) -> pd.DataFrame | None:
    """Explicit feedback built from review texts (scentmatch.data.feedback), training period only."""
    path = proc / "feedback.parquet"
    if not path.exists():
        return None
    fb = pd.read_parquet(path)
    return fb[fb["period"] == "train"]


def build_model(cfg: dict, data: EvalData, use_notes: bool) -> tuple[NLGCL, pd.DataFrame | None]:
    proc = cfg["paths"]["processed_dir"]
    item_notes, n_notes = (load_item_notes(proc) if use_notes else (None, 0))
    graph = build_graph(data.train, data.n_users, data.n_items, item_notes, n_notes,
                        note_edge_weight=cfg["model"].get("note_edge_weight", 1.0))
    return NLGCL(graph, **cfg["model"]), item_notes


def make_score_fn(model: NLGCL):
    model.eval()
    users_e, items_e, _ = model.final_embeddings()

    def score(users: np.ndarray) -> np.ndarray:
        return (users_e[torch.as_tensor(users)] @ items_e.T).numpy()
    return score


def train(cfg: dict, epochs: int | None = None, use_notes: bool | None = None,
          run_name: str | None = None, forward_only: bool = False, train_file: str = "train.parquet",
          verbose: bool = True) -> dict:
    tcfg, mcfg = cfg["train"], cfg["model"]
    use_notes = mcfg["use_notes"] if use_notes is None else use_notes
    mcfg["use_notes"] = use_notes
    epochs = epochs or tcfg["epochs"]
    set_seed(tcfg["seed"])
    rng = np.random.default_rng(tcfg["seed"])
    proc = cfg["paths"]["processed_dir"]

    data = EvalData.load(proc, with_test=False, train_file=train_file)
    if train_file != "train.parquet":            # e.g. cold-start training: validate on users still in train
        data.val = data.val[data.val["user_idx"].isin(data.train["user_idx"].unique())]
    feedback = load_feedback(proc) if (tcfg.get("dislike_neg_prob", 0) > 0 or tcfg.get("rescued_positives")) else None
    if feedback is not None and tcfg.get("rescued_positives"):
        # low-star reviews whose text praises the scent ("smells amazing but does not last")
        extra = feedback[(feedback["label"] == "like") & feedback["user_idx"].isin(data.train["user_idx"].unique())]
        data.train = pd.concat([data.train, extra[["user_idx", "item_idx", "timestamp", "rating"]]], ignore_index=True)
        data.__post_init__()
    model, item_notes = build_model(cfg, data, use_notes)
    if verbose:
        print(f"graph: {data.n_users} users, {data.n_items} perfumes, {model.n_notes} notes, "
              f"{model.edge_index.shape[1]} directed edges; params={sum(p.numel() for p in model.parameters()):,}")

    if forward_only:                                   # Week 2 check: one forward pass
        t0 = time.time()
        users_e, items_e, notes_e, layers = model()
        u = torch.arange(8)
        out = model.loss(u, torch.as_tensor(data.train["item_idx"].values[:8]), torch.randint(0, data.n_items, (8,)))
        info = {"users_emb": list(users_e.shape), "items_emb": list(items_e.shape),
                "notes_emb": list(notes_e.shape), "n_layers": len(layers) - 1,
                "loss": out["loss"].item(), "bpr": out["bpr"].item(), "cl": out["cl"].item(),
                "finite": bool(torch.isfinite(out["loss"])), "seconds": round(time.time() - t0, 3)}
        print(json.dumps(info))
        return info

    optim = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    tr_u = data.train["user_idx"].to_numpy()
    tr_i = data.train["item_idx"].to_numpy()
    seen_keys = np.sort(tr_u.astype(np.int64) * data.n_items + tr_i)
    note_csr = None
    if item_notes is not None:
        note_csr = Csr(item_notes["note_idx"].to_numpy(), item_notes["item_idx"].to_numpy(), model.n_notes)
        notes_with_items = np.flatnonzero(note_csr.counts)
    dis_p = float(tcfg.get("dislike_neg_prob", 0.0))
    dis_csr = None
    if dis_p > 0 and feedback is not None:
        dis = feedback[feedback["label"] == "dislike"]
        dis_csr = Csr(dis["user_idx"].to_numpy(), dis["item_idx"].to_numpy(), data.n_users)

    run_name = run_name or ("nlgcl_notes" if use_notes else "nlgcl")
    ckpt_dir = cfg["paths"]["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "logs" / "runs" / f"{run_name}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best, best_epoch, patience = -1.0, -1, 0
    history = []
    bs = tcfg["batch_size"]
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        perm = rng.permutation(len(tr_u))
        sums = {"loss": 0.0, "bpr": 0.0, "cl": 0.0, "cl_note": 0.0, "reg": 0.0}
        n_batches = 0
        for start in range(0, len(perm), bs):
            idx = perm[start:start + bs]
            users, pos = tr_u[idx], tr_i[idx]
            neg = sample_negatives(users, data.n_items, seen_keys, rng)
            if dis_csr is not None:                       # explicit dislikes as hard negatives
                has = dis_csr.counts[users] > 0
                swap = has & (rng.random(len(users)) < dis_p)
                if swap.any():
                    neg[swap] = dis_csr.sample(users[swap], rng)
            note_pairs = None
            if note_csr is not None:                      # distinct notes, one random perfume each
                notes = rng.permutation(notes_with_items)[:bs]
                note_pairs = (torch.as_tensor(note_csr.sample(notes, rng)), torch.as_tensor(notes))
            out = model.loss(torch.as_tensor(users), torch.as_tensor(pos), torch.as_tensor(neg), note_pairs)
            optim.zero_grad()
            out["loss"].backward()
            optim.step()
            for k in sums:
                sums[k] += float(out[k].detach())
            n_batches += 1
        row = {"epoch": epoch, **{k: v / n_batches for k, v in sums.items()}, "train_seconds": time.time() - t0}

        model.eval()
        with torch.no_grad():
            val = evaluate_scores(make_score_fn(model), data, "val")
            # validation BPR loss on (user, val item, random negative) for the loss curves
            vu = torch.as_tensor(data.val["user_idx"].to_numpy().copy())
            vp = torch.as_tensor(data.val["item_idx"].to_numpy().copy())
            vn = torch.as_tensor(sample_negatives(vu.numpy(), data.n_items, seen_keys,
                                                  np.random.default_rng(epoch)))
            u_e, i_e, _ = model.final_embeddings()
            val_bpr = float(torch.nn.functional.softplus(
                -((u_e[vu] * i_e[vp]).sum(1) - (u_e[vu] * i_e[vn]).sum(1))).mean())
        row.update({"val_bpr": val_bpr, "val_NDCG@10": val["NDCG@10"], "val_Recall@20": val["Recall@20"],
                    "val_Coverage@10": val["Coverage@10"], "val_AvgPop@10": val["AvgPop@10"]})
        history.append(row)
        if verbose:
            print({k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()}, flush=True)

        if not np.isfinite(row["loss"]):
            raise RuntimeError("loss is NaN/inf - stopping")
        if val["NDCG@10"] > best:
            best, best_epoch, patience = val["NDCG@10"], epoch, 0
            u_e, i_e, n_e = model.final_embeddings()
            torch.save({"state_dict": model.state_dict(), "config": dict(cfg["model"]),
                        "train_config": dict(tcfg), "train_file": train_file, "epoch": epoch,
                        "val": val, "n_users": data.n_users, "n_items": data.n_items, "n_notes": model.n_notes,
                        "embeddings": {"users": u_e.clone(), "items": i_e.clone(), "notes": n_e.clone()}},
                       ckpt_dir / f"{run_name}.pt")
        else:
            patience += 1
            if patience >= tcfg["early_stop_patience"]:
                if verbose:
                    print(f"early stopping at epoch {epoch}")
                break

    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    summary = {"run": run_name, "best_epoch": best_epoch, "best_val_NDCG@10": best,
               "best_val": history[best_epoch - 1], "epochs_run": len(history),
               "seconds": round(time.time() - t_start, 1),
               "checkpoint": str(ckpt_dir / f"{run_name}.pt"), "log": str(log_path)}
    print(json.dumps({k: summary[k] for k in ("run", "best_epoch", "best_val_NDCG@10", "epochs_run", "seconds")}))
    return summary


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    m, t = cfg["model"], cfg["train"]
    for flag, (section, key) in {
        "lr": (t, "lr"), "batch_size": (t, "batch_size"), "seed": (t, "seed"),
        "dislike_neg_prob": (t, "dislike_neg_prob"), "patience": (t, "early_stop_patience"),
        "cl_weight": (m, "cl_weight"), "note_cl_weight": (m, "note_cl_weight"), "reg_weight": (m, "reg_weight"),
        "n_layers": (m, "n_layers"), "cl_layers": (m, "cl_layers"), "emb_size": (m, "embedding_size"),
        "cl_temp": (m, "cl_temp"), "note_edge_weight": (m, "note_edge_weight"),
    }.items():
        value = getattr(args, flag, None)
        if value is not None:
            section[key] = value
    if args.rescued_positives:
        t["rescued_positives"] = True
    m["cl_layers"] = min(m["cl_layers"], m["n_layers"])
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--use-notes", action="store_true")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--train-file", default="train.parquet", help="e.g. cold/train.parquet for cold-start runs")
    for flag, typ in (("lr", float), ("batch-size", int), ("seed", int), ("cl-weight", float),
                      ("note-cl-weight", float), ("reg-weight", float), ("n-layers", int), ("cl-layers", int),
                      ("emb-size", int), ("cl-temp", float), ("note-edge-weight", float),
                      ("dislike-neg-prob", float), ("patience", int)):
        ap.add_argument(f"--{flag}", type=typ, default=None)
    ap.add_argument("--rescued-positives", action="store_true",
                    help="add 1-3 star reviews whose text praises the scent as training positives")
    ap.add_argument("--forward-only", action="store_true", help="build the graph and run one forward pass")
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    train(cfg, args.epochs, args.use_notes or None, args.run_name, args.forward_only, args.train_file)


if __name__ == "__main__":
    main()
