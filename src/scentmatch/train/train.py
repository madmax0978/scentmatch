"""Training / validation loop (ML Engineer - Ops).

* BPR with one uniformly sampled negative per training positive (re-drawn if seen),
  plus NLGCL neighbour-layer contrastive loss and L2 regularisation;
* validation NDCG@10 after every epoch through the shared evaluation harness;
* early stopping on validation NDCG@10, best checkpoint saved to checkpoints/;
* per-epoch losses and metrics written to a CSV log (used for the loss curves).

Run:  python -m scentmatch.train.train [--epochs N] [--use-notes] [--forward-only]
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


def build_model(cfg: dict, data: EvalData, use_notes: bool) -> tuple[NLGCL, pd.DataFrame | None]:
    proc = cfg["paths"]["processed_dir"]
    item_notes, n_notes = None, 0
    if use_notes:
        item_notes = pd.read_parquet(proc / "item_notes.parquet")
        n_notes = len(json.loads((proc / "notes.json").read_text()))
    graph = build_graph(data.train, data.n_users, data.n_items, item_notes, n_notes)
    return NLGCL(graph, **cfg["model"]), item_notes


def make_score_fn(model: NLGCL):
    model.eval()
    users_e, items_e, _ = model.final_embeddings()

    def score(users: np.ndarray) -> np.ndarray:
        return (users_e[torch.as_tensor(users)] @ items_e.T).numpy()
    return score


def train(cfg: dict, epochs: int | None = None, use_notes: bool | None = None,
          run_name: str | None = None, forward_only: bool = False) -> dict:
    tcfg, mcfg = cfg["train"], cfg["model"]
    use_notes = mcfg["use_notes"] if use_notes is None else use_notes
    mcfg["use_notes"] = use_notes
    epochs = epochs or tcfg["epochs"]
    set_seed(tcfg["seed"])
    rng = np.random.default_rng(tcfg["seed"])

    data = EvalData.load(cfg["paths"]["processed_dir"])
    model, item_notes = build_model(cfg, data, use_notes)
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
    if item_notes is not None:
        note_items = item_notes["item_idx"].to_numpy()
        note_ids = item_notes["note_idx"].to_numpy()

    run_name = run_name or ("nlgcl_notes" if use_notes else "nlgcl")
    ckpt_dir = cfg["paths"]["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(cfg["paths"]["processed_dir"]).parents[1] / "logs" / f"train_{run_name}.csv"
    log_path.parent.mkdir(exist_ok=True)
    best, best_epoch, patience = -1.0, -1, 0
    history = []
    bs = tcfg["batch_size"]

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        perm = rng.permutation(len(tr_u))
        sums = {"loss": 0.0, "bpr": 0.0, "cl": 0.0, "reg": 0.0}
        n_batches = 0
        for start in range(0, len(perm), bs):
            idx = perm[start:start + bs]
            users, pos = tr_u[idx], tr_i[idx]
            neg = sample_negatives(users, data.n_items, seen_keys, rng)
            note_pairs = None
            if item_notes is not None:
                j = rng.integers(0, len(note_items), size=min(bs, len(note_items)))
                note_pairs = (torch.as_tensor(note_items[j]), torch.as_tensor(note_ids[j]))
            out = model.loss(torch.as_tensor(users), torch.as_tensor(pos), torch.as_tensor(neg), note_pairs)
            optim.zero_grad()
            out["loss"].backward()
            optim.step()
            for k in sums:
                sums[k] += out[k].item()
            n_batches += 1
        row = {"epoch": epoch, **{k: v / n_batches for k, v in sums.items()}, "train_seconds": time.time() - t0}

        model.eval()
        with torch.no_grad():
            val = evaluate_scores(make_score_fn(model), data, "val")
            # validation BPR loss on (user, val item, random negative) for the loss curves
            vu = torch.as_tensor(data.val["user_idx"].to_numpy())
            vp = torch.as_tensor(data.val["item_idx"].to_numpy())
            vn = torch.as_tensor(sample_negatives(vu.numpy(), data.n_items, seen_keys, rng))
            u_e, i_e, _ = model.final_embeddings()
            val_bpr = float(torch.nn.functional.softplus(
                -((u_e[vu] * i_e[vp]).sum(1) - (u_e[vu] * i_e[vn]).sum(1))).mean())
        row.update({"val_bpr": val_bpr, "val_NDCG@10": val["NDCG@10"], "val_Recall@20": val["Recall@20"],
                    "val_Coverage@10": val["Coverage@10"]})
        history.append(row)
        print({k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()}, flush=True)

        if not np.isfinite(row["loss"]):
            raise RuntimeError("loss is NaN/inf - stopping")
        if val["NDCG@10"] > best:
            best, best_epoch, patience = val["NDCG@10"], epoch, 0
            torch.save({"state_dict": model.state_dict(), "config": cfg["model"], "epoch": epoch,
                        "val": val, "n_users": data.n_users, "n_items": data.n_items,
                        "n_notes": model.n_notes}, ckpt_dir / f"{run_name}.pt")
        else:
            patience += 1
            if patience >= tcfg["early_stop_patience"]:
                print(f"early stopping at epoch {epoch}")
                break

    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    summary = {"run": run_name, "best_epoch": best_epoch, "best_val_NDCG@10": best,
               "epochs_run": len(history), "checkpoint": str(ckpt_dir / f"{run_name}.pt"), "log": str(log_path)}
    print(json.dumps(summary))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--use-notes", action="store_true")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--forward-only", action="store_true", help="build the graph and run one forward pass")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.lr:
        cfg["train"]["lr"] = args.lr
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    train(cfg, args.epochs, args.use_notes or None, args.run_name, args.forward_only)


if __name__ == "__main__":
    main()
