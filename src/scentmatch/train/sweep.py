"""Hyper-parameter search on the VALIDATION split only (ML Engineer - Ops, Week 3).

Each line of a sweep is a set of overrides on configs/default.yaml. Runs go in parallel
(2 torch threads each); every result is appended to reports/sweeps/<name>.csv with the
best validation metrics and the best epoch. The test split is never loaded (train.py
uses EvalData.load(with_test=False)).

Run:  python -m scentmatch.train.sweep <stage>      stages: core, layers, notes, feedback, seeds
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports" / "sweeps"


def _run(job: dict) -> dict:
    import torch
    torch.set_num_threads(2)
    from scentmatch.config import load_config
    from scentmatch.train.train import train

    cfg = load_config()
    over = dict(job["overrides"])
    use_notes = over.pop("use_notes", False)
    train_file = over.pop("train_file", "train.parquet")
    for key, value in over.items():
        section = "train" if key in ("seed", "dislike_neg_prob", "rescued_positives", "lr", "batch_size") else "model"
        cfg[section][key] = value
    cfg["model"]["cl_layers"] = min(cfg["model"]["cl_layers"], cfg["model"]["n_layers"])
    s = train(cfg, use_notes=use_notes, run_name=job["name"], train_file=train_file, verbose=False)
    best = s["best_val"]
    return {"name": job["name"], **job["overrides"], "best_epoch": s["best_epoch"], "epochs_run": s["epochs_run"],
            "val_NDCG@10": best["val_NDCG@10"], "val_Recall@20": best["val_Recall@20"],
            "val_Coverage@10": best["val_Coverage@10"], "val_AvgPop@10": best["val_AvgPop@10"],
            "train_bpr": best["bpr"], "val_bpr": best["val_bpr"], "seconds": s["seconds"]}


def stage_jobs(stage: str, best: dict) -> list[dict]:
    jobs = []
    if stage == "core":          # contrastive weight (0 = LightGCN) x L2 regularisation
        for cl, reg in itertools.product([0.0, 5e-5, 2e-4, 1e-3], [1e-4, 1e-3, 1e-2]):
            jobs.append({"name": f"core_cl{cl:g}_reg{reg:g}", "overrides": {"cl_weight": cl, "reg_weight": reg}})
    elif stage == "core2":       # finer: small contrastive weights x stronger regularisation
        for cl, reg in itertools.product([0.0, 1e-5, 2e-5, 5e-5], [0.03, 0.1]):
            jobs.append({"name": f"core2_cl{cl:g}_reg{reg:g}", "overrides": {"cl_weight": cl, "reg_weight": reg}})
        for cl in (1e-5, 2e-5):
            jobs.append({"name": f"core2_cl{cl:g}_reg0.01", "overrides": {"cl_weight": cl, "reg_weight": 0.01}})
    elif stage == "layers":
        for L in (1, 2, 3):
            for cl in (0.0, best["cl_weight"]):
                jobs.append({"name": f"layers_L{L}_cl{cl:g}",
                             "overrides": {"n_layers": L, "cl_weight": cl, "reg_weight": best["reg_weight"]}})
    elif stage == "notes":       # note-edge weight x note contrastive weight
        for w, ncl in itertools.product([1.0, 0.3, 0.1], [0.0, best["cl_weight"], 5 * best["cl_weight"]]):
            jobs.append({"name": f"notes_w{w:g}_ncl{ncl:g}",
                         "overrides": {**best, "use_notes": True, "note_edge_weight": w, "note_cl_weight": ncl}})
    elif stage == "feedback":    # explicit dislikes as hard negatives, rescued positives
        for notes in (False, True):
            base = {**best, **(best.get("_notes") or {})} if notes else dict(best)
            base.pop("_notes", None)
            for p, rescued in [(0.0, True), (0.3, False), (0.6, False), (0.3, True)]:
                jobs.append({"name": f"fb_{'notes' if notes else 'plain'}_p{p:g}_r{int(rescued)}",
                             "overrides": {**base, "use_notes": notes, "dislike_neg_prob": p,
                                           "rescued_positives": rescued}})
    return [j for j in jobs if "_notes" not in j["overrides"]]


def run_jobs(jobs: list[dict], name: str, workers: int = 4) -> list[dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.csv"
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for row in ex.map(_run, jobs):
            rows.append(row)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "name", k))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", help="core | core2 | layers | notes | feedback, or a name used with --jobs")
    ap.add_argument("--jobs", default=None, help="JSON file with a list of {name, overrides} jobs")
    ap.add_argument("--best", default="{}", help="JSON of the best overrides from the previous stage")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    jobs = json.loads(Path(args.jobs).read_text()) if args.jobs else stage_jobs(args.stage, json.loads(args.best))
    run_jobs(jobs, args.stage, args.workers)


if __name__ == "__main__":
    main()
