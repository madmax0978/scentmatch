"""Serving logic shared by the API: one interface, two implementations.

* MockRecommender  - Week 2: no model needed; returns popular perfumes (deterministic per
                     user) with real catalogue data, so API and UI can be built in parallel.
* ModelRecommender - loads a trained NLGCL checkpoint; personalised ranking for known users,
                     quiz fold-in for new users, note-based explanations.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scentmatch.model.graph import build_graph
from scentmatch.model.nlgcl import NLGCL


@dataclass
class Rec:
    item_idx: int
    score: float
    explanation: str


class Catalogue:
    def __init__(self, processed_dir: Path):
        self.items = pd.read_parquet(processed_dir / "items.parquet").set_index("item_idx").sort_index()
        self.note_names: list[str] = json.loads((processed_dir / "notes.json").read_text())
        self.note_idx = {n: i for i, n in enumerate(self.note_names)}
        inotes = pd.read_parquet(processed_dir / "item_notes.parquet")
        self.item_notes = {i: set(g) for i, g in inotes.groupby("item_idx")["note_idx"]}
        self.items_by_note = {n: g.to_numpy() for n, g in inotes.groupby("note_idx")["item_idx"]}
        users = pd.read_parquet(processed_dir / "users.parquet")
        self.user_idx = dict(zip(users["user_id"], users["user_idx"]))
        known = pd.concat([pd.read_parquet(processed_dir / f"{s}.parquet") for s in ("train", "val", "test")])
        self.user_items = {u: g.to_numpy() for u, g in known.groupby("user_idx")["item_idx"]}
        self.popularity = self.items["train_popularity"].to_numpy().astype(float)

    @property
    def n_items(self) -> int:
        return len(self.items)

    def notes_of(self, item_idx: int) -> set[int]:
        return self.item_notes.get(int(item_idx), set())

    def explain(self, item_idx: int, reference_notes: set[int], order: np.ndarray | None = None) -> str:
        shared = self.notes_of(item_idx) & reference_notes
        if not shared:
            return "Popular with people who share your taste"
        names = [self.note_names[n] for n in shared]
        if order is not None:                         # order by note-user similarity
            names = [self.note_names[n] for n in sorted(shared, key=lambda n: -order[n])]
        return "Shares " + ", ".join(names[:3]) + " with your profile"


class MockRecommender:
    mode = "mock"

    def __init__(self, catalogue: Catalogue):
        self.cat = catalogue
        self.pool = np.argsort(-catalogue.popularity)[:200]

    def _pick(self, key: str, k: int, exclude=()) -> list[Rec]:
        seed = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        pool = [i for i in self.pool if i not in set(exclude)]
        chosen = rng.choice(pool, size=min(k, len(pool)), replace=False)
        return [Rec(int(i), round(1.0 - r / (2 * k), 3), "[mock] popular perfume") for r, i in enumerate(chosen)]

    def recommend_user(self, user_id: str, k: int = 10) -> list[Rec]:
        return self._pick(user_id, k)

    def recommend_notes(self, notes: list[str], k: int = 10) -> list[Rec]:
        ref = {self.cat.note_idx[n] for n in notes if n in self.cat.note_idx}
        recs = self._pick("|".join(sorted(notes)), k)
        return [Rec(r.item_idx, r.score, "[mock] " + self.cat.explain(r.item_idx, ref)) for r in recs]


class ModelRecommender:
    mode = "model"

    def __init__(self, catalogue: Catalogue, checkpoint: Path, processed_dir: Path):
        self.cat = catalogue
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        train = pd.read_parquet(processed_dir / "train.parquet")
        inotes = pd.read_parquet(processed_dir / "item_notes.parquet") if ckpt["n_notes"] else None
        graph = build_graph(train, ckpt["n_users"], ckpt["n_items"], inotes, ckpt["n_notes"])
        model = NLGCL(graph, **cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        u, i, n = model.final_embeddings()
        self.user_e, self.item_e = u.numpy(), i.numpy()
        self.note_e = n.numpy() if ckpt["n_notes"] else None
        self.epoch = ckpt["epoch"]

    def predict(self, user_vec: np.ndarray, exclude: np.ndarray, k: int) -> list[tuple[int, float]]:
        scores = self.item_e @ user_vec
        scores[exclude] = -np.inf
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]

    def recommend_user(self, user_id: str, k: int = 10) -> list[Rec]:
        uidx = self.cat.user_idx.get(user_id)
        if uidx is None:
            raise KeyError(user_id)
        seen = self.cat.user_items.get(uidx, np.array([], dtype=int))
        ref = set().union(*(self.cat.notes_of(i) for i in seen)) if len(seen) else set()
        order = self.note_e @ self.user_e[uidx] if self.note_e is not None else None
        return [Rec(i, s, self.cat.explain(i, ref, order))
                for i, s in self.predict(self.user_e[uidx], seen, k)]

    def notes_vector(self, note_ids: list[int]) -> np.ndarray:
        """Fold-in: mean of seed-note embeddings, or (no note nodes) of perfumes carrying them."""
        if self.note_e is not None:
            return self.note_e[note_ids].mean(0)
        items = np.unique(np.concatenate([self.cat.items_by_note.get(n, []) for n in note_ids]))
        return self.item_e[items.astype(int)].mean(0)

    def recommend_notes(self, notes: list[str], k: int = 10) -> list[Rec]:
        ids = [self.cat.note_idx[n] for n in notes if n in self.cat.note_idx]
        if not ids:
            raise ValueError("none of the seed notes is in the vocabulary")
        vec = self.notes_vector(ids)
        order = self.note_e @ vec if self.note_e is not None else None
        return [Rec(i, s, self.cat.explain(i, set(ids), order))
                for i, s in self.predict(vec, np.array([], dtype=int), k)]
