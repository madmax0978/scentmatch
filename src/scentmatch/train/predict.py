"""Serving logic shared by the API: one interface, two implementations.

* MockRecommender  - Week 2: no model needed; returns popular perfumes (deterministic per
                     user) with real catalogue data, so API and UI can be built in parallel.
* ModelRecommender - loads a trained NLGCL checkpoint; personalised ranking for known users,
                     quiz fold-in for new users, note-based explanations.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scentmatch.model.graph import build_graph
from scentmatch.model.nlgcl import NLGCL


_SIZE = re.compile(r"\b\d+(\.\d+)?\s*(ml|oz|ounce|ounces|fl\.?\s*oz|fl)\b.*$", re.I)


def short_title(title: str) -> str:
    """'Yara by Lattafa Perfumes | Eau De Parfum - 100ml (3.4 fl oz)' -> 'Yara by Lattafa Perfumes'."""
    t = re.sub(r"^\s*\d+(\.\d+)?\s*(ml|oz)\b\s*", "", str(title), flags=re.I)
    t = re.sub(r"^(men'?s|women'?s)\s+(perfume|cologne)?\s*(fragrance)?\s*,?\s*(by\s+)?", "", t, flags=re.I)
    t = re.split(r"\s[|\-–—]\s|,|\(", t)[0]
    t = _SIZE.sub("", t)
    t = re.sub(r"\b(eau de (parfum|toilette|cologne)|edp|edt|spray|perfume|cologne)\b.*$", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip(" -|,.")
    return t or str(title)[:60]


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
        train = pd.read_parquet(processed_dir / "train.parquet")
        self.train_items = {u: g.to_numpy() for u, g in train.groupby("user_idx")["item_idx"]}
        self.popularity = self.items["train_popularity"].to_numpy().astype(float)
        # perfume x note matrix, weighted by IDF (rare notes say more about a taste than
        # "musk" or "vanilla", which half the catalogue carries), rows L2-normalised
        n_items, n_notes = len(self.items), len(self.note_names)
        df = np.bincount(inotes["note_idx"], minlength=n_notes).astype(float)
        self.idf = np.log((1 + n_items) / (1 + df)) + 1.0
        x = np.zeros((n_items, n_notes), dtype=np.float32)
        x[inotes["item_idx"].to_numpy(), inotes["note_idx"].to_numpy()] = 1.0
        x *= self.idf
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        self.note_matrix = np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)
        self.short_titles = [short_title(t) for t in self.items["title"]]
        self.n_notes_per_item = (x > 0).sum(1)
        # quality prior in [0, 1]: Amazon rating x log(number of ratings); sets, minis and body
        # products are kept in the catalogue but pushed down for the quiz showcase
        rating = self.items["average_rating"].fillna(3.5).to_numpy()
        count = np.log1p(self.items["rating_number"].fillna(0).to_numpy())
        self.quality = np.clip((rating - 3.0) / 2.0, 0, 1) * (count / count.max())
        self.is_set = self.items["title"].str.contains(
            r"gift set|\bset\b|piece|body mist|body spray|lotion|shower gel|sampler|sample|travel|mini\b",
            case=False, regex=True).to_numpy()

    @property
    def n_items(self) -> int:
        return len(self.items)

    def notes_of(self, item_idx: int) -> set[int]:
        return self.item_notes.get(int(item_idx), set())

    def content_scores(self, note_ids: list[int]) -> np.ndarray:
        """How well each perfume matches the seed notes, in [0, 1]:
        half TF-IDF cosine (precision: the perfume is *about* these notes) and half
        coverage (recall: share of the seed notes, IDF-weighted, that the perfume carries),
        so a perfume matching one rare note cannot beat one matching five."""
        q = np.zeros(len(self.note_names), dtype=np.float32)
        np.add.at(q, note_ids, 1.0)
        q *= self.idf
        cosine = self.note_matrix @ (q / (np.linalg.norm(q) or 1.0))
        coverage = (self.note_matrix > 0).astype(np.float32) @ q / (q.sum() or 1.0)
        # descriptions listing 30-40 notes would match every quiz: damp beyond 15 notes
        damp = np.minimum(1.0, np.sqrt(15.0 / np.maximum(self.n_notes_per_item, 1)))
        return (0.5 * cosine + 0.5 * np.sqrt(coverage)) * damp

    def diversify(self, ranked: np.ndarray, k: int, max_per_brand: int | None = None) -> list[int]:
        """Walk a ranked list, skipping other sizes/editions of a perfume already picked
        and (optionally) capping how many perfumes one brand can take."""
        out, seen_titles, brands = [], set(), {}
        for i in ranked:
            t = self.short_titles[i].lower()
            b = str(self.items["store"].iat[i]).lower()
            if t in seen_titles or (max_per_brand and brands.get(b, 0) >= max_per_brand):
                continue
            out.append(int(i)); seen_titles.add(t); brands[b] = brands.get(b, 0) + 1
            if len(out) == k:
                break
        return out

    def explain(self, item_idx: int, reference_notes: set[int], order: np.ndarray | None = None,
                quiz: bool = False) -> str:
        shared = self.notes_of(item_idx) & reference_notes
        if not shared:
            return "Popular with people who share your taste"
        names = [self.note_names[n] for n in shared]
        if order is not None:                         # order by note-user similarity
            names = [self.note_names[n] for n in sorted(shared, key=lambda n: -order[n])]
        names = names[:3]
        listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        return (f"Because you chose {listed}" if quiz else f"Shares {listed} with perfumes you loved")


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
        top = np.argpartition(-scores, 5 * k)[:5 * k]
        top = top[np.argsort(-scores[top])]
        return [(i, float(scores[i])) for i in self.cat.diversify(top, k)]

    def recommend_user(self, user_id: str, k: int = 10) -> list[Rec]:
        uidx = self.cat.user_idx.get(user_id)
        if uidx is None:
            raise KeyError(user_id)
        seen = self.cat.user_items.get(uidx, np.array([], dtype=int))
        ref = set().union(*(self.cat.notes_of(i) for i in seen)) if len(seen) else set()
        order = self.note_e @ self.user_e[uidx] if self.note_e is not None else None
        return [Rec(i, s, self.cat.explain(i, ref, order))
                for i, s in self.predict(self.user_e[uidx], seen, k)]

    def recommend_notes(self, notes: list[str], k: int = 10) -> list[Rec]:
        """New user from quiz notes (hybrid fold-in).

        1. content: TF-IDF cosine between the seed notes and each perfume's notes
           -> perfumes that really carry the chosen notes;
        2. collaborative: NLGCL embedding of the best content matches, averaged into a
           taste vector; cosine with every perfume -> what people with that taste also love;
        3. score = 0.6 content + 0.25 collaborative + 0.15 quality (Amazon rating and
           number of ratings), then de-duplicate sizes and cap brands at 2.
        """
        ids = [self.cat.note_idx[n] for n in notes if n in self.cat.note_idx]
        if not ids:
            raise ValueError("none of the seed notes is in the vocabulary")
        content = self.cat.content_scores(ids)
        anchors = np.argsort(-content)[:30]
        anchors = anchors[content[anchors] > 0]
        item_n = self.item_e / (np.linalg.norm(self.item_e, axis=1, keepdims=True) + 1e-9)
        taste = (item_n[anchors] * content[anchors, None]).sum(0) if len(anchors) else item_n.mean(0)
        collab = item_n @ (taste / (np.linalg.norm(taste) + 1e-9))
        score = 0.6 * content + 0.25 * np.clip(collab, 0, None) + 0.15 * self.cat.quality
        score[content == 0] *= 0.5                       # perfumes sharing no chosen note go last
        score[self.cat.is_set] *= 0.7                    # gift sets / minis / body products go lower
        ranked = np.argsort(-score)[: 20 * k]
        chosen = self.cat.diversify(ranked, k, max_per_brand=2)
        order = np.zeros(len(self.cat.note_names)); order[ids] = self.cat.idf[ids]
        return [Rec(i, round(float(score[i]), 4), self.cat.explain(i, set(ids), order, quiz=True)) for i in chosen]
