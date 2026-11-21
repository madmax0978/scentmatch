"""Serving logic shared by the API and by the offline evaluation (Week 3).

One scoring function for everybody, built so that every number can be explained:

    score(i) = w_notes * NOTES(i) + w_collab * COLLAB(i) - w_avoid * AVOID(i)

* NOTES(i)  in [0, 1]: how well perfume i carries the user's notes. The note profile is the
            quiz notes + the notes of every perfume the user liked (IDF-weighted: rare notes
            such as "oud" say more about a taste than "musk"). Half cosine (the perfume is
            *about* these notes), half coverage (it carries *many* of them). Each note's share
            of NOTES(i) is reported.
* COLLAB(i) in [-1, 1]: cosine between perfume i and the user's taste vector in the NLGCL
            embedding space, multiplied by (|e_i| / max |e|)^beta. The length of a perfume's
            embedding is how popular the model thinks it is (we measured it); beta is the one
            knob that says how much popularity we let in: 0 = pure taste, 1 = the model as
            trained. beta and the weights are tuned on held-out users. The taste vector is the sum of the (normalised) embeddings of
            the user's anchors: the learned user embedding for a known user, every liked
            perfume, and - for a quiz-only visitor - the perfumes that best match the quiz
            notes. Because it is a sum, COLLAB(i) splits exactly into one term per anchor:
            "close to Yara, which you loved" is a real part of the score, not a story.
* AVOID(i)  in [0, 1]: closeness to what the user rejected ("Not for me"): the highest cosine
            to a rejected perfume, and the match with the rejected notes the user never
            liked. Rejected perfumes themselves are removed.

No Amazon star rating or review count is used anywhere (professor's feedback: it pushes
best-sellers and biases everything). Popularity is not an input either.

Known users ("Your collection") keep the pure NLGCL ranking (dot product of learned
embeddings), which is what the offline evaluation measures, until they give feedback in the
app; then the same formula as above is used with their learned embedding as one anchor.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_SIZE = re.compile(r"\b\d+(\.\d+)?\s*(ml|oz|ounce|ounces|fl\.?\s*oz|fl)\b.*$", re.I)

# Default weights, tuned on the cold-start *validation* users (reports/week3_coldstart.json):
# before any like we trust the notes more, once the visitor has liked perfumes we trust what
# people who loved them also loved. W_AVOID was tuned on the 1,553 users who disliked a perfume
# in a review (reports/week3_feedback_val.json): the strongest penalty costing <= 5% NDCG.
SETTINGS = {"quiz": {"w_notes": 0.2, "beta": 1.0}, "feedback": {"w_notes": 0.2, "beta": 0.75}}
# Known users ("Your collection"): NLGCL ranking cos(e_u, e_i) * |e_i|^beta. beta = 1 is the plain
# dot product; on validation beta = 0.5 is MORE accurate and covers far more of the catalogue
# (reports/week3_analysis.json): the trained model leans too much on popularity.
BETA_KNOWN = 0.5
W_AVOID = 0.1
N_QUIZ_ANCHORS = 30


def short_title(title: str) -> str:
    """'Yara by Lattafa Perfumes | Eau De Parfum - 100ml (3.4 fl oz)' -> 'Yara by Lattafa Perfumes'."""
    t = re.sub(r"^\s*\d+(\.\d+)?\s*(ml|oz)\b\s*", "", str(title), flags=re.I)
    t = re.sub(r"^(men'?s|women'?s)\s+(perfume|cologne)?\s*(fragrance)?\s*,?\s*(by\s+)?", "", t, flags=re.I)
    t = re.split(r"\s[|\-–—]\s|,|\(", t)[0]
    t = _SIZE.sub("", t)
    t = re.sub(r"\b(eau de (parfum|toilette|cologne)|edp|edt|spray|perfume|cologne)\b.*$", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip(" -|,.")
    return t or str(title)[:60]


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-9)


@dataclass
class Profile:
    """Everything we know about a visitor. Item ids are item_idx."""
    seed_notes: list[int] = field(default_factory=list)
    liked: list[int] = field(default_factory=list)
    disliked: list[int] = field(default_factory=list)
    user_idx: int | None = None          # known Amazon user: learned embedding is an anchor

    @property
    def has_feedback(self) -> bool:
        return bool(self.liked or self.disliked)


@dataclass
class Rec:
    item_idx: int
    score: float
    explanation: str
    breakdown: dict = field(default_factory=dict)


class Catalogue:
    def __init__(self, processed_dir: Path):
        self.items = pd.read_parquet(processed_dir / "items.parquet").set_index("item_idx").sort_index()
        self.asin_to_idx = {a: int(i) for i, a in self.items["parent_asin"].items()}
        self.note_names: list[str] = json.loads((processed_dir / "notes.json").read_text())
        self.note_idx = {n: i for i, n in enumerate(self.note_names)}
        inotes = pd.read_parquet(processed_dir / "item_notes.parquet")
        self.item_notes = {i: set(g) for i, g in inotes.groupby("item_idx")["note_idx"]}
        self.items_by_note = {n: g.to_numpy() for n, g in inotes.groupby("note_idx")["item_idx"]}
        users = pd.read_parquet(processed_dir / "users.parquet")
        self.user_idx = dict(zip(users["user_id"], users["user_idx"]))
        known = pd.concat([pd.read_parquet(processed_dir / f"{s}.parquet") for s in ("train", "val", "test")])
        # serving only: never re-propose a perfume the user already rated (any split)
        self.user_items = {u: g.to_numpy() for u, g in known.groupby("user_idx")["item_idx"]}
        train = pd.read_parquet(processed_dir / "train.parquet")
        self.train_items = {u: g.to_numpy() for u, g in train.groupby("user_idx")["item_idx"]}
        self.popularity = self.items["train_popularity"].to_numpy().astype(float)
        n_items, n_notes = len(self.items), len(self.note_names)
        df = np.bincount(inotes["note_idx"], minlength=n_notes).astype(float)
        self.idf = np.log((1 + n_items) / (1 + df)) + 1.0
        x = np.zeros((n_items, n_notes), dtype=np.float32)
        x[inotes["item_idx"].to_numpy(), inotes["note_idx"].to_numpy()] = 1.0
        self.has_note = x                                  # binary perfume x note matrix
        xw = x * self.idf
        norm = np.linalg.norm(xw, axis=1, keepdims=True)
        self.note_matrix = np.divide(xw, norm, out=np.zeros_like(xw), where=norm > 0)
        self.n_notes_per_item = x.sum(1)
        self.short_titles = [short_title(t) for t in self.items["title"]]
        ptype = self.items["product_type"] if "product_type" in self.items else pd.Series("perfume", self.items.index)
        self.product_type = ptype.to_numpy()

    @property
    def n_items(self) -> int:
        return len(self.items)

    def notes_of(self, item_idx: int) -> set[int]:
        return self.item_notes.get(int(item_idx), set())

    def note_profile(self, profile: Profile) -> np.ndarray:
        """Weight of each note for this visitor: quiz notes count 1, each liked perfume adds 1
        to each of its notes (so a note shared by several liked perfumes weighs more)."""
        q = np.zeros(len(self.note_names), dtype=np.float32)
        np.add.at(q, profile.seed_notes, 1.0)
        for i in profile.liked:
            q += self.has_note[i]
        return q

    def avoid_profile(self, profile: Profile, liked_q: np.ndarray) -> np.ndarray:
        """Notes of rejected perfumes that the visitor never asked for nor liked."""
        q = np.zeros(len(self.note_names), dtype=np.float32)
        for i in profile.disliked:
            q += self.has_note[i]
        q[liked_q > 0] = 0
        return q

    def note_match(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """NOTES(i) for every perfume, and the IDF-weighted query used (for explanations)."""
        qw = q * self.idf
        if qw.sum() == 0:
            return np.zeros(self.n_items, dtype=np.float32), qw
        cosine = self.note_matrix @ (qw / np.linalg.norm(qw))
        coverage = self.has_note @ qw / qw.sum()
        # descriptions listing 30-40 notes would match every profile: damp beyond 15 notes
        damp = np.minimum(1.0, np.sqrt(15.0 / np.maximum(self.n_notes_per_item, 1)))
        return ((0.5 * cosine + 0.5 * np.sqrt(coverage)) * damp).astype(np.float32), qw

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

    def names(self, ids) -> list[str]:
        return [self.short_titles[int(i)] for i in ids]


def _listing(names: list[str]) -> str:
    names = names[:3]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


class Scorer:
    """The explainable scoring formula of the module docstring, on top of item embeddings."""

    def __init__(self, cat: Catalogue, item_e: np.ndarray, user_e: np.ndarray | None = None,
                 w_notes: float | None = None, beta: float | None = None, w_avoid: float = W_AVOID,
                 settings: dict | None = None):
        self.cat, self.item_e, self.user_e = cat, item_e, user_e
        self.item_n = _unit(item_e)
        norms = np.linalg.norm(item_e, axis=1)
        self.rel_norm = norms / norms.max()                # "how popular the model thinks it is"
        self.settings = {k: dict(v) for k, v in (settings or SETTINGS).items()}
        if w_notes is not None or beta is not None:        # fixed weights (evaluation grids)
            for v in self.settings.values():
                v.update({k: x for k, x in (("w_notes", w_notes), ("beta", beta)) if x is not None})
        self.w_avoid = w_avoid
        self.w_notes = self.w_collab = None                 # set per profile in score()

    def anchors(self, profile: Profile, notes_score: np.ndarray) -> list[tuple[str, np.ndarray, float]]:
        """(label, unit vector, weight) of each anchor of the taste vector."""
        out = []
        if profile.user_idx is not None and self.user_e is not None:
            out.append(("your Amazon history", _unit(self.user_e[profile.user_idx]), 1.0))
        for i in profile.liked:
            out.append((f"item:{i}", self.item_n[i], 1.0))
        if not out and notes_score.any():            # quiz only: the best note matches
            top = np.argsort(-notes_score)[:N_QUIZ_ANCHORS]
            top = top[notes_score[top] > 0]
            for i in top:
                out.append((f"quiz:{i}", self.item_n[i], float(notes_score[i])))
        return out

    def score(self, profile: Profile) -> dict:
        cat = self.cat
        mode = "feedback" if (profile.liked or profile.user_idx is not None) else "quiz"
        self.w_notes = self.settings[mode]["w_notes"]
        self.w_collab = 1.0 - self.w_notes
        beta = self.settings[mode]["beta"]
        pop_factor = self.rel_norm ** beta
        q = cat.note_profile(profile)
        notes_score, qw = cat.note_match(q)
        anchors = self.anchors(profile, notes_score)
        if anchors:
            taste = sum(w * v for _, v, w in anchors)
            norm = float(np.linalg.norm(taste)) or 1.0
            collab = (self.item_n @ (taste / norm)) * pop_factor
        else:
            norm, collab = 1.0, np.zeros(cat.n_items, dtype=np.float32)
        avoid = np.zeros(cat.n_items, dtype=np.float32)
        aq = np.zeros(0)
        if profile.disliked:
            sim = self.item_n @ self.item_n[profile.disliked].T
            aq = cat.avoid_profile(profile, q)
            avoid_notes, _ = cat.note_match(aq)
            avoid = 0.5 * np.clip(sim.max(1), 0, None) + 0.5 * avoid_notes
        total = self.w_notes * notes_score + self.w_collab * collab - self.w_avoid * avoid
        return {"total": total, "notes": notes_score, "collab": collab, "avoid": avoid,
                "qw": qw, "aq": aq, "anchors": anchors, "norm": norm, "pop_factor": pop_factor,
                "mode": mode, "beta": beta}

    def rank(self, profile: Profile, k: int, exclude=None) -> list[int]:
        """Ranking only (no explanation, no shop-window adjustment): used by the offline evaluation."""
        s = self.score(profile)["total"].copy()
        drop = list(profile.liked) + list(profile.disliked) + (list(exclude) if exclude is not None else [])
        s[drop] = -np.inf
        top = np.argpartition(-s, k)[:k]
        return top[np.argsort(-s[top])].tolist()

    def explain(self, i: int, parts: dict, profile: Profile, quiz: bool, similar_to: bool = False) -> tuple[str, dict]:
        cat = self.cat
        n_part = self.w_notes * float(parts["notes"][i])
        c_part = self.w_collab * float(parts["collab"][i])
        a_part = self.w_avoid * float(parts["avoid"][i])
        # per-note contributions (exact for the cosine half of NOTES, proportional for the coverage half)
        shared = [n for n in cat.notes_of(i) if parts["qw"][n] > 0]
        shared.sort(key=lambda n: -parts["qw"][n])
        note_w = np.array([parts["qw"][n] for n in shared], dtype=float)
        note_share = (note_w / note_w.sum() * n_part) if len(shared) else np.zeros(0)
        # per-anchor contributions to COLLAB (exact: the taste vector is a sum)
        contrib = []
        for label, v, w in parts["anchors"]:
            contrib.append((label, w * float(self.item_n[i] @ v) / parts["norm"] * float(parts["pop_factor"][i])
                            * self.w_collab))
        liked = sorted([(int(l[5:]), c) for l, c in contrib if l.startswith("item:")], key=lambda t: -t[1])
        history = [c for l, c in contrib if l == "your Amazon history"]
        avoided = [n for n in np.flatnonzero(parts["aq"]) if n not in cat.notes_of(i)] if len(parts["aq"]) else []
        breakdown = {
            "mode": parts["mode"], "w_notes": self.w_notes, "w_collab": round(self.w_collab, 3),
            "beta": parts["beta"], "popularity_factor": round(float(parts["pop_factor"][i]), 4),
            "total": round(n_part + c_part - a_part, 4),
            "notes_part": round(n_part, 4), "collab_part": round(c_part, 4), "avoid_part": round(-a_part, 4),
            "notes": [{"note": cat.note_names[n], "contribution": round(float(s), 4)} for n, s in zip(shared, note_share)][:6],
            "similar_liked": [{"name": cat.short_titles[j], "parent_asin": cat.items["parent_asin"].iat[j],
                               "contribution": round(c, 4)} for j, c in liked[:3]],
            "history_contribution": round(history[0], 4) if history else None,
            "avoided_notes": [cat.note_names[n] for n in avoided[:4]],
        }
        # the sentence: the biggest positive part first, then the other one, then what was avoided
        names = [cat.note_names[n] for n in shared]
        phrases = []
        if similar_to and liked:                      # "More like this" on one perfume
            ref = cat.short_titles[liked[0][0]]
            if names:
                phrases.append((n_part, f"Shares {_listing(names)} with {ref}"))
            phrases.append((liked[0][1], f"Loved by the same people as {ref}"))
        if names and n_part > 0 and not similar_to:
            phrases.append((n_part, f"Matches your notes: {_listing(names)}" if quiz and not profile.liked
                            else f"Shares {_listing(names)} with what you love"))
        if liked and liked[0][1] > 0.02 and not similar_to:
            phrases.append((liked[0][1], f"Close to {cat.short_titles[liked[0][0]]}, which you loved"))
        elif history and history[0] > 0.02 and not similar_to:
            phrases.append((history[0], "Fits the taste of your past purchases"))
        elif c_part > 0.05 and not similar_to:
            phrases.append((c_part, "Loved by people with your taste"))
        phrases = [t for _, t in sorted(phrases, key=lambda x: -x[0])]
        if breakdown["avoided_notes"]:
            phrases.append(f"Avoids {_listing(breakdown['avoided_notes'])}, which you turned down")
        text = " · ".join(phrases) if phrases else "A discovery outside your usual notes"
        return text, breakdown

    def recommend(self, profile: Profile, k: int = 10, exclude: np.ndarray | None = None,
                  serving: bool = True, quiz: bool = False, similar_to: bool = False) -> list[Rec]:
        parts = self.score(profile)
        s = parts["total"].copy()
        drop = list(profile.liked) + list(profile.disliked)
        if exclude is not None:
            drop += list(exclude)
        s[drop] = -np.inf
        adjust = np.zeros_like(s)
        if serving:
            # product choice for the shop window: full perfumes first, sets/minis/body sprays
            # are still recommendable but slightly lower; perfumes sharing no note go last
            adjust[self.cat.product_type != "perfume"] -= 0.05
            if parts["notes"].any():
                adjust[parts["notes"] == 0] -= 0.1
            s = s + adjust
            ranked = np.argsort(-s)[: 20 * k]
            chosen = self.cat.diversify(ranked, k, max_per_brand=2 if quiz else 3)
        else:
            chosen = np.argsort(-s)[:k].tolist()
        out = []
        for i in chosen:
            text, br = self.explain(i, parts, profile, quiz, similar_to)
            br["shop_adjustment"] = round(float(adjust[i]), 4)
            br["total"] = round(br["total"] + float(adjust[i]), 4)
            out.append(Rec(int(i), br["total"], text, br))
        return out


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
        return self._pick("|".join(sorted(notes)), k)

    def recommend_session(self, profile: Profile, k: int = 10) -> list[Rec]:
        return self._pick(f"{profile.seed_notes}{profile.liked}{profile.disliked}", k,
                          exclude=list(profile.liked) + list(profile.disliked))

    def similar(self, item_idx: int, k: int = 6) -> list[Rec]:
        return self._pick(str(item_idx), k, exclude=[item_idx])


def load_embeddings(checkpoint: Path, processed_dir: Path) -> dict:
    """Final embeddings of a checkpoint (Week 3 checkpoints store them; older ones are rebuilt)."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "embeddings" in ckpt:
        e = ckpt["embeddings"]
        users, items, notes = e["users"], e["items"], e["notes"]
    else:
        from scentmatch.model.graph import build_graph
        from scentmatch.model.nlgcl import NLGCL
        train = pd.read_parquet(processed_dir / "train.parquet")
        inotes = pd.read_parquet(processed_dir / "item_notes.parquet") if ckpt["n_notes"] else None
        model = NLGCL(build_graph(train, ckpt["n_users"], ckpt["n_items"], inotes, ckpt["n_notes"]), **ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        users, items, notes = model.final_embeddings()
    return {"users": users.numpy(), "items": items.numpy(), "notes": notes.numpy() if ckpt["n_notes"] else None,
            "epoch": ckpt["epoch"], "config": ckpt["config"], "val": ckpt.get("val")}


class ModelRecommender:
    mode = "model"

    def __init__(self, catalogue: Catalogue, checkpoint: Path, processed_dir: Path):
        self.cat = catalogue
        e = load_embeddings(checkpoint, processed_dir)
        self.user_e, self.item_e, self.note_e = e["users"], e["items"], e["notes"]
        self.epoch, self.checkpoint = e["epoch"], Path(checkpoint).name
        self.scorer = Scorer(catalogue, self.item_e, self.user_e)

    # ---------------------------------------------------------------- known users
    def recommend_user(self, user_id: str, k: int = 10) -> list[Rec]:
        """NLGCL ranking with the popularity knob: score(u, i) = cos(e_u, e_i) * |e_i|^BETA_KNOWN
        (what the offline evaluation measures as "beta = 0.5")."""
        uidx = self.cat.user_idx.get(user_id)
        if uidx is None:
            raise KeyError(user_id)
        seen = self.cat.user_items.get(uidx, np.array([], dtype=int))
        u = self.user_e[uidx] / max(np.linalg.norm(self.user_e[uidx]), 1e-9)
        scores = (self.scorer.item_n @ u) * self.scorer.rel_norm ** BETA_KNOWN
        scores[seen] = -np.inf
        top = np.argpartition(-scores, 5 * k)[:5 * k]
        top = top[np.argsort(-scores[top])]
        chosen = self.cat.diversify(top, k)
        # explanation: the learned score, split into the perfumes of the history it is closest to
        hist = self.cat.train_items.get(uidx, seen)
        profile = Profile(liked=[int(i) for i in hist])
        parts = self.scorer.score(profile)
        out = []
        for i in chosen:
            text, br = self.scorer.explain(i, parts, profile, quiz=False)
            br["model_score"] = round(float(scores[i]), 4)
            br["explanation_type"] = ("post-hoc: the ranking is the NLGCL score cos(e_u, e_i) x popularity^0.5; "
                                      "the parts show which loved perfumes and notes it is closest to")
            br["beta"] = BETA_KNOWN
            out.append(Rec(int(i), round(float(scores[i]), 4), text, br))
        return out

    # ---------------------------------------------------------------- quiz / session
    def recommend_notes(self, notes: list[str], k: int = 10) -> list[Rec]:
        ids = [self.cat.note_idx[n] for n in notes if n in self.cat.note_idx]
        if not ids:
            raise ValueError("none of the seed notes is in the vocabulary")
        return self.scorer.recommend(Profile(seed_notes=ids), k, quiz=True)

    def recommend_session(self, profile: Profile, k: int = 10) -> list[Rec]:
        exclude = None
        if profile.user_idx is not None:
            exclude = self.cat.user_items.get(profile.user_idx)
        if not profile.seed_notes and not profile.has_feedback and profile.user_idx is None:
            raise ValueError("empty profile: answer the quiz or like a perfume first")
        return self.scorer.recommend(profile, k, exclude=exclude, quiz=bool(profile.seed_notes))

    def similar(self, item_idx: int, k: int = 6) -> list[Rec]:
        """'More like this': same formula with the perfume as the only liked anchor."""
        return self.scorer.recommend(Profile(liked=[item_idx]), k, similar_to=True)
