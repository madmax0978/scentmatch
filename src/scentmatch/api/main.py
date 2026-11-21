"""ScentMatch REST API (Systems Engineer).

    GET  /health                          status, serving mode, checkpoint, catalogue size
    GET  /notes                           note vocabulary (for the "notes you love" picker)
    GET  /quiz                            the taste-quiz questions
    GET  /recommendations/{user_id}       top-k for a known Amazon user (pure NLGCL ranking)
    POST /recommendations/quiz            top-k for a new visitor from quiz answers / seed notes
    POST /recommendations/session         top-k from everything the visitor did: quiz answers,
                                          perfumes liked and rejected (read back from SQLite)
    GET  /perfumes/{parent_asin}          catalogue entry
    GET  /perfumes/{parent_asin}/similar  "more like this"
    GET  /session/{user_id}               what the visitor did so far: quiz answers, loved and
                                          rejected perfumes (to show and edit them in the app)
    POST /log_interaction                 store quiz / click / like / not_for_me / undo events in SQLite

Every recommended perfume carries `explanation` (one sentence) and `breakdown` (the parts of
its score: notes, collaborative, avoided - see scentmatch.train.predict). No Amazon star
rating is returned or used.

Mode: SCENTMATCH_MODE=model (default when a checkpoint exists) or mock.
Run:  uvicorn scentmatch.api.main:app --port 8000
"""
from __future__ import annotations

import json
import os
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from scentmatch.api import db
from scentmatch.api.quiz import QUIZ, seed_notes
from scentmatch.config import load_config
from scentmatch.train.predict import Catalogue, MockRecommender, ModelRecommender, Profile, Rec

cfg = load_config(os.getenv("SCENTMATCH_CONFIG"))
PROCESSED = cfg["paths"]["processed_dir"]
DB_PATH = cfg["paths"]["db_path"]
db.init_db(DB_PATH)
catalogue = Catalogue(PROCESSED)
ASINS = set(catalogue.items["parent_asin"])

ckpt = cfg["paths"]["checkpoints_dir"] / os.getenv("SCENTMATCH_CHECKPOINT", cfg.get("serving", {}).get(
    "checkpoint", "nlgcl.pt"))
mode = os.getenv("SCENTMATCH_MODE", "model" if ckpt.exists() else "mock")
recommender = ModelRecommender(catalogue, ckpt, PROCESSED) if mode == "model" and ckpt.exists() \
    else MockRecommender(catalogue)

app = FastAPI(title="ScentMatch API", version="0.3.0")


class Perfume(BaseModel):
    rank: int
    parent_asin: str
    title: str
    name: str                       # short display name, e.g. "Yara" instead of the full Amazon title
    brand: str
    notes: list[str]
    image: str | None
    score: float
    explanation: str
    breakdown: dict = Field(default_factory=dict)


class Reaction(BaseModel):
    parent_asin: str
    name: str
    brand: str
    image: str | None
    notes: list[str]
    reaction: Literal["like", "not_for_me"]


class RecResponse(BaseModel):
    user_id: str | None
    mode: str
    seed_notes: list[str] | None = None
    history: list[str] | None = None   # known user: perfumes they loved in the training data
    liked: list[str] | None = None     # session: perfumes liked in the app
    disliked: list[str] | None = None  # session: perfumes rejected in the app
    reactions: list[Reaction] | None = None  # session: the same, with ids, so the app can edit them
    items: list[Perfume]


class SessionState(BaseModel):
    user_id: str
    answers: dict[str, str]
    extra_notes: list[str]
    seed_notes: list[str]
    reactions: list[Reaction]


class QuizRequest(BaseModel):
    user_id: str | None = None
    answers: dict[str, str] = Field(default_factory=dict, examples=[{"moment": "Evening out", "taste": "Sweet"}])
    extra_notes: list[str] = Field(default_factory=list)
    k: int = 10


class SessionRequest(BaseModel):
    user_id: str
    answers: dict[str, str] | None = None      # if omitted: the visitor's last logged quiz
    extra_notes: list[str] | None = None
    liked: list[str] = Field(default_factory=list)       # added to the likes found in the log
    disliked: list[str] = Field(default_factory=list)
    use_log: bool = True
    k: int = 9


class InteractionIn(BaseModel):
    user_id: str
    item_id: str | None = None
    event: Literal["quiz", "click", "like", "not_for_me", "undo", "view"]   # undo = forget my reaction
    payload: dict | None = None


def _to_response(recs: list[Rec], user_id: str | None, **extra) -> RecResponse:
    items = []
    for rank, r in enumerate(recs, start=1):
        row = catalogue.items.loc[r.item_idx]
        items.append(Perfume(rank=rank, parent_asin=row["parent_asin"], title=row["title"],
                             name=catalogue.short_titles[r.item_idx], brand=row["store"],
                             notes=[n for n in row["notes"].split(", ") if n], image=row["image"],
                             score=round(float(r.score), 4), explanation=r.explanation, breakdown=r.breakdown))
    return RecResponse(user_id=user_id, mode=recommender.mode, items=items, **extra)


def _reaction(asin: str, event: str) -> Reaction:
    i = catalogue.asin_to_idx[asin]
    row = catalogue.items.loc[i]
    return Reaction(parent_asin=asin, name=catalogue.short_titles[i], brand=row["store"], image=row["image"],
                    notes=[n for n in row["notes"].split(", ") if n], reaction=event)


def _idx(asin: str) -> int:
    if asin not in catalogue.asin_to_idx:
        raise HTTPException(404, f"unknown perfume '{asin}'")
    return catalogue.asin_to_idx[asin]


@app.get("/health")
def health():
    return {"status": "ok", "mode": recommender.mode, "n_perfumes": catalogue.n_items,
            "checkpoint": getattr(recommender, "checkpoint", None),
            "n_logged_interactions": db.count_interactions(DB_PATH)}


@app.get("/quiz")
def quiz():
    return [{"id": q["id"], "question": q["question"], "options": list(q["options"])} for q in QUIZ]


@app.get("/notes")
def notes():
    """Notes carried by at least one perfume (for the "notes you love" picker), most common first."""
    counts = sorted(((len(v), catalogue.note_names[n]) for n, v in catalogue.items_by_note.items()), reverse=True)
    return [name for _, name in counts]


@app.get("/recommendations/{user_id}", response_model=RecResponse)
def recommend_user(user_id: str, k: int = Query(10, ge=1, le=50)):
    try:
        recs = recommender.recommend_user(user_id, k)
    except KeyError:
        raise HTTPException(404, f"unknown user '{user_id}': use POST /recommendations/quiz")
    seen = catalogue.train_items.get(catalogue.user_idx.get(user_id), [])
    return _to_response(recs, user_id, history=catalogue.names(seen))


@app.post("/recommendations/quiz", response_model=RecResponse)
def recommend_quiz(req: QuizRequest):
    try:
        notes = seed_notes(req.answers, req.extra_notes)
        recs = recommender.recommend_notes(notes, req.k)
    except (KeyError, ValueError) as e:
        raise HTTPException(422, str(e))
    return _to_response(recs, req.user_id, seed_notes=notes)


def replay_log(user_id: str) -> tuple[dict, dict[str, str]]:
    """Replay a visitor's log, oldest first: the last quiz, and the current reaction to each perfume.
    The latest event on a perfume wins: like, not_for_me, or undo (= no reaction any more).
    The log itself is never edited, so every change of mind stays in the data."""
    last_quiz: dict = {}
    reaction: dict[str, str] = {}
    for ev in db.user_events(DB_PATH, user_id):
        if ev["event"] == "quiz" and ev["payload"]:
            last_quiz = json.loads(ev["payload"])
        elif ev["event"] in ("like", "not_for_me", "undo") and ev["item_id"]:
            reaction.pop(ev["item_id"], None)          # re-insert: the dict stays in "last changed" order
            if ev["event"] != "undo":
                reaction[ev["item_id"]] = ev["event"]
    return last_quiz, reaction


def build_profile(req: SessionRequest) -> tuple[Profile, list[str], dict[str, str]]:
    """Rebuild a visitor's profile from the request and the interaction log.
    Quiz: the answers in the request, else the visitor's last logged quiz.
    Reactions: see replay_log (the latest reaction to a perfume wins, undo removes it)."""
    answers, extra = req.answers, req.extra_notes
    last_quiz, reaction = replay_log(req.user_id) if req.use_log else ({}, {})
    if answers is None:
        answers = last_quiz.get("answers") or {}
    if extra is None:
        extra = last_quiz.get("extra_notes") or []
    for a in req.liked:
        reaction[a] = "like"
    for a in req.disliked:
        reaction[a] = "not_for_me"
    notes = seed_notes(answers, extra)
    profile = Profile(
        seed_notes=[catalogue.note_idx[n] for n in notes if n in catalogue.note_idx],
        liked=[_idx(a) for a, e in reaction.items() if e == "like"],
        disliked=[_idx(a) for a, e in reaction.items() if e == "not_for_me"],
        user_idx=catalogue.user_idx.get(req.user_id))
    return profile, notes, reaction


@app.post("/recommendations/session", response_model=RecResponse)
def recommend_session(req: SessionRequest):
    try:
        profile, notes, reaction = build_profile(req)
        recs = recommender.recommend_session(profile, req.k)
    except (KeyError, ValueError) as e:
        raise HTTPException(422, str(e))
    return _to_response(recs, req.user_id, seed_notes=notes or None,
                        liked=catalogue.names(profile.liked), disliked=catalogue.names(profile.disliked),
                        reactions=[_reaction(a, e) for a, e in reversed(reaction.items())])


@app.get("/session/{user_id}", response_model=SessionState)
def session(user_id: str):
    """What a visitor did so far, latest change first (empty for an unknown visitor)."""
    last_quiz, reaction = replay_log(user_id)
    answers, extra = last_quiz.get("answers") or {}, last_quiz.get("extra_notes") or []
    try:
        notes = seed_notes(answers, extra)
    except KeyError:                                  # a quiz logged with an option that no longer exists
        notes = []
    return SessionState(user_id=user_id, answers=answers, extra_notes=extra, seed_notes=notes,
                        reactions=[_reaction(a, e) for a, e in reversed(reaction.items())])


@app.post("/log_interaction", status_code=201)
def log_interaction(ev: InteractionIn):
    if ev.item_id is not None and ev.item_id not in ASINS:
        raise HTTPException(404, f"unknown perfume '{ev.item_id}'")
    row_id = db.log_interaction(DB_PATH, ev.user_id, ev.event, ev.item_id,
                                json.dumps(ev.payload) if ev.payload else None)
    return {"id": row_id, "status": "logged"}


@app.get("/perfumes/{parent_asin}")
def perfume(parent_asin: str):
    i = _idx(parent_asin)
    row = catalogue.items.loc[i].drop(["categories", "average_rating", "rating_number"], errors="ignore")
    out = {"item_idx": i}
    for k, v in row.items():                      # numpy scalars / arrays -> plain JSON types
        if hasattr(v, "tolist"):
            v = v.tolist()
        elif isinstance(v, float) and v != v:     # NaN
            v = None
        out[k] = v
    out["notes"] = [n for n in (out.get("notes") or "").split(", ") if n]   # same format as /recommendations
    return out


@app.get("/perfumes/{parent_asin}/similar", response_model=RecResponse)
def similar(parent_asin: str, k: int = Query(6, ge=1, le=30)):
    return _to_response(recommender.similar(_idx(parent_asin), k), None)
