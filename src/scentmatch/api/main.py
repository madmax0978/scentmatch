"""ScentMatch REST API (Systems Engineer).

    GET  /health                         status, serving mode, catalogue size
    GET  /notes                          note vocabulary (for the "notes you love" picker)
    GET  /quiz                           the taste-quiz questions
    GET  /recommendations/{user_id}      top-k for a known user
    POST /recommendations/quiz           top-k for a new user from quiz answers / seed notes
    POST /log_interaction                store quiz / click / like / not_for_me events in SQLite
    GET  /perfumes/{parent_asin}         catalogue entry

Mode: SCENTMATCH_MODE=mock (default in Week 2) or model (needs a checkpoint).
Run:  uvicorn scentmatch.api.main:app --reload --port 8000
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
from scentmatch.train.predict import Catalogue, MockRecommender, ModelRecommender, Rec

cfg = load_config(os.getenv("SCENTMATCH_CONFIG"))
PROCESSED = cfg["paths"]["processed_dir"]
DB_PATH = cfg["paths"]["db_path"]
db.init_db(DB_PATH)
catalogue = Catalogue(PROCESSED)

mode = os.getenv("SCENTMATCH_MODE", "mock")
ckpt = cfg["paths"]["checkpoints_dir"] / os.getenv("SCENTMATCH_CHECKPOINT", "nlgcl.pt")
if mode == "model" and ckpt.exists():
    recommender = ModelRecommender(catalogue, ckpt, PROCESSED)
else:
    recommender = MockRecommender(catalogue)

app = FastAPI(title="ScentMatch API", version="0.2.0")


class Perfume(BaseModel):
    rank: int
    parent_asin: str
    title: str
    name: str                       # short display name, e.g. "Yara" instead of the full Amazon title
    brand: str
    rating: float | None = None
    notes: list[str]
    image: str | None
    score: float
    explanation: str


class RecResponse(BaseModel):
    user_id: str | None
    mode: str
    seed_notes: list[str] | None = None
    history: list[str] | None = None   # known user: perfumes they loved in the training data
    items: list[Perfume]


class QuizRequest(BaseModel):
    user_id: str | None = None
    answers: dict[str, str] = Field(default_factory=dict, examples=[{"moment": "Evening out", "taste": "Sweet"}])
    extra_notes: list[str] = Field(default_factory=list)
    k: int = 10


class InteractionIn(BaseModel):
    user_id: str
    item_id: str | None = None
    event: Literal["quiz", "click", "like", "not_for_me", "view"]
    payload: dict | None = None


def _to_response(recs: list[Rec], user_id: str | None, notes: list[str] | None = None,
                 history: list[str] | None = None) -> RecResponse:
    info = catalogue.items.loc[[r.item_idx for r in recs]]
    items = []
    for rank, (r, (_, row)) in enumerate(zip(recs, info.iterrows()), start=1):
        rating = row["average_rating"]
        items.append(Perfume(rank=rank, parent_asin=row["parent_asin"], title=row["title"],
                             name=catalogue.short_titles[r.item_idx], brand=row["store"],
                             rating=None if rating != rating else float(rating), notes=[n for n in row["notes"].split(", ") if n],
                             image=row["image"], score=round(float(r.score), 4), explanation=r.explanation))
    return RecResponse(user_id=user_id, mode=recommender.mode, seed_notes=notes, history=history, items=items)


@app.get("/health")
def health():
    return {"status": "ok", "mode": recommender.mode, "n_perfumes": catalogue.n_items,
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
        seen = catalogue.train_items.get(catalogue.user_idx.get(user_id), [])
        return _to_response(recs, user_id, history=[catalogue.short_titles[i] for i in seen])
    except KeyError:
        raise HTTPException(404, f"unknown user '{user_id}': use POST /recommendations/quiz")


@app.post("/recommendations/quiz", response_model=RecResponse)
def recommend_quiz(req: QuizRequest):
    try:
        notes = seed_notes(req.answers, req.extra_notes)
        recs = recommender.recommend_notes(notes, req.k)
    except (KeyError, ValueError) as e:
        raise HTTPException(422, str(e))
    return _to_response(recs, req.user_id, notes)


@app.post("/log_interaction", status_code=201)
def log_interaction(ev: InteractionIn):
    if ev.item_id is not None and ev.item_id not in set(catalogue.items["parent_asin"]):
        raise HTTPException(404, f"unknown perfume '{ev.item_id}'")
    row_id = db.log_interaction(DB_PATH, ev.user_id, ev.event, ev.item_id,
                                json.dumps(ev.payload) if ev.payload else None)
    return {"id": row_id, "status": "logged"}


@app.get("/perfumes/{parent_asin}")
def perfume(parent_asin: str):
    hit = catalogue.items[catalogue.items["parent_asin"] == parent_asin]
    if hit.empty:
        raise HTTPException(404, "unknown perfume")
    row = hit.iloc[0].drop("categories")
    out = {"item_idx": int(hit.index[0])}
    for k, v in row.items():                      # numpy scalars / arrays -> plain JSON types
        if hasattr(v, "tolist"):
            v = v.tolist()
        elif isinstance(v, float) and v != v:     # NaN
            v = None
        out[k] = v
    out["notes"] = [n for n in (out.get("notes") or "").split(", ") if n]   # same format as /recommendations
    return out
