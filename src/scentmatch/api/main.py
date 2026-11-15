"""ScentMatch REST API (Systems Engineer).

    GET  /health                         status, serving mode, catalogue size
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
    brand: str
    notes: list[str]
    image: str | None
    score: float
    explanation: str


class RecResponse(BaseModel):
    user_id: str | None
    mode: str
    seed_notes: list[str] | None = None
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


def _to_response(recs: list[Rec], user_id: str | None, notes: list[str] | None = None) -> RecResponse:
    info = catalogue.items.loc[[r.item_idx for r in recs]]
    items = []
    for rank, (r, (_, row)) in enumerate(zip(recs, info.iterrows()), start=1):
        items.append(Perfume(rank=rank, parent_asin=row["parent_asin"], title=row["title"],
                             brand=row["store"], notes=[n for n in row["notes"].split(", ") if n],
                             image=row["image"], score=round(float(r.score), 4), explanation=r.explanation))
    return RecResponse(user_id=user_id, mode=recommender.mode, seed_notes=notes, items=items)


@app.get("/health")
def health():
    return {"status": "ok", "mode": recommender.mode, "n_perfumes": catalogue.n_items,
            "n_logged_interactions": db.count_interactions(DB_PATH)}


@app.get("/quiz")
def quiz():
    return [{"id": q["id"], "question": q["question"], "options": list(q["options"])} for q in QUIZ]


@app.get("/recommendations/{user_id}", response_model=RecResponse)
def recommend_user(user_id: str, k: int = Query(10, ge=1, le=50)):
    try:
        return _to_response(recommender.recommend_user(user_id, k), user_id)
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
    row = hit.iloc[0]
    return {"item_idx": int(hit.index[0]), **{k: (None if v != v else v) for k, v in row.drop("categories").items()}}
