"""API contract in mock mode (needs the processed data from the pipeline)."""
import json
import os

import pytest

from scentmatch.config import load_config

cfg = load_config()
SERVED = cfg["paths"]["checkpoints_dir"] / cfg["serving"]["checkpoint"]
pytestmark = pytest.mark.skipif(not (cfg["paths"]["processed_dir"] / "items.parquet").exists(),
                                reason="run `python -m scentmatch.data.pipeline` first")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["SCENTMATCH_MODE"] = "mock"
    from fastapi.testclient import TestClient
    from scentmatch.api import main
    main.DB_PATH = tmp_path_factory.mktemp("db") / "test.db"
    main.db.init_db(main.DB_PATH)
    return TestClient(main.app)


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["mode"] == "mock" and r["n_perfumes"] > 1000


def test_quiz_notes_exist_in_graph_vocabulary():
    from scentmatch.api.quiz import all_quiz_notes
    notes = set(json.loads((cfg["paths"]["processed_dir"] / "notes.json").read_text()))
    assert all_quiz_notes() - notes == set()


def test_mock_recommendations_for_user(client):
    r = client.get("/recommendations/any-user?k=10")
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "mock" and len(body["items"]) == 10
    assert [p["rank"] for p in body["items"]] == list(range(1, 11))


def test_quiz_recommendations(client):
    r = client.post("/recommendations/quiz", json={"answers": {"moment": "Evening out", "taste": "Sweet"}})
    assert r.status_code == 200
    assert "vanilla" in r.json()["seed_notes"]


def test_log_interaction_reaches_db(client):
    from scentmatch.api import main
    before = main.db.count_interactions(main.DB_PATH)
    asin = client.get("/recommendations/u1?k=1").json()["items"][0]["parent_asin"]
    r = client.post("/log_interaction", json={"user_id": "u1", "item_id": asin, "event": "like"})
    assert r.status_code == 201
    assert main.db.count_interactions(main.DB_PATH) == before + 1


def test_bad_event_rejected(client):
    assert client.post("/log_interaction", json={"user_id": "u1", "event": "buy"}).status_code == 422


def test_perfume_endpoint_serialises(client):
    r = client.post("/recommendations/quiz", json={"answers": {"taste": "Sweet"}})
    asin = r.json()["items"][0]["parent_asin"]
    r = client.get(f"/perfumes/{asin}")
    assert r.status_code == 200 and r.json()["parent_asin"] == asin
    assert isinstance(r.json()["notes"], list)
    assert client.get("/perfumes/NOPE").status_code == 404


@pytest.mark.skipif(not SERVED.exists(), reason="needs the served checkpoint (scripts/fetch_release.sh)")
def test_model_quiz_follows_the_answers():
    """Two opposite quiz profiles must get (almost) disjoint lists that carry the chosen notes."""
    from scentmatch.api.quiz import seed_notes
    from scentmatch.train.predict import Catalogue, ModelRecommender
    proc = cfg["paths"]["processed_dir"]
    cat = Catalogue(proc)
    rec = ModelRecommender(cat, SERVED, proc)
    sweet = seed_notes({"taste": "Sweet", "season": "Winter", "intensity": "Bold, noticed from afar"})
    fresh = seed_notes({"taste": "Fresh", "season": "Summer", "place": "The beach"})
    a, b = rec.recommend_notes(sweet), rec.recommend_notes(fresh)
    assert len({r.item_idx for r in a} & {r.item_idx for r in b}) <= 2
    for recs, notes in ((a, sweet), (b, fresh)):
        ids = {cat.note_idx[n] for n in notes}
        assert all(cat.notes_of(r.item_idx) & ids for r in recs)


def test_session_endpoint_reads_the_log(client):
    """A like logged through the API is part of the next session request."""
    asin = client.get("/recommendations/u2?k=1").json()["items"][0]["parent_asin"]
    client.post("/log_interaction", json={"user_id": "web-test", "event": "quiz",
                                          "payload": {"answers": {"taste": "Sweet"}, "extra_notes": []}})
    client.post("/log_interaction", json={"user_id": "web-test", "item_id": asin, "event": "like"})
    body = client.post("/recommendations/session", json={"user_id": "web-test"}).json()
    assert body["liked"] and "vanilla" in body["seed_notes"]
    assert asin not in [p["parent_asin"] for p in body["items"]]


def test_no_amazon_rating_in_responses(client):
    p = client.post("/recommendations/quiz", json={"answers": {"taste": "Fresh"}}).json()["items"][0]
    assert "rating" not in p
    assert "average_rating" not in client.get(f"/perfumes/{p['parent_asin']}").json()


@pytest.mark.skipif(not SERVED.exists(), reason="needs the served checkpoint (scripts/fetch_release.sh)")
def test_feedback_changes_the_selection_and_is_explained():
    """Like -> the list moves towards the liked perfume; Not for me -> it and its neighbours go."""
    import numpy as np
    from scentmatch.api.quiz import seed_notes
    from scentmatch.train.predict import Catalogue, ModelRecommender, Profile
    proc = cfg["paths"]["processed_dir"]
    cat = Catalogue(proc)
    rec = ModelRecommender(cat, SERVED, proc)
    ids = [cat.note_idx[n] for n in seed_notes({"taste": "Sweet", "season": "Winter"})]
    quiz = rec.recommend_session(Profile(seed_notes=ids), 9)
    liked = quiz[0].item_idx
    after_like = rec.recommend_session(Profile(seed_notes=ids, liked=[liked]), 9)
    assert liked not in [r.item_idx for r in after_like]
    assert len({r.item_idx for r in quiz} & {r.item_idx for r in after_like}) < 9
    assert any(b["name"] == cat.short_titles[liked] for r in after_like for b in r.breakdown["similar_liked"])
    rejected = after_like[0].item_idx
    after_no = rec.recommend_session(Profile(seed_notes=ids, liked=[liked], disliked=[rejected]), 9)
    assert rejected not in [r.item_idx for r in after_no]
    # every score goes down (or stays), and the closer a perfume is to the rejected one, the more
    before = rec.scorer.score(Profile(seed_notes=ids, liked=[liked]))["total"]
    after = rec.scorer.score(Profile(seed_notes=ids, liked=[liked], disliked=[rejected]))["total"]
    drop = before - after
    assert (drop >= -1e-6).all()
    sim = rec.scorer.item_n @ rec.scorer.item_n[rejected]
    near, far = np.argsort(-sim)[1:51], np.argsort(sim)[:50]
    assert drop[near].mean() > drop[far].mean()
    for r in after_no:                     # the parts add up to the score shown
        b = r.breakdown
        assert abs(b["notes_part"] + b["collab_part"] + b["avoid_part"] + b["shop_adjustment"] - r.score) < 1e-3
        cited = [n["note"] for n in b["notes"]]
        assert set(cited) <= {cat.note_names[n] for n in cat.notes_of(r.item_idx)}


def test_undo_and_switch_a_reaction(client):
    """A visitor can take a reaction back (undo) or switch it; the latest event wins,
    GET /session shows the current state and the log keeps every change."""
    from scentmatch.api import main
    items = client.post("/recommendations/quiz", json={"answers": {"taste": "Sweet"}, "k": 3}).json()["items"]
    a, b = items[0]["parent_asin"], items[1]["parent_asin"]
    uid = "web-undo"
    client.post("/log_interaction", json={"user_id": uid, "event": "quiz",
                                          "payload": {"answers": {"taste": "Sweet"}, "extra_notes": []}})
    for asin, ev in ((a, "like"), (b, "like"), (b, "not_for_me"), (a, "undo")):
        assert client.post("/log_interaction", json={"user_id": uid, "item_id": asin, "event": ev}).status_code == 201
    state = client.get(f"/session/{uid}").json()
    assert [(r["parent_asin"], r["reaction"]) for r in state["reactions"]] == [(b, "not_for_me")]
    assert "vanilla" in state["seed_notes"] and state["answers"] == {"taste": "Sweet"}
    body = client.post("/recommendations/session", json={"user_id": uid}).json()
    assert body["liked"] == [] and len(body["disliked"]) == 1
    assert [r["parent_asin"] for r in body["reactions"]] == [b]
    assert len(main.db.user_events(main.DB_PATH, uid)) == 5          # nothing was deleted
    assert client.get("/session/nobody").json()["reactions"] == []


def test_old_database_is_migrated(tmp_path):
    """A database created before the 'undo' event keeps its rows and accepts 'undo'."""
    import sqlite3
    from scentmatch.api import db
    path = tmp_path / "old.db"
    old = db.SCHEMA.replace(",'undo'", "")
    with sqlite3.connect(path) as conn:
        conn.executescript(old)
        conn.execute("INSERT INTO interactions (user_id, item_id, event) VALUES ('u', 'x', 'like')")
    db.init_db(path)
    db.log_interaction(path, "u", "undo", "x")
    assert [e["event"] for e in db.user_events(path, "u")] == ["like", "undo"]
