"""API contract in mock mode (needs the processed data from the pipeline)."""
import json
import os

import pytest

from scentmatch.config import load_config

cfg = load_config()
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
