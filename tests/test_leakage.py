"""Q3 (Week 3) - no data leakage. Every check runs on the real processed data.

What could leak, and the test that rules it out:
  1. the same (user, perfume) pair in two splits            -> test_splits_are_disjoint
  2. validation/test happening before training in time      -> test_split_is_chronological
  3. a validation/test edge inside the training graph       -> test_graph_uses_training_edges_only
  4. popularity counted on validation/test                  -> test_popularity_from_training_only
  5. the evaluation not masking the right "seen" perfumes   -> test_harness_masks_the_right_items
  6. training reading the test file at all                  -> test_training_never_loads_test
  7. explicit feedback used by the model coming from after
     the user's validation interaction                      -> test_feedback_used_in_training_is_past_only
  8. cold-start users still present in the cold training    -> test_cold_users_absent_from_cold_train
"""
import json

import numpy as np
import pandas as pd
import pytest

from scentmatch.config import load_config

cfg = load_config()
PROC = cfg["paths"]["processed_dir"]
pytestmark = pytest.mark.skipif(not (PROC / "train.parquet").exists(),
                                reason="run `python -m scentmatch.data.pipeline` first")


@pytest.fixture(scope="module")
def splits():
    return {s: pd.read_parquet(PROC / f"{s}.parquet") for s in ("train", "val", "test")}


def pairs(df: pd.DataFrame) -> set:
    return set(zip(df["user_idx"], df["item_idx"]))


def test_splits_are_disjoint(splits):
    tr, va, te = (pairs(splits[s]) for s in ("train", "val", "test"))
    assert not tr & va and not tr & te and not va & te


def test_one_validation_and_one_test_item_per_user(splits):
    n_users = json.loads((PROC / "stats.json").read_text())["users"]
    for s in ("val", "test"):
        counts = splits[s].groupby("user_idx").size()
        assert len(counts) == n_users and (counts == 1).all()


def test_split_is_chronological(splits):
    last_train = splits["train"].groupby("user_idx")["timestamp"].max()
    val = splits["val"].set_index("user_idx")["timestamp"]
    test = splits["test"].set_index("user_idx")["timestamp"]
    assert (last_train <= val.reindex(last_train.index)).all()
    assert (val <= test.reindex(val.index)).all()


def test_graph_uses_training_edges_only(splits):
    from scentmatch.model.graph import build_graph
    stats = json.loads((PROC / "stats.json").read_text())
    n_users, n_items = stats["users"], stats["items"]
    inotes = pd.read_parquet(PROC / "item_notes.parquet")
    g = build_graph(splits["train"], n_users, n_items, inotes, int(inotes["note_idx"].max()) + 1)
    src, dst = g.edge_index.numpy()
    ui = (src < n_users) & (dst >= n_users) & (dst < n_users + n_items)
    graph_pairs = set(zip(src[ui], dst[ui] - n_users))
    assert graph_pairs == pairs(splits["train"])
    assert not graph_pairs & (pairs(splits["val"]) | pairs(splits["test"]))


def test_popularity_from_training_only(splits):
    from scentmatch.eval.harness import EvalData
    items = pd.read_parquet(PROC / "items.parquet").sort_values("item_idx")
    expected = np.bincount(splits["train"]["item_idx"], minlength=len(items))
    assert (items["train_popularity"].to_numpy() == expected).all()
    data = EvalData.load(PROC, with_test=False)
    assert (data.popularity == expected).all()


def test_harness_masks_the_right_items():
    """Toy data: when scoring validation only training items are hidden; for test,
    training + validation items are hidden. The target itself is never hidden."""
    from scentmatch.eval.harness import EvalData, rank_scores
    tr = pd.DataFrame({"user_idx": [0, 0], "item_idx": [0, 1]})
    va = pd.DataFrame({"user_idx": [0], "item_idx": [2]})
    te = pd.DataFrame({"user_idx": [0], "item_idx": [3]})
    data = EvalData(1, 6, tr, va, te)
    score = lambda users: np.tile(np.array([9, 8, 7, 6, 5, 4], dtype=float), (len(users), 1))  # noqa: E731
    assert rank_scores(score, data, "val", top=3)[0] == [2, 3, 4]
    assert rank_scores(score, data, "test", top=3)[0] == [3, 4, 5]


def test_training_never_loads_test(monkeypatch):
    from scentmatch.eval.harness import EvalData
    data = EvalData.load(PROC, with_test=False)
    assert data.test is None
    with pytest.raises(RuntimeError):
        data.targets("test")
    # train.py must call the loader with with_test=False
    src = (cfg["paths"]["processed_dir"].parents[1] / "src" / "scentmatch" / "train" / "train.py").read_text()
    assert "EvalData.load(proc, with_test=False" in src


@pytest.mark.skipif(not (PROC / "feedback.parquet").exists(), reason="run scentmatch.data.feedback first")
def test_feedback_used_in_training_is_past_only(splits):
    fb = pd.read_parquet(PROC / "feedback.parquet")
    used = fb[fb["period"] == "train"].merge(splits["val"][["user_idx", "timestamp"]], on="user_idx",
                                               suffixes=("", "_val"))
    assert (used["timestamp"] < used["timestamp_val"]).all()
    # a low-star review is never also a positive of the same user
    assert not pairs(fb) & (pairs(splits["train"]) | pairs(splits["val"]) | pairs(splits["test"]))


@pytest.mark.skipif(not (PROC / "cold" / "users.parquet").exists(), reason="run scentmatch.data.coldstart first")
def test_cold_users_absent_from_cold_train():
    cold_users = pd.read_parquet(PROC / "cold" / "users.parquet")
    cold_train = pd.read_parquet(PROC / "cold" / "train.parquet")
    assert not set(cold_users["user_idx"]) & set(cold_train["user_idx"])
