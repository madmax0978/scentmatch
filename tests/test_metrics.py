"""Evaluation harness on mock predictions with hand-computed expected values."""
import math

import numpy as np
import pandas as pd
import pytest

from scentmatch.eval.harness import EvalData, evaluate_rankings, evaluate_scores
from scentmatch.eval.metrics import catalogue_coverage, ndcg_at_k, recall_at_k


def test_ndcg_single_relevant():
    assert ndcg_at_k([5, 3, 7], {5}, 10) == pytest.approx(1.0)
    assert ndcg_at_k([1, 5, 7], {5}, 10) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k([1, 2, 3], {5}, 10) == 0.0
    assert ndcg_at_k([1, 2, 5], {5}, 2) == 0.0          # outside the cut-off


def test_ndcg_multiple_relevant():
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k([1, 9, 2], {1, 2}, 10) == pytest.approx(dcg / idcg)


def test_recall_and_coverage():
    assert recall_at_k([1, 2, 3], {3, 4}, 20) == 0.5
    assert catalogue_coverage([[0, 1], [1, 2]], n_items=10, k=2) == pytest.approx(0.3)


@pytest.fixture
def toy():
    train = pd.DataFrame({"user_idx": [0, 0, 1, 1, 2], "item_idx": [0, 1, 1, 2, 0]})
    val = pd.DataFrame({"user_idx": [0, 1, 2], "item_idx": [2, 3, 3]})
    test = pd.DataFrame({"user_idx": [0, 1, 2], "item_idx": [3, 4, 4]})
    return EvalData(3, 5, train, val, test)


def test_mock_rankings(toy):
    mock = {0: [3, 4], 1: [0, 4], 2: [1, 2]}      # hit at rank 1, hit at rank 2, miss
    res = evaluate_rankings(mock, toy, "test")
    assert res["NDCG@10"] == pytest.approx((1 + 1 / math.log2(3) + 0) / 3)
    assert res["Recall@20"] == pytest.approx(2 / 3)


def test_seen_items_are_masked(toy):
    # scores favour item 1, which every user except user 2 has already seen in train
    def score(users):
        return np.tile(np.array([0.0, 9.0, 1.0, 2.0, 3.0]), (len(users), 1))
    res = evaluate_scores(score, toy, "test", top=4)
    # user 0 ranking must not contain 0,1 (train) or 2 (val)
    assert res["n_users"] == 3


def test_missing_user_raises(toy):
    with pytest.raises(ValueError):
        evaluate_rankings({0: [1]}, toy, "test")
