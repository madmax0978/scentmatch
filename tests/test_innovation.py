"""Week 3 notes v2: main-notes filter, diversity metrics and note grounding (hand-computed)."""
import numpy as np
import pandas as pd
import pytest

from scentmatch.eval.metrics import brand_diversity, intra_list_diversity
from scentmatch.model.graph import build_graph, main_notes


@pytest.fixture
def notes():
    # note 0 is on every perfume (generic), note 1 on two, notes 2 and 3 on one each (rare)
    return pd.DataFrame({"item_idx": [0, 0, 0, 1, 1, 2, 2],
                         "note_idx": [0, 1, 2, 0, 1, 0, 3]})


def test_main_notes_keeps_rarest(notes):
    m = main_notes(notes, 1)
    assert dict(zip(m["item_idx"], m["note_idx"])) == {0: 2, 1: 1, 2: 3}
    m2 = main_notes(notes, 2)
    assert sorted(m2[m2["item_idx"] == 0]["note_idx"]) == [1, 2]      # generic note 0 dropped
    assert m2.groupby("item_idx").size().max() == 2


def test_main_notes_none_keeps_all(notes):
    assert main_notes(notes, None).equals(notes)
    assert len(main_notes(notes, 10)) == len(notes)


def test_graph_with_main_notes_has_fewer_edges(notes):
    train = pd.DataFrame({"user_idx": [0, 1], "item_idx": [0, 2]})
    full = build_graph(train, 2, 3, notes, 4, note_edge_weight=0.3)
    cut = build_graph(train, 2, 3, main_notes(notes, 1), 4, note_edge_weight=0.3)
    assert full.edge_index.shape[1] == 2 * (2 + 7)
    assert cut.edge_index.shape[1] == 2 * (2 + 3)


def test_intra_list_diversity():
    v = np.array([[1, 0], [1, 0], [0, 1], [0, 0]], dtype=float)
    assert intra_list_diversity([[0, 1]], v, 10) == pytest.approx(0.0)    # same profile
    assert intra_list_diversity([[0, 2]], v, 10) == pytest.approx(1.0)    # nothing shared
    # 0,1,2: pairs (0,1)=0, (0,2)=1, (1,2)=1 -> 2/3; item 3 has no notes and is ignored
    assert intra_list_diversity([[0, 1, 2, 3]], v, 10) == pytest.approx(2 / 3)
    assert intra_list_diversity([[3, 0]], v, 10) == 0.0                   # < 2 items with notes


def test_brand_diversity():
    brands = np.array(["a", "a", "b", "c"])
    assert brand_diversity([[0, 1], [2, 3]], brands, 10) == pytest.approx((0.5 + 1.0) / 2)
