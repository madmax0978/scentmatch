"""NLGCL skeleton: forward pass, loss and one optimisation step on a toy graph."""
import pandas as pd
import torch

from scentmatch.model.graph import build_graph
from scentmatch.model.nlgcl import NLGCL


def toy_graph(with_notes: bool):
    train = pd.DataFrame({"user_idx": [0, 0, 1, 2, 3, 3], "item_idx": [0, 1, 1, 2, 2, 3]})
    notes = pd.DataFrame({"item_idx": [0, 1, 2, 3], "note_idx": [0, 0, 1, 1]}) if with_notes else None
    return build_graph(train, n_users=4, n_items=4, item_notes=notes, n_notes=2 if with_notes else 0)


def test_graph_is_symmetric_and_normalised():
    g = toy_graph(False)
    assert g.edge_index.shape[1] == 12                       # 6 edges x 2 directions
    assert torch.all(g.edge_weight > 0) and torch.all(g.edge_weight <= 1)


def test_forward_and_step():
    for with_notes in (False, True):
        g = toy_graph(with_notes)
        model = NLGCL(g, embedding_size=8, n_layers=2, cl_layers=2)
        users, items, notes, layers = model()
        assert users.shape == (4, 8) and items.shape == (4, 8)
        assert notes.shape == ((2, 8) if with_notes else (0, 8))
        assert len(layers) == 3
        pairs = (torch.tensor([0, 2]), torch.tensor([0, 1])) if with_notes else None
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        out = model.loss(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]), torch.tensor([3, 3, 0]), pairs)
        assert torch.isfinite(out["loss"])
        before = model.embedding.weight.detach().clone()
        out["loss"].backward()
        opt.step()
        assert not torch.equal(before, model.embedding.weight)
