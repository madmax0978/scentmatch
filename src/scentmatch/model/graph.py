"""Graph construction (Project Lead): user-perfume(-note) graph from TRAINING edges only.

Node index layout:  [0, n_users) users | [n_users, n_users+n_items) perfumes | then notes.
Edges are undirected (both directions stored) and symmetrically normalised
(D^-1/2 A D^-1/2, no self-loops) as in LightGCN / NLGCL.

Week 3: perfume-note edges can be given a weight w < 1 (`note_edge_weight`). A perfume has
~7 notes but only ~8 training ratings, so with w = 1 half of its neighbourhood is notes and
the note signal drowns the collaborative one; w is tuned on validation.

Week 3 (notes v2): `main_notes(item_notes, k)` keeps only the k most distinctive notes of each
perfume (highest IDF over the catalogue). Notes are extracted from Amazon text, so long
descriptions list 20-45 notes, most of them generic ("musk", "amber"); they add edges that
link almost every perfume to every other one. k is tuned on validation (`max_notes_per_item`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm


@dataclass
class Graph:
    n_users: int
    n_items: int
    n_notes: int
    edge_index: torch.Tensor      # [2, E]
    edge_weight: torch.Tensor     # [E]

    @property
    def n_nodes(self) -> int:
        return self.n_users + self.n_items + self.n_notes


def main_notes(item_notes: pd.DataFrame, k: int | None) -> pd.DataFrame:
    """Keep, for each perfume, its k notes with the highest IDF (rarest = most characteristic).
    Ties are broken by note id so the result is deterministic. k = None or 0 keeps everything."""
    if not k:
        return item_notes
    df = item_notes.groupby("note_idx")["item_idx"].transform("size")
    out = item_notes.assign(_df=df).sort_values(["item_idx", "_df", "note_idx"], kind="stable")
    return out.groupby("item_idx", sort=False).head(int(k)).drop(columns="_df").reset_index(drop=True)


def build_graph(train: pd.DataFrame, n_users: int, n_items: int,
                item_notes: pd.DataFrame | None = None, n_notes: int = 0,
                note_edge_weight: float = 1.0) -> Graph:
    u = train["user_idx"].to_numpy()
    i = train["item_idx"].to_numpy() + n_users
    src, dst = [u, i], [i, u]
    weights = [np.ones(2 * len(u), dtype=np.float32)]
    if item_notes is not None and n_notes > 0:
        it = item_notes["item_idx"].to_numpy() + n_users
        nt = item_notes["note_idx"].to_numpy() + n_users + n_items
        src += [it, nt]
        dst += [nt, it]
        weights.append(np.full(2 * len(it), note_edge_weight, dtype=np.float32))
    else:
        n_notes = 0
    edge_index = torch.from_numpy(np.stack([np.concatenate(src), np.concatenate(dst)])).long()
    weight = torch.from_numpy(np.concatenate(weights))
    n_nodes = n_users + n_items + n_notes
    edge_index, edge_weight = gcn_norm(edge_index, weight, n_nodes, add_self_loops=False)
    return Graph(n_users, n_items, n_notes, edge_index, edge_weight)
