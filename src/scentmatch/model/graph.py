"""Graph construction (Project Lead): user-perfume(-note) graph from TRAINING edges only.

Node index layout:  [0, n_users) users | [n_users, n_users+n_items) perfumes | then notes.
Edges are undirected (both directions stored) and symmetrically normalised
(D^-1/2 A D^-1/2, no self-loops) as in LightGCN / NLGCL.
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


def build_graph(train: pd.DataFrame, n_users: int, n_items: int,
                item_notes: pd.DataFrame | None = None, n_notes: int = 0) -> Graph:
    u = train["user_idx"].to_numpy()
    i = train["item_idx"].to_numpy() + n_users
    src, dst = [u, i], [i, u]
    if item_notes is not None and n_notes > 0:
        it = item_notes["item_idx"].to_numpy() + n_users
        nt = item_notes["note_idx"].to_numpy() + n_users + n_items
        src += [it, nt]
        dst += [nt, it]
    else:
        n_notes = 0
    edge_index = torch.from_numpy(np.stack([np.concatenate(src), np.concatenate(dst)])).long()
    n_nodes = n_users + n_items + n_notes
    edge_index, edge_weight = gcn_norm(edge_index, None, n_nodes, add_self_loops=False)
    return Graph(n_users, n_items, n_notes, edge_index, edge_weight)
