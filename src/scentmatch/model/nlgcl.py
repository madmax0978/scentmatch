"""NLGCL (Xu et al., ACM RecSys 2025) in PyTorch Geometric, with ScentMatch's note extension.

* Propagation: LightGCN (LGConv) over the normalised graph, L layers, final embedding =
  mean of layers 0..L.
* Neighbour-layer contrastive loss (heterogeneous scope, Eq. 6-7): for each training pair
  (u, i) and each of the first G layers g, the layer-(g+1) embedding of perfume i is pulled
  towards the layer-g embedding of user u (and symmetrically), against in-batch negatives,
  with cosine similarity and temperature tau - following the authors' public code
  (github.com/Jinfeng-Xu/NLGCL).
* ScentMatch extension: optional note nodes. The same neighbour-layer contrast is applied
  to (perfume, note) edges, pulling perfumes that share notes together.

Loss:  L = L_bpr + lambda_1 * (alpha * L_nl^u + (1 - alpha) * L_nl^i) + lambda_note * L_nl^note
           + lambda_2 ||Theta||^2

Week 3 changes
* the note term has its own weight lambda_note (it used to share lambda_1);
* the trainer feeds (perfume, note) pairs with *distinct* notes in a batch: with only ~470
  notes, random pairs repeated the same note, so InfoNCE pushed a note away from itself
  (false negatives) - the likely reason why notes hurt accuracy in Week 2;
* `cl_weight = 0` gives LightGCN, the baseline NLGCL extends (ablation of the paper's idea).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LGConv

from scentmatch.model.graph import Graph


class NLGCL(nn.Module):
    def __init__(self, graph: Graph, embedding_size: int = 64, n_layers: int = 2,
                 cl_layers: int = 2, cl_temp: float = 0.1, cl_weight: float = 5e-5,
                 reg_weight: float = 1e-4, alpha: float = 0.6, note_cl_weight: float | None = None, **_):
        super().__init__()
        assert 1 <= cl_layers <= n_layers, "G must satisfy 1 <= G <= L"
        self.n_users, self.n_items, self.n_notes = graph.n_users, graph.n_items, graph.n_notes
        self.n_layers, self.cl_layers = n_layers, cl_layers
        self.cl_temp, self.cl_weight, self.reg_weight, self.alpha = cl_temp, cl_weight, reg_weight, alpha
        self.note_cl_weight = cl_weight if note_cl_weight is None else note_cl_weight
        self.embedding = nn.Embedding(graph.n_nodes, embedding_size)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.conv = LGConv(normalize=False)          # weights already normalised in Graph
        self.register_buffer("edge_index", graph.edge_index)
        self.register_buffer("edge_weight", graph.edge_weight)

    # ------------------------------------------------------------------ forward
    def propagate(self) -> list[torch.Tensor]:
        """Return the list of layer embeddings [E^(0), ..., E^(L)] for all nodes."""
        x = self.embedding.weight
        layers = [x]
        for _ in range(self.n_layers):
            x = self.conv(x, self.edge_index, self.edge_weight)
            layers.append(x)
        return layers

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        layers = self.propagate()
        final = torch.stack(layers, dim=1).mean(dim=1)
        users, items, notes = torch.split(final, [self.n_users, self.n_items, self.n_notes])
        return users, items, notes, layers

    # ------------------------------------------------------------------ losses
    def _split(self, x: torch.Tensor):
        return torch.split(x, [self.n_users, self.n_items, self.n_notes])

    def _info_nce(self, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        """In-batch InfoNCE: row k of `positive` is the positive of row k of `anchor`."""
        a, p = F.normalize(anchor, dim=1), F.normalize(positive, dim=1)
        logits = a @ p.T / self.cl_temp
        return F.cross_entropy(logits, torch.arange(len(a), device=a.device), reduction="sum")

    def neighbor_layer_cl(self, layers, users, items, note_pairs=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (user-perfume contrastive loss, perfume-note contrastive loss)."""
        cl_u = cl_i = cl_n = torch.zeros(())
        for g in range(self.cl_layers):
            u_g, i_g, n_g = self._split(layers[g])
            u_next, i_next, n_next = self._split(layers[g + 1])
            cl_u = cl_u + self._info_nce(i_next[items], u_g[users])    # user-side view pair
            cl_i = cl_i + self._info_nce(u_next[users], i_g[items])    # item-side view pair
            if note_pairs is not None and self.n_notes > 0:
                p_items, p_notes = note_pairs
                cl_n = cl_n + self._info_nce(n_next[p_notes], i_g[p_items]) \
                            + self._info_nce(i_next[p_items], n_g[p_notes])
        return self.alpha * cl_u + (1 - self.alpha) * cl_i, cl_n

    def loss(self, users: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor,
             note_pairs: tuple[torch.Tensor, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        u_all, i_all, _, layers = self.forward()
        u, p, n = u_all[users], i_all[pos], i_all[neg]
        bpr = F.softplus(-((u * p).sum(1) - (u * n).sum(1))).mean()
        ego = self.embedding.weight
        reg = (ego[users].pow(2).sum() + ego[pos + self.n_users].pow(2).sum()
               + ego[neg + self.n_users].pow(2).sum()) / 2 / len(users)
        if self.cl_weight > 0 or (note_pairs is not None and self.note_cl_weight > 0):
            cl, cl_note = self.neighbor_layer_cl(layers, users, pos, note_pairs)
        else:                                            # LightGCN: no contrastive term at all
            cl = cl_note = torch.zeros(())
        total = bpr + self.cl_weight * cl + self.note_cl_weight * cl_note + self.reg_weight * reg
        return {"loss": total, "bpr": bpr.detach(), "cl": cl.detach(), "cl_note": cl_note.detach(),
                "reg": reg.detach()}

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def final_embeddings(self):
        users, items, notes, _ = self.forward()
        return users, items, notes
