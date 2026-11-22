"""Ranking metrics. Every function takes ranked item lists and sets of relevant items."""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np


def ndcg_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    dcg = sum(1.0 / math.log2(r + 2) for r, item in enumerate(ranked[:k]) if item in relevant)
    ideal = sum(1.0 / math.log2(r + 2) for r in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def catalogue_coverage(rankings: Iterable[Sequence[int]], n_items: int, k: int) -> float:
    """Share of the catalogue that appears in at least one top-k list."""
    seen: set[int] = set()
    for ranked in rankings:
        seen.update(ranked[:k])
    return len(seen) / n_items


def average_popularity(rankings: Iterable[Sequence[int]], popularity: np.ndarray, k: int) -> float:
    """Mean training popularity (number of training positives) of recommended items."""
    vals = [popularity[list(r[:k])].mean() for r in rankings if len(r)]
    return float(np.mean(vals)) if vals else 0.0


def intra_list_diversity(rankings: Iterable[Sequence[int]], item_vectors: np.ndarray, k: int) -> float:
    """Mean pairwise cosine *distance* between the items of each top-k list (Week 3, notes v2).

    `item_vectors` are unit-norm rows (here: IDF-weighted note vectors, so the measure is the
    same for every model). Items with an all-zero vector (no notes) are left out of their list;
    lists with fewer than 2 such items are skipped. 0 = every list is k copies of one scent
    profile, 1 = no two recommended perfumes share a note."""
    vals = []
    for ranked in rankings:
        v = item_vectors[list(ranked[:k])]
        v = v[np.abs(v).sum(1) > 0]
        n = len(v)
        if n < 2:
            continue
        sim = v @ v.T
        vals.append(1.0 - (sim.sum() - np.trace(sim)) / (n * (n - 1)))
    return float(np.mean(vals)) if vals else 0.0


def brand_diversity(rankings: Iterable[Sequence[int]], brands: np.ndarray, k: int) -> float:
    """Mean share of distinct brands in a top-k list (1 = k different brands)."""
    vals = [len({brands[i] for i in r[:k]}) / len(r[:k]) for r in rankings if len(r)]
    return float(np.mean(vals)) if vals else 0.0
