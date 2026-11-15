"""Sanity-floor rankers (not control models): random and most-popular."""
from __future__ import annotations

import numpy as np


def random_scores(n_items: int, seed: int = 0):
    rng = np.random.default_rng(seed)

    def score(users: np.ndarray) -> np.ndarray:
        return rng.random((len(users), n_items), dtype=np.float32)
    return score


def most_popular_scores(popularity: np.ndarray):
    """popularity must be computed on training interactions only (no leakage)."""
    pop = popularity.astype(np.float32)

    def score(users: np.ndarray) -> np.ndarray:
        return np.broadcast_to(pop, (len(users), len(pop)))
    return score
