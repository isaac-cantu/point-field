"""Retrieval metrics and paired bootstrap used in the object experiments.

Protocol (thesis): queries are compared against a gallery; a gallery item is
relevant when it has the query's class. mAP is the mean over queries of the
average precision of the ranked gallery. Two methods are compared on the
*same* queries with a stratified paired bootstrap of the per-query AP.
"""

from __future__ import annotations

import numpy as np


def _dist(Q: np.ndarray, G: np.ndarray) -> np.ndarray:
    Q, G = np.asarray(Q, float), np.asarray(G, float)
    d2 = (Q ** 2).sum(1)[:, None] + (G ** 2).sum(1)[None, :] - 2 * Q @ G.T
    return np.sqrt(np.clip(d2, 0.0, None))


def average_precision(dist_row: np.ndarray, relevant: np.ndarray) -> float:
    """AP of one query: mean of precision@k at the ranks of relevant items
    (ties broken by a stable sort)."""
    order = np.argsort(dist_row, kind="stable")
    rel = np.asarray(relevant, bool)[order]
    if not rel.any():
        return float("nan")
    hits = np.cumsum(rel)
    ranks = np.flatnonzero(rel) + 1
    return float((hits[rel] / ranks).mean())


def per_query_ap(Q, yq, G, yg) -> np.ndarray:
    D = _dist(Q, G)
    yq, yg = np.asarray(yq), np.asarray(yg)
    return np.array([average_precision(D[i], yg == yq[i]) for i in range(len(yq))])


def mean_average_precision(Q, yq, G, yg) -> float:
    return float(np.nanmean(per_query_ap(Q, yq, G, yg)))


def nn_accuracy(Q, yq, G, yg) -> float:
    """1-NN classification accuracy of the queries against the gallery."""
    D = _dist(Q, G)
    return float((np.asarray(yg)[np.argmin(D, axis=1)] == np.asarray(yq)).mean())


def precision_at_k(Q, yq, G, yg, k: int = 10) -> float:
    D = _dist(Q, G)
    idx = np.argsort(D, axis=1, kind="stable")[:, :k]
    return float((np.asarray(yg)[idx] == np.asarray(yq)[:, None]).mean())


def paired_bootstrap(a: np.ndarray, b: np.ndarray, strata: np.ndarray | None = None,
                     n_boot: int = 3000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Bootstrap of mean(a) - mean(b) for per-query scores of two methods on
    the same queries, resampling within each stratum (class). Returns the
    difference, a percentile CI and the fraction of resamples <= 0."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("a and b must be paired (same shape)")
    diff = a - b
    strata = np.zeros(len(diff), int) if strata is None else np.asarray(strata)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for t in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        stats[t] = diff[idx].mean()
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_le_0": float((stats <= 0).mean())}
