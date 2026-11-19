"""Whole-object descriptors for 3D point clouds (exploratory stage).

CD descriptor: the moment matrix M_d of the (centred, RMS-normalised) cloud,
made rotation-invariant by averaging over random rotations (empirical
Reynolds operator, E_g[M_d(gX)]), trace-normalised and compared with the
Log-Euclidean distance on the SPD manifold, ||log M1 - log M2||_F.

Classical training-free baselines for the same task:
  * D2 shape distribution (Osada et al., 2002): histogram of pairwise distances.
  * PCA eigenvalue features (linearity / planarity / sphericity), the usual
    covariance descriptor in point-cloud/LiDAR work.
"""

from __future__ import annotations

import numpy as np

from pointfield.core.pose import random_rotation
from pointfield.learning.christoffel import monomial_basis, monomial_exponents


def _normalize(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0)
    return Xc / np.sqrt((Xc ** 2).sum(axis=1).mean())


def cd_moment_descriptor(X: np.ndarray, degree: int = 4, n_rotations: int = 30,
                         seed: int | None = 0, reg: float = 1e-8) -> np.ndarray:
    """Rotation-averaged, trace-normalised moment matrix (SPD)."""
    X = _normalize(np.asarray(X, dtype=float))
    terms = monomial_exponents(X.shape[1], degree)
    rng = np.random.default_rng(seed)
    M = np.zeros((len(terms), len(terms)))
    for _ in range(n_rotations):
        R = random_rotation(X.shape[1], seed=int(rng.integers(2**31 - 1)))
        B = monomial_basis(X @ R.T, terms)
        M += B.T @ B / len(X)
    M /= n_rotations
    M /= np.trace(M)
    return M + reg * np.eye(len(M))


def spd_log(M: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(M)
    return (V * np.log(np.clip(w, 1e-300, None))) @ V.T


def log_euclidean_distances(mats: list[np.ndarray]) -> np.ndarray:
    L = np.array([spd_log(M).ravel() for M in mats])
    sq = (L ** 2).sum(1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * L @ L.T, 0.0)
    return np.sqrt(D2)


def d2_histogram(X: np.ndarray, n_pairs: int = 20000, bins: int = 32, seed: int | None = 0) -> np.ndarray:
    X = _normalize(np.asarray(X, dtype=float))
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, len(X), n_pairs), rng.integers(0, len(X), n_pairs)
    d = np.linalg.norm(X[i] - X[j], axis=1)
    h, _ = np.histogram(d, bins=bins, range=(0, 4.0))
    return h / h.sum()


def pca_eigen_features(X: np.ndarray) -> np.ndarray:
    X = _normalize(np.asarray(X, dtype=float))
    w = np.sort(np.linalg.eigvalsh(np.cov(X.T)))[::-1]
    w = w / w.sum()
    l1, l2, l3 = w
    return np.array([l1, l2, l3, (l1 - l2) / l1, (l2 - l3) / l1, l3 / l1])


def pairwise(features: np.ndarray, metric: str = "l2") -> np.ndarray:
    F = np.asarray(features, dtype=float)
    if metric == "l1":
        return np.abs(F[:, None, :] - F[None, :, :]).sum(-1)
    return np.linalg.norm(F[:, None, :] - F[None, :, :], axis=-1)


def loo_1nn_accuracy(D: np.ndarray, y: np.ndarray) -> float:
    D = D.copy()
    np.fill_diagonal(D, np.inf)
    return float((y[np.argmin(D, axis=1)] == y).mean())
