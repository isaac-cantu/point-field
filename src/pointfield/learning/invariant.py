"""Invariant whole-object descriptors built on the Christoffel–Darboux function.

This is the post-mid-term object pipeline used for the thesis (synthetic
families and ModelNet10). One vector per point cloud:

    normalise (centre + one RMS radius)
      -> weighted tensor basis, degree 2   (rotation-compatible ridge)
      -> M, eps = reg * tr(M) / s, Cholesky
      -> blocks:  logspec  log(1 + eta_j), eigenvalues of M            (s)
                  scores   19 quantiles of log(q(x_i) / s)             (19)
                  corr     19 quantiles of K(x_i,x_j)/sqrt(q_i q_j)    (19)
                  angular  radial-angular power invariants              (30)

Exact invariances (same points transformed): translation, positive scale,
rotations *and reflections* (the group is O(d), so chiral objects collide),
permutation of the points. NOT invariant to resampling, density, noise,
occlusion or outliers: those are measured, not assumed.

Controls that must be reported next to any CD result:

* :func:`moment_descriptor` ("M_U") -- spectra of the moment matrices of
  degree 2 and 3 + the same angular block, no inversion. In the thesis
  experiments CD and this control tie on clean data; CD wins under sampling
  density bias.
* :func:`pointfield.learning.descriptors.d2_histogram` -- D2 (Osada 2002).

Variants: ``sample_weight`` from :func:`knn_density_weights` ("CD_W", only
changes M), :func:`geometric_outlier_filter`, :class:`MultiViewMetric`
(Mahalanobis metric learnt from re-sampled views of reference objects).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import eval_gegenbauer, eval_legendre
from sklearn.neighbors import NearestNeighbors

from pointfield.learning.christoffel import (
    ChristoffelDarboux,
    monomial_basis,
    monomial_exponents,
    monomial_weights,
)

QUANTILES = np.linspace(0.05, 0.95, 19)


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------
def normalize_similarity(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Centre at the mean and divide by the RMS radius (one isotropic factor).

    For X' = a R X + t (a > 0, R orthogonal) the output is Y' = R Y.
    Not robust: outliers and occlusion move the centre and the scale.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) < 2 or not np.all(np.isfinite(X)):
        raise ValueError("X must be a finite (N, d) array with N >= 2")
    c = X.mean(axis=0)
    Y = X - c
    r = float(np.sqrt((Y ** 2).sum(axis=1).mean()))
    if r <= 0:
        raise ValueError("degenerate cloud (zero RMS radius)")
    return Y / r, c, r


# --------------------------------------------------------------------------
# angular block
# --------------------------------------------------------------------------
def _zonal(l: int, t: np.ndarray, dim: int) -> np.ndarray:
    """Zonal polynomial of degree l in cos(angle) for S^{dim-1}:
    Chebyshev cos(l*phi) in 2D, Legendre in 3D, normalised Gegenbauer beyond."""
    t = np.clip(t, -1.0, 1.0)
    if dim == 2:
        return np.cos(l * np.arccos(t))
    if dim == 3:
        return eval_legendre(l, t)
    a = (dim - 2) / 2.0
    return eval_gegenbauer(l, a, t) / eval_gegenbauer(l, a, 1.0)


def angular_block(Y: np.ndarray, centers=(0.5, 1.0, 1.5), width: float = 0.4,
                  l_max: int = 4) -> np.ndarray:
    """Radial-angular power invariants of a normalised cloud.

    P_l^{ab} = (1/N^2) sum_{i,j} b_a(r_i) b_b(r_j) Z_l(u_i . u_j), with Gaussian
    radial channels b_a and zonal polynomials Z_l. Keeps the upper triangle
    (a <= b) of each 3x3 matrix: 6 * (l_max + 1) = 30 numbers. Invariant to
    O(d) and permutations (addition theorem). Not the full SOAP descriptor.
    """
    Y = np.asarray(Y, dtype=float)
    N, dim = Y.shape
    r = np.linalg.norm(Y, axis=1)
    U = np.zeros_like(Y)
    nz = r > 1e-12
    U[nz] = Y[nz] / r[nz, None]
    G = U @ U.T
    Bm = np.exp(-0.5 * ((r[:, None] - np.asarray(centers)[None, :]) / width) ** 2)   # N x C
    iu = np.triu_indices(len(centers))
    out = []
    for l in range(l_max + 1):
        Z = _zonal(l, G, dim)
        if l > 0:                          # points at the origin carry no direction
            Z = Z * np.outer(nz, nz)
        P = Bm.T @ Z @ Bm / N ** 2
        out.append(P[iu])
    return np.concatenate(out)


# --------------------------------------------------------------------------
# CD descriptor ("CD_U" / "CD_W")
# --------------------------------------------------------------------------
@dataclass
class CDDescriptor:
    """CD object descriptor (see module docstring).

    Parameters
    ----------
    degree : polynomial degree (2 in all thesis experiments).
    reg : ridge, eps = reg * tr(M) / s.
    angular : append the 30-number angular block.
    max_pairs : number of point pairs used for the correlation quantiles
        (all pairs when N(N-1)/2 <= max_pairs; otherwise a seeded subsample,
        which breaks exact permutation invariance only through that sample).
    """

    degree: int = 2
    reg: float = 0.01
    angular: bool = True
    max_pairs: int = 200_000
    seed: int = 0
    blocks_: dict = field(default_factory=dict, init=False, repr=False)

    def __call__(self, X: np.ndarray, sample_weight: np.ndarray | None = None) -> np.ndarray:
        return self.transform(X, sample_weight)

    def transform(self, X: np.ndarray, sample_weight: np.ndarray | None = None) -> np.ndarray:
        Y, _, _ = normalize_similarity(X)
        cd = ChristoffelDarboux(self.degree, self.reg, basis="weighted").fit(Y, sample_weight)
        s = cd.basis_size_
        eta = np.clip(np.linalg.eigvalsh(cd.moment_matrix_), 0.0, None)
        K = cd.kernel(Y)
        q = np.diag(K).copy()
        iu = np.triu_indices(len(Y), k=1)
        corr = K[iu] / np.sqrt(q[iu[0]] * q[iu[1]])
        if len(corr) > self.max_pairs:
            corr = np.random.default_rng(self.seed).choice(corr, self.max_pairs, replace=False)
        blocks = {
            "logspec": np.log1p(eta),
            "scores": np.quantile(np.log(q / s), QUANTILES),
            "corr": np.quantile(np.clip(corr, -1.0, 1.0), QUANTILES),
        }
        if self.angular:
            blocks["angular"] = angular_block(Y)
        self.blocks_ = blocks
        return np.concatenate(list(blocks.values()))

    def block_slices(self, dim: int) -> dict[str, slice]:
        from math import comb
        sizes = {"logspec": comb(dim + self.degree, self.degree), "scores": 19, "corr": 19}
        if self.angular:
            sizes["angular"] = 30
        out, start = {}, 0
        for k, n in sizes.items():
            out[k] = slice(start, start + n)
            start += n
        return out


def moment_descriptor(X: np.ndarray, degrees=(2, 3), angular: bool = True) -> np.ndarray:
    """Control without CD inversion ("M_U"): log(1 + eigenvalues) of the
    weighted-basis moment matrices of each degree, + the angular block."""
    Y, _, _ = normalize_similarity(X)
    parts = []
    for n in degrees:
        terms = monomial_exponents(Y.shape[1], n)
        V = monomial_basis(Y, terms, monomial_weights(terms))
        M = V.T @ V / len(V)                       # no ridge, no inversion
        parts.append(np.log1p(np.clip(np.linalg.eigvalsh(M), 0.0, None)))
    if angular:
        parts.append(angular_block(Y))
    return np.concatenate(parts)


def moment_block_slices(dim: int, degrees=(2, 3), angular: bool = True) -> dict[str, slice]:
    from math import comb
    out, start = {}, 0
    for n in degrees:
        out[f"spec{n}"] = slice(start, start + comb(dim + n, n))
        start += comb(dim + n, n)
    if angular:
        out["angular"] = slice(start, start + 30)
    return out


# --------------------------------------------------------------------------
# density weights and outlier filter
# --------------------------------------------------------------------------
def knn_density_weights(X: np.ndarray, k: int = 16, exponent: float | None = None,
                        clip=(0.25, 4.0)) -> np.ndarray:
    """w_i ∝ clip[(r_i / median r)^m, lo, hi], r_i = distance to the k-th
    neighbour, m = intrinsic dimension (default d - 1, surfaces). Up-weights
    sparsely sampled regions. Lowered clean mAP in the thesis; use only when
    a density bias is expected."""
    X = np.asarray(X, dtype=float)
    m = X.shape[1] - 1 if exponent is None else exponent
    k = min(k, len(X) - 1)
    dist, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    r = dist[:, -1]
    w = np.clip((r / np.median(r)) ** m, *clip)
    return w / w.sum()


def geometric_outlier_filter(X: np.ndarray, k: int = 16, n_mad: float = 3.0) -> np.ndarray:
    """Boolean mask of points to keep: mean distance to the k nearest
    neighbours <= median + n_mad * 1.4826 * MAD. In the thesis it rescues
    retrieval with 5% uniform outliers but erodes thin parts of clean objects
    (lower clean accuracy): enable only with known contamination."""
    X = np.asarray(X, dtype=float)
    k = min(k, len(X) - 1)
    dist, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    m = dist[:, 1:].mean(axis=1)
    med = np.median(m)
    mad = 1.4826 * np.median(np.abs(m - med))
    return m <= med + n_mad * mad


# --------------------------------------------------------------------------
# block standardisation and metric
# --------------------------------------------------------------------------
class BlockScaler:
    """Standardise with reference statistics, divide each block by sqrt(its
    size) and the whole vector by sqrt(number of blocks), so that no block
    dominates the Euclidean distance because of its length."""

    def __init__(self, slices: dict[str, slice]):
        self.slices = slices

    def fit(self, F: np.ndarray) -> BlockScaler:
        F = np.asarray(F, dtype=float)
        self.mean_ = F.mean(axis=0)
        sd = F.std(axis=0)
        self.scale_ = np.where(sd > 1e-12, sd, 1.0)
        return self

    def transform(self, F: np.ndarray) -> np.ndarray:
        Z = (np.asarray(F, dtype=float) - self.mean_) / self.scale_
        out = np.empty_like(Z)
        for sl in self.slices.values():
            out[..., sl] = Z[..., sl] / np.sqrt(sl.stop - sl.start)
        return out / np.sqrt(len(self.slices))

    def fit_transform(self, F):
        return self.fit(F).transform(F)


def make_view(X: np.ndarray, rng: np.random.Generator, n_sub: int = 128,
              density_bias: float = 2.0, noise: float = 0.01) -> np.ndarray:
    """A re-sampled view of a cloud for metric learning: subsample ``n_sub``
    points with probability ∝ exp(density_bias * clip(t, -2, 2)), t the
    normalised coordinate along a random direction, plus small noise (in RMS
    units). Default views of the thesis multi-view metric."""
    Y, c, r = normalize_similarity(X)
    u = rng.normal(size=Y.shape[1])
    u /= np.linalg.norm(u)
    t = Y @ u
    p = np.exp(density_bias * np.clip(t / (t.std() + 1e-12), -2, 2))
    idx = rng.choice(len(Y), size=min(n_sub, len(Y)), replace=False, p=p / p.sum())
    V = Y[idx] + noise * rng.normal(size=(len(idx), Y.shape[1]))
    return V * r + c


class MultiViewMetric:
    """Mahalanobis metric learnt from within-object variation.

    ``fit`` takes features of shape (n_objects, n_views, p) of *reference*
    objects only. C = mean over objects and views of (f - mean_object)(...)^T;
    the metric is (C + eta * tr(C)/p * I)^(-1), eta chosen in ``etas`` by
    identity retrieval (view -> its own object) on a held-out 20% of the
    objects, then C is re-estimated on all of them. ``transform`` maps
    features so that Euclidean distance equals the learnt metric.
    """

    def __init__(self, etas=(0.1, 1.0, 10.0), holdout: float = 0.2, seed: int = 0):
        self.etas, self.holdout, self.seed = tuple(etas), holdout, seed

    @staticmethod
    def _cov(V: np.ndarray) -> np.ndarray:
        D = V - V.mean(axis=1, keepdims=True)
        D = D.reshape(-1, V.shape[2])
        return D.T @ D / len(D)

    @staticmethod
    def _whitener(C: np.ndarray, eta: float) -> np.ndarray:
        p = len(C)
        w, Q = np.linalg.eigh(C + eta * np.trace(C) / p * np.eye(p))
        return Q @ np.diag(1.0 / np.sqrt(w)) @ Q.T

    def fit(self, V: np.ndarray) -> MultiViewMetric:
        V = np.asarray(V, dtype=float)
        if V.ndim != 3 or V.shape[1] < 2:
            raise ValueError("V must be (n_objects, n_views >= 2, p)")
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(V))
        n_ho = max(2, round(self.holdout * len(V)))
        ho, tr = perm[:n_ho], perm[n_ho:]
        C = self._cov(V[tr])
        scores = {}
        for eta in self.etas:
            W = self._whitener(C, eta)
            G = V[ho, 1:].mean(axis=1) @ W            # gallery: mean of the other views
            Q = V[ho, 0] @ W                          # query: first view
            D = np.linalg.norm(Q[:, None] - G[None], axis=-1)
            scores[eta] = float((np.argmin(D, axis=1) == np.arange(len(ho))).mean())
        self.identity_accuracy_ = scores
        self.eta_ = max(self.etas, key=lambda e: (scores[e], -e))
        self.W_ = self._whitener(self._cov(V), self.eta_)
        return self

    def transform(self, F: np.ndarray) -> np.ndarray:
        return np.asarray(F, dtype=float) @ self.W_
