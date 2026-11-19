"""Clustering and embedding built on the Christoffel–Darboux function.

Three stages, in the order they were developed (each motivated by a
measured failure of the previous one):

1. CDGlobalClustering   (CD inicial)  one CD model per group, points assigned
   to the model under which they are most typical (lowest P_d). k-means/GMM
   analogue; needs a density gap between groups.
2. CDLocalSpectral      (mejora 1)    one *local* CD model per neighbourhood
   defines an affinity graph; spectral embedding + k-means on it.
3. SzegoPhase           (mejora 2)    angular phase psi = theta - c*g(r)
   (Szegő / trigonometric CD) appended as features *only if* a label-free
   gate says the phase carries cluster structure.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import spectral_embedding
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from pointfield.learning.christoffel import ChristoffelDarboux, monomial_basis, monomial_exponents


# --------------------------------------------------------------------------
# 1. CD inicial: global mixture of CD models
# --------------------------------------------------------------------------
class CDGlobalClustering:
    """Assign each point to the group whose CD model finds it most typical.

    Initialised with k-means; alternates (fit one CD per group) / (reassign by
    argmin_k log P_d^{(k)}(x)). Because (1/N_k) sum P^{(k)} = s(n,d) for every
    group, raw P_d values are already on a common scale.
    """

    def __init__(self, n_clusters: int, degree: int = 2, n_iter: int = 30, reg: float = 1e-6,
                 random_state: int | None = 0):
        self.n_clusters, self.degree, self.n_iter = n_clusters, degree, n_iter
        self.reg, self.random_state = reg, random_state

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        labels = KMeans(self.n_clusters, n_init=10, random_state=self.random_state).fit_predict(X)
        s = len(monomial_exponents(X.shape[1], self.degree))
        for _ in range(self.n_iter):
            scores = np.full((len(X), self.n_clusters), np.inf)
            for k in range(self.n_clusters):
                members = X[labels == k]
                if len(members) < s + 2:          # too few points for a degree-d model
                    continue
                scores[:, k] = np.log(ChristoffelDarboux(self.degree, self.reg).fit(members).score(X))
            new = np.argmin(scores, axis=1)
            if np.array_equal(new, labels):
                break
            labels = new
        self.labels_ = labels
        return labels


# --------------------------------------------------------------------------
# 2. Mejora 1: local CD affinity + spectral embedding
# --------------------------------------------------------------------------
def local_cd_affinity(X: np.ndarray, n_neighbors: int = 10, degree: int = 1,
                      reg: float = 1e-6) -> np.ndarray:
    """A_ij from a CD model fitted only on the neighbourhood of x_i.

    P_i(x_j) = v_d(x_j)^T (M_d^{(i)})^{-1} v_d(x_j), M^{(i)} from the k+1 nearest
    points of x_i. Robust z-score inside the neighbourhood,
    z_ij = (P_i(x_j) - median_i) / MAD_i, and A_ij = exp(-z_ij^2 / 2),
    symmetrised with max(A, A^T). With degree=1, P_i - 1 is the Mahalanobis
    distance under the local covariance, so the graph follows the local
    orientation of the data (it is anisotropic, unlike an RBF kernel).
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    k = min(n_neighbors + 1, n)
    idx = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X, return_distance=False)
    terms = monomial_exponents(X.shape[1], degree)
    A = np.zeros((n, n))
    for i, nb in enumerate(idx):
        B = monomial_basis(X[nb], terms)
        M = B.T @ B / len(nb)
        M += reg * np.trace(M) / len(M) * np.eye(len(M))
        P = np.einsum("ij,jk,ik->i", B, np.linalg.pinv(M, hermitian=True), B)
        med = np.median(P)
        mad = np.median(np.abs(P - med)) + 1e-12
        z = (P - med) / mad
        A[i, nb] = np.exp(-0.5 * z ** 2)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return A


class CDLocalSpectral:
    """Spectral clustering / embedding on the local CD affinity graph."""

    def __init__(self, n_clusters: int = 2, n_neighbors: int = 10, degree: int = 1,
                 random_state: int | None = 0):
        self.n_clusters, self.n_neighbors = n_clusters, n_neighbors
        self.degree, self.random_state = degree, random_state

    def affinity(self, X):
        self.affinity_ = local_cd_affinity(X, self.n_neighbors, self.degree)
        return self.affinity_

    def embed(self, X: np.ndarray, n_components: int = 2) -> np.ndarray:
        A = self.affinity(X)
        return spectral_embedding(A, n_components=n_components, drop_first=True,
                                  random_state=self.random_state)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        A = self.affinity(X)
        E = spectral_embedding(A, n_components=self.n_clusters, drop_first=False,
                               random_state=self.random_state)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)   # Ng–Jordan–Weiss row normalisation
        self.embedding_ = E
        self.labels_ = KMeans(self.n_clusters, n_init=10, random_state=self.random_state).fit_predict(E)
        return self.labels_


# --------------------------------------------------------------------------
# 3. Mejora 2: Szegő coupled phase, gated
# --------------------------------------------------------------------------
def _fast_kmeans(F: np.ndarray, k: int, rng: np.random.Generator, n_init: int = 3, n_iter: int = 30) -> np.ndarray:
    """Minimal Lloyd k-means (k-means++ init) for the tiny 2-feature phase
    search; avoids sklearn's per-call overhead inside a grid of ~300 fits."""
    best_lab, best_inertia = None, np.inf
    for _ in range(n_init):
        C = [F[rng.integers(len(F))]]
        for _ in range(1, k):
            d2 = np.min(((F[:, None, :] - np.array(C)[None]) ** 2).sum(-1), axis=1)
            C.append(F[rng.choice(len(F), p=d2 / d2.sum())] if d2.sum() > 0 else F[rng.integers(len(F))])
        C = np.array(C)
        for _ in range(n_iter):
            lab = np.argmin(((F[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
            newC = np.array([F[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        lab = np.argmin(((F[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        inertia = ((F - C[lab]) ** 2).sum()
        if inertia < best_inertia:
            best_lab, best_inertia = lab, inertia
    return best_lab


class SzegoPhase:
    """Search psi(x) = theta(x) - c * g(r(x)), g in {identity, log}, so that the
    points form n_clusters tight arcs on the unit circle (cos psi, sin psi).

    c is parametrised scale-free as ``turns`` (windings across the radial
    span): c = 2*pi*turns / (g(r_max) - g(r_min)). turns and g are chosen,
    without labels, by maximising the silhouette of k-means on
    (cos psi, sin psi): coarse grid, then a local refinement. Requires
    centred data with a single (isotropic) scale factor.
    """

    def __init__(self, n_clusters: int, turns_grid=None, forms=("lin", "log"),
                 random_state: int | None = 0, max_points: int = 500):
        self.n_clusters = n_clusters
        self.turns_grid = np.linspace(-3.0, 3.0, 121) if turns_grid is None else np.asarray(turns_grid)
        self.forms, self.random_state, self.max_points = forms, random_state, max_points

    @staticmethod
    def _g(r, form):
        return r if form == "lin" else np.log(r + 1e-3)

    def _c(self, X, turns, form):
        g = self._g(np.linalg.norm(X[:, :2], axis=1), form)
        lo, hi = np.percentile(g, [1, 99])
        return 2 * np.pi * turns / max(hi - lo, 1e-9)

    @classmethod
    def phase(cls, X: np.ndarray, c: float, form: str) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        theta = np.arctan2(X[:, 1], X[:, 0])
        return theta - c * cls._g(np.linalg.norm(X[:, :2], axis=1), form)

    def _sil(self, sub, c, form):
        psi = self.phase(sub, c, form)
        F = np.column_stack((np.cos(psi), np.sin(psi)))
        lab = _fast_kmeans(F, self.n_clusters, np.random.default_rng(self.random_state))
        if len(np.unique(lab)) < 2:
            return -1.0
        return float(silhouette_score(F, lab))

    def fit(self, X: np.ndarray) -> "SzegoPhase":
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        sub = X if len(X) <= self.max_points else X[rng.choice(len(X), self.max_points, replace=False)]
        best = (-np.inf, 0.0, "lin")
        step = float(np.diff(self.turns_grid[:2])[0]) if len(self.turns_grid) > 1 else 0.1
        for form in self.forms:
            scores = [self._sil(sub, self._c(X, t, form), form) for t in self.turns_grid]
            t0 = self.turns_grid[int(np.argmax(scores))]
            for t in np.linspace(t0 - step, t0 + step, 11):
                sil = self._sil(sub, self._c(X, t, form), form)
                if sil > best[0]:
                    best = (sil, float(t), form)
        self.silhouette_, self.turns_, self.form_ = best
        self.c_ = float(self._c(X, self.turns_, self.form_))
        # silhouette of the untwisted angle (c = 0): the gain over it measures
        # how much *winding* (coupling of angle and radius) the data shows
        self.baseline_silhouette_ = self._sil(sub, 0.0, "lin")
        self.winding_gain_ = self.silhouette_ - self.baseline_silhouette_
        return self

    def features(self, X: np.ndarray) -> np.ndarray:
        psi = self.phase(X, self.c_, self.form_)
        return np.column_stack((np.cos(psi), np.sin(psi)))


class CDSzegoSpectral:
    """CD-Local-Spectral with the gated Szegő phase extension.

    1. phase = SzegoPhase.fit(X)  (label-free search of turns and g).
    2. gate: winding_gain = sil(best phase) - sil(plain angle, c = 0).
       A genuine winding makes the twisted angle far more clustered than the
       plain angle; blobs or rings do not. Activate only if gain >= ``gate``
       (0.20, fixed on development seeds 0-2, never on evaluation seeds).
    3. CDLocalSpectral on [X, w cos psi, w sin psi] if active, else on X.
    """

    def __init__(self, n_clusters: int = 2, n_neighbors: int = 10, degree: int = 1,
                 gate: float = 0.20, weight: float = 1.0, random_state: int | None = 0):
        self.n_clusters, self.n_neighbors, self.degree = n_clusters, n_neighbors, degree
        self.gate, self.weight, self.random_state = gate, weight, random_state

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        self.phase_ = SzegoPhase(self.n_clusters, random_state=self.random_state).fit(X)
        self.active_ = bool(self.phase_.winding_gain_ >= self.gate)
        Z = np.column_stack((X, self.weight * self.phase_.features(X))) if self.active_ else X
        model = CDLocalSpectral(self.n_clusters, self.n_neighbors, self.degree, self.random_state)
        self.labels_ = model.fit_predict(Z)
        self.features_ = Z
        return self.labels_
