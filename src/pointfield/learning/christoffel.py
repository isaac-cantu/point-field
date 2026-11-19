"""Christoffel–Darboux (CD) core.

Empirical moment matrix, CD kernel and Christoffel function for a point
cloud X = {x_1..x_N} in R^d seen as the empirical measure mu_N
(notation: d = dimension, n = degree, s = number of monomials):

    v_n(x)   = (w_alpha x^alpha)_{|alpha| <= n}           size s = C(n+d, d)
    M        = (1/N) sum_i v_n(x_i) v_n(x_i)^T             (or a weighted mean)
    A        = M + eps I,  eps = reg * tr(M) / s           (ridge)
    q(x)     = K(x, x) = v_n(x)^T A^{-1} v_n(x)            large off the support
    Lambda(x)= 1 / q(x)                                    Christoffel function

Variational meaning: Lambda(z) = min { a^T A a : a^T v(z) = 1 }, i.e. the
smallest (regularised) mean square of a degree-<=n polynomial equal to 1 at z.

Identities used as checks: (1/N) sum_i q(x_i) = s with eps = 0 and full rank;
with eps > 0 it equals sum_j eta_j / (eta_j + eps) <= s. Degree 1 gives
q(x) = 1 + (x - m)^T Sigma^{-1} (x - m) (Mahalanobis, Sigma with divisor N).

Numerics: A is factorised with Cholesky; nothing is inverted explicitly.
``pinv`` must not be used: it discards near-null directions, which are exactly
the polynomial equations satisfied by the support (circle: x^2 + y^2 - 1), and
then reports points off the support as typical.

Basis: ``basis="monomial"`` (raw x^alpha, as in the mid-term benchmark) or
``basis="weighted"`` (w_alpha = sqrt(k! / alpha!), k = |alpha|). With the
weighted basis <phi_k(x), phi_k(y)> = (x^T y)^k, so phi(Rx) = U(R) phi(x)
with U orthogonal: M -> U M U^T, eps I is preserved, and q, K and the
spectrum of M are invariant to rotations and reflections of centred data.
"""

from __future__ import annotations

import warnings
from itertools import combinations_with_replacement
from math import comb, factorial

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve, solve_triangular


def monomial_exponents(n_vars: int, degree: int) -> list[tuple[int, ...]]:
    """All monomials of total degree <= ``degree`` as tuples of column indices
    (with repetition): () is the constant, (0, 0, 1) is x0^2 x1, ..."""
    terms: list[tuple[int, ...]] = []
    for k in range(degree + 1):
        terms.extend(combinations_with_replacement(range(n_vars), k))
    return terms


def monomial_weights(terms) -> np.ndarray:
    """sqrt(k! / alpha!) for each monomial (multinomial weights)."""
    w = np.empty(len(terms))
    for j, combo in enumerate(terms):
        counts = np.bincount(np.asarray(combo, dtype=int)) if combo else np.zeros(0, dtype=int)
        denom = np.prod([factorial(int(c)) for c in counts]) if len(counts) else 1
        w[j] = np.sqrt(factorial(len(combo)) / denom)
    return w


def monomial_basis(X: np.ndarray, terms, weights: np.ndarray | None = None) -> np.ndarray:
    """Design matrix B with B[i, j] = w_j * x_i^{alpha_j}."""
    X = np.asarray(X, dtype=float)
    B = np.ones((X.shape[0], len(terms)))
    for j, combo in enumerate(terms):
        for idx in combo:
            B[:, j] *= X[:, idx]
    if weights is not None:
        B *= weights
    return B


def basis_size(n_vars: int, degree: int) -> int:
    return comb(n_vars + degree, degree)


class ChristoffelDarboux:
    """Global CD model of one point cloud.

    Parameters
    ----------
    degree : polynomial degree n.
    reg : ridge eps relative to tr(M)/s (eps = reg * tr(M) / s).
    basis : "monomial" (raw powers) or "weighted" (rotation-compatible).
    """

    def __init__(self, degree: int = 2, reg: float = 1e-8, basis: str = "monomial"):
        if not isinstance(degree, (int, np.integer)) or degree < 0:
            raise ValueError("degree must be a non-negative integer")
        if not np.isfinite(reg) or reg < 0:
            raise ValueError("reg must be finite and non-negative")
        if basis not in ("monomial", "weighted"):
            raise ValueError("basis must be 'monomial' or 'weighted'")
        self.degree = int(degree)
        self.reg = float(reg)
        self.basis = basis

    # ------------------------------------------------------------------ fit
    def design(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_ or not np.isfinite(X).all():
            raise ValueError("query must be finite with the fitted dimension")
        return monomial_basis(X, self.terms_, self.weights_)

    def fit(self, X: np.ndarray, sample_weight: np.ndarray | None = None) -> "ChristoffelDarboux":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) == 0 or not np.all(np.isfinite(X)):
            raise ValueError("X must be a finite (N, d) array with N > 0")
        self.n_features_ = X.shape[1]
        self.terms_ = monomial_exponents(self.n_features_, self.degree)
        self.weights_ = monomial_weights(self.terms_) if self.basis == "weighted" else None
        B = self.design(X)
        if sample_weight is None:
            M = (B.T @ B) / len(X)
        else:
            w = np.asarray(sample_weight, dtype=float)
            if w.shape != (len(X),) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
                raise ValueError("sample_weight must be non-negative with positive sum")
            M = (B * (w / w.sum())[:, None]).T @ B
        s = len(M)
        self.eps_ = self.reg * np.trace(M) / s
        self.moment_matrix_ = M
        self.M_reg_ = M + self.eps_ * np.eye(s)
        try:
            self._chol = cho_factor(self.M_reg_, lower=True)
        except LinAlgError as err:
            raise LinAlgError("M + eps I is not positive definite; use reg > 0 "
                              "(the support satisfies a polynomial equation)") from err
        self.n_samples_ = len(X)
        return self

    # --------------------------------------------------------------- queries
    def _whiten(self, X: np.ndarray) -> np.ndarray:
        """Z = L^{-1} v(x)^T, so that q(x) = ||Z[:, i]||^2."""
        L = self._chol[0]
        return solve_triangular(L, self.design(X).T, lower=True, check_finite=False)

    def score(self, X: np.ndarray) -> np.ndarray:
        """q(x) = v(x)^T (M + eps I)^{-1} v(x)  (called P_d in the mid-term report)."""
        Z = self._whiten(X)
        return np.einsum("ij,ij->j", Z, Z)

    def christoffel(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / self.score(X)

    def kernel(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        ZX = self._whiten(X)
        ZY = ZX if Y is None else self._whiten(Y)
        return ZX.T @ ZY

    @property
    def M_inv_(self) -> np.ndarray:
        """(M + eps I)^{-1}, computed from the Cholesky factor (kept for compatibility)."""
        return cho_solve(self._chol, np.eye(len(self.M_reg_)))

    @property
    def basis_size_(self) -> int:
        return len(self.terms_)

    @property
    def condition_number_(self) -> float:
        """cond(M) of the *unregularised* moment matrix (diagnostic)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(np.linalg.cond(self.moment_matrix_))

    @property
    def effective_dof_(self) -> float:
        """sum_j eta_j / (eta_j + eps): mean of q on the training data."""
        eta = np.clip(np.linalg.eigvalsh(self.moment_matrix_), 0.0, None)
        return float(np.sum(eta / (eta + self.eps_)))

    def trace_check(self, X: np.ndarray) -> float:
        """Relative error of (1/N) sum q(x_i) vs s on the training data
        (exact only with eps -> 0 and full rank; see ``effective_dof_``)."""
        return float(abs(self.score(X).mean() - self.basis_size_) / self.basis_size_)


def degree_diagnostics(X: np.ndarray, degrees=(1, 2, 3, 4, 5, 6), reg: float = 1e-8) -> list[dict]:
    """Basis size, samples per parameter, cond(M) and the trace check per degree."""
    rows = []
    for d in degrees:
        cd = ChristoffelDarboux(d, reg).fit(X)
        rows.append({
            "degree": d,
            "basis_size": cd.basis_size_,
            "N_per_param": len(X) / cd.basis_size_,
            "cond_M": cd.condition_number_,
            "trace_rel_error": cd.trace_check(X),
        })
    return rows
