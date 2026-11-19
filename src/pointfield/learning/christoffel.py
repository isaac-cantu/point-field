"""Christoffel–Darboux (CD) core.

Empirical moment matrix, CD kernel and Christoffel function for a point
cloud X = {x_1..x_N} in R^n seen as the empirical measure mu_N:

    v_d(x)  = (x^alpha)_{|alpha| <= d}                  monomial vector, size s(n,d) = C(n+d, d)
    M_d     = (1/N) sum_i v_d(x_i) v_d(x_i)^T + eps I    (Tikhonov-regularised)
    K_d(x,y)= v_d(x)^T M_d^{-1} v_d(y)
    P_d(x)  = K_d(x,x)          small on the support, grows off it
    Lambda_d(x) = 1 / P_d(x)    Christoffel function

Sanity identity (eps -> 0): (1/N) sum_i P_d(x_i) = s(n,d).
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from math import comb

import numpy as np


def monomial_exponents(n_vars: int, degree: int) -> list[tuple[int, ...]]:
    """All monomials of total degree <= ``degree`` as tuples of column indices
    (with repetition): () is the constant, (0, 0, 1) is x0^2 x1, ..."""
    terms: list[tuple[int, ...]] = []
    for k in range(degree + 1):
        terms.extend(combinations_with_replacement(range(n_vars), k))
    return terms


def monomial_basis(X: np.ndarray, terms) -> np.ndarray:
    """Design matrix B with B[i, j] = v_d(x_i)_j."""
    X = np.asarray(X, dtype=float)
    B = np.ones((X.shape[0], len(terms)))
    for j, combo in enumerate(terms):
        for idx in combo:
            B[:, j] *= X[:, idx]
    return B


def basis_size(n_vars: int, degree: int) -> int:
    return comb(n_vars + degree, degree)


class ChristoffelDarboux:
    """Global CD model of one point cloud.

    Parameters
    ----------
    degree : polynomial degree d.
    reg : Tikhonov ridge eps added to M_d (relative to its trace / size).
    """

    def __init__(self, degree: int = 2, reg: float = 1e-8):
        if degree < 0:
            raise ValueError("degree must be non-negative")
        self.degree = int(degree)
        self.reg = float(reg)

    def fit(self, X: np.ndarray) -> "ChristoffelDarboux":
        X = np.asarray(X, dtype=float)
        self.n_features_ = X.shape[1]
        self.terms_ = monomial_exponents(self.n_features_, self.degree)
        B = monomial_basis(X, self.terms_)
        M = (B.T @ B) / len(X)
        eps = self.reg * np.trace(M) / len(M)
        self.moment_matrix_ = M
        self.M_reg_ = M + eps * np.eye(len(M))
        self.M_inv_ = np.linalg.pinv(self.M_reg_, hermitian=True)
        self.n_samples_ = len(X)
        return self

    # --------------------------------------------------------------- queries
    def score(self, X: np.ndarray) -> np.ndarray:
        """P_d(x) = v_d(x)^T M_d^{-1} v_d(x)."""
        B = monomial_basis(X, self.terms_)
        return np.einsum("ij,jk,ik->i", B, self.M_inv_, B)

    def christoffel(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / self.score(X)

    def kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return monomial_basis(X, self.terms_) @ self.M_inv_ @ monomial_basis(Y, self.terms_).T

    @property
    def basis_size_(self) -> int:
        return len(self.terms_)

    @property
    def condition_number_(self) -> float:
        return float(np.linalg.cond(self.moment_matrix_))

    def trace_check(self, X: np.ndarray) -> float:
        """Relative error of (1/N) sum P_d(x_i) vs s(n,d) on the training data."""
        return float(abs(self.score(X).mean() - self.basis_size_) / self.basis_size_)


def degree_diagnostics(X: np.ndarray, degrees=(1, 2, 3, 4, 5, 6), reg: float = 1e-8) -> list[dict]:
    """Basis size, samples per parameter, cond(M_d) and the trace check per degree."""
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
