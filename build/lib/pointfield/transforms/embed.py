"""Embeddings of low-dimensional point sets into higher-dimensional spaces."""

import numpy as np

from pointfield.core.pose import random_rotation
from pointfield.geometry.manifolds.swiss_roll import roll


def embed_linear(points: np.ndarray, ambient_dim: int, seed: int | None = None,
                 noise: float = 0.0) -> np.ndarray:
    """Isometric linear embedding R^d -> R^D via a random orthonormal frame."""
    P = np.asarray(points, dtype=float)
    d = P.shape[1]
    if ambient_dim < d:
        raise ValueError("ambient_dim must be >= the input dimension")
    rng = np.random.default_rng(seed)
    Q = random_rotation(ambient_dim, seed=int(rng.integers(2**31 - 1)))[:, :d]
    out = P @ Q.T
    if noise > 0:
        out = out + rng.normal(scale=noise, size=out.shape)
    return out


def embed_roll(points: np.ndarray, **kwargs) -> np.ndarray:
    """Nonlinear 2D -> 3D embedding: curl the plane into a swiss roll."""
    return roll(points, **kwargs)


__all__ = ["embed_linear", "embed_roll"]
