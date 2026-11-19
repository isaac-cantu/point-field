import numpy as np


def affine(points: np.ndarray, A, b=None) -> np.ndarray:
    """x -> A x + b."""
    P = np.asarray(points, dtype=float)
    A = np.asarray(A, dtype=float)
    out = P @ A.T
    return out if b is None else out + np.asarray(b, dtype=float)
