import numpy as np


def scale(points: np.ndarray, factor, center=None) -> np.ndarray:
    """Uniform (scalar) or per-axis (vector) scaling about ``center``."""
    P = np.asarray(points, dtype=float)
    c = np.zeros(P.shape[1]) if center is None else np.asarray(center, dtype=float)
    return (P - c) * np.asarray(factor, dtype=float) + c
