import numpy as np


def translate(points: np.ndarray, offset) -> np.ndarray:
    return np.asarray(points, dtype=float) + np.asarray(offset, dtype=float)
