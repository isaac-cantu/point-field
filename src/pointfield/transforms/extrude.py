import numpy as np


def extrude(points: np.ndarray, height: float = 1.0, seed: int | None = None) -> np.ndarray:
    """2D -> 3D prism: each point gets a uniform z in [-height/2, height/2]."""
    P = np.asarray(points, dtype=float)
    if P.shape[1] != 2:
        raise ValueError("extrude expects 2D points")
    rng = np.random.default_rng(seed)
    return np.column_stack((P, rng.uniform(-height / 2, height / 2, len(P))))
