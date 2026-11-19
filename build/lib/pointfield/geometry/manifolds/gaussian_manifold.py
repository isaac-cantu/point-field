import numpy as np

from pointfield.core.bounds import AABB
from pointfield.core.geometry import Geometry


class GaussianBlob(Geometry):
    """Anisotropic Gaussian cloud N(0, Sigma) in any dimension (compact group).

    Sigma = R diag(std**2) R^T with R given by ``rotation`` (2D: an angle).
    Bounds are the 3-sigma box.
    """

    def __init__(self, std=(1.0, 1.0), rotation: float | np.ndarray = 0.0):
        std = np.atleast_1d(np.asarray(std, dtype=float))
        if np.any(std <= 0):
            raise ValueError("std must be positive")
        self.std = std
        if np.isscalar(rotation):
            if len(std) != 2 and rotation != 0.0:
                raise ValueError("scalar rotation only valid in 2D")
            c, s = np.cos(rotation), np.sin(rotation)
            R = np.array([[c, -s], [s, c]]) if len(std) == 2 else np.eye(len(std))
        else:
            R = np.asarray(rotation, dtype=float)
        self._R = R
        self.rotation = rotation if np.isscalar(rotation) else "matrix"

    @property
    def ambient_dimension(self) -> int:
        return len(self.std)

    @property
    def intrinsic_dimension(self) -> int:
        return len(self.std)

    @property
    def covariance(self) -> np.ndarray:
        return self._R @ np.diag(self.std ** 2) @ self._R.T

    @property
    def bounds(self) -> AABB:
        ext = 3 * np.sqrt(np.diag(self.covariance))
        return AABB(-ext, ext)

    def sample(self, n_points: int, seed: int | None = None) -> np.ndarray:
        self._check_n(n_points)
        rng = np.random.default_rng(seed)
        Z = rng.normal(size=(n_points, len(self.std))) * self.std
        return Z @ self._R.T
