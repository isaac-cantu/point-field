import numpy as np

from pointfield.core.bounds import AABB
from pointfield.core.geometry import Geometry


class HyperSphere(Geometry):
    """Unit (n-1)-sphere S^{n-1} in R^n (``filled=True`` gives the n-ball)."""

    def __init__(self, dim: int = 3, radius: float = 1.0, filled: bool = False):
        if dim < 2:
            raise ValueError("dim must be at least 2")
        self.dim, self.radius, self.filled = int(dim), float(radius), bool(filled)

    @property
    def ambient_dimension(self) -> int:
        return self.dim

    @property
    def intrinsic_dimension(self) -> int:
        return self.dim if self.filled else self.dim - 1

    @property
    def bounds(self) -> AABB:
        return AABB(-self.radius * np.ones(self.dim), self.radius * np.ones(self.dim))

    def sample(self, n_points: int, seed: int | None = None) -> np.ndarray:
        self._check_n(n_points)
        rng = np.random.default_rng(seed)
        Z = rng.normal(size=(n_points, self.dim))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        if self.filled:
            Z *= rng.uniform(size=(n_points, 1)) ** (1.0 / self.dim)
        return self.radius * Z
