import numpy as np

from pointfield.core.bounds import AABB
from pointfield.core.geometry import Geometry
from pointfield.geometry._sampling import segment_distance


class Line(Geometry):
    """Straight segment between ``start`` and ``end`` (1-D)."""

    def __init__(self, start=(-1.0, 0.0), end=(1.0, 0.0)):
        a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        if a.shape != (2,) or b.shape != (2,):
            raise ValueError("start and end must be 2D points")
        if np.allclose(a, b):
            raise ValueError("start and end must differ")
        self.start, self.end = a, b

    @property
    def ambient_dimension(self) -> int:
        return 2

    @property
    def intrinsic_dimension(self) -> int:
        return 1

    @property
    def centroid(self) -> np.ndarray:
        return 0.5 * (self.start + self.end)

    @property
    def bounds(self) -> AABB:
        return AABB(np.minimum(self.start, self.end), np.maximum(self.start, self.end))

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        return segment_distance(np.asarray(points, dtype=float), self.start, self.end)

    def contains(self, points: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
        return self.signed_distance(points) <= tolerance

    def sample(self, n_points: int, seed: int | None = None) -> np.ndarray:
        self._check_n(n_points)
        rng = np.random.default_rng(seed)
        t = rng.uniform(0.0, 1.0, size=n_points)
        return self.start + t[:, None] * (self.end - self.start)
