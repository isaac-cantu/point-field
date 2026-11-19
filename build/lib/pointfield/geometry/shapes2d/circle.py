import numpy as np

from pointfield.core.bounds import AABB
from pointfield.core.geometry import Geometry


class Circle(Geometry):
    """Circumference of radius ``radius`` (a 1-D curve in the plane)."""

    def __init__(self, radius: float = 1.0):
        if radius <= 0:
            raise ValueError("radius must be greater than zero")
        self.radius = float(radius)

    @property
    def ambient_dimension(self) -> int:
        return 2

    @property
    def intrinsic_dimension(self) -> int:
        return 1

    @property
    def bounds(self) -> AABB:
        r = self.radius
        return AABB(minimum=np.array([-r, -r]), maximum=np.array([r, r]))

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points)
        return np.linalg.norm(points, axis=1) - self.radius

    def contains(self, points: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
        return np.abs(self.signed_distance(points)) <= tolerance

    def sample(self, n_points: int, seed: int | None = None) -> np.ndarray:
        self._check_n(n_points)
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n_points)
        return self.radius * np.column_stack((np.cos(theta), np.sin(theta)))
