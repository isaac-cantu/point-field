import numpy as np

from pointfield.core.bounds import AABB
from pointfield.core.geometry import Geometry


class Annulus(Geometry):
    """Ring region inner_radius <= |x| <= outer_radius (area-uniform)."""

    def __init__(self, inner_radius: float = 0.5, outer_radius: float = 1.0):
        if not 0 <= inner_radius < outer_radius:
            raise ValueError("require 0 <= inner_radius < outer_radius")
        self.inner_radius = float(inner_radius)
        self.outer_radius = float(outer_radius)

    @property
    def ambient_dimension(self) -> int:
        return 2

    @property
    def intrinsic_dimension(self) -> int:
        return 2

    @property
    def bounds(self) -> AABB:
        r = self.outer_radius
        return AABB(minimum=np.array([-r, -r]), maximum=np.array([r, r]))

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(np.asarray(points), axis=1)
        mid = 0.5 * (self.inner_radius + self.outer_radius)
        half = 0.5 * (self.outer_radius - self.inner_radius)
        return np.abs(r - mid) - half

    def contains(self, points: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
        return self.signed_distance(points) <= tolerance

    def sample(self, n_points: int, seed: int | None = None) -> np.ndarray:
        self._check_n(n_points)
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n_points)
        a2, b2 = self.inner_radius ** 2, self.outer_radius ** 2
        r = np.sqrt(rng.uniform(a2, b2, size=n_points))
        return np.column_stack((r * np.cos(theta), r * np.sin(theta)))
