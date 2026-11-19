import numpy as np

from pointfield.core.bounds import AABB
from pointfield.core.geometry import Geometry
from pointfield.geometry._sampling import (
    points_in_polygon,
    polygon_boundary_distance,
    sample_polygon_area,
    sample_polyline,
)


class Polygon(Geometry):
    """Simple polygon (convex or not) given by its vertices in order.

    ``filled=True`` samples the interior (area-uniform); ``filled=False``
    samples the boundary (arc-length uniform). Vertices are re-centred so the
    polygon's vertex mean sits at the origin.
    """

    def __init__(self, vertices, filled: bool = True, center: bool = True):
        V = np.asarray(vertices, dtype=float)
        if V.ndim != 2 or V.shape[1] != 2 or len(V) < 3:
            raise ValueError("vertices must be an (m >= 3, 2) array")
        if center:
            V = V - V.mean(axis=0)
        self.vertices = V
        self.filled = bool(filled)

    @property
    def ambient_dimension(self) -> int:
        return 2

    @property
    def intrinsic_dimension(self) -> int:
        return 2 if self.filled else 1

    @property
    def bounds(self) -> AABB:
        return AABB(minimum=self.vertices.min(axis=0), maximum=self.vertices.max(axis=0))

    @property
    def area(self) -> float:
        x, y = self.vertices[:, 0], self.vertices[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=float)
        d = polygon_boundary_distance(p, self.vertices)
        return np.where(points_in_polygon(p, self.vertices), -d, d)

    def contains(self, points: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
        d = self.signed_distance(points)
        return d <= tolerance if self.filled else np.abs(d) <= tolerance

    def sample(self, n_points: int, seed: int | None = None) -> np.ndarray:
        self._check_n(n_points)
        rng = np.random.default_rng(seed)
        if self.filled:
            return sample_polygon_area(self.vertices, n_points, rng)
        return sample_polyline(self.vertices, n_points, rng, closed=True)


class RegularPolygon(Polygon):
    def __init__(self, n_sides: int = 6, radius: float = 1.0, filled: bool = True,
                 rotation: float = 0.0):
        if n_sides < 3:
            raise ValueError("n_sides must be at least 3")
        ang = rotation + 2 * np.pi * np.arange(n_sides) / n_sides
        super().__init__(radius * np.column_stack((np.cos(ang), np.sin(ang))), filled=filled)
        self.n_sides = int(n_sides)
        self.radius = float(radius)
