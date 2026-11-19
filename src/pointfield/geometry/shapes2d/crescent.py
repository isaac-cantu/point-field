import numpy as np

from pointfield.core.bounds import AABB
from pointfield.core.geometry import Geometry


class Crescent(Geometry):
    """Thick circular arc (annular sector): the "moon" of the two-moons set.

    Points satisfy |r - radius| <= thickness/2 and
    angle_start <= theta <= angle_start + angle_span. ``thickness=0`` gives
    the bare arc (1-D).
    """

    def __init__(self, radius: float = 1.0, thickness: float = 0.2,
                 angle_start: float = 0.0, angle_span: float = np.pi):
        if radius <= 0:
            raise ValueError("radius must be greater than zero")
        if not 0 <= thickness < 2 * radius:
            raise ValueError("require 0 <= thickness < 2 * radius")
        if not 0 < angle_span <= 2 * np.pi:
            raise ValueError("angle_span must be in (0, 2*pi]")
        self.radius = float(radius)
        self.thickness = float(thickness)
        self.angle_start = float(angle_start)
        self.angle_span = float(angle_span)

    @property
    def ambient_dimension(self) -> int:
        return 2

    @property
    def intrinsic_dimension(self) -> int:
        return 2 if self.thickness > 0 else 1

    @property
    def bounds(self) -> AABB:
        th = self.angle_start + np.linspace(0.0, self.angle_span, 2001)
        rs = (self.radius - self.thickness / 2, self.radius + self.thickness / 2)
        P = np.vstack([np.column_stack((r * np.cos(th), r * np.sin(th))) for r in rs])
        return AABB.from_points(P)

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=float)
        r = np.linalg.norm(p, axis=1)
        th = np.mod(np.arctan2(p[:, 1], p[:, 0]) - self.angle_start, 2 * np.pi)
        radial = np.abs(r - self.radius) - self.thickness / 2
        inside_arc = th <= self.angle_span
        ends = []
        for a in (self.angle_start, self.angle_start + self.angle_span):
            e = self.radius * np.array([np.cos(a), np.sin(a)])
            ends.append(np.linalg.norm(p - e, axis=1) - self.thickness / 2)
        return np.where(inside_arc, radial, np.minimum(*ends))

    def contains(self, points: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
        return self.signed_distance(points) <= tolerance

    def sample(self, n_points: int, seed: int | None = None) -> np.ndarray:
        self._check_n(n_points)
        rng = np.random.default_rng(seed)
        th = self.angle_start + rng.uniform(0.0, self.angle_span, size=n_points)
        if self.thickness > 0:
            a = self.radius - self.thickness / 2
            b = self.radius + self.thickness / 2
            r = np.sqrt(rng.uniform(a * a, b * b, size=n_points))
        else:
            r = np.full(n_points, self.radius)
        return np.column_stack((r * np.cos(th), r * np.sin(th)))
