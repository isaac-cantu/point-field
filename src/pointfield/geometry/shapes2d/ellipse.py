import numpy as np

from pointfield.core.bounds import AABB
from pointfield.core.geometry import Geometry
from pointfield.geometry._sampling import sample_parametric_curve


class Ellipse(Geometry):
    """Ellipse with semi-axes (a, b). ``filled=True`` samples the region,
    ``filled=False`` samples the boundary curve uniformly in arc length."""

    def __init__(self, a: float = 1.0, b: float = 0.5, filled: bool = True):
        if a <= 0 or b <= 0:
            raise ValueError("semi-axes must be greater than zero")
        self.a = float(a)
        self.b = float(b)
        self.filled = bool(filled)

    @property
    def ambient_dimension(self) -> int:
        return 2

    @property
    def intrinsic_dimension(self) -> int:
        return 2 if self.filled else 1

    @property
    def bounds(self) -> AABB:
        return AABB(minimum=np.array([-self.a, -self.b]), maximum=np.array([self.a, self.b]))

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """First-order approximation of the signed distance to the boundary."""
        p = np.asarray(points, dtype=float)
        k0 = np.sqrt((p[:, 0] / self.a) ** 2 + (p[:, 1] / self.b) ** 2)
        k1 = np.sqrt((p[:, 0] / self.a ** 2) ** 2 + (p[:, 1] / self.b ** 2) ** 2)
        return k0 * (k0 - 1.0) / np.maximum(k1, 1e-12)

    def contains(self, points: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
        d = self.signed_distance(points)
        return d <= tolerance if self.filled else np.abs(d) <= tolerance

    def sample(self, n_points: int, seed: int | None = None) -> np.ndarray:
        self._check_n(n_points)
        rng = np.random.default_rng(seed)
        if self.filled:
            theta = rng.uniform(0.0, 2.0 * np.pi, size=n_points)
            r = np.sqrt(rng.uniform(0.0, 1.0, size=n_points))
            return np.column_stack((self.a * r * np.cos(theta), self.b * r * np.sin(theta)))
        curve = lambda t: np.column_stack((self.a * np.cos(t), self.b * np.sin(t)))
        P, _ = sample_parametric_curve(curve, 0.0, 2.0 * np.pi, n_points, rng)
        return P
