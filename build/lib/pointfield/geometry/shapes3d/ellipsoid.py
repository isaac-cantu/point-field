import numpy as np

from pointfield.geometry.shapes3d._base import Surface3D


class Ellipsoid(Surface3D):
    """Ellipsoid surface with semi-axes (a, b, c), area-uniform: points of the
    unit sphere are accepted with probability proportional to the area
    element of the map u -> (a u1, b u2, c u3), which is exact."""

    def __init__(self, a: float = 1.0, b: float = 0.7, c: float = 0.4):
        if min(a, b, c) <= 0:
            raise ValueError("semi-axes must be positive")
        self.a, self.b, self.c = float(a), float(b), float(c)

    def sample(self, n_points: int, seed: int | None = None) -> np.ndarray:
        self._check_n(n_points)
        rng = np.random.default_rng(seed)
        axes = np.array([self.a, self.b, self.c])
        out, remaining = [], n_points
        gmax = max(self.a * self.b, self.a * self.c, self.b * self.c)
        while remaining > 0:
            Z = rng.normal(size=(4 * remaining + 64, 3))
            Z /= np.linalg.norm(Z, axis=1, keepdims=True)
            g = np.sqrt((self.b * self.c * Z[:, 0]) ** 2 + (self.a * self.c * Z[:, 1]) ** 2
                        + (self.a * self.b * Z[:, 2]) ** 2)
            keep = Z[rng.uniform(size=len(Z)) < g / gmax] * axes
            out.append(keep[:remaining])
            remaining -= len(out[-1])
        return np.vstack(out)
