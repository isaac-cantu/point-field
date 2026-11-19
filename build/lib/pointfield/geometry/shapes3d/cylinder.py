import numpy as np

from pointfield.geometry.shapes3d._base import Surface3D


class Cylinder(Surface3D):
    """Closed cylinder surface (lateral wall + caps) along z, area-uniform."""

    def __init__(self, radius: float = 0.5, height: float = 1.5, caps: bool = True):
        if radius <= 0 or height <= 0:
            raise ValueError("radius and height must be positive")
        self.radius, self.height, self.caps = float(radius), float(height), bool(caps)

    def sample(self, n_points: int, seed: int | None = None) -> np.ndarray:
        self._check_n(n_points)
        rng = np.random.default_rng(seed)
        r, H = self.radius, self.height
        a_side = 2 * np.pi * r * H
        a_cap = np.pi * r * r if self.caps else 0.0
        part = rng.choice(3, size=n_points, p=np.array([a_side, a_cap, a_cap]) / (a_side + 2 * a_cap))
        th = rng.uniform(0, 2 * np.pi, n_points)
        rr = np.where(part == 0, r, r * np.sqrt(rng.uniform(size=n_points)))
        z = np.where(part == 0, rng.uniform(-H / 2, H / 2, n_points), np.where(part == 1, H / 2, -H / 2))
        return np.column_stack((rr * np.cos(th), rr * np.sin(th), z))
