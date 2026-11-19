import numpy as np

from pointfield.geometry.shapes3d._base import Surface3D


class Cone(Surface3D):
    """Cone surface (lateral + base disk) along z, apex up, centred, area-uniform."""

    def __init__(self, radius: float = 0.6, height: float = 1.4, base: bool = True):
        if radius <= 0 or height <= 0:
            raise ValueError("radius and height must be positive")
        self.radius, self.height, self.base = float(radius), float(height), bool(base)

    def sample(self, n_points: int, seed: int | None = None) -> np.ndarray:
        self._check_n(n_points)
        rng = np.random.default_rng(seed)
        r, H = self.radius, self.height
        slant = np.hypot(r, H)
        a_lat, a_base = np.pi * r * slant, (np.pi * r * r if self.base else 0.0)
        lateral = rng.uniform(size=n_points) < a_lat / (a_lat + a_base)
        th = rng.uniform(0, 2 * np.pi, n_points)
        s = np.sqrt(rng.uniform(size=n_points))           # fraction from apex (lateral) / radius frac (base)
        rr = r * s
        z = np.where(lateral, H / 2 - H * s, -H / 2)
        return np.column_stack((rr * np.cos(th), rr * np.sin(th), z))
