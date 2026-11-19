import numpy as np

from pointfield.geometry.shapes3d._base import Surface3D


class Cube(Surface3D):
    """Surface of an axis-aligned box (default: unit cube), area-uniform.
    Name kept as ``Cube``; pass ``size=(lx, ly, lz)`` for a box."""

    def __init__(self, size=1.0):
        s = np.broadcast_to(np.asarray(size, dtype=float), (3,)).copy()
        if np.any(s <= 0):
            raise ValueError("size must be positive")
        self.size = s

    def sample(self, n_points: int, seed: int | None = None) -> np.ndarray:
        self._check_n(n_points)
        rng = np.random.default_rng(seed)
        h = self.size / 2
        areas = np.array([self.size[1] * self.size[2], self.size[0] * self.size[2], self.size[0] * self.size[1]])
        face_axis = rng.choice(3, size=n_points, p=areas / areas.sum())
        P = rng.uniform(-h, h, size=(n_points, 3))
        sign = rng.choice([-1.0, 1.0], size=n_points)
        P[np.arange(n_points), face_axis] = sign * h[face_axis]
        return P
