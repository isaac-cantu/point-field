import numpy as np

from pointfield.geometry.shapes3d._base import Surface3D


class Sphere(Surface3D):
    def __init__(self, radius: float = 1.0):
        if radius <= 0:
            raise ValueError("radius must be greater than zero")
        self.radius = float(radius)

    def signed_distance(self, points):
        return np.linalg.norm(np.asarray(points), axis=1) - self.radius

    def sample(self, n_points: int, seed: int | None = None) -> np.ndarray:
        self._check_n(n_points)
        rng = np.random.default_rng(seed)
        Z = rng.normal(size=(n_points, 3))
        return self.radius * Z / np.linalg.norm(Z, axis=1, keepdims=True)
