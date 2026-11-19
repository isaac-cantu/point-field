import numpy as np

from pointfield.geometry.shapes2d.polygon import Polygon


class Triangle(Polygon):
    """Triangle from three vertices (default: equilateral of circumradius 1).

    The filled version uses exact barycentric sampling (no rejection).
    """

    def __init__(self, vertices=None, filled: bool = True):
        if vertices is None:
            ang = np.pi / 2 + 2 * np.pi * np.arange(3) / 3
            vertices = np.column_stack((np.cos(ang), np.sin(ang)))
        V = np.asarray(vertices, dtype=float)
        if V.shape != (3, 2):
            raise ValueError("a triangle needs exactly 3 vertices in 2D")
        super().__init__(V, filled=filled)

    def sample(self, n_points: int, seed: int | None = None) -> np.ndarray:
        if not self.filled:
            return super().sample(n_points, seed)
        self._check_n(n_points)
        rng = np.random.default_rng(seed)
        u, v = rng.uniform(size=(2, n_points))
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        A, B, C = self.vertices
        return A + u[:, None] * (B - A) + v[:, None] * (C - A)
