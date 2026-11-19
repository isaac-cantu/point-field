import numpy as np

from pointfield.geometry.shapes2d.polygon import Polygon


class Star(Polygon):
    """Star polygon with ``n_tips`` tips (non-convex region or boundary)."""

    def __init__(self, n_tips: int = 5, outer_radius: float = 1.0,
                 inner_radius: float = 0.45, filled: bool = True):
        if n_tips < 3:
            raise ValueError("n_tips must be at least 3")
        if not 0 < inner_radius < outer_radius:
            raise ValueError("require 0 < inner_radius < outer_radius")
        k = np.arange(2 * n_tips)
        ang = np.pi / 2 + np.pi * k / n_tips
        r = np.where(k % 2 == 0, outer_radius, inner_radius)
        super().__init__(np.column_stack((r * np.cos(ang), r * np.sin(ang))), filled=filled)
        self.n_tips = int(n_tips)
        self.outer_radius = float(outer_radius)
        self.inner_radius = float(inner_radius)
