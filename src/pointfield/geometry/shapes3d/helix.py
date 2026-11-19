import numpy as np

from pointfield.core.bounds import AABB
from pointfield.core.geometry import Geometry
from pointfield.geometry._sampling import sample_parametric_curve


class Helix(Geometry):
    """Circular helix (radius, pitch per turn, turns), optional phase and jitter."""

    def __init__(self, radius: float = 1.0, pitch: float = 1.0, turns: float = 2.0,
                 phase: float = 0.0, thickness: float = 0.0):
        self.radius, self.pitch, self.turns = float(radius), float(pitch), float(turns)
        self.phase, self.thickness = float(phase), float(thickness)

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def intrinsic_dimension(self) -> int:
        return 1

    @property
    def bounds(self) -> AABB:
        r, H = self.radius, self.pitch * self.turns
        return AABB(np.array([-r, -r, -H / 2]), np.array([r, r, H / 2]))

    def curve(self, t):
        H = self.pitch * self.turns
        return np.column_stack((self.radius * np.cos(t + self.phase), self.radius * np.sin(t + self.phase),
                                self.pitch * t / (2 * np.pi) - H / 2))

    def sample(self, n_points: int, seed: int | None = None) -> np.ndarray:
        self._check_n(n_points)
        rng = np.random.default_rng(seed)
        P, _ = sample_parametric_curve(self.curve, 0.0, 2 * np.pi * self.turns, n_points, rng)
        if self.thickness > 0:
            P = P + rng.normal(scale=self.thickness, size=P.shape)
        return P
