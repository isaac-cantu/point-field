import numpy as np

from pointfield.core.bounds import AABB
from pointfield.core.geometry import Geometry
from pointfield.geometry._sampling import sample_parametric_curve


class Spiral(Geometry):
    """Planar spiral arm, sampled uniformly in arc length.

    kind="archimedean":  r(t) = a + b * t
    kind="logarithmic":  r(t) = a * exp(b * t)
    with t in [0, 2*pi*turns] and the arm rotated by ``phase``. ``thickness``
    adds isotropic Gaussian jitter of that standard deviation (0 = bare curve).
    The latent arc parameter of the last sample is kept in ``last_t``.
    """

    def __init__(self, a: float = 0.5, b: float = 0.5, turns: float = 1.5,
                 kind: str = "archimedean", phase: float = 0.0, thickness: float = 0.0):
        if kind not in ("archimedean", "logarithmic"):
            raise ValueError("kind must be 'archimedean' or 'logarithmic'")
        if turns <= 0:
            raise ValueError("turns must be greater than zero")
        if kind == "logarithmic" and a <= 0:
            raise ValueError("logarithmic spirals need a > 0")
        if thickness < 0:
            raise ValueError("thickness must be non-negative")
        self.a, self.b, self.turns = float(a), float(b), float(turns)
        self.kind, self.phase, self.thickness = kind, float(phase), float(thickness)
        self.last_t: np.ndarray | None = None

    @property
    def ambient_dimension(self) -> int:
        return 2

    @property
    def intrinsic_dimension(self) -> int:
        return 1

    @property
    def t_max(self) -> float:
        return 2.0 * np.pi * self.turns

    def radius_at(self, t: np.ndarray) -> np.ndarray:
        if self.kind == "archimedean":
            return self.a + self.b * t
        return self.a * np.exp(self.b * t)

    def curve(self, t: np.ndarray) -> np.ndarray:
        r = self.radius_at(t)
        return np.column_stack((r * np.cos(t + self.phase), r * np.sin(t + self.phase)))

    @property
    def bounds(self) -> AABB:
        R = float(self.radius_at(np.array([self.t_max]))[0]) + 3 * self.thickness
        return AABB(minimum=np.array([-R, -R]), maximum=np.array([R, R]))

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Distance to a dense polyline of the arm (unsigned, a curve has no inside)."""
        p = np.asarray(points, dtype=float)
        C = self.curve(np.linspace(0, self.t_max, 3000))
        d2 = ((p[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        return np.sqrt(d2.min(axis=1))

    def contains(self, points: np.ndarray, tolerance: float = 1e-2) -> np.ndarray:
        return self.signed_distance(points) <= tolerance

    def sample(self, n_points: int, seed: int | None = None) -> np.ndarray:
        self._check_n(n_points)
        rng = np.random.default_rng(seed)
        P, t = sample_parametric_curve(self.curve, 0.0, self.t_max, n_points, rng)
        self.last_t = t
        if self.thickness > 0:
            P = P + rng.normal(scale=self.thickness, size=P.shape)
        return P
