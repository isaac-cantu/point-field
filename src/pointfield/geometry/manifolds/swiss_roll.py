import numpy as np

from pointfield.core.bounds import AABB
from pointfield.core.geometry import Geometry


def roll(latent: np.ndarray, t_min: float = 1.5 * np.pi, t_max: float = 4.5 * np.pi,
         height: float | None = None) -> np.ndarray:
    """Curl 2D latent coordinates (u, v) into a 3D swiss roll.

    u is rescaled to t in [t_min, t_max] and mapped to (t cos t, v', t sin t);
    v is rescaled to have the same range as the roll diameter if ``height`` is
    None. Isometric up to the rescaling, so the latent 2D coordinates are the
    ground-truth "unrolled" embedding.
    """
    L = np.asarray(latent, dtype=float)
    u, v = L[:, 0], L[:, 1]
    span_u = max(u.max() - u.min(), 1e-12)
    t = t_min + (u - u.min()) / span_u * (t_max - t_min)
    scale = (t_max - t_min) / span_u  # same factor for v keeps the map near-isometric
    h = (v - v.mean()) * scale if height is None else (v - v.mean()) / max(np.ptp(v), 1e-12) * height
    return np.column_stack((t * np.cos(t), h, t * np.sin(t)))


class SwissRoll(Geometry):
    """Classic swiss roll surface; ``last_latent`` stores the (t, h) coordinates."""

    def __init__(self, t_min: float = 1.5 * np.pi, t_max: float = 4.5 * np.pi, height: float = 21.0):
        self.t_min, self.t_max, self.height = float(t_min), float(t_max), float(height)
        self.last_latent: np.ndarray | None = None

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def intrinsic_dimension(self) -> int:
        return 2

    @property
    def bounds(self) -> AABB:
        R = self.t_max
        return AABB(np.array([-R, -self.height / 2, -R]), np.array([R, self.height / 2, R]))

    def sample(self, n_points: int, seed: int | None = None) -> np.ndarray:
        self._check_n(n_points)
        rng = np.random.default_rng(seed)
        # area-uniform: density of t proportional to arc length ~ t
        t = np.sqrt(rng.uniform(self.t_min ** 2, self.t_max ** 2, size=n_points))
        h = rng.uniform(-self.height / 2, self.height / 2, size=n_points)
        self.last_latent = np.column_stack((t, h))
        return np.column_stack((t * np.cos(t), h, t * np.sin(t)))
