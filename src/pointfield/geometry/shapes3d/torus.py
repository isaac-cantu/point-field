import numpy as np

from pointfield.geometry.shapes3d._base import Surface3D


class Torus(Surface3D):
    """Ring torus (major radius R, minor radius r), area-uniform via rejection."""

    def __init__(self, major_radius: float = 1.0, minor_radius: float = 0.35):
        if not 0 < minor_radius < major_radius:
            raise ValueError("require 0 < minor_radius < major_radius")
        self.major_radius, self.minor_radius = float(major_radius), float(minor_radius)

    def signed_distance(self, points):
        p = np.asarray(points, dtype=float)
        q = np.linalg.norm(p[:, :2], axis=1) - self.major_radius
        return np.sqrt(q ** 2 + p[:, 2] ** 2) - self.minor_radius

    def sample(self, n_points: int, seed: int | None = None) -> np.ndarray:
        self._check_n(n_points)
        rng = np.random.default_rng(seed)
        R, r = self.major_radius, self.minor_radius
        u_all, v_all, remaining = [], [], n_points
        while remaining > 0:
            u = rng.uniform(0, 2 * np.pi, 3 * remaining + 32)
            v = rng.uniform(0, 2 * np.pi, 3 * remaining + 32)
            keep = rng.uniform(size=len(v)) < (R + r * np.cos(v)) / (R + r)
            u_all.append(u[keep][:remaining]); v_all.append(v[keep][:remaining])
            remaining -= len(u_all[-1])
        u, v = np.concatenate(u_all), np.concatenate(v_all)
        return np.column_stack(((R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)))
