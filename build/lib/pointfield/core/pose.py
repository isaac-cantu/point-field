"""Rigid placement (rotation + translation + uniform scale) of a geometry."""

from dataclasses import dataclass, field

import numpy as np


def rotation_matrix_2d(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def rotation_matrix_3d(angles: tuple[float, float, float]) -> np.ndarray:
    """Rotation from XYZ Euler angles (radians)."""
    ax, ay, az = angles
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx


def random_rotation(dim: int, seed: int | None = None) -> np.ndarray:
    """Haar-uniform rotation in SO(dim) (QR of a Gaussian matrix)."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(dim, dim)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


@dataclass
class Pose:
    """x_world = scale * R @ x_local + translation."""

    translation: np.ndarray = field(default_factory=lambda: np.zeros(2))
    rotation: np.ndarray | float = 0.0
    scale: float = 1.0

    def matrix(self, dim: int) -> np.ndarray:
        if np.isscalar(self.rotation):
            if dim != 2:
                raise ValueError("scalar rotation is only valid in 2D")
            return rotation_matrix_2d(float(self.rotation))
        return np.asarray(self.rotation, dtype=float)

    def apply(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        dim = points.shape[1]
        R = self.matrix(dim)
        t = np.asarray(self.translation, dtype=float)
        if t.shape[0] != dim:
            raise ValueError("translation dimension does not match the points")
        return self.scale * points @ R.T + t
