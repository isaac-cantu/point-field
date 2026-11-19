"""Triangle meshes: OFF reader and area-uniform surface sampling.

Sampling a mesh defines a measure: faces are chosen with probability
proportional to their area and points inside a face with barycentric
coordinates (1 - sqrt(u1), sqrt(u1)(1 - u2), sqrt(u1) u2), which is uniform
on the triangle. Sampling vertices is *not* surface sampling.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from pointfield.core.bounds import AABB
from pointfield.core.geometry import Geometry


def read_off(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an OFF file (as used by ModelNet). Polygons are fan-triangulated.

    Handles the ModelNet quirk where the counts follow "OFF" on the same line
    (e.g. ``OFF490 518 0``).
    """
    tokens: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if line:
                tokens.extend(line.split())
    if not tokens or not tokens[0].startswith("OFF"):
        raise ValueError(f"{path}: not an OFF file")
    head = tokens[0][3:]
    pos = 1
    if head:
        tokens.insert(1, head)
    n_v, n_f = int(tokens[pos]), int(tokens[pos + 1])
    pos += 3
    V = np.array(tokens[pos:pos + 3 * n_v], dtype=float).reshape(n_v, 3)
    pos += 3 * n_v
    faces = []
    for _ in range(n_f):
        k = int(tokens[pos])
        idx = [int(t) for t in tokens[pos + 1:pos + 1 + k]]
        pos += 1 + k
        for j in range(1, k - 1):
            faces.append((idx[0], idx[j], idx[j + 1]))
    return V, np.asarray(faces, dtype=int).reshape(-1, 3)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TriangleMesh(Geometry):
    """Surface of a triangle mesh in R^3, sampled uniformly in area."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        V = np.asarray(vertices, dtype=float)
        F = np.asarray(faces, dtype=int)
        if V.ndim != 2 or V.shape[1] != 3 or F.ndim != 2 or F.shape[1] != 3:
            raise ValueError("vertices must be (n, 3) and faces (m, 3)")
        if not np.isfinite(V).all() or len(F) == 0 or np.any(F < 0) or np.any(F >= len(V)):
            raise ValueError("finite vertices and valid non-negative face indices required")
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        keep = areas > 0
        if not np.any(keep):
            raise ValueError("mesh has zero total area")
        self.vertices, self.faces, self.areas = V, F[keep], areas[keep]

    @classmethod
    def from_off(cls, path: str | Path) -> TriangleMesh:
        return cls(*read_off(path))

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def intrinsic_dimension(self) -> int:
        return 2

    @property
    def area(self) -> float:
        return float(self.areas.sum())

    @property
    def bounds(self) -> AABB:
        return AABB(minimum=self.vertices.min(axis=0), maximum=self.vertices.max(axis=0))

    def sample(self, n_points: int, seed: int | None = None) -> np.ndarray:
        self._check_n(n_points)
        rng = np.random.default_rng(seed)
        f = rng.choice(len(self.faces), size=n_points, p=self.areas / self.areas.sum())
        a, b, c = (self.vertices[self.faces[f, j]] for j in range(3))
        u1, u2 = rng.uniform(size=n_points), rng.uniform(size=n_points)
        s = np.sqrt(u1)
        return (1 - s)[:, None] * a + (s * (1 - u2))[:, None] * b + (s * u2)[:, None] * c
