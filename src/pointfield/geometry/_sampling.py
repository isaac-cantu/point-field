"""Shared sampling utilities (arc-length and area-uniform sampling)."""

import numpy as np


def sample_polyline(vertices: np.ndarray, n_points: int, rng: np.random.Generator,
                    closed: bool = True) -> np.ndarray:
    """Uniform-in-arc-length sample on a polyline."""
    V = np.asarray(vertices, dtype=float)
    if closed:
        V = np.vstack([V, V[:1]])
    seg = np.diff(V, axis=0)
    lengths = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(lengths)])
    s = rng.uniform(0.0, cum[-1], size=n_points)
    idx = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
    t = (s - cum[idx]) / np.where(lengths[idx] > 0, lengths[idx], 1.0)
    return V[idx] + t[:, None] * seg[idx]


def sample_parametric_curve(curve, t_min: float, t_max: float, n_points: int,
                            rng: np.random.Generator, resolution: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Arc-length-uniform sample of a parametric curve t -> curve(t) (vectorised).

    Returns (points, t) so callers can keep the latent parameter.
    """
    tt = np.linspace(t_min, t_max, resolution)
    P = curve(tt)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = rng.uniform(0.0, cum[-1], size=n_points)
    t = np.interp(s, cum, tt)
    return curve(t), t


def points_in_polygon(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Even-odd rule point-in-polygon test (works for non-convex polygons)."""
    x, y = points[:, 0], points[:, 1]
    V = np.asarray(vertices, dtype=float)
    inside = np.zeros(len(points), dtype=bool)
    j = len(V) - 1
    for i in range(len(V)):
        xi, yi = V[i]
        xj, yj = V[j]
        cond = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (xj - xi) * (y - yi) / (yj - yi) + xi
        inside ^= cond & (x < xint)
        j = i
    return inside


def sample_polygon_area(vertices: np.ndarray, n_points: int, rng: np.random.Generator) -> np.ndarray:
    """Area-uniform sample inside a (possibly non-convex) simple polygon by rejection."""
    V = np.asarray(vertices, dtype=float)
    lo, hi = V.min(axis=0), V.max(axis=0)
    out = []
    remaining = n_points
    while remaining > 0:
        cand = rng.uniform(lo, hi, size=(max(4 * remaining, 64), 2))
        keep = cand[points_in_polygon(cand, V)]
        out.append(keep[:remaining])
        remaining -= len(out[-1])
    return np.vstack(out)


def segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((points - a) @ ab) / max(ab @ ab, 1e-300), 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(points - proj, axis=1)


def polygon_boundary_distance(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    V = np.asarray(vertices, dtype=float)
    d = np.full(len(points), np.inf)
    for i in range(len(V)):
        d = np.minimum(d, segment_distance(points, V[i], V[(i + 1) % len(V)]))
    return d
