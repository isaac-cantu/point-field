"""Controlled perturbations applied to a :class:`PointDataset`.

Each function returns a new dataset and appends a record to
``meta["perturbations"]`` so the exact corruption is reproducible.
Scales are relative to the dataset's RMS radius so the same parameters mean
the same difficulty on every geometry.
"""

import numpy as np

from pointfield.core.dataset import PointDataset


def _rms_radius(X: np.ndarray) -> float:
    Xc = X - X.mean(axis=0)
    return float(np.sqrt((Xc ** 2).sum(axis=1).mean()))


def add_noise(ds: PointDataset, sigma: float = 0.05, seed: int | None = None) -> PointDataset:
    """Isotropic Gaussian jitter with std = sigma * RMS radius."""
    rng = np.random.default_rng(seed)
    s = sigma * _rms_radius(ds.X)
    return ds.with_points(ds.X + rng.normal(scale=s, size=ds.X.shape),
                          note=f"noise(sigma={sigma}, seed={seed})")


def add_occlusion(ds: PointDataset, fraction: float = 0.25, direction=None,
                  seed: int | None = None) -> PointDataset:
    """Remove the ``fraction`` of points that lie furthest along a direction
    (a sensor that cannot see one side). Direction random if not given."""
    if not 0 <= fraction < 1:
        raise ValueError("fraction must be in [0, 1)")
    rng = np.random.default_rng(seed)
    if direction is None:
        direction = rng.normal(size=ds.dim)
    u = np.asarray(direction, dtype=float)
    u = u / np.linalg.norm(u)
    proj = (ds.X - ds.X.mean(axis=0)) @ u
    keep = proj <= np.quantile(proj, 1 - fraction)
    return ds.subset(keep, note=f"occlusion(fraction={fraction}, direction={np.round(u, 3).tolist()}, seed={seed})")


def add_variable_density(ds: PointDataset, min_keep: float = 0.15, direction=None,
                         seed: int | None = None) -> PointDataset:
    """Subsample with keep-probability decreasing linearly from 1 to
    ``min_keep`` along a direction (non-uniform scan density)."""
    rng = np.random.default_rng(seed)
    if direction is None:
        direction = rng.normal(size=ds.dim)
    u = np.asarray(direction, dtype=float)
    u = u / np.linalg.norm(u)
    proj = (ds.X - ds.X.mean(axis=0)) @ u
    t = (proj - proj.min()) / max(np.ptp(proj), 1e-12)
    p_keep = 1.0 - (1.0 - min_keep) * t
    keep = rng.uniform(size=ds.n_points) < p_keep
    return ds.subset(keep, note=f"variable_density(min_keep={min_keep}, direction={np.round(u, 3).tolist()}, seed={seed})")


def add_outliers(ds: PointDataset, fraction: float = 0.05, margin: float = 0.2,
                 seed: int | None = None, label: int = -1) -> PointDataset:
    """Append uniform background points in the (enlarged) bounding box, label -1."""
    rng = np.random.default_rng(seed)
    lo, hi = ds.X.min(axis=0), ds.X.max(axis=0)
    pad = margin * (hi - lo)
    n = int(round(fraction * ds.n_points))
    O = rng.uniform(lo - pad, hi + pad, size=(n, ds.dim))
    out = PointDataset(np.vstack([ds.X, O]), np.concatenate([ds.y, np.full(n, label)]),
                       name=ds.name, latent=None, meta=dict(ds.meta))
    out.meta.setdefault("perturbations", []).append(f"outliers(fraction={fraction}, seed={seed})")
    return out


PERTURBATIONS = {
    "ruido": lambda ds, seed: add_noise(ds, sigma=0.06, seed=seed),
    "oclusion": lambda ds, seed: add_occlusion(ds, fraction=0.25, seed=seed),
    "densidad_variable": lambda ds, seed: add_variable_density(ds, min_keep=0.15, seed=seed),
}
