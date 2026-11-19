"""Labelled point clouds and scene composition.

A :class:`PointDataset` is what every benchmark consumes: the points, the
reference labels (reserved for evaluation), optional latent coordinates
(ground-truth intrinsic coordinates, also evaluation-only) and a metadata
dictionary with everything needed to regenerate it (geometries, parameters,
seed, perturbations applied).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from pointfield.core.geometry import Geometry
from pointfield.core.pose import Pose


@dataclass
class PointDataset:
    X: np.ndarray
    y: np.ndarray
    name: str = "dataset"
    latent: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.y = np.asarray(self.y, dtype=int)
        if len(self.X) != len(self.y):
            raise ValueError("X and y must have the same length")

    # ------------------------------------------------------------------ basics
    @property
    def n_points(self) -> int:
        return len(self.X)

    @property
    def dim(self) -> int:
        return self.X.shape[1]

    @property
    def n_groups(self) -> int:
        return int(len(np.unique(self.y[self.y >= 0])))

    def subset(self, mask: np.ndarray, note: str | None = None) -> "PointDataset":
        mask = np.asarray(mask)
        out = PointDataset(
            X=self.X[mask],
            y=self.y[mask],
            name=self.name,
            latent=None if self.latent is None else self.latent[mask],
            meta=copy.deepcopy(self.meta),
        )
        if note:
            out.meta.setdefault("perturbations", []).append(note)
        return out

    def with_points(self, X: np.ndarray, note: str | None = None) -> "PointDataset":
        out = PointDataset(
            X=X, y=self.y.copy(), name=self.name,
            latent=None if self.latent is None else self.latent.copy(),
            meta=copy.deepcopy(self.meta),
        )
        if note:
            out.meta.setdefault("perturbations", []).append(note)
        return out

    def normalized(self) -> "PointDataset":
        """Centre and divide by a single (isotropic) RMS radius.

        Isotropic on purpose: per-axis standardisation distorts angular
        structure (it breaks the phase extension used for spirals).
        """
        Xc = self.X - self.X.mean(axis=0)
        s = np.sqrt((Xc ** 2).sum(axis=1).mean())
        return self.with_points(Xc / s)

    def __repr__(self) -> str:
        return (f"PointDataset(name={self.name!r}, n={self.n_points}, dim={self.dim}, "
                f"groups={self.n_groups})")


@dataclass
class SceneItem:
    geometry: Geometry
    n_points: int
    pose: Pose = field(default_factory=Pose)
    label: int | None = None


def compose(items: list[SceneItem], name: str = "scene", seed: int | None = None) -> PointDataset:
    """Sample every item, place it with its pose and stack with labels."""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for i, item in enumerate(items):
        sub_seed = int(rng.integers(0, 2**31 - 1))
        P = item.geometry.sample(item.n_points, seed=sub_seed)
        if item.pose is not None:
            pose = item.pose
            if np.asarray(pose.translation).shape[0] != P.shape[1]:
                pose = Pose(np.zeros(P.shape[1]), pose.rotation if P.shape[1] == 2 else np.eye(P.shape[1]), pose.scale)
            P = pose.apply(P)
        Xs.append(P)
        ys.append(np.full(len(P), i if item.label is None else item.label))
    meta = {
        "seed": seed,
        "items": [
            {"geometry": it.geometry.describe(), "n_points": it.n_points,
             "translation": np.asarray(it.pose.translation).tolist(),
             "rotation": (float(it.pose.rotation) if np.isscalar(it.pose.rotation)
                          else np.asarray(it.pose.rotation).tolist()),
             "scale": it.pose.scale}
            for it in items
        ],
    }
    return PointDataset(np.vstack(Xs), np.concatenate(ys), name=name, meta=meta)
