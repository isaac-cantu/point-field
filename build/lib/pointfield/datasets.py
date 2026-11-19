"""Benchmark scenarios built from point-field geometries.

Every scenario has a known reference partition (y) and is built only from
library primitives + poses, so it can be regenerated from (name, n, seed).
Four difficulty families required by the benchmark protocol:

  compacto      3 anisotropic Gaussian groups that nearly touch
  no_convexo    2 interleaved crescents ("two moons")
  entrelazado   2 interleaved Archimedean spiral arms
  anidado       2 concentric rings (disconnected, not linearly separable)

plus held-out scenarios that were NOT used while developing improvements:

  espiral_3brazos   3-arm Archimedean spiral
  espiral_log       2 interleaved logarithmic spirals
"""

from __future__ import annotations

import numpy as np

from pointfield.core.dataset import PointDataset, SceneItem, compose
from pointfield.core.pose import Pose
from pointfield.geometry.manifolds.gaussian_manifold import GaussianBlob
from pointfield.geometry.shapes2d import Annulus, Crescent, Spiral


def make_compacto(n_per_group: int = 150, seed: int | None = 0) -> PointDataset:
    items = [
        SceneItem(GaussianBlob(std=(0.55, 0.30), rotation=0.5), n_per_group, Pose(np.array([0.0, 0.0]))),
        SceneItem(GaussianBlob(std=(0.35, 0.35)), n_per_group, Pose(np.array([1.9, 0.6]))),
        SceneItem(GaussianBlob(std=(0.45, 0.25), rotation=-0.9), n_per_group, Pose(np.array([0.9, -1.5]))),
    ]
    return compose(items, name="compacto", seed=seed)


def make_no_convexo(n_per_group: int = 150, seed: int | None = 0, thickness: float = 0.25) -> PointDataset:
    moon = dict(radius=1.0, thickness=thickness, angle_span=np.pi)
    items = [
        SceneItem(Crescent(angle_start=0.0, **moon), n_per_group, Pose(np.array([0.0, 0.0]))),
        SceneItem(Crescent(angle_start=np.pi, **moon), n_per_group, Pose(np.array([1.0, 0.45]))),
    ]
    return compose(items, name="no_convexo", seed=seed)


def make_entrelazado(n_per_group: int = 150, seed: int | None = 0, n_arms: int = 2,
                     turns: float = 1.5, thickness: float = 0.08, kind: str = "archimedean") -> PointDataset:
    if kind == "archimedean":
        a, b = 0.5, 0.5
    else:
        a, b = 0.4, 0.22
    items = [
        SceneItem(Spiral(a=a, b=b, turns=turns, kind=kind, phase=2 * np.pi * k / n_arms, thickness=thickness),
                  n_per_group)
        for k in range(n_arms)
    ]
    name = "entrelazado" if (n_arms == 2 and kind == "archimedean") else (
        f"espiral_{n_arms}brazos" if kind == "archimedean" else "espiral_log")
    return compose(items, name=name, seed=seed)


def make_anidado(n_per_group: int = 150, seed: int | None = 0) -> PointDataset:
    items = [
        SceneItem(Annulus(0.30, 0.55), n_per_group),
        SceneItem(Annulus(1.25, 1.50), n_per_group),
    ]
    return compose(items, name="anidado", seed=seed)


def make_espiral_3brazos(n_per_group: int = 150, seed: int | None = 0) -> PointDataset:
    return make_entrelazado(n_per_group, seed, n_arms=3, turns=1.2, thickness=0.06)


def make_espiral_log(n_per_group: int = 150, seed: int | None = 0) -> PointDataset:
    return make_entrelazado(n_per_group, seed, n_arms=2, turns=2.0, thickness=0.05, kind="logarithmic")


SCENARIOS = {
    "compacto": make_compacto,
    "no_convexo": make_no_convexo,
    "entrelazado": make_entrelazado,
    "anidado": make_anidado,
}

HELD_OUT = {
    "espiral_3brazos": make_espiral_3brazos,
    "espiral_log": make_espiral_log,
}

DESCRIPTIONS = {
    "compacto": "3 grupos gaussianos anisotrópicos casi en contacto",
    "no_convexo": "2 medias lunas entrelazadas (Crescent)",
    "entrelazado": "2 brazos de espiral arquimediana",
    "anidado": "2 anillos concéntricos (Annulus), desconectados",
    "espiral_3brazos": "3 brazos de espiral (no usado en desarrollo)",
    "espiral_log": "2 espirales logarítmicas (no usado en desarrollo)",
}


def make(name: str, n_per_group: int = 150, seed: int | None = 0) -> PointDataset:
    table = {**SCENARIOS, **HELD_OUT}
    if name not in table:
        raise KeyError(f"unknown scenario {name!r}; options: {sorted(table)}")
    ds = table[name](n_per_group=n_per_group, seed=seed)
    ds.meta["scenario"] = name
    ds.latent = ds.X.copy()      # 2D coordinates are the ground-truth intrinsic coordinates
    return ds
