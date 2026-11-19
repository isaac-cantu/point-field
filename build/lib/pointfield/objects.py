"""Synthetic object families for whole-object recognition experiments.

Each family is a class of shapes with intra-class shape variation (random
aspect ratios / proportions). Every generated object receives a random
similarity transform x -> a Q x + t with Q in O(d) (reflections included,
because the CD descriptor is O(d)-invariant), scale a in [0.5, 2] and a
random translation. Perturbations from :mod:`pointfield.perturbations` can
then be applied to the sampled cloud.

    X, y = make_object_dataset(dim=3, n_per_class=20, n_points=512, seed=0)
"""

from __future__ import annotations

import numpy as np

from pointfield.core.pose import random_rotation
from pointfield.geometry.shapes2d import (
    Annulus,
    Ellipse,
    Rectangle,
    RegularPolygon,
    Star,
    Triangle,
)
from pointfield.geometry.shapes3d import (
    Cone,
    Cube,
    Cylinder,
    Ellipsoid,
    Pyramid,
    Sphere,
    Torus,
)


def _u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


FAMILIES_2D = {
    "ellipse": lambda r: Ellipse(1.0, _u(r, 0.35, 0.8), filled=False),
    "rectangle": lambda r: Rectangle(1.0, _u(r, 0.35, 0.8), filled=False),
    "triangle": lambda r: Triangle(np.array([[0, 0], [1, 0], [_u(r, 0.2, 0.8), _u(r, 0.5, 1.0)]]),
                                   filled=False),
    "hexagon": lambda r: RegularPolygon(6, 1.0, filled=False),
    "star": lambda r: Star(5, 1.0, _u(r, 0.35, 0.55), filled=False),
    "annulus": lambda r: Annulus(_u(r, 0.4, 0.7), 1.0),
}

FAMILIES_3D = {
    "sphere": lambda r: Sphere(1.0),
    "ellipsoid": lambda r: Ellipsoid(1.0, _u(r, 0.5, 0.8), _u(r, 0.25, 0.45)),
    "cube": lambda r: Cube(1.0),
    "cylinder": lambda r: Cylinder(_u(r, 0.3, 0.6), _u(r, 1.2, 2.0)),
    "cone": lambda r: Cone(_u(r, 0.4, 0.8), _u(r, 1.0, 1.8)),
    "torus": lambda r: Torus(1.0, _u(r, 0.2, 0.45)),
    "pyramid": lambda r: Pyramid(1.0, _u(r, 0.6, 1.4)),
}


def random_similarity(X: np.ndarray, rng: np.random.Generator, reflect: bool = True,
                      scale=(0.5, 2.0), shift: float = 5.0) -> np.ndarray:
    d = X.shape[1]
    Q = random_rotation(d, seed=int(rng.integers(2 ** 31)))
    if reflect and rng.random() < 0.5:
        Q[:, 0] = -Q[:, 0]
    a = rng.uniform(*scale)
    t = rng.uniform(-shift, shift, size=d)
    return a * X @ Q.T + t


def make_object(family: str, dim: int, n_points: int, rng: np.random.Generator,
                transform: bool = True) -> np.ndarray:
    fam = FAMILIES_2D if dim == 2 else FAMILIES_3D
    geom = fam[family](rng)
    X = geom.sample(n_points, seed=int(rng.integers(2 ** 31)))
    return random_similarity(X, rng) if transform else X


def make_object_dataset(dim: int = 3, n_per_class: int = 20, n_points: int = 512,
                        families=None, seed: int = 0, transform: bool = True):
    """List of clouds and integer labels; ``families`` defaults to all."""
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")
    fam = list(families or (FAMILIES_2D if dim == 2 else FAMILIES_3D))
    rng = np.random.default_rng(seed)
    clouds, labels = [], []
    for k, name in enumerate(fam):
        for _ in range(n_per_class):
            clouds.append(make_object(name, dim, n_points, rng, transform))
            labels.append(k)
    return clouds, np.array(labels), fam
