"""Bloque 03: point-field library — primitives, benchmark scenarios, perturbations
and parameter variations. Seed 0 everywhere (any seed reproduces).
Outputs: figures/fig_primitivas.png, figures/fig_libreria.png, results/escenarios.json
"""

import json
import os

import numpy as np

from style import INK, clean, colors_for, plt
from pointfield.core.dataset import SceneItem, compose
from pointfield.core.pose import Pose
from pointfield.datasets import DESCRIPTIONS, make
from pointfield.geometry.shapes2d import (Annulus, Circle, Crescent, Disk, Ellipse, Line, Polygon,
                                          Rectangle, Spiral, Star, Triangle)
from pointfield.geometry.shapes3d import Cone, Cube, Cylinder, Ellipsoid, Helix, Sphere, Torus
from pointfield.perturbations import add_noise, add_occlusion, add_variable_density

HERE = os.path.dirname(__file__)


def primitives():
    g2 = [("Circle", Circle()), ("Disk", Disk()), ("Annulus", Annulus(0.5, 1)), ("Ellipse", Ellipse(1, .5)),
          ("Ellipse (borde)", Ellipse(1, .5, filled=False)), ("Rectangle", Rectangle(2, 1)),
          ("Triangle", Triangle()), ("Star", Star()), ("Polygon no convexo", Polygon([[0, 0], [2, 0], [2, 2], [1, .8], [0, 2]])),
          ("Line", Line()), ("Crescent", Crescent()), ("Spiral (log)", Spiral(a=.3, b=.25, turns=2, kind="logarithmic"))]
    g3 = [("Sphere", Sphere()), ("Cube", Cube()), ("Torus", Torus()), ("Cylinder", Cylinder()),
          ("Cone", Cone()), ("Ellipsoid", Ellipsoid()), ("Helix", Helix(turns=3))]
    fig = plt.figure(figsize=(14, 6.2))
    for i, (n, g) in enumerate(g2):
        ax = fig.add_subplot(3, 7, i + 1)
        X = g.sample(500, seed=0)
        ax.scatter(X[:, 0], X[:, 1], s=2, c="#2a6f97"); ax.set_aspect("equal"); clean(ax)
        ax.set_title(f"{n}\ndim. intrínseca {g.intrinsic_dimension}", fontsize=7.5)
    for i, (n, g) in enumerate(g3):
        ax = fig.add_subplot(3, 7, 15 + i, projection="3d")
        X = g.sample(700, seed=0)
        ax.scatter(*X.T, s=1.5, c="#c8553d", depthshade=True)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.set_box_aspect((1, 1, 1))
        ax.set_title(n, fontsize=7.5)
    fig.suptitle("point-field: primitivas 2D (fila 1–2) y superficies 3D (fila 3) · muestreo uniforme en "
                 "longitud de arco / área · semilla 0", color=INK, fontsize=10)
    fig.savefig(os.path.join(HERE, "figures", "fig_primitivas.png"))


def scatter(ax, ds, title):
    ax.scatter(ds.X[:, 0], ds.X[:, 1], s=3, c=colors_for(ds.y))
    ax.set_aspect("equal", adjustable="datalim"); clean(ax); ax.set_title(title, fontsize=8)


def scenarios():
    names = ["compacto", "no_convexo", "entrelazado", "anidado", "espiral_3brazos", "espiral_log"]
    fig, axs = plt.subplots(3, 6, figsize=(15, 8))
    meta = {}
    for j, n in enumerate(names):
        ds = make(n, seed=0)
        short = {"compacto": "3 gaussianas casi en contacto", "no_convexo": "2 lunas (Crescent)",
                 "entrelazado": "2 brazos arquimedianos", "anidado": "2 anillos (Annulus)",
                 "espiral_3brazos": "reservado: 3 brazos", "espiral_log": "reservado: logarítmica"}[n]
        scatter(axs[0, j], ds, f"{n}\n{short}")
        meta[n] = ds.meta
    e, m = make("entrelazado", seed=0), make("no_convexo", seed=0)
    pert = [(add_noise(e, .06, seed=1), "entrelazado + ruido\nσ = 0.06·RMS"),
            (add_occlusion(e, .25, seed=1), "entrelazado + oclusión\n25 % en una dirección"),
            (add_variable_density(e, .15, seed=1), "entrelazado + densidad var.\np_keep 1 → 0.15"),
            (add_noise(m, .06, seed=1), "no_convexo + ruido\nσ = 0.06·RMS"),
            (add_occlusion(m, .25, seed=1), "no_convexo + oclusión\n25 %"),
            (add_variable_density(m, .15, seed=1), "no_convexo + densidad var.\np_keep 1 → 0.15")]
    for j, (ds, t) in enumerate(pert):
        scatter(axs[1, j], ds, t)

    def spiral(turns, thick):
        return compose([SceneItem(Spiral(turns=turns, phase=p, thickness=thick), 150) for p in (0, np.pi)], seed=0)

    def moons(th, dy):
        mo = dict(radius=1.0, thickness=th, angle_span=np.pi)
        return compose([SceneItem(Crescent(angle_start=0, **mo), 150),
                        SceneItem(Crescent(angle_start=np.pi, **mo), 150, Pose(np.array([1.0, dy])))], seed=0)
    var = [(spiral(1.0, .08), "Spiral turns = 1.0"), (spiral(2.5, .08), "Spiral turns = 2.5"),
           (moons(.1, .45), "Crescent thickness = 0.10"), (moons(.5, .2), "Crescent thickness = 0.50,\ndesplaz. y = 0.2"),
           (add_noise(make("anidado", seed=0), .03, seed=1), "anidado + ruido σ = 0.03"),
           (add_noise(make("anidado", seed=0), .15, seed=1), "anidado + ruido σ = 0.15")]
    for j, (ds, t) in enumerate(var):
        scatter(axs[2, j], ds, t)
    for r, lab in enumerate(["Escenarios\n(referencia conocida)", "Perturbaciones", "Variación de\nparámetros"]):
        axs[r, 0].set_ylabel(lab, fontsize=9, color=INK, fontweight="bold")
    fig.suptitle("Librería de generación (pointfield.datasets / perturbations) · color = grupo de referencia · semilla 0",
                 color=INK, fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "fig_libreria.png"))
    with open(os.path.join(HERE, "results", "escenarios.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)


if __name__ == "__main__":
    primitives()
    scenarios()
