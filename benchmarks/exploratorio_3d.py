"""Bloque 08 (exploratorio): primera implementación de CD en 3D.

(a) Representación del soporte: isosuperficie P_d = percentil 95 sobre una
    malla 44^3 (marching cubes) para un toro y un cono, grados 2/4/6.
(b) Descriptor de objeto completo: M_d promediada sobre 30 rotaciones
    (operador de Reynolds) + traza unitaria + distancia Log-Euclidiana.
    Clasificación 1-NN leave-one-out, 7 clases x 10 instancias, 400 puntos,
    pose aleatoria (rotación Haar, escala 0.5-2, traslación). Variantes:
    base, ruido (σ = 0.03·RMS), oclusión (20 %). Semillas de evaluación 100-104.
    Baselines sin entrenamiento: distribución D2 (Osada 2002) y
    eigen-rasgos PCA (linealidad/planaridad/esfericidad).

Outputs: figures/fig_3d_isosuperficies.png, figures/fig_3d_clases.png,
         results/clasificacion_3d.csv
"""

import os
import time

import numpy as np
import pandas as pd
from skimage.measure import marching_cubes

from style import INK, plt
from pointfield.core.dataset import PointDataset
from pointfield.core.pose import random_rotation
from pointfield.geometry.shapes3d import Cone, Cube, Cylinder, Ellipsoid, Sphere, Torus
from pointfield.learning import ChristoffelDarboux
from pointfield.learning.descriptors import (cd_moment_descriptor, d2_histogram, log_euclidean_distances,
                                             loo_1nn_accuracy, pairwise, pca_eigen_features)
from pointfield.perturbations import add_noise, add_occlusion

HERE = os.path.dirname(__file__)
SEEDS = [100, 101, 102, 103, 104]
CLASSES = {"esfera": Sphere(), "cubo": Cube(), "caja": Cube((1.0, 1.0, 0.55)), "toro": Torus(),
           "cilindro": Cylinder(), "cono": Cone(), "elipsoide": Ellipsoid()}


def isosurfaces():
    """Cloud, isosurface at d=2 and planar sections of log10 P_d for d=2,4,6.

    For d >= 4 both objects lie (piecewise) on algebraic surfaces of low degree
    (torus: quartic; cone: quadric x plane), so M_d becomes numerically singular
    and {P_d small} collapses onto the surface itself: the shell is thinner than
    any grid cell, hence the section view.
    """
    objs = [("toro", Torus(1.0, 0.35), 2), ("cono", Cone(), 1)]   # section plane: z=0 / y=0
    fig = plt.figure(figsize=(16, 7.4))
    rows = []
    lim = 1.9
    for r, (name, g, axis) in enumerate(objs):
        X = PointDataset(g.sample(1500, seed=0), np.zeros(1500, int)).normalized().X
        ax = fig.add_subplot(2, 5, 5 * r + 1, projection="3d")
        ax.scatter(*X.T, s=1, c="#2a6f97"); ax.set_title(f"{name}: nube (N = 1500)")
        # isosurface d = 2
        cd2 = ChristoffelDarboux(2, reg=1e-8).fit(X)
        n = 44
        g1 = np.linspace(-lim, lim, n)
        G = np.stack(np.meshgrid(g1, g1, g1, indexing="ij"), -1).reshape(-1, 3)
        P = np.log10(cd2.score(G)).reshape(n, n, n)
        verts, faces, _, _ = marching_cubes(P, np.log10(np.percentile(cd2.score(X), 95)), spacing=(g1[1] - g1[0],) * 3)
        verts -= lim
        ax = fig.add_subplot(2, 5, 5 * r + 2, projection="3d")
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color="#d9a21b", alpha=0.6, linewidth=0)
        ax.set_title("d = 2: isosuperficie P_d = p95")
        for a3 in fig.axes[-2:]:
            a3.set_xticks([]); a3.set_yticks([]); a3.set_zticks([]); a3.set_box_aspect((1, 1, 1))
            a3.set_xlim(-lim, lim); a3.set_ylim(-lim, lim); a3.set_zlim(-lim, lim); a3.view_init(28, 35)
        # sections
        u = np.linspace(-lim, lim, 300)
        U, V = np.meshgrid(u, u)
        plane = np.zeros((U.size, 3))
        other = [i for i in range(3) if i != axis]
        plane[:, other[0]], plane[:, other[1]] = U.ravel(), V.ravel()
        near = np.abs(X[:, axis]) < 0.04
        for c, d in enumerate((2, 4, 6)):
            cd = ChristoffelDarboux(d, reg=1e-8).fit(X)
            L = np.log10(cd.score(plane)).reshape(U.shape)
            ax = fig.add_subplot(2, 5, 5 * r + 3 + c)
            ax.contourf(U, V, L, levels=30, cmap="magma_r")
            ax.contour(U, V, cd.score(plane).reshape(U.shape), levels=[np.percentile(cd.score(X), 95)],
                       colors=["#d9a21b"], linewidths=1.2)
            ax.scatter(X[near][:, other[0]], X[near][:, other[1]], s=2, c="#e9fbf7")
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"corte {'xyz'[axis]} = 0 · d = {d}\ncond(M_d) = {cd.condition_number_:.1e}", fontsize=8.5)
            rows.append({"objeto": name, "degree": d, "basis_size": cd.basis_size_, "cond_M": cd.condition_number_})
    fig.suptitle("CD en 3D (exploratorio) · color: log10 P_d en un corte plano · línea dorada: P_d = percentil 95 de los datos"
                 " · semilla 0", color=INK, fontsize=10)
    fig.savefig(os.path.join(HERE, "figures", "fig_3d_isosuperficies.png"))
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "results", "cd_3d_condicion.csv"), index=False)
    return pd.DataFrame(rows)


def make_objects(seed, variant, n_inst=10, n_pts=400):
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for ci, (name, g) in enumerate(CLASSES.items()):
        for _ in range(n_inst):
            s = int(rng.integers(2**31 - 1))
            P = g.sample(n_pts, seed=s)
            P = rng.uniform(0.5, 2.0) * P @ random_rotation(3, seed=s + 1).T + rng.normal(scale=2, size=3)
            ds = PointDataset(P, np.zeros(len(P), int))
            if variant == "ruido":
                ds = add_noise(ds, 0.03, seed=s + 2)
            elif variant == "oclusion":
                ds = add_occlusion(ds, 0.20, seed=s + 2)
            Xs.append(ds.X); ys.append(ci)
    return Xs, np.array(ys)


def classify():
    rows = []
    for variant in ["base", "ruido", "oclusion"]:
        for seed in SEEDS:
            Xs, y = make_objects(seed, variant)
            res = {}
            # configuraciones: (grado, rotaciones). 30 rotaciones fijado a priori; 300 añadido tras
            # un diagnóstico en la semilla de desarrollo 7 (la varianza Monte Carlo del promedio
            # de Reynolds era el cuello de botella: 30 -> 0.94, 100 -> 0.99, 300 -> 1.00).
            for d, R in ((2, 30), (4, 30), (4, 300)):
                t0 = time.perf_counter()
                D = log_euclidean_distances([cd_moment_descriptor(X, degree=d, n_rotations=R, seed=seed) for X in Xs])
                res[f"CD + Log-Euclidiana (d={d}, {R} rot.)"] = (loo_1nn_accuracy(D, y), time.perf_counter() - t0)
            t0 = time.perf_counter()
            D = pairwise(np.array([d2_histogram(X, seed=seed) for X in Xs]), "l1")
            res["D2 (Osada 2002)"] = (loo_1nn_accuracy(D, y), time.perf_counter() - t0)
            t0 = time.perf_counter()
            D = pairwise(np.array([pca_eigen_features(X) for X in Xs]))
            res["Eigen-rasgos PCA"] = (loo_1nn_accuracy(D, y), time.perf_counter() - t0)
            for m, (acc, dt) in res.items():
                rows.append({"variant": variant, "seed": seed, "method": m, "accuracy_1nn_loo": acc, "time_s": dt})
            print(variant, seed, {k: round(v[0], 3) for k, v in res.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "results", "clasificacion_3d.csv"), index=False)
    return df


def class_gallery():
    fig = plt.figure(figsize=(14, 2.6))
    for i, (name, g) in enumerate(CLASSES.items()):
        ax = fig.add_subplot(1, 7, i + 1, projection="3d")
        X = g.sample(400, seed=i)
        ax.scatter(*X.T, s=1.5, c="#c8553d"); ax.set_title(name)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.set_box_aspect((1, 1, 1))
    fig.savefig(os.path.join(HERE, "figures", "fig_3d_clases.png"))


if __name__ == "__main__":
    import sys
    if "--solo-clasificacion" not in sys.argv:
        class_gallery()
        print(isosurfaces().to_string())
    df = classify()
    print(df.groupby(["variant", "method"]).accuracy_1nn_loo.agg(["mean", "std"]).round(3))
