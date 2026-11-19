"""Bloque 02: P_d(x) on 2D clouds, level sets for several degrees.

Data: anillo = Annulus(0.75, 1.0), N=300; lunas = no_convexo (2x150); seed 0;
isotropic normalisation; Tikhonov reg 1e-8 (relative). Level set drawn at the
97th percentile of P_d over the data (the "support" estimate).
Outputs: figures/fig_cd_2d.png, results/cd_2d_degree_diagnostics.csv
"""

import os

import numpy as np
import pandas as pd

from style import GOLD, INK, clean, plt
from pointfield.core.dataset import PointDataset
from pointfield.datasets import make
from pointfield.geometry.shapes2d import Annulus
from pointfield.learning import ChristoffelDarboux

HERE = os.path.dirname(__file__)
SEED = 0


def anillo():
    X = Annulus(0.75, 1.0).sample(300, seed=SEED)
    return PointDataset(X, np.zeros(300, int), name="anillo").normalized()


def panel(ax, X, d, outliers=None, lim=2.2):
    cd = ChristoffelDarboux(d, reg=1e-8).fit(X)
    g = np.linspace(-lim, lim, 260)
    GX, GY = np.meshgrid(g, g)
    P = cd.score(np.column_stack((GX.ravel(), GY.ravel()))).reshape(GX.shape)
    level = np.percentile(cd.score(X), 97)
    ax.contourf(GX, GY, np.log10(P), levels=24, cmap="magma_r", alpha=0.95)
    ax.contour(GX, GY, P, levels=[level], colors=[GOLD], linewidths=1.6)
    ax.scatter(X[:, 0], X[:, 1], s=3, c="#e9fbf7", edgecolors="none")
    if outliers is not None:
        ax.scatter(outliers[:, 0], outliers[:, 1], marker="X", s=40, c="#e03b3b", edgecolors="white", linewidths=0.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_title(f"d = {d}    cond(M_d) = {cd.condition_number_:.1e}", fontsize=8.5)
    return clean(ax), cd


def main():
    ring = anillo().X
    moons = make("no_convexo", seed=SEED).normalized().X
    out_ring = np.array([[0.0, 0.0], [1.7, 1.7], [-1.6, 0.2]])
    degrees = [1, 2, 4, 6]
    fig, axs = plt.subplots(2, 5, figsize=(15, 6.8), gridspec_kw={"hspace": 0.28, "wspace": 0.06})
    rows = []
    for r, (name, X, outl) in enumerate([("anillo (con hueco)", ring, out_ring), ("lunas (no convexo)", moons, None)]):
        axs[r, 0].scatter(X[:, 0], X[:, 1], s=4, c="#2a6f97")
        axs[r, 0].set_xlim(-2.2, 2.2); axs[r, 0].set_ylim(-2.2, 2.2)
        axs[r, 0].set_aspect("equal"); clean(axs[r, 0])
        axs[r, 0].set_title(f"Nube original: {name}\nN = {len(X)}")
        if outl is not None:
            axs[r, 0].scatter(outl[:, 0], outl[:, 1], marker="X", s=40, c="#e03b3b")
        for c, d in enumerate(degrees):
            _, cd = panel(axs[r, c + 1], X, d, outl)
            Ptr = cd.score(X)
            row = {"dataset": name, "degree": d, "basis_size": cd.basis_size_,
                   "N_per_param": len(X) / cd.basis_size_, "cond_M": cd.condition_number_,
                   "trace_rel_error": cd.trace_check(X), "P_median_data": float(np.median(Ptr))}
            if outl is not None:
                Po = cd.score(outl)
                row.update({"ratio_hueco": Po[0] / np.median(Ptr), "ratio_lejano": Po[1] / np.median(Ptr),
                            "hueco_fuera_soporte": bool(Po[0] > np.percentile(Ptr, 97))})
            rows.append(row)
    fig.suptitle("Función de Christoffel–Darboux  P_d(x) = v_d(x)ᵀ M_d⁻¹ v_d(x)   ·   color: log₁₀ P_d   ·   "
                 "línea dorada: conjunto de nivel P_d = percentil 97 de los datos", fontsize=10, color=INK, y=1.01)
    fig.savefig(os.path.join(HERE, "figures", "fig_cd_2d.png"))
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "results", "cd_2d_degree_diagnostics.csv"), index=False)
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    main()
