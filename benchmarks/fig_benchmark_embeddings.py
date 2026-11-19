"""Figures/tables for the embedding benchmark.
  figures/fig_grid_embeddings.png   input (rolled 3D) + 6 embeddings, seed 100
  results/tabla_embeddings_resumen.csv
"""

import os

import pandas as pd
from sklearn.manifold import trustworthiness

from benchmark_embeddings_2d import CASES, EMBEDDERS, K, embed, rolled_case
from style import INK, colors_for, plt

HERE = os.path.dirname(__file__)


def main(seed=100):
    df = pd.read_csv(os.path.join(HERE, "results", "embeddings_2d_eval.csv"))
    t = df.groupby(["case", "method"], sort=False)[["T", "C", "kNN_acc", "time_s"]].agg(["mean", "std"]).round(3)
    t.to_csv(os.path.join(HERE, "results", "tabla_embeddings_resumen.csv"))
    names = list(EMBEDDERS)
    fig = plt.figure(figsize=(2.1 * (len(names) + 1), 2.2 * len(CASES)))
    for i, case in enumerate(CASES):
        X, Z, y = rolled_case(case, seed)
        ax = fig.add_subplot(len(CASES), len(names) + 1, i * (len(names) + 1) + 1, projection="3d")
        ax.scatter(*X.T, s=1.5, c=colors_for(y)); ax.view_init(12, -70)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(f"{case}: entrada 3D\n(enrollada)", fontsize=8)
        for j, n in enumerate(names):
            E = embed(n, X)
            ax = fig.add_subplot(len(CASES), len(names) + 1, i * (len(names) + 1) + j + 2)
            ax.scatter(E[:, 0], E[:, 1], s=1.8, c=colors_for(y)); ax.set_xticks([]); ax.set_yticks([])
            m = df[(df.case == case) & (df.method == n)]
            ax.text(0.98, 0.02, f"T {trustworthiness(Z, E, n_neighbors=K):.2f}", transform=ax.transAxes,
                    ha="right", fontsize=7.5, fontweight="bold", color=INK)
            if i == 0:
                ax.set_title(n, fontsize=8.5)
    fig.suptitle("Embeddings 3D → 2D de escenarios enrollados · color = grupo de referencia (solo evaluación) · "
                 "T = trustworthiness respecto a las coordenadas 2D originales (k = 10) · semilla 100",
                 color=INK, fontsize=9.5)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "fig_grid_embeddings.png"))


if __name__ == "__main__":
    main()
