"""Figures and tables for the 2D clustering benchmark (reads results/*.csv).

  figures/fig_grid_clustering.png      scikit-learn style grid (seed 100, base cases)
  figures/fig_grid_perturbaciones.png  same grid for the perturbed variants of 2 cases
  figures/fig_heatmap_ari.png          mean ARI over evaluation seeds, all cases x methods
  results/tabla_clustering_resumen.csv mean ± std (ARI, NMI) per case/variant/method
  results/tabla_oracle.csv             default vs label-selected (oracle) ARI
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from common import HELD_OUT_CASES, MAIN_CASES, METHODS, VARIANTS, load_case, timed_fit_predict
from style import INK, colors_for, plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
SHORT = {"K-Means": "K-Means", "DBSCAN": "DBSCAN", "OPTICS": "OPTICS", "Spectral (kNN)": "Spectral\n(kNN)",
         "CD global (inicial)": "CD global\n(inicial)", "CD local-espectral (mejora 1)": "CD local-espectral\n(mejora 1)",
         "CD + fase Szegő (mejora 2)": "CD + fase Szegő\n(mejora 2)"}
VSHORT = {"base": "base", "ruido": "+ ruido", "oclusion": "+ oclusión", "densidad_variable": "+ dens. var."}


def grid(cases, fname, title, seed=100, methods=None):
    methods = list(METHODS) if methods is None else methods
    fig, axs = plt.subplots(len(cases), len(methods), figsize=(2.05 * len(methods), 2.05 * len(cases)))
    for i, (case, variant) in enumerate(cases):
        ds = load_case(case, variant, seed)
        for j, m in enumerate(methods):
            model = METHODS[m][0](ds.n_groups)
            lab, dt = timed_fit_predict(model, ds.X)
            ax = axs[i, j]
            ax.scatter(ds.X[:, 0], ds.X[:, 1], s=2.2, c=colors_for(lab))
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal", adjustable="datalim")
            ari = adjusted_rand_score(ds.y, lab)
            ax.text(0.98, 0.02, f"ARI {ari:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                    fontweight="bold", color="#0f4c4a" if ari > 0.8 else ("#9a6b00" if ari > 0.4 else "#b03030"))
            ax.text(0.02, 0.02, f"{dt:.2f}s", transform=ax.transAxes, fontsize=6.5, color="#6b7b7a")
            if getattr(model, "active_", None) is not None:
                ax.text(0.02, 0.98, "fase ON" if model.active_ else "fase OFF", transform=ax.transAxes,
                        va="top", fontsize=6.5, color="#d9a21b" if model.active_ else "#9aa5a4", fontweight="bold")
            if i == 0:
                ax.set_title(SHORT[m], fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"{case}\n{VSHORT[variant]}", fontsize=8.5, color=INK, fontweight="bold")
    fig.suptitle(title, color=INK, fontsize=10.5, y=1.0)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", fname))


def heatmap(df):
    methods = list(METHODS)
    cases = MAIN_CASES + HELD_OUT_CASES
    idx = [(c, v) for c in cases for v in VARIANTS]
    M = np.array([[df[(df.case == c) & (df.variant == v) & (df.method == m)].ARI.mean() for m in methods] for c, v in idx])
    S = np.array([[df[(df.case == c) & (df.variant == v) & (df.method == m)].ARI.std() for m in methods] for c, v in idx])
    fig, ax = plt.subplots(figsize=(10.5, 10))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}\n±{S[i, j]:.2f}", ha="center", va="center", fontsize=6.6,
                    color="black" if 0.25 < M[i, j] < 0.85 else "white")
    ax.set_xticks(range(len(methods)), [SHORT[m] for m in methods], fontsize=8)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(idx)), [f"{'[reservado] ' if c in HELD_OUT_CASES else ''}{c} {VSHORT[v]}"
                                   for c, v in idx], fontsize=8)
    ax.axhline(len(MAIN_CASES) * len(VARIANTS) - 0.5, color=INK, lw=2)
    for k in range(1, len(cases)):
        ax.axhline(k * len(VARIANTS) - 0.5, color="white", lw=1)
    ax.axvline(3.5, color=INK, lw=1.5)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="ARI (media de 5 semillas de evaluación)")
    ax.set_title("ARI con configuración por defecto · semillas 100–104 · K-Means/Spectral/CD reciben k; "
                 "DBSCAN/OPTICS no", fontsize=9.5, pad=46)
    fig.savefig(os.path.join(HERE, "figures", "fig_heatmap_ari.png"))


def heatmap_compact(df):
    """Slide version: mean ARI only, larger text."""
    methods = list(METHODS)
    cases = MAIN_CASES + HELD_OUT_CASES
    idx = [(c, v) for c in cases for v in VARIANTS]
    M = np.array([[df[(df.case == c) & (df.variant == v) & (df.method == m)].ARI.mean() for m in methods] for c, v in idx])
    lab = ["K-Means", "DBSCAN", "OPTICS", "Spectral", "CD\ninicial", "CD\nmejora 1", "CD\nmejora 2"]
    fig, ax = plt.subplots(figsize=(7.2, 8.4))
    ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=10,
                    color="black" if 0.25 < M[i, j] < 0.85 else "white")
    ax.set_xticks(range(len(methods)), lab, fontsize=10.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(idx)), [(f"{c}{'*' if c in HELD_OUT_CASES else ''} base" if v == "base" else VSHORT[v]) for c, v in idx], fontsize=10)
    for k in range(1, len(cases)):
        ax.axhline(k * len(VARIANTS) - 0.5, color="white", lw=2)
    ax.axhline(len(MAIN_CASES) * len(VARIANTS) - 0.5, color=INK, lw=2.5)
    ax.axvline(3.5, color=INK, lw=2)
    ax.tick_params(length=0)
    fig.savefig(os.path.join(HERE, "figures", "fig_heatmap_ari_slide.png"))


def tables(df):
    g = df.groupby(["case", "variant", "method"], sort=False)
    t = g.agg(ARI_mean=("ARI", "mean"), ARI_std=("ARI", "std"), NMI_mean=("NMI", "mean"), NMI_std=("NMI", "std"),
              time_s=("time_s", "mean"), szego_active_frac=("szego_active", lambda s: s.dropna().astype(float).mean()
                                                            if s.notna().any() else np.nan)).reset_index()
    t.to_csv(os.path.join(RES, "tabla_clustering_resumen.csv"), index=False)
    # per-method aggregates
    df = df.assign(grupo=np.where(df.held_out, "reservado", "principal"),
                   cond=np.where(df.variant == "base", "base", "perturbado"))
    agg = df.groupby(["grupo", "cond", "method"]).ARI.agg(["mean", "std"]).unstack(["grupo", "cond"]).round(3)
    agg.to_csv(os.path.join(RES, "tabla_clustering_por_metodo.csv"))
    print(agg.to_string())
    if os.path.exists(os.path.join(RES, "clustering_2d_oracle.csv")):
        o = pd.read_csv(os.path.join(RES, "clustering_2d_oracle.csv"))
        o = o.groupby(["case", "variant", "method"]).ARI_oracle.mean().reset_index()
        d = df.groupby(["case", "variant", "method"]).ARI.mean().reset_index().rename(columns={"ARI": "ARI_default"})
        m = d.merge(o, on=["case", "variant", "method"], how="left")
        m.to_csv(os.path.join(RES, "tabla_oracle.csv"), index=False)


def winding_table(df):
    w = df[df.method == "CD + fase Szegő (mejora 2)"].groupby(["case", "variant"], sort=False).agg(
        ganancia_devanado=("winding_gain", "mean"), ganancia_std=("winding_gain", "std"),
        fase_activada=("szego_active", lambda s: s.astype(float).mean())).reset_index()
    w.to_csv(os.path.join(RES, "tabla_compuerta_szego.csv"), index=False)
    print(w.round(2).to_string())


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(RES, "clustering_2d_eval.csv"))
    tables(df)
    winding_table(df)
    heatmap(df)
    heatmap_compact(df)
    grid([(c, "base") for c in MAIN_CASES + HELD_OUT_CASES], "fig_grid_clustering.png",
         "Agrupamiento en 2D · casos base · semilla 100 · color = grupo asignado (gris = ruido)")
    grid([(c, v) for c in ("no_convexo", "entrelazado") for v in VARIANTS[1:]], "fig_grid_perturbaciones.png",
         "Agrupamiento en 2D · perturbaciones · semilla 100")
    grid([(c, "base") for c in MAIN_CASES], "fig_grid_inicial.png",
         "Evaluación inicial · casos base · semilla de evaluación 100",
         methods=["K-Means", "DBSCAN", "OPTICS", "Spectral (kNN)", "CD global (inicial)"])
