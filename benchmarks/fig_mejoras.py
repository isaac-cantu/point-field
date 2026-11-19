"""Bloque 06: before/after of the CD stages on the SAME data, plus how the
Szegő phase works (search curve and the phase circle).
Data: entrelazado, evaluation seed 100 (and anidado, where the gate stays off).
Output: figures/fig_mejoras.png
"""

import os

import numpy as np
from sklearn.metrics import adjusted_rand_score

from common import load_case
from style import GOLD, INK, colors_for, plt
from pointfield.learning import CDGlobalClustering, CDLocalSpectral, CDSzegoSpectral, SzegoPhase

HERE = os.path.dirname(__file__)


def main(seed=100):
    fig, axs = plt.subplots(2, 5, figsize=(16, 6.8))
    for r, case in enumerate(["entrelazado", "anidado"]):
        ds = load_case(case, "base", seed)
        X, y, k = ds.X, ds.y, ds.n_groups
        stages = [("CD global (inicial)", CDGlobalClustering(k, 2)),
                  ("CD local-espectral (mejora 1)", CDLocalSpectral(k, 10, 1)),
                  ("CD + fase Szegő (mejora 2)", CDSzegoSpectral(k, 10, 1))]
        for j, (name, m) in enumerate(stages):
            lab = m.fit_predict(X)
            ax = axs[r, j]
            ax.scatter(X[:, 0], X[:, 1], s=4, c=colors_for(lab)); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            extra = ""
            if isinstance(m, CDSzegoSpectral):
                extra = f"\ncompuerta: ganancia {m.phase_.winding_gain_:.2f} → {'ON' if m.active_ else 'OFF'}"
            ax.set_title(f"{name}\nARI = {adjusted_rand_score(y, lab):.2f}{extra}", fontsize=8.5)
        ph = SzegoPhase(k).fit(X)
        sub = X
        ts = np.linspace(-3, 3, 121)
        sil = [ph._sil(sub, ph._c(X, t, ph.form_), ph.form_) for t in ts]
        ax = axs[r, 3]
        ax.plot(ts, sil, color=INK)
        ax.axvline(ph.turns_, color=GOLD, ls="--")
        ax.axhline(ph.baseline_silhouette_, color="#9aa5a4", ls=":")
        ax.set_xlabel("vueltas (c en unidades libres de escala)"); ax.set_ylabel("silhouette en (cos ψ, sin ψ)")
        ax.set_title(f"Búsqueda sin etiquetas de ψ = θ − c·g(r)\nóptimo: {ph.turns_:+.2f} vueltas, g = {ph.form_}", fontsize=8.5)
        ax.text(0.02, ph.baseline_silhouette_, " ángulo sin torsión (c = 0)", fontsize=7, color="#6b7b7a",
                va="bottom", transform=ax.get_yaxis_transform())
        F = ph.features(X)
        ax = axs[r, 4]
        jit = 1 + 0.06 * np.random.default_rng(0).normal(size=len(F))
        ax.scatter(F[:, 0] * jit, F[:, 1] * jit, s=4, c=colors_for(y)); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Fase óptima (cos ψ, sin ψ)\ncolor = grupo de referencia (solo evaluación)", fontsize=8.5)
        axs[r, 0].set_ylabel(f"{case}\nsemilla {seed}", fontsize=9, fontweight="bold", color=INK)
    fig.suptitle("Mejoras de CD sobre los mismos datos: inicial → local-espectral → fase Szegő con compuerta",
                 color=INK, fontsize=10.5)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "fig_mejoras.png"))


if __name__ == "__main__":
    main()
