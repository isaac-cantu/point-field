"""Figuras explicativas (paso a paso) para la presentación y el documento.

Cada función genera UNA figura que muestra cómo funciona una pieza del
método o del protocolo, con números reales calculados aquí mismo:

  fig_paso_cd.png             cálculo de P_d en un ejemplo de 6 puntos (d = 1)
  fig_paso_grado.png          misma nube, cómo cambia P_d al subir d (1, 2, 3)
  fig_paso_perturbaciones.png cómo actúa cada perturbación sobre los puntos
  fig_casos_benchmark.png     los 24 casos del benchmark (= filas del mapa de calor)
  fig_paso_metricas.png       qué mide el ARI con particiones de ejemplo
  fig_paso_trust.png          qué mide trustworthiness (intrusos en la vecindad)
  fig_paso_knee.png           cómo se elige eps de DBSCAN sin etiquetas
  fig_paso_enrollado.png      experimento de embeddings: 2D -> 3D -> 2D
  fig_paso_afinidad.png       afinidad local CD vs. núcleo RBF en un punto
  fig_paso_szego.png          la fase de Szegő paso a paso en una espiral
  fig_paso_descriptor.png     descriptor 3D: rotar, promediar M_d, comparar
  fig_paso_corte3d.png        qué es un corte plano de P_d en 3D

Uso:  python benchmarks/fig_explicativas.py            (todas)
      python benchmarks/fig_explicativas.py szego knee (solo algunas)
"""

import os
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import Isomap, trustworthiness
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from common import knee_eps, load_case
from style import GOLD, INK, MUTED, TEAL, clean, colors_for, plt
from pointfield.core.dataset import PointDataset
from pointfield.core.pose import random_rotation
from pointfield.datasets import make
from pointfield.geometry.shapes2d import Annulus
from pointfield.geometry.shapes3d import Cone, Sphere, Torus
from pointfield.learning import ChristoffelDarboux, SzegoPhase, monomial_basis, monomial_exponents
from pointfield.learning.descriptors import cd_moment_descriptor, log_euclidean_distances
from pointfield.perturbations import add_noise, add_occlusion, add_variable_density
from pointfield.transforms import embed_roll

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figures")
RED = "#c8553d"


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("ok", name)


# ---------------------------------------------------------------------------
def paso_cd():
    """Ejemplo de 6 puntos, grado 1: v(x) = (1, x1, x2), M = B^T B / N, P(x) = v^T M^-1 v."""
    X = np.array([[-1.0, -0.5], [-0.5, 0.0], [0.0, 0.2], [0.5, 0.5], [1.0, 0.6], [0.2, -0.3]])
    terms = monomial_exponents(2, 1)
    B = monomial_basis(X, terms)
    M = B.T @ B / len(X)
    Minv = np.linalg.inv(M)
    P = np.einsum("ij,jk,ik->i", B, Minv, B)
    q_in, q_out = np.array([0.1, 0.1]), np.array([-0.8, 0.9])

    def Pq(q):
        v = np.array([1.0, q[0], q[1]])
        return float(v @ Minv @ v)

    fig = plt.figure(figsize=(16, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1.05, 1.05, 1.25], wspace=0.35)
    ax = fig.add_subplot(gs[0])
    ax.scatter(X[:, 0], X[:, 1], s=60, c="#2a6f97", zorder=3)
    for i, (a, b) in enumerate(X):
        ax.annotate(f"x{i + 1}", (a, b), textcoords="offset points", xytext=(6, 5), fontsize=9)
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal"); ax.grid(alpha=0.25)
    ax.set_title("① Nube: N = 6 puntos en ℝ²")

    ax = fig.add_subplot(gs[1]); ax.axis("off")
    rows = [["", "1", "x₁", "x₂"]] + [[f"x{i + 1}"] + [f"{v:+.2f}" for v in B[i]] for i in range(len(X))]
    t = ax.table(cellText=rows, loc="center", cellLoc="center")
    t.scale(1, 1.5); t.set_fontsize(10)
    ax.set_title("② Vector de monomios v₁(xᵢ) = (1, x₁, x₂)\n(una fila por punto: matriz B)")

    ax = fig.add_subplot(gs[2])
    ax.imshow(M, cmap="Blues", vmin=-0.2, vmax=1.1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=11, color="white" if M[i, j] > 0.7 else INK)
    ax.set_xticks(range(3), ["1", "x₁", "x₂"]); ax.set_yticks(range(3), ["1", "x₁", "x₂"])
    ax.set_title("③ Matriz de momentos M₁ = BᵀB / N\n(promedios: 1, medias, segundos momentos)")

    ax = fig.add_subplot(gs[3])
    g = np.linspace(-1.6, 1.6, 240)
    GX, GY = np.meshgrid(g, g)
    V = np.stack([np.ones_like(GX), GX, GY], -1)
    PG = np.einsum("...i,ij,...j->...", V, Minv, V)
    cs = ax.contourf(GX, GY, np.log10(PG), levels=20, cmap="magma_r")
    ax.contour(GX, GY, PG, levels=[2, 3, 6], colors=[GOLD, "white", "white"], linewidths=[1.6, 0.8, 0.8])
    ax.scatter(X[:, 0], X[:, 1], s=30, c="#e9fbf7", edgecolors=INK, linewidths=0.4, zorder=3)
    ax.scatter(*q_in, marker="o", s=70, c="#3ddc97", edgecolors="k", zorder=4)
    ax.scatter(*q_out, marker="X", s=90, c=RED, edgecolors="white", zorder=4)
    ax.annotate(f"P = {Pq(q_in):.2f}", q_in, xytext=(8, -14), textcoords="offset points", color="white", fontsize=10, fontweight="bold")
    ax.annotate(f"P = {Pq(q_out):.1f}", q_out, xytext=(8, 4), textcoords="offset points", color="white", fontsize=10, fontweight="bold")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("④ P₁(x) = v₁(x)ᵀ M₁⁻¹ v₁(x) en todo el plano\nclaro = típico · oscuro = atípico")
    fig.colorbar(cs, ax=ax, fraction=0.046, label="log₁₀ P₁(x)")
    fig.suptitle(f"Paso a paso con grado 1: en los datos P = {', '.join(f'{p:.2f}' for p in P)}  ·  promedio = {P.mean():.2f} = s(2,1) = 3 (identidad de traza)",
                 color=INK, fontsize=10.5, y=1.02)
    save(fig, "fig_paso_cd.png")
    return {"M": M.round(3).tolist(), "P_datos": P.round(3).tolist(), "P_in": round(Pq(q_in), 3), "P_out": round(Pq(q_out), 3)}


def paso_grado():
    X = PointDataset(Annulus(0.75, 1.0).sample(300, seed=0), np.zeros(300, int)).normalized().X
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.9))
    axs[0].scatter(X[:, 0], X[:, 1], s=4, c="#2a6f97"); axs[0].set_aspect("equal"); clean(axs[0])
    axs[0].set_title("Anillo, N = 300")
    g = np.linspace(-2, 2, 220)
    GX, GY = np.meshgrid(g, g)
    G = np.column_stack((GX.ravel(), GY.ravel()))
    names = {1: "1, x, y (3 monomios)", 2: "+ x², xy, y² (6)", 3: "+ x³, x²y, xy², y³ (10)"}
    for c, d in enumerate((1, 2, 3)):
        cd = ChristoffelDarboux(d, reg=1e-8).fit(X)
        P = cd.score(G).reshape(GX.shape)
        lvl = np.percentile(cd.score(X), 97)
        ax = axs[c + 1]
        ax.contourf(GX, GY, np.log10(P), levels=22, cmap="magma_r")
        ax.contour(GX, GY, P, levels=[lvl], colors=[GOLD], linewidths=1.6)
        ax.scatter(X[:, 0], X[:, 1], s=2, c="#e9fbf7")
        ax.set_aspect("equal"); clean(ax)
        ax.set_title(f"d = {d}: {names[d]}\ncond(M_d) = {cd.condition_number_:.1e}", fontsize=9)
    fig.suptitle("Con más monomios, el polinomio de Christoffel puede «doblarse» y dejar fuera el hueco · línea dorada = frontera del soporte estimado (percentil 97)",
                 color=INK, fontsize=10)
    save(fig, "fig_paso_grado.png")


def paso_perturbaciones():
    ds = make("no_convexo", seed=0)
    rng = np.random.default_rng(1)
    u = rng.normal(size=2); u /= np.linalg.norm(u)
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.8))
    axs[0].scatter(ds.X[:, 0], ds.X[:, 1], s=6, c=colors_for(ds.y)); axs[0].set_title("Base\nCrescent × 2 (no_convexo)")
    nz = add_noise(ds, 0.06, seed=1)
    ax = axs[1]
    for a, b in zip(ds.X[::6], nz.X[::6]):
        ax.annotate("", b, a, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.6))
    ax.scatter(nz.X[:, 0], nz.X[:, 1], s=6, c=colors_for(nz.y))
    ax.set_title("Ruido gaussiano\nx ← x + ε,  ε ~ N(0, (0.06·RMS)² I)\nflechas: desplazamiento de algunos puntos")
    oc = add_occlusion(ds, 0.25, seed=1)
    ax = axs[2]
    proj = (ds.X - ds.X.mean(0)) @ np.array(oc.meta["perturbations"][-1].split("direction=")[1].split("]")[0].strip("[").split(","), dtype=float)
    thr = np.quantile(proj, 0.75)
    ax.scatter(ds.X[proj > thr, 0], ds.X[proj > thr, 1], s=6, c="#d0d0d0")
    ax.scatter(oc.X[:, 0], oc.X[:, 1], s=6, c=colors_for(oc.y))
    d = np.array(oc.meta["perturbations"][-1].split("direction=")[1].split("]")[0].strip("[").split(","), dtype=float)
    c0 = ds.X.mean(0)
    ax.annotate("", c0 + 1.0 * d, c0, arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
    ax.text(*(c0 + 1.1 * d), "u", color=RED, fontsize=12, fontweight="bold")
    ax.set_title("Oclusión direccional\nse quita el 25 % con mayor (x − x̄)·u\ngris = lo que «el sensor no ve»")
    dv = add_variable_density(ds, 0.15, seed=1)
    ax = axs[3]
    d = np.array(dv.meta["perturbations"][-1].split("direction=")[1].split("]")[0].strip("[").split(","), dtype=float)
    pj = (ds.X - ds.X.mean(0)) @ d
    tt = (pj - pj.min()) / np.ptp(pj)
    sc = ax.scatter(ds.X[:, 0], ds.X[:, 1], s=14, c=1 - 0.85 * tt, cmap="Greys", vmin=0, vmax=1.3, alpha=0.35)
    ax.scatter(dv.X[:, 0], dv.X[:, 1], s=6, c=colors_for(dv.y))
    ax.annotate("", c0 + 1.0 * d, c0, arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
    ax.set_title("Densidad variable\ncada punto se conserva con p = 1 − 0.85·t\nt ∈ [0, 1] = posición a lo largo de u")
    for a in axs:
        a.set_aspect("equal", adjustable="datalim"); clean(a)
    save(fig, "fig_paso_perturbaciones.png")


def casos_benchmark():
    cases = ["compacto", "no_convexo", "entrelazado", "anidado", "espiral_3brazos", "espiral_log"]
    vs = [("base", "base"), ("ruido", "+ ruido"), ("oclusion", "+ oclusión"), ("densidad_variable", "+ dens. var.")]
    fig, axs = plt.subplots(4, 6, figsize=(15, 10.2))
    for j, c in enumerate(cases):
        for i, (v, vn) in enumerate(vs):
            ds = load_case(c, v, 100)
            ax = axs[i, j]
            ax.scatter(ds.X[:, 0], ds.X[:, 1], s=2.5, c=colors_for(ds.y))
            ax.set_aspect("equal", adjustable="datalim"); clean(ax)
            if i == 0:
                ax.set_title(c + (" *" if c.startswith("espiral_") else ""), fontsize=11)
            if j == 0:
                ax.set_ylabel(vn, fontsize=11, fontweight="bold", color=INK)
            ax.text(0.98, 0.02, f"N={ds.n_points}", transform=ax.transAxes, ha="right", fontsize=7.5, color=MUTED)
    fig.suptitle("Los 24 casos del benchmark (6 escenarios × 4 variantes) · cada uno es una fila del mapa de calor · semilla 100 · * reservado",
                 color=INK, fontsize=11)
    plt.tight_layout()
    save(fig, "fig_casos_benchmark.png")


def paso_metricas():
    ds = load_case("no_convexo", "base", 100)
    X, y = ds.X, ds.y
    rng = np.random.default_rng(0)
    parts = [("Partición perfecta", y),
             ("Misma partición, colores\nintercambiados", 1 - y),
             ("K-Means (corte recto)", KMeans(2, n_init=10, random_state=0).fit_predict(X)),
             ("Etiquetas al azar", rng.integers(0, 2, len(y))),
             ("Mitad de una luna\nmal asignada", np.where((y == 0) & (X[:, 0] < X[:, 0].mean()), 1, y))]
    fig, axs = plt.subplots(1, 5, figsize=(16, 3.7))
    for ax, (t, lab) in zip(axs, parts):
        ax.scatter(X[:, 0], X[:, 1], s=5, c=colors_for(lab)); ax.set_aspect("equal"); clean(ax)
        a = adjusted_rand_score(y, lab)
        ax.set_title(f"{t}\nARI = {a:.2f}", fontsize=10)
    fig.suptitle("ARI compara la partición obtenida con la de referencia por pares de puntos: 1 = idéntica (sin importar los nombres), ≈ 0 = como al azar",
                 color=INK, fontsize=10.5, y=1.0)
    plt.tight_layout()
    save(fig, "fig_paso_metricas.png")


def paso_trust():
    ds = make("no_convexo", seed=100)
    Z = ds.latent
    X = embed_roll(Z)
    E_good = Isomap(n_neighbors=10, n_components=2).fit_transform(X)
    from sklearn.decomposition import PCA
    E_bad = PCA(2).fit_transform(X)
    i = int(np.argmin(np.linalg.norm(Z - np.array([0.0, 1.0]), axis=1)))
    k = 10

    def nn(A):
        return NearestNeighbors(n_neighbors=k + 1).fit(A).kneighbors(A[i:i + 1], return_distance=False)[0][1:]
    true_nb = set(nn(Z))
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (A, t) in zip(axs, [(Z, "Coordenadas 2D reales (referencia)"), (E_good, "Embedding Isomap"), (E_bad, "Embedding PCA")]):
        nb = nn(A)
        ax.scatter(A[:, 0], A[:, 1], s=4, c="#cfd8d7")
        intr = [j for j in nb if j not in true_nb]
        keep = [j for j in nb if j in true_nb]
        ax.scatter(A[keep, 0], A[keep, 1], s=28, c=TEAL, label="vecino que también lo es en 2D")
        if intr:
            ax.scatter(A[intr, 0], A[intr, 1], s=36, c=RED, marker="X", label="intruso (no era vecino)")
        ax.scatter(A[i, 0], A[i, 1], s=110, c=GOLD, edgecolors="k", marker="*", zorder=5, label="punto i")
        T = trustworthiness(Z, A, n_neighbors=k)
        sub = "" if A is Z else f"\nintrusos de i: {len(intr)} de {k} · T global = {T:.2f}"
        ax.set_title(t + sub, fontsize=10); clean(ax)
    axs[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8.5, frameon=False)
    fig.suptitle("Trustworthiness: ¿los k = 10 vecinos de cada punto en el embedding eran vecinos en los datos reales? Cada intruso resta según qué tan lejos estaba",
                 color=INK, fontsize=10.5, y=1.04)
    save(fig, "fig_paso_trust.png")


def paso_knee():
    ds = load_case("anidado", "base", 100)
    X = ds.X
    d = np.sort(NearestNeighbors(n_neighbors=5).fit(X).kneighbors(X)[0][:, -1])
    eps = knee_eps(X, 5)
    x = np.linspace(0, 1, len(d)); yy = (d - d[0]) / (d[-1] - d[0])
    kpos = int(np.argmax(x - yy))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    ax = axs[0]
    ax.plot(np.arange(len(d)), d, color=INK)
    ax.plot([0, len(d) - 1], [d[0], d[-1]], ls=":", color=MUTED, label="cuerda")
    ax.scatter([kpos], [d[kpos]], s=80, c=GOLD, edgecolors="k", zorder=4, label=f"rodilla → eps = {eps:.3f}")
    ax.set_xlabel("puntos ordenados por su distancia al 5.º vecino"); ax.set_ylabel("distancia al 5.º vecino")
    ax.legend(frameon=False); ax.set_title("① Curva k-distancia (Ester et al., 1996)")
    from sklearn.cluster import DBSCAN
    lab = DBSCAN(eps=eps, min_samples=5).fit_predict(X)
    ax = axs[1]
    ax.scatter(X[:, 0], X[:, 1], s=6, c=colors_for(lab)); ax.set_aspect("equal"); clean(ax)
    ax.set_title(f"② DBSCAN con ese eps: {len(set(lab) - {-1})} grupos, ARI = {adjusted_rand_score(ds.y, lab):.2f}\n(gris = ruido; la rodilla no usa etiquetas)")
    save(fig, "fig_paso_knee.png")


def paso_enrollado():
    ds = make("entrelazado", seed=100)
    Z = ds.latent
    X = embed_roll(Z)
    E = Isomap(n_neighbors=10, n_components=2).fit_transform(X)
    fig = plt.figure(figsize=(14, 4.2))
    ax = fig.add_subplot(1, 3, 1); ax.scatter(Z[:, 0], Z[:, 1], s=4, c=colors_for(ds.y)); ax.set_aspect("equal"); clean(ax)
    ax.set_title("① Escenario 2D (coordenadas reales\n= respuesta correcta, solo para evaluar)")
    ax = fig.add_subplot(1, 3, 2, projection="3d"); ax.scatter(*X.T, s=2, c=colors_for(ds.y)); ax.view_init(14, -70)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title("② Se enrolla en 3D: (u, v) → (t cos t, v, t sin t)\nlo que recibe cada método")
    ax = fig.add_subplot(1, 3, 3); ax.scatter(E[:, 0], E[:, 1], s=4, c=colors_for(ds.y)); clean(ax)
    ax.set_title(f"③ Embedding 2D (aquí Isomap)\nT = {trustworthiness(Z, E, n_neighbors=10):.2f} frente a ①")
    save(fig, "fig_paso_enrollado.png")


def paso_afinidad():
    ds = load_case("entrelazado", "base", 100)
    X, y = ds.X, ds.y
    i = int(np.argmin(np.linalg.norm(X - np.array([0.62, -0.1]), axis=1)))
    k = 10
    nb = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X[i:i + 1], return_distance=False)[0]
    cd = ChristoffelDarboux(1, reg=1e-6).fit(X[nb])
    Pn = cd.score(X[nb]); med = np.median(Pn); mad = np.median(np.abs(Pn - med)) + 1e-12
    A = np.exp(-0.5 * ((Pn - med) / mad) ** 2)
    c = X[i]
    w = 0.9
    g = np.linspace(-w, w, 200)
    GX, GY = np.meshgrid(c[0] + g, c[1] + g)
    G = np.column_stack((GX.ravel(), GY.ravel()))
    PG = cd.score(G).reshape(GX.shape)
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2))
    ax = axs[0]
    ax.scatter(X[:, 0], X[:, 1], s=4, c=colors_for(y), alpha=0.5)
    ax.add_patch(plt.Rectangle((c[0] - w, c[1] - w), 2 * w, 2 * w, fill=False, ec=INK, lw=1.2))
    ax.scatter(*c, marker="*", s=160, c=GOLD, edgecolors="k", zorder=5)
    ax.set_aspect("equal"); clean(ax); ax.set_title("① Punto i y su zona (recuadro)")
    ax = axs[1]
    ax.contourf(GX, GY, np.log10(PG), levels=18, cmap="magma_r", alpha=0.85)
    ax.contour(GX, GY, PG, levels=[2, 3, 5], colors=["white"], linewidths=0.8)
    m = (np.abs(X[:, 0] - c[0]) < w) & (np.abs(X[:, 1] - c[1]) < w)
    ax.scatter(X[m, 0], X[m, 1], s=10, c=colors_for(y[m]))
    ax.scatter(X[nb, 0], X[nb, 1], s=50, facecolors="none", edgecolors="white", linewidths=1.2)
    ax.scatter(*c, marker="*", s=160, c=GOLD, edgecolors="k", zorder=5)
    ax.set_xlim(c[0] - w, c[0] + w); ax.set_ylim(c[1] - w, c[1] + w); ax.set_aspect("equal"); clean(ax)
    ax.set_title("② Modelo CD local (grado 1) ajustado\nsolo con los k = 10 vecinos (círculos)\nsus curvas siguen la dirección del brazo")
    ax = axs[2]
    ax.scatter(X[m, 0], X[m, 1], s=8, c="#d6dcdb")
    sc = ax.scatter(X[nb, 0], X[nb, 1], s=40 + 200 * A, c=A, cmap="viridis", vmin=0, vmax=1, edgecolors="k", linewidths=0.4)
    ax.scatter(*c, marker="*", s=160, c=GOLD, edgecolors="k", zorder=5)
    ax.set_xlim(c[0] - w, c[0] + w); ax.set_ylim(c[1] - w, c[1] + w); ax.set_aspect("equal"); clean(ax)
    fig.colorbar(sc, ax=ax, fraction=0.046, label="afinidad A_ij")
    ax.set_title("③ Peso de cada arista i–j del grafo\nA_ij = exp(−z²/2), z = (P_i(x_j) − med.)/MAD\n(tamaño y color = afinidad)")
    save(fig, "fig_paso_afinidad.png")


def paso_szego():
    ds = load_case("entrelazado", "base", 100)
    X, y = ds.X, ds.y
    r = np.linalg.norm(X, axis=1); th = np.arctan2(X[:, 1], X[:, 0])
    ph = SzegoPhase(2).fit(X)
    psi = SzegoPhase.phase(X, ph.c_, ph.form_)
    fig, axs = plt.subplots(1, 5, figsize=(21, 4.6), gridspec_kw={"width_ratios": [1, 1.15, 1.15, 1, 1.15], "wspace": 0.38})
    ax = axs[0]; ax.scatter(X[:, 0], X[:, 1], s=4, c=colors_for(y)); ax.set_aspect("equal"); clean(ax)
    ax.set_title("① Espiral en (x, y)\n(el color es la referencia,\nsolo para explicar)")
    ax = axs[1]; ax.scatter(r, np.degrees(th), s=4, c=colors_for(y))
    ax.set_xlabel("radio r"); ax.set_ylabel("ángulo θ (grados)"); ax.set_yticks([-180, 0, 180])
    ax.set_title("② En polares cada brazo es\nuna recta inclinada: θ ≈ r / b\n(el ángulo crece con el radio)")
    ax = axs[2]; ax.scatter(r, np.degrees(np.angle(np.exp(1j * psi))), s=4, c=colors_for(y))
    ax.set_xlabel("radio r"); ax.set_ylabel("ψ = θ − c·r (grados)"); ax.set_yticks([-180, 0, 180])
    ax.set_title(f"③ Se resta c·r  (c = {ph.c_:.2f}; {ph.turns_:.2f} vueltas)\nla inclinación desaparece y cada\nbrazo queda en su propia franja")
    ax = axs[3]
    F = np.column_stack((np.cos(psi), np.sin(psi)))
    jit = 1 + 0.05 * np.random.default_rng(0).normal(size=len(F))
    ax.scatter(F[:, 0] * jit, F[:, 1] * jit, s=4, c=colors_for(y)); ax.set_aspect("equal"); clean(ax)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, ec=MUTED, ls=":"))
    ax.set_title("④ Coordenadas (cos ψ, sin ψ):\ncada brazo se vuelve un\narco del círculo unitario")
    ax = axs[4]
    ts = np.linspace(-3, 3, 121)
    sil = [ph._sil(X, ph._c(X, t, ph.form_), ph.form_) for t in ts]
    ax.plot(ts, sil, color=INK); ax.axvline(ph.turns_, color=GOLD, ls="--")
    ax.axhline(ph.baseline_silhouette_, color=MUTED, ls=":")
    ax.annotate(f"ganancia = {ph.winding_gain_:.2f}", (ph.turns_, ph.silhouette_), xytext=(-110, -6), textcoords="offset points",
                fontsize=9, color=GOLD)
    ax.set_xlabel("vueltas probadas"); ax.set_ylabel("silhouette en ④")
    ax.set_title("⑤ Búsqueda sin etiquetas: c que\nmaximiza el silhouette de ④\n(línea punteada: c = 0, ángulo simple)")
    save(fig, "fig_paso_szego.png")
    return {"c": ph.c_, "turns": ph.turns_, "gain": ph.winding_gain_, "sil": ph.silhouette_, "sil0": ph.baseline_silhouette_}


def paso_descriptor():
    objs = [("esfera", Sphere()), ("toro", Torus()), ("cono", Cone())]
    fig = plt.figure(figsize=(18, 5))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1.5, 1.3], hspace=0.35, wspace=0.3)
    mats = []
    for j, (n, g) in enumerate(objs):
        X = g.sample(500, seed=j)
        R = random_rotation(3, seed=10 + j)
        Xr = X @ R.T
        ax = fig.add_subplot(gs[0, j], projection="3d"); ax.scatter(*Xr.T, s=1, c="#c8553d"); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(f"{n} (pose al azar)", fontsize=9)
        M = cd_moment_descriptor(Xr, degree=2, n_rotations=300, seed=0)
        mats.append(M)
        ax = fig.add_subplot(gs[1, j]); ax.imshow(np.log10(np.abs(M) + 1e-6), cmap="viridis"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("M_2 promedio (10×10, log|·|)", fontsize=8.5)
    ax = fig.add_subplot(gs[:, 3]); ax.axis("off")
    ax.text(0, 0.95, "Pasos", fontsize=11, fontweight="bold", color=INK, va="top")
    steps = ["1. centrar y escalar (RMS = 1)", "2. rotar R veces al azar (Haar)", "3. M_d de cada rotación y promediar\n    (operador de Reynolds)",
             "4. dividir entre la traza", "5. distancia Log-Euclidiana\n    ‖log M₁ − log M₂‖_F", "6. clasificar por vecino más cercano"]
    for k, s in enumerate(steps):
        ax.text(0, 0.82 - k * 0.15, s, fontsize=9, color=INK, va="top")
    D = log_euclidean_distances(mats)
    ax = fig.add_subplot(gs[:, 4])
    ax.imshow(D, cmap="magma_r")
    for a in range(3):
        for b in range(3):
            ax.text(b, a, f"{D[a, b]:.2f}", ha="center", va="center", color="white" if D[a, b] > D.max() / 2 else INK)
    ax.set_xticks(range(3), [o[0] for o in objs]); ax.set_yticks(range(3), [o[0] for o in objs])
    ax.set_title("Distancias entre descriptores\n(0 en la diagonal; mayor = más distintos)", fontsize=9)
    save(fig, "fig_paso_descriptor.png")


def paso_corte3d():
    X = PointDataset(Torus(1.0, 0.35).sample(1500, seed=0), np.zeros(1500, int)).normalized().X
    fig = plt.figure(figsize=(12, 4.3))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(*X.T, s=1, c="#2a6f97", alpha=0.5)
    gg = np.linspace(-2, 2, 2)
    PX, PY = np.meshgrid(gg, gg)
    ax.plot_surface(PX, PY, np.zeros_like(PX), color=GOLD, alpha=0.25)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.view_init(32, 35); ax.set_box_aspect((1, 1, 0.45))
    ax.set_title("① Plano z = 0 que atraviesa el toro")
    lim = 2.1
    u = np.linspace(-lim, lim, 260)
    U, V = np.meshgrid(u, u)
    plane = np.column_stack((U.ravel(), V.ravel(), np.zeros(U.size)))
    for c, d in enumerate((2, 4)):
        cd = ChristoffelDarboux(d, reg=1e-8).fit(X)
        ax = fig.add_subplot(1, 3, 2 + c)
        ax.contourf(U, V, np.log10(cd.score(plane)).reshape(U.shape), levels=26, cmap="magma_r")
        near = np.abs(X[:, 2]) < 0.04
        ax.scatter(X[near, 0], X[near, 1], s=3, c="#e9fbf7")
        ax.set_aspect("equal"); clean(ax)
        ax.set_title(f"{'②' if c == 0 else '③'} log₁₀ P_{d} evaluado en el plano\n(puntos blancos = datos cerca del plano)")
    fig.suptitle("Un corte muestra P_d sobre una rebanada del espacio 3D: claro = típico (soporte), oscuro = atípico",
                 color=INK, fontsize=10.5, y=1.02)
    save(fig, "fig_paso_corte3d.png")


def heatmap_miniaturas():
    """Mapa de calor del ARI con una miniatura de cada caso al inicio de su fila:
    así se ve qué representa cada fila. Dos figuras: principales y reservados."""
    import pandas as pd
    from matplotlib.colors import Normalize
    from common import METHODS
    df = pd.read_csv(os.path.join(HERE, "results", "clustering_2d_eval.csv"))
    methods = list(METHODS)
    lab = ["K-Means", "DBSCAN", "OPTICS", "Spectral", "CD\ninicial", "CD\nmejora 1", "CD\nmejora 2"]
    vs = [("base", "base"), ("ruido", "+ ruido"), ("oclusion", "+ oclusión"), ("densidad_variable", "+ dens. var.")]
    cmap = plt.get_cmap("RdYlGn"); norm = Normalize(0, 1)
    for tag, cases in (("principales", ["compacto", "no_convexo", "entrelazado", "anidado"]),
                       ("reservados", ["espiral_3brazos", "espiral_log"])):
        rows = [(c, v, vn) for c in cases for v, vn in vs]
        n = len(rows)
        fig = plt.figure(figsize=(9.2, 0.62 * n + 1.0))
        gs = fig.add_gridspec(n + 1, 9, width_ratios=[0.75, 1.6] + [1] * 7, height_ratios=[0.9] + [1] * n,
                              hspace=0.08, wspace=0.06)
        for j, l in enumerate(lab):
            ax = fig.add_subplot(gs[0, j + 2]); ax.axis("off")
            ax.text(0.5, 0.15, l, ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=INK)
        for i, (c, v, vn) in enumerate(rows):
            ds = load_case(c, v, 100)
            ax = fig.add_subplot(gs[i + 1, 0])
            ax.scatter(ds.X[:, 0], ds.X[:, 1], s=0.6, c=colors_for(ds.y)); ax.set_aspect("equal", adjustable="datalim")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#c9d1d0")
            ax = fig.add_subplot(gs[i + 1, 1]); ax.axis("off")
            ax.text(0.02, 0.5, (c + "\n" if v == "base" else "") + vn, va="center", fontsize=9,
                    fontweight="bold" if v == "base" else "normal", color=INK)
            for j, m in enumerate(methods):
                val = df[(df.case == c) & (df.variant == v) & (df.method == m)].ARI.mean()
                ax = fig.add_subplot(gs[i + 1, j + 2])
                ax.set_facecolor(cmap(norm(val))); ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.text(0.5, 0.5, f"{val:.2f}", ha="center", va="center", fontsize=9.5,
                        color="black" if 0.25 < val < 0.85 else "white", transform=ax.transAxes)
        save(fig, f"fig_heatmap_miniaturas_{tag}.png")


ALL = {"cd": paso_cd, "grado": paso_grado, "perturbaciones": paso_perturbaciones, "casos": casos_benchmark,
       "metricas": paso_metricas, "trust": paso_trust, "knee": paso_knee, "enrollado": paso_enrollado,
       "afinidad": paso_afinidad, "szego": paso_szego, "descriptor": paso_descriptor, "corte3d": paso_corte3d,
       "heatmap": heatmap_miniaturas}

if __name__ == "__main__":
    which = sys.argv[1:] or list(ALL)
    for w in which:
        out = ALL[w]()
        if out:
            print(w, out)
