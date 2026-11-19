"""Comprueba que las cifras citadas en la presentación y en el documento
explicativo coinciden con los CSV generados por los benchmarks.

Uso:  python benchmarks/verificar_cifras.py
Sale con código 1 y lista las diferencias si alguna cifra no coincide
(tolerancia: el redondeo a 2 decimales usado en el texto).
"""

import os
import sys

import numpy as np
import pandas as pd

R = os.path.join(os.path.dirname(__file__), "results")
fails, checks = [], 0


def check(name, value, expected, tol=0.006):
    global checks
    checks += 1
    ok = abs(float(value) - expected) <= tol
    print(f"{'OK ' if ok else 'ERR'}  {name:70s} {float(value):9.3f}  (texto: {expected})")
    if not ok:
        fails.append(name)


# ---------------------------------------------------------------- Bloque 02
d = pd.read_csv(os.path.join(R, "cd_2d_degree_diagnostics.csv"))
ring = d[d.dataset.str.startswith("anillo")].set_index("degree")
check("anillo d=1 cond(M_d)", ring.loc[1, "cond_M"], 2.2, 0.05)
check("anillo d=2 P(centro)/mediana", ring.loc[2, "ratio_hueco"], 7.8, 0.05)
check("anillo d=2 P(lejano)/mediana", ring.loc[2, "ratio_lejano"], 188, 0.5)
check("anillo d=6 log10 cond(M_d) (2.5e7)", np.log10(ring.loc[6, "cond_M"]), np.log10(2.5e7), 0.02)
check("traza: error relativo máximo < 1e-3", d.trace_rel_error.max() < 1e-3, 1, 0)

# ---------------------------------------------------------- Bloques 05 y 07
c = pd.read_csv(os.path.join(R, "clustering_2d_eval.csv"))
assert sorted(c.seed.unique()) == [100, 101, 102, 103, 104], "semillas de evaluación"
m = c.groupby(["case", "variant", "method"]).ARI.mean()
check("compacto base · CD global", m["compacto", "base", "CD global (inicial)"], 0.96)
check("no_convexo base · CD global", m["no_convexo", "base", "CD global (inicial)"], 0.33)
check("entrelazado base · CD global", m["entrelazado", "base", "CD global (inicial)"], 0.06)
check("anidado base · CD global", m["anidado", "base", "CD global (inicial)"], 0.00)
check("entrelazado base · CD local (mejora 1)", m["entrelazado", "base", "CD local-espectral (mejora 1)"], 0.17)
check("entrelazado base · CD + Szegő (mejora 2)", m["entrelazado", "base", "CD + fase Szegő (mejora 2)"], 1.00)
check("entrelazado ruido · CD + Szegő", m["entrelazado", "ruido", "CD + fase Szegő (mejora 2)"], 0.42)
check("espiral_3brazos base · CD + Szegő", m["espiral_3brazos", "base", "CD + fase Szegő (mejora 2)"], 1.00)
check("espiral_log base · CD + Szegő", m["espiral_log", "base", "CD + fase Szegő (mejora 2)"], 0.92)
check("no_convexo ruido · CD + Szegő", m["no_convexo", "ruido", "CD + fase Szegő (mejora 2)"], 0.94)
c["grupo"] = np.where(c.held_out, "reservado", "principal")
c["cond"] = np.where(c.variant == "base", "base", "perturbado")
g = c.groupby(["method", "grupo", "cond"]).ARI.mean()
for meth, vals in {"K-Means": (0.31, 0.31, 0.04, 0.04), "DBSCAN": (0.49, 0.55, 0.18, 0.21),
                   "OPTICS": (0.33, 0.25, 0.27, 0.16), "Spectral (kNN)": (0.79, 0.71, 0.36, 0.19),
                   "CD global (inicial)": (0.34, 0.31, 0.04, 0.04),
                   "CD local-espectral (mejora 1)": (0.78, 0.72, 0.48, 0.22),
                   "CD + fase Szegő (mejora 2)": (0.99, 0.75, 0.96, 0.27)}.items():
    for (gr, co), v in zip([("principal", "base"), ("principal", "perturbado"), ("reservado", "base"),
                            ("reservado", "perturbado")], vals):
        check(f"media {meth} · {gr} {co}", g[meth, gr, co], v)
sz = c[c.method.str.contains("Szeg")]
act = sz.groupby(["case", "variant"]).szego_active.mean()
check("compuerta nunca activa en compacto", act["compacto"].max(), 0.0)
check("compuerta nunca activa en anidado", act["anidado"].max(), 0.0)
check("compuerta activa 2/5 en entrelazado + ruido", act["entrelazado", "ruido"], 0.4)

o = pd.read_csv(os.path.join(R, "tabla_oracle.csv")).set_index(["case", "variant", "method"])
check("oráculo DBSCAN entrelazado base", o.loc[("entrelazado", "base", "DBSCAN"), "ARI_oracle"], 0.94)
check("oráculo CD local entrelazado base", o.loc[("entrelazado", "base", "CD local-espectral (mejora 1)"), "ARI_oracle"], 0.64)
check("oráculo Spectral entrelazado base", o.loc[("entrelazado", "base", "Spectral (kNN)"), "ARI_oracle"], 0.26)

dev = pd.read_csv(os.path.join(R, "clustering_2d_dev.csv"))
assert sorted(dev.seed.unique()) == [0, 1, 2], "semillas de desarrollo"
dz = dev[dev.method.str.contains("Szeg") & ~dev.held_out]
check("dev: ganancia mínima espiral limpia (0.36)", dz[(dz.case == "entrelazado") & (dz.variant == "base")].winding_gain.min(), 0.36, 0.01)
check("dev: ganancia máxima anidado (≤ 0.13)", dz[dz.case == "anidado"].winding_gain.max(), 0.13, 0.01)

# -------------------------------------------------------------- embeddings
e = pd.read_csv(os.path.join(R, "embeddings_2d_eval.csv")).groupby("method")[["T", "C", "kNN_acc"]].mean()
for meth, (t, k) in {"PCA": (0.81, 0.80), "Isomap": (0.86, 0.88), "UMAP": (0.95, 0.92),
                     "Laplacian Eigenmaps": (0.86, 0.835), "CD local-espectral": (0.86, 0.83)}.items():
    check(f"embedding T · {meth}", e.loc[meth, "T"], t)
    check(f"embedding 5-NN · {meth}", e.loc[meth, "kNN_acc"], k)

# ----------------------------------------------------------------------- 3D
k3 = pd.read_csv(os.path.join(R, "clasificacion_3d.csv")).groupby(["method", "variant"]).accuracy_1nn_loo.mean()
check("3D CD d=4 300 rot · base", k3["CD + Log-Euclidiana (d=4, 300 rot.)", "base"], 0.99)
check("3D CD d=4 300 rot · oclusión", k3["CD + Log-Euclidiana (d=4, 300 rot.)", "oclusion"], 0.66)
check("3D CD d=4 30 rot · base", k3["CD + Log-Euclidiana (d=4, 30 rot.)", "base"], 0.89)
check("3D D2 · base", k3["D2 (Osada 2002)", "base"], 1.00)
check("3D D2 · oclusión", k3["D2 (Osada 2002)", "oclusion"], 0.96)
check("3D eigen PCA · base", k3["Eigen-rasgos PCA", "base"], 0.86)
c3 = pd.read_csv(os.path.join(R, "cd_3d_condicion.csv")).set_index(["objeto", "degree"])
check("toro d=4 log10 cond (1.1e17)", np.log10(c3.loc[("toro", 4), "cond_M"]), 17.03, 0.05)

print(f"\n{checks} comprobaciones, {len(fails)} diferencias")
sys.exit(1 if fails else 0)
