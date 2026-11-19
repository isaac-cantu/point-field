"""Shared configuration for every benchmark script (methods, seeds, cases).

Protocol (applies to all scripts):
  * Every method receives exactly the same realisation of the data
    (same scenario, same seed, same perturbation seed), isotropically
    normalised (centred, single RMS scale factor).
  * Reference labels are used ONLY to compute ARI/NMI. The single exception is
    the "oracle" grid (``oracle_grid``), which picks each method's best
    hyperparameter with the labels and is reported only as an upper bound.
  * Default hyperparameters were fixed a priori (library defaults or the
    configuration of the scikit-learn clustering comparison) or by a
    label-free heuristic (DBSCAN eps from the k-distance knee). The Szegő gate
    threshold was fixed on DEVELOPMENT seeds 0-2; all reported results use
    EVALUATION seeds 100-104.
  * Methods marked needs_k receive the true number of groups; DBSCAN and
    OPTICS do not.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.cluster import DBSCAN, OPTICS, KMeans, SpectralClustering
from sklearn.neighbors import NearestNeighbors

from pointfield.datasets import HELD_OUT, SCENARIOS, make
from pointfield.learning import CDGlobalClustering, CDLocalSpectral, CDSzegoSpectral
from pointfield.perturbations import PERTURBATIONS

warnings.filterwarnings("ignore")

DEV_SEEDS = [0, 1, 2]
EVAL_SEEDS = [100, 101, 102, 103, 104]
N_PER_GROUP = 150
VARIANTS = ["base", "ruido", "oclusion", "densidad_variable"]
MAIN_CASES = list(SCENARIOS)
HELD_OUT_CASES = list(HELD_OUT)


def load_case(name: str, variant: str, seed: int):
    ds = make(name, n_per_group=N_PER_GROUP, seed=seed)
    if variant != "base":
        ds = PERTURBATIONS[variant](ds, seed + 1000)
    return ds.normalized()


def knee_eps(X: np.ndarray, min_samples: int) -> float:
    """Label-free DBSCAN eps: knee of the sorted k-distance curve
    (Ester et al., 1996), located as the point furthest from the chord."""
    d = NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)[0][:, -1]
    d = np.sort(d)
    x = np.linspace(0, 1, len(d))
    y = (d - d[0]) / max(d[-1] - d[0], 1e-12)
    return float(d[np.argmax(x - y)])


class DBSCANKnee:
    def __init__(self, min_samples: int = 5):
        self.min_samples = min_samples

    def fit_predict(self, X):
        self.eps_ = knee_eps(X, self.min_samples)
        return DBSCAN(eps=self.eps_, min_samples=self.min_samples).fit_predict(X)


# name -> (factory(k), needs_k, family, default config text)
METHODS = {
    "K-Means": (lambda k: KMeans(k, n_init=10, random_state=0), True, "referencia",
                "n_init=10"),
    "DBSCAN": (lambda k: DBSCANKnee(min_samples=5), False, "referencia",
               "min_samples=5, eps = rodilla k-distancia"),
    "OPTICS": (lambda k: OPTICS(min_samples=7, xi=0.05, min_cluster_size=0.1), False, "referencia",
               "min_samples=7, xi=0.05, min_cluster_size=0.1"),
    "Spectral (kNN)": (lambda k: SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=10,
                                                     random_state=0), True, "referencia",
                       "afinidad kNN, n_neighbors=10"),
    "CD global (inicial)": (lambda k: CDGlobalClustering(k, degree=2), True, "CD",
                            "grado 2, init k-means, reasignación argmin log P_d"),
    "CD local-espectral (mejora 1)": (lambda k: CDLocalSpectral(k, n_neighbors=10, degree=1), True, "CD",
                                      "k=10, grado local 1"),
    "CD + fase Szegő (mejora 2)": (lambda k: CDSzegoSpectral(k, n_neighbors=10, degree=1, gate=0.20), True,
                                   "CD", "k=10, grado 1, compuerta ganancia de devanado >= 0.20"),
}

# label-using upper bound (backup slides only)
ORACLE_GRID = {
    "DBSCAN": [{"eps": e, "min_samples": 5} for e in np.linspace(0.03, 0.6, 20)],
    "OPTICS": [{"min_samples": m} for m in (5, 7, 10, 15, 20)],
    "Spectral (kNN)": [{"n_neighbors": k} for k in (5, 7, 10, 15, 20, 30)],
    "CD global (inicial)": [{"degree": d} for d in (1, 2, 3, 4)],
    "CD local-espectral (mejora 1)": [{"n_neighbors": k} for k in (5, 7, 10, 15, 20, 30)],
}


def oracle_factory(method: str, k: int, params: dict):
    if method == "DBSCAN":
        return DBSCAN(**params)
    if method == "OPTICS":
        return OPTICS(xi=0.05, min_cluster_size=0.1, **params)
    if method == "Spectral (kNN)":
        return SpectralClustering(k, affinity="nearest_neighbors", random_state=0, **params)
    if method == "CD global (inicial)":
        return CDGlobalClustering(k, **params)
    if method == "CD local-espectral (mejora 1)":
        return CDLocalSpectral(k, degree=1, **params)
    raise KeyError(method)


def timed_fit_predict(model, X):
    t0 = time.perf_counter()
    lab = model.fit_predict(X)
    return np.asarray(lab), time.perf_counter() - t0
