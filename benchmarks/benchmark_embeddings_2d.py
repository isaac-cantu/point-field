"""Embedding benchmark: recover the intrinsic 2D layout of rolled scenarios.

Each 2D scenario is curled into a 3D swiss roll (pointfield.transforms.embed_roll),
so the original 2D coordinates are a known ground-truth intrinsic embedding.
Every method maps the 3D cloud to 2D. Metrics (labels/latent used only here):
  * T(k), C(k): trustworthiness / continuity of the embedding w.r.t. the latent
    2D coordinates (Venna & Kaski), k = 10.
  * kNN-acc: leave-one-out 5-NN accuracy of the reference groups in the
    embedding (are groups kept apart?).
All methods use the same neighbourhood size k = 10 where applicable.

Outputs: results/embeddings_2d_eval.csv
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, SpectralEmbedding, trustworthiness
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from common import EVAL_SEEDS, N_PER_GROUP
from pointfield.datasets import make
from pointfield.learning import CDLocalSpectral
from pointfield.transforms import embed_roll

warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__)
K = 10
CASES = ["compacto", "no_convexo", "entrelazado", "anidado"]


def _umap():
    import umap
    return umap.UMAP(n_neighbors=K, n_components=2, random_state=0)


EMBEDDERS = {
    "PCA": lambda: PCA(2),
    "Kernel PCA (RBF)": lambda: KernelPCA(2, kernel="rbf"),
    "Isomap": lambda: Isomap(n_neighbors=K, n_components=2),
    "Laplacian Eigenmaps": lambda: SpectralEmbedding(2, affinity="nearest_neighbors", n_neighbors=K,
                                                     random_state=0),
    "UMAP": _umap,
    "CD local-espectral": None,
}


def rolled_case(case, seed):
    ds = make(case, n_per_group=N_PER_GROUP, seed=seed)
    Z = ds.latent
    X = embed_roll(Z)
    X = X - X.mean(0)
    X = X / np.sqrt((X ** 2).sum(1).mean())
    return X, Z, ds.y


def embed(name, X):
    if name == "CD local-espectral":
        return CDLocalSpectral(n_neighbors=K, degree=1).embed(X, 2)
    return EMBEDDERS[name]().fit_transform(X)


def main():
    rows = []
    for case in CASES:
        for seed in EVAL_SEEDS:
            X, Z, y = rolled_case(case, seed)
            for name in EMBEDDERS:
                t0 = time.perf_counter()
                E = embed(name, X)
                dt = time.perf_counter() - t0
                rows.append({
                    "case": case, "seed": seed, "method": name, "time_s": dt,
                    "T": trustworthiness(Z, E, n_neighbors=K),
                    "C": trustworthiness(E, Z, n_neighbors=K),
                    "T_input": trustworthiness(X, E, n_neighbors=K),
                    "kNN_acc": cross_val_score(KNeighborsClassifier(5), E, y, cv=5).mean(),
                })
            print(case, seed, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "results", "embeddings_2d_eval.csv"), index=False)


if __name__ == "__main__":
    main()
