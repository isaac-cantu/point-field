"""Whole-object retrieval with invariant descriptors (post-mid-term pipeline).

Synthetic families (``pointfield.objects``) by default; ``modelnet10.py``
reuses :func:`run_protocol` on real meshes.

Protocol
--------
* Disjoint reference (gallery) and query objects, every object under a
  random similarity (rotation/reflection, scale, translation).
* Queries are perturbed (condition); the gallery is always clean.
* Standardisation (BlockScaler) and the multi-view metric are fitted ONLY on
  reference objects; eta is chosen by identity retrieval on 20% of them.
* Methods: CD_U, CD_U_metric, CD_W (density weights), M_U, M_U_metric, D2.
* Metrics: mAP and 1-NN; paired stratified bootstrap (3000) for
  CD_U_metric - M_U_metric and CD_W - CD_U.

    python benchmarks/objetos_invariantes.py --dim 3 --out benchmarks/results/objetos_3d.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pointfield.learning import (
    BlockScaler,
    CDDescriptor,
    MultiViewMetric,
    knn_density_weights,
    make_view,
    moment_block_slices,
    moment_descriptor,
    nn_accuracy,
    paired_bootstrap,
    per_query_ap,
)
from pointfield.learning.descriptors import d2_histogram
from pointfield.objects import make_object_dataset

N_VIEWS = 4


# ------------------------------------------------------------------ conditions
def _unit_dir(rng, d):
    u = rng.normal(size=d)
    return u / np.linalg.norm(u)


def perturb(X: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
    c = X.mean(0)
    r = np.sqrt(((X - c) ** 2).sum(1).mean())
    if condition == "clean":
        return X
    if condition == "noise":
        return X + 0.03 * r * rng.normal(size=X.shape)
    if condition == "density":                     # keep prob. decays along a direction
        t = (X - c) @ _unit_dir(rng, X.shape[1]) / r
        keep = rng.random(len(X)) < np.clip(0.15 + 0.85 * (t - t.min()) / np.ptp(t), 0, 1)
        return X[keep]
    if condition == "occlusion":                   # remove the 20% of points furthest along u
        t = (X - c) @ _unit_dir(rng, X.shape[1])
        return X[t < np.quantile(t, 0.8)]
    if condition == "outliers":                    # 5% uniform in the enlarged bounding box
        lo, hi = X.min(0), X.max(0)
        pad = 0.2 * (hi - lo)
        O = rng.uniform(lo - pad, hi + pad, size=(int(0.05 * len(X)), X.shape[1]))
        return np.vstack([X, O])
    raise ValueError(condition)


CONDITIONS = ("clean", "noise", "density", "occlusion", "outliers")


# ------------------------------------------------------------------ features
def features(clouds, dim):
    cd = CDDescriptor()
    out = {"CD_U": np.array([cd(x) for x in clouds]),
           "CD_W": np.array([cd(x, knn_density_weights(x)) for x in clouds]),
           "M_U": np.array([moment_descriptor(x) for x in clouds]),
           "D2": np.array([d2_histogram(x) for x in clouds])}
    return out


def run_protocol(ref, y_ref, qry, y_qry, dim, conditions=CONDITIONS, seed=0, n_boot=3000):
    rng = np.random.default_rng(seed)
    cd = CDDescriptor()
    slices = {"CD": cd.block_slices(dim), "M": moment_block_slices(dim)}
    Fr = features(ref, dim)
    sc_cd = BlockScaler(slices["CD"]).fit(Fr["CD_U"])
    sc_cdw = BlockScaler(slices["CD"]).fit(Fr["CD_W"])
    sc_m = BlockScaler(slices["M"]).fit(Fr["M_U"])
    views = [[make_view(x, rng) for _ in range(N_VIEWS)] for x in ref]
    Vcd = np.array([[sc_cd.transform(cd(v)) for v in vs] for vs in views])
    Vm = np.array([[sc_m.transform(moment_descriptor(v)) for v in vs] for vs in views])
    met_cd, met_m = MultiViewMetric(seed=seed).fit(Vcd), MultiViewMetric(seed=seed).fit(Vm)

    gallery = {"CD_U": sc_cd.transform(Fr["CD_U"]), "CD_W": sc_cdw.transform(Fr["CD_W"]),
               "M_U": sc_m.transform(Fr["M_U"]), "D2": Fr["D2"]}
    gallery["CD_U_metric"] = met_cd.transform(gallery["CD_U"])
    gallery["M_U_metric"] = met_m.transform(gallery["M_U"])

    rows = []
    for cond in conditions:
        Q = [perturb(x, cond, rng) for x in qry]
        Fq = features(Q, dim)
        query = {"CD_U": sc_cd.transform(Fq["CD_U"]), "CD_W": sc_cdw.transform(Fq["CD_W"]),
                 "M_U": sc_m.transform(Fq["M_U"]), "D2": Fq["D2"]}
        query["CD_U_metric"] = met_cd.transform(query["CD_U"])
        query["M_U_metric"] = met_m.transform(query["M_U"])
        ap = {m: per_query_ap(query[m], y_qry, gallery[m], y_ref) for m in query}
        for m in query:
            rows.append({"condition": cond, "method": m, "mAP": float(np.nanmean(ap[m])),
                         "nn": nn_accuracy(query[m], y_qry, gallery[m], y_ref)})
        for a, b in (("CD_U_metric", "M_U_metric"), ("CD_W", "CD_U")):
            bs = paired_bootstrap(ap[a], ap[b], strata=y_qry, n_boot=n_boot, seed=seed)
            rows.append({"condition": cond, "method": f"{a} - {b}", "mAP": bs["diff"],
                         "nn": float("nan"), "ci_low": bs["ci_low"], "ci_high": bs["ci_high"]})
    meta = {"eta_CD": met_cd.eta_, "eta_M": met_m.eta_}
    return rows, meta


def write_rows(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["condition", "method", "mAP", "nn", "ci_low", "ci_high"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def print_rows(rows):
    for r in rows:
        ci = f"  [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]" if "ci_low" in r else ""
        nn = "" if np.isnan(r["nn"]) else f"  1-NN {r['nn']:.3f}"
        print(f"{r['condition']:10s} {r['method']:26s} mAP {r['mAP']:+.3f}{nn}{ci}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--n-ref", type=int, default=20, help="reference objects per class")
    ap.add_argument("--n-query", type=int, default=10, help="query objects per class")
    ap.add_argument("--points", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--n-boot", type=int, default=3000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ref, y_ref, fam = make_object_dataset(a.dim, a.n_ref, a.points, seed=a.seed)
    qry, y_qry, _ = make_object_dataset(a.dim, a.n_query, a.points, seed=a.seed + 1)
    rows, meta = run_protocol(ref, y_ref, qry, y_qry, a.dim, seed=a.seed, n_boot=a.n_boot)
    print(f"families: {fam}  |  {meta}")
    print_rows(rows)
    write_rows(rows, a.out or f"benchmarks/results/objetos_{a.dim}d.csv")


if __name__ == "__main__":
    main()
