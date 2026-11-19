"""Clustering benchmark in 2D (scikit-learn "comparing clustering algorithms" style).

Usage:
    python benchmarks/benchmark_clustering_2d.py            # evaluation seeds 100-104 (+ oracle)
    python benchmarks/benchmark_clustering_2d.py --dev      # development seeds 0-2

Outputs (benchmarks/results/):
    clustering_2d_{eval|dev}.csv      one row per (case, variant, seed, method)
    clustering_2d_oracle.csv          label-selected upper bound (backup only)
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from common import (DEV_SEEDS, EVAL_SEEDS, HELD_OUT_CASES, MAIN_CASES, METHODS, ORACLE_GRID,
                    VARIANTS, load_case, oracle_factory, timed_fit_predict)

HERE = os.path.dirname(__file__)


def run(seeds, tag, with_oracle):
    rows, orows = [], []
    for case in MAIN_CASES + HELD_OUT_CASES:
        for variant in VARIANTS:
            for seed in seeds:
                ds = load_case(case, variant, seed)
                k = ds.n_groups
                for mname, (factory, needs_k, family, cfg) in METHODS.items():
                    model = factory(k)
                    lab, dt = timed_fit_predict(model, ds.X)
                    rows.append({
                        "case": case, "held_out": case in HELD_OUT_CASES, "variant": variant,
                        "seed": seed, "method": mname, "family": family, "receives_k": needs_k,
                        "n_points": ds.n_points,
                        "ARI": adjusted_rand_score(ds.y, lab),
                        "NMI": normalized_mutual_info_score(ds.y, lab),
                        "n_found": len(set(lab) - {-1}), "noise_frac": float((lab == -1).mean()),
                        "time_s": dt,
                        "szego_active": getattr(model, "active_", None),
                        "winding_gain": getattr(getattr(model, "phase_", None), "winding_gain_", None),
                        "dbscan_eps": getattr(model, "eps_", None),
                    })
                if with_oracle:
                    for mname, grid in ORACLE_GRID.items():
                        scored = [(adjusted_rand_score(ds.y, oracle_factory(mname, k, p).fit_predict(ds.X)), p)
                                  for p in grid]
                        best = max(scored, key=lambda t: t[0])
                        orows.append({"case": case, "variant": variant, "seed": seed, "method": mname,
                                      "ARI_oracle": best[0], "best_params": str(best[1])})
                print(f"{tag} {case:16s} {variant:18s} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "results", f"clustering_2d_{tag}.csv"), index=False)
    if with_oracle:
        pd.DataFrame(orows).to_csv(os.path.join(HERE, "results", "clustering_2d_oracle.csv"), index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--no-oracle", action="store_true")
    a = ap.parse_args()
    if a.dev:
        run(DEV_SEEDS, "dev", with_oracle=False)
    else:
        run(EVAL_SEEDS, "eval", with_oracle=not a.no_oracle)
