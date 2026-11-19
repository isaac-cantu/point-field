"""ModelNet10 retrieval with the current invariant CD pipeline (NEW protocol, not A8 replication).

The dataset is not downloaded by this script. Unpack ModelNet10 locally
(``<root>/<class>/{train,test}/*.off``) and pass ``--root``; ``data/`` is
git-ignored.

Protocol (fixed before looking at test results):
  * read every .off (handles the "OFF490 518 0" header quirk), drop exact
    duplicates by SHA-256 of the file;
  * per class, from the official train split draw 400 reference objects in
    total (stratified, seed 20261025), 100 development queries, and from the
    official test split 200 test queries (stratified);
  * 512 area-uniform surface points per mesh, random similarity incl.
    reflection;
  * same methods / conditions / bootstrap as ``objetos_invariantes.py``.

Thesis numbers (S2, clean, test): CD_U_metric mAP 0.402 vs M_U_metric
0.398 (tie, CI includes 0); density bias: CD_U_metric - M_U_metric +0.017 / +0.020 with
CI > 0; 5% outliers collapse every unfiltered method (~0.10).

    python benchmarks/modelnet10.py --root data/ModelNet10 --split test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from objetos_invariantes import CONDITIONS, print_rows, run_protocol, write_rows

from pointfield.geometry.mesh import TriangleMesh, file_sha256
from pointfield.objects import random_similarity

SEED = 20261025


def index(root: Path, split: str):
    """Sorted, de-duplicated list of (path, class) for one official split."""
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    seen, items = set(), []
    for c in classes:
        for f in sorted((root / c / split).glob("*.off")):
            h = file_sha256(f)
            if h in seen:
                continue
            seen.add(h)
            items.append((f, c))
    return items, classes


def stratified(items, classes, n_total, rng, exclude=()):
    ex = set(exclude)
    per = n_total // len(classes)
    out = []
    for c in classes:
        pool = [it for it in items if it[1] == c and it[0] not in ex]
        k = min(per, len(pool))
        out += [pool[i] for i in rng.choice(len(pool), size=k, replace=False)]
    return out


def load(items, classes, n_points, rng):
    clouds, y = [], []
    for f, c in items:
        X = TriangleMesh.from_off(f).sample(n_points, seed=int(rng.integers(2 ** 31)))
        clouds.append(random_similarity(X, rng))
        y.append(classes.index(c))
    return clouds, np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--split", choices=("dev", "test"), default="dev",
                    help="dev = 100 development queries from train; test = 200 official-test queries")
    ap.add_argument("--n-ref", type=int, default=400)
    ap.add_argument("--n-dev", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--points", type=int, default=512)
    ap.add_argument("--n-boot", type=int, default=3000)
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not a.root.is_dir():
        raise SystemExit(f"{a.root} not found: unpack ModelNet10 there first")

    rng = np.random.default_rng(SEED)
    train, classes = index(a.root, "train")
    ref = stratified(train, classes, a.n_ref, rng)
    if a.split == "dev":
        qry = stratified(train, classes, a.n_dev, rng, exclude=[f for f, _ in ref])
    else:
        test, _ = index(a.root, "test")
        train_hashes = {file_sha256(f) for f, _ in train}
        test = [(f, c) for f, c in test if file_sha256(f) not in train_hashes]
        qry = stratified(test, classes, a.n_test, rng)
    R, y_ref = load(ref, classes, a.points, rng)
    Q, y_qry = load(qry, classes, a.points, rng)
    rows, meta = run_protocol(R, y_ref, Q, y_qry, 3, conditions=a.conditions, seed=SEED,
                              n_boot=a.n_boot)
    print(f"{len(R)} reference, {len(Q)} queries ({a.split})  |  {meta}")
    print_rows(rows)
    write_rows(rows, a.out or f"benchmarks/results/modelnet10_{a.split}.csv")


if __name__ == "__main__":
    main()
