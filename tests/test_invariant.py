"""Tests for the post-mid-term object pipeline and the CD core fixes."""

import numpy as np
import pytest

from pointfield.geometry.mesh import TriangleMesh, read_off
from pointfield.geometry.shapes2d import Circle
from pointfield.geometry.shapes3d import Pyramid
from pointfield.learning import (
    BlockScaler,
    CDDescriptor,
    ChristoffelDarboux,
    MultiViewMetric,
    geometric_outlier_filter,
    knn_density_weights,
    make_view,
    mean_average_precision,
    moment_block_slices,
    moment_descriptor,
    monomial_basis,
    monomial_exponents,
    monomial_weights,
    nn_accuracy,
    paired_bootstrap,
    per_query_ap,
)
from pointfield.objects import make_object_dataset, random_similarity


# ---------------------------------------------------------------- CD core
def test_weighted_basis_reproduces_power_of_inner_product():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(2, 3))
    for k in (1, 2, 3):
        terms = [t for t in monomial_exponents(3, k) if len(t) == k]
        w = monomial_weights(terms)
        bx = monomial_basis(x[None], terms, w)[0]
        by = monomial_basis(y[None], terms, w)[0]
        assert np.isclose(bx @ by, (x @ y) ** k)


def test_cholesky_matches_explicit_inverse():
    X = np.random.default_rng(1).normal(size=(300, 3))
    cd = ChristoffelDarboux(2, reg=1e-6, basis="weighted").fit(X)
    V = cd.design(X[:20])
    A = cd.moment_matrix_ + cd.eps_ * np.eye(cd.basis_size_)
    ref = np.einsum("ij,jk,ik->i", V, np.linalg.inv(A), V)
    assert np.allclose(cd.score(X[:20]), ref, rtol=1e-8)


def test_degree_one_is_mahalanobis_plus_one():
    X = np.random.default_rng(2).normal(size=(500, 2)) @ np.array([[2, 0.3], [0, 0.5]])
    cd = ChristoffelDarboux(1, reg=1e-12).fit(X)
    mu, S = X.mean(0), np.cov(X.T, bias=True)
    d = X - mu
    maha = np.einsum("ij,jk,ik->i", d, np.linalg.inv(S), d)
    assert np.allclose(cd.score(X), 1 + maha, rtol=1e-6)


def test_circle_ridge_versus_pseudoinverse():
    # singular M: rank 5 of 6 (support equation x^2 + y^2 = 1)
    X = Circle(1.0).sample(400, seed=0)
    cd = ChristoffelDarboux(2, reg=0.01, basis="weighted").fit(X)
    assert np.linalg.matrix_rank(cd.moment_matrix_, tol=1e-10) == 5
    o = np.zeros((1, 2))
    q_ridge = cd.score(o)[0]
    V = cd.design(o)
    q_pinv = (V @ np.linalg.pinv(cd.moment_matrix_) @ V.T)[0, 0]
    assert q_ridge > 50                   # ridge: centre is far off the support
    assert q_pinv < 1                     # pinv: centre looks *more* typical than the data
    assert cd.score(X).mean() < 10


def test_fit_rejects_non_finite():
    X = np.ones((10, 2)); X[0, 0] = np.nan
    with pytest.raises(ValueError):
        ChristoffelDarboux(2).fit(X)


# ---------------------------------------------------------------- invariance
@pytest.mark.parametrize("dim", [2, 3])
def test_cd_descriptor_similarity_and_permutation_invariant(dim):
    clouds, _, _ = make_object_dataset(dim, 1, 300, seed=3)
    X = clouds[0]
    rng = np.random.default_rng(4)
    Y = random_similarity(X, rng, reflect=True)[rng.permutation(len(X))]
    Y = random_similarity(Y, rng, reflect=True)  # two transforms, at least one likely reflects
    cd = CDDescriptor()
    assert np.max(np.abs(cd(X) - cd(Y))) < 1e-9
    assert np.max(np.abs(moment_descriptor(X) - moment_descriptor(Y))) < 1e-9


def test_descriptor_sizes():
    clouds, _, _ = make_object_dataset(3, 1, 200, seed=0)
    cd = CDDescriptor()
    f = cd(clouds[0])
    assert f.shape == (78,)
    assert sum(s.stop - s.start for s in cd.block_slices(3).values()) == 78
    m = moment_descriptor(clouds[0])
    assert m.shape == (sum(s.stop - s.start for s in moment_block_slices(3).values()),)


def test_density_weights_and_filter():
    rng = np.random.default_rng(0)
    clouds, _, _ = make_object_dataset(3, 1, 400, seed=1, transform=False)
    X = clouds[0]
    w = knn_density_weights(X)
    assert np.isclose(w.sum(), 1) and np.all(w > 0)
    assert np.isfinite(CDDescriptor()(X, sample_weight=w)).all()
    noisy = np.vstack([X, rng.uniform(-3, 3, size=(20, 3))])
    keep = geometric_outlier_filter(noisy)
    assert keep[len(X):].mean() < 0.3 and keep[:len(X)].mean() > 0.9


# ---------------------------------------------------------------- evaluation
def test_map_and_nn_toy():
    G = np.array([[0.0], [1.0], [10.0], [11.0]]); yg = np.array([0, 0, 1, 1])
    Q = np.array([[0.4], [10.6]]); yq = np.array([0, 1])
    assert mean_average_precision(Q, yq, G, yg) == 1.0
    assert nn_accuracy(Q, yq, G, yg) == 1.0
    # one swap: relevant at ranks 1 and 3 -> AP = (1 + 2/3)/2
    ap = per_query_ap(np.array([[0.0]]), [0], np.array([[0.0], [0.5], [1.0]]), [0, 1, 0])
    assert np.isclose(ap[0], (1 + 2 / 3) / 2)


def test_paired_bootstrap_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.uniform(size=200); b = a - 0.05
    r = paired_bootstrap(a, b, strata=np.arange(200) % 4, n_boot=500)
    assert np.isclose(r["diff"], 0.05) and r["ci_low"] > 0


def test_multiview_metric_end_to_end():
    clouds, y, _ = make_object_dataset(3, 4, 256, seed=5)
    cd = CDDescriptor()
    rng = np.random.default_rng(0)
    F = np.array([cd(x) for x in clouds])
    scaler = BlockScaler(cd.block_slices(3)).fit(F)
    V = np.array([[scaler.transform(cd(make_view(x, rng))) for _ in range(3)] for x in clouds])
    metric = MultiViewMetric().fit(V)
    Z = metric.transform(scaler.transform(F))
    assert Z.shape == F.shape and metric.eta_ in (0.1, 1.0, 10.0)
    assert mean_average_precision(Z, y, Z, y) > 0.5


# ---------------------------------------------------------------- meshes
def test_pyramid_area_sampling_and_mesh():
    P = Pyramid(1.0, 1.0)
    X = P.sample(20000, seed=0)
    assert X.shape == (20000, 3)
    lo, hi = P.bounds.minimum, P.bounds.maximum
    assert np.all(X >= np.asarray(lo) - 1e-9) and np.all(X <= np.asarray(hi) + 1e-9)
    # base carries area 1 out of the total: fraction of points on the base plane
    on_base = np.isclose(X[:, 2], X[:, 2].min(), atol=1e-9).mean()
    assert abs(on_base - 1.0 / P.area) < 0.02


def test_read_off_header_quirk(tmp_path):
    p = tmp_path / "quad.off"
    p.write_text("OFF4 1 0\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n4 0 1 2 3\n")
    V, F = read_off(p)
    assert V.shape == (4, 3) and F.shape == (2, 3)       # fan triangulation
    m = TriangleMesh(V, F)
    assert np.isclose(m.area, 1.0)
    assert m.sample(100, seed=0).shape == (100, 3)
