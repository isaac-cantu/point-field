import numpy as np

from pointfield.core.pose import random_rotation
from pointfield.datasets import make
from pointfield.geometry.shapes2d import Annulus
from pointfield.geometry.shapes3d import Torus
from pointfield.learning import CDLocalSpectral, ChristoffelDarboux, SzegoPhase, basis_size
from pointfield.learning.descriptors import cd_moment_descriptor, log_euclidean_distances


def test_basis_size():
    assert basis_size(2, 3) == 10 and basis_size(3, 4) == 35


def test_trace_identity():
    X = Annulus(0.8, 1.0).sample(400, seed=0)
    cd = ChristoffelDarboux(degree=3, reg=1e-12).fit(X)
    assert cd.trace_check(X) < 1e-6


def test_outlier_scores_higher():
    X = Annulus(0.8, 1.0).sample(400, seed=0)
    cd = ChristoffelDarboux(degree=4).fit(X)
    typical = np.median(cd.score(X))
    assert cd.score(np.array([[0.0, 0.0]]))[0] > 10 * typical     # hole
    assert cd.score(np.array([[3.0, 3.0]]))[0] > 1e3 * typical     # far away


def test_rotation_invariant_descriptor():
    X = Torus().sample(600, seed=0)
    M1 = cd_moment_descriptor(X, degree=2, n_rotations=200, seed=1)
    M2 = cd_moment_descriptor(X @ random_rotation(3, 5).T, degree=2, n_rotations=200, seed=2)
    M3 = cd_moment_descriptor(np.random.default_rng(0).normal(size=(600, 3)), degree=2, n_rotations=200)
    D = log_euclidean_distances([M1, M2, M3])
    assert D[0, 1] < 0.25 * D[0, 2]


def test_szego_recovers_winding():
    ds = make("entrelazado", seed=0).normalized()
    ph = SzegoPhase(2).fit(ds.X)
    assert ph.form_ == "lin" and abs(ph.turns_ - 1.5) < 0.2 and ph.winding_gain_ > 0.2


def test_local_spectral_runs():
    ds = make("anidado", seed=0).normalized()
    lab = CDLocalSpectral(2, 10).fit_predict(ds.X)
    assert len(np.unique(lab)) == 2
