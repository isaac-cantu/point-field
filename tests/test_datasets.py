import numpy as np
import pytest

from pointfield.datasets import HELD_OUT, SCENARIOS, make
from pointfield.perturbations import PERTURBATIONS, add_occlusion, add_outliers


@pytest.mark.parametrize("name", list(SCENARIOS) + list(HELD_OUT))
def test_scenarios_reproducible(name):
    a, b = make(name, seed=5), make(name, seed=5)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.y, b.y)
    assert a.n_groups >= 2
    assert a.meta["seed"] == 5


@pytest.mark.parametrize("pname", list(PERTURBATIONS))
def test_perturbations_logged(pname):
    ds = make("entrelazado", seed=0)
    out = PERTURBATIONS[pname](ds, 1)
    assert out.meta["perturbations"]
    assert out.n_points <= ds.n_points


def test_occlusion_fraction():
    ds = make("anidado", seed=0)
    out = add_occlusion(ds, fraction=0.25, seed=0)
    assert abs(out.n_points / ds.n_points - 0.75) < 0.01


def test_outliers_label():
    ds = make("compacto", seed=0)
    out = add_outliers(ds, fraction=0.1, seed=0)
    assert (out.y == -1).sum() == round(0.1 * ds.n_points)


def test_normalized_isotropic():
    ds = make("no_convexo", seed=0).normalized()
    assert np.allclose(ds.X.mean(axis=0), 0)
    assert abs(np.sqrt((ds.X ** 2).sum(1).mean()) - 1) < 1e-12
