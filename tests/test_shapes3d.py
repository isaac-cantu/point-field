import numpy as np
import pytest

from pointfield.geometry.shapes3d import Cone, Cube, Cylinder, Ellipsoid, Helix, Sphere, Torus


@pytest.mark.parametrize("g", [Sphere(), Torus(), Cube(), Cylinder(), Cone(), Ellipsoid(), Helix()],
                         ids=lambda g: type(g).__name__)
def test_3d_samples(g):
    X = g.sample(400, seed=0)
    assert X.shape == (400, 3)
    np.testing.assert_array_equal(X, g.sample(400, seed=0))


def test_sphere_torus_on_surface():
    assert np.allclose(Sphere(2.0).signed_distance(Sphere(2.0).sample(100, 0)), 0)
    t = Torus(1.0, 0.3)
    assert np.allclose(t.signed_distance(t.sample(100, 0)), 0, atol=1e-12)
