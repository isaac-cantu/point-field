import numpy as np
import pytest

from pointfield.geometry.shapes2d import (
    Annulus, Circle, Crescent, Disk, Ellipse, Line, Polygon, Rectangle, Spiral, Star, Triangle,
)

ALL = [
    Circle(1.0), Disk(1.0), Annulus(0.5, 1.0), Ellipse(1.0, 0.5), Ellipse(1.0, 0.5, filled=False),
    Rectangle(2.0, 1.0), Rectangle(2.0, 1.0, filled=False), Triangle(), Star(), Star(filled=False),
    Polygon([[0, 0], [2, 0], [2, 2], [1, 1], [0, 2]]), Line(), Crescent(), Spiral(),
]


@pytest.mark.parametrize("g", ALL, ids=lambda g: type(g).__name__)
def test_sample_shape_and_reproducible(g):
    a, b = g.sample(300, seed=3), g.sample(300, seed=3)
    assert a.shape == (300, 2)
    np.testing.assert_array_equal(a, b)
    assert g.ambient_dimension == 2


@pytest.mark.parametrize("g", [x for x in ALL if hasattr(x, "contains") and not isinstance(x, Spiral)],
                         ids=lambda g: type(g).__name__)
def test_samples_lie_on_geometry(g):
    X = g.sample(500, seed=1)
    assert g.contains(X, tolerance=1e-6).mean() > 0.99


@pytest.mark.parametrize("g", ALL, ids=lambda g: type(g).__name__)
def test_samples_inside_bounds(g):
    X = g.sample(500, seed=2)
    bb = g.bounds
    assert np.all(X >= bb.minimum - 1e-6) and np.all(X <= bb.maximum + 1e-6)


def test_disk_area_uniform():
    X = Disk(1.0).sample(20000, seed=0)
    inner = (np.linalg.norm(X, axis=1) < np.sqrt(0.5)).mean()
    assert abs(inner - 0.5) < 0.02          # half the area within r = 1/sqrt(2)


def test_intrinsic_dimensions():
    assert Circle().intrinsic_dimension == 1
    assert Disk().intrinsic_dimension == 2
    assert Ellipse(filled=False).intrinsic_dimension == 1
    assert Spiral().intrinsic_dimension == 1


def test_rectangle_size():
    r = Rectangle(4.0, 1.0)
    np.testing.assert_allclose(r.bounds.size, [4.0, 1.0])
    assert abs(r.area - 4.0) < 1e-12


def test_spiral_on_curve():
    s = Spiral(a=0.5, b=0.5, turns=1.5)
    X = s.sample(200, seed=0)
    assert np.all(s.signed_distance(X) < 1e-2)


def test_invalid_parameters():
    with pytest.raises(ValueError):
        Disk(-1)
    with pytest.raises(ValueError):
        Annulus(1.0, 0.5)
    with pytest.raises(ValueError):
        Spiral(kind="hyperbolic")
