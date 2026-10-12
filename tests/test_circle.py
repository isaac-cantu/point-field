import numpy as np

from pointfield.geometry.shapes2d.circle import Circle


def test_circle_dimensions():
    circle = Circle(radius=1.0)

    assert circle.ambient_dimension == 2
    assert circle.intrinsic_dimension == 1


def test_circle_centroid():
    circle = Circle(radius=1.0)

    np.testing.assert_allclose(
        circle.centroid,
        [0.0, 0.0],
    )


def test_circle_bounds():
    circle = Circle(radius=2.0)

    np.testing.assert_allclose(
        circle.bounds.minimum,
        [-2.0, -2.0],
    )

    np.testing.assert_allclose(
        circle.bounds.maximum,
        [2.0, 2.0],
    )


def test_circle_sample_shape():
    circle = Circle(radius=1.0)

    X = circle.sample(
        n_points=100,
        seed=42,
    )

    assert X.shape == (100, 2)


def test_circle_sample_radius():
    circle = Circle(radius=3.0)

    X = circle.sample(
        n_points=1000,
        seed=42,
    )

    radii = np.linalg.norm(X, axis=1)

    np.testing.assert_allclose(
        radii,
        3.0,
        atol=1e-12,
    )