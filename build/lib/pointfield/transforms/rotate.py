import numpy as np

from pointfield.core.pose import rotation_matrix_2d, rotation_matrix_3d


def rotate(points: np.ndarray, rotation, center=None) -> np.ndarray:
    """Rotate about ``center`` (default origin).

    rotation: angle (2D), 3 Euler angles (3D) or an explicit (d, d) matrix.
    """
    P = np.asarray(points, dtype=float)
    d = P.shape[1]
    if np.isscalar(rotation):
        R = rotation_matrix_2d(float(rotation))
    else:
        rot = np.asarray(rotation, dtype=float)
        R = rotation_matrix_3d(tuple(rot)) if rot.shape == (3,) else rot
    if R.shape != (d, d):
        raise ValueError("rotation does not match the point dimension")
    c = np.zeros(d) if center is None else np.asarray(center, dtype=float)
    return (P - c) @ R.T + c
