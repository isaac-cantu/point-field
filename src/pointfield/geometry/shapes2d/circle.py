import numpy as np

from pointfield.core.bounds import AABB
from pointfield.core.geometry import Geometry


class Circle(Geometry):

    def __init__(
        self,
        radius: float = 1.0,
    ):
        if radius <= 0:
            raise ValueError("radius must be greater than zero")

        self.radius = float(radius)

    @property
    def ambient_dimension(self) -> int:
        return 2

    @property
    def intrinsic_dimension(self) -> int:
        return 1

    @property
    def centroid(self) -> np.ndarray:
        return np.zeros(2)

    @property
    def bounds(self) -> AABB:
        r = self.radius

        return AABB(
            minimum=np.array([-r, -r]),
            maximum=np.array([r, r]),
        )

    def contains(
        self,
        points: np.ndarray,
        tolerance: float = 1e-6,
    ) -> np.ndarray:

        points = np.asarray(points)

        distance = np.linalg.norm(
            points,
            axis=1,
        )

        return np.abs(
            distance - self.radius
        ) <= tolerance


    def signed_distance(
        self,
        points: np.ndarray,
    ) -> np.ndarray:

        points = np.asarray(points)

        return (
            np.linalg.norm(points, axis=1)
            - self.radius
        )

    def sample(
        self,
        n_points: int,
        seed: int | None = None,
    ) -> np.ndarray:

        if n_points <= 0:
            raise ValueError(
                "n_points must be greater than zero"
            )

        rng = np.random.default_rng(seed)

        theta = rng.uniform(
            0.0,
            2.0 * np.pi,
            size=n_points,
        )

        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)

        return np.column_stack((x, y))