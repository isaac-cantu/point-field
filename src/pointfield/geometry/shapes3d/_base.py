import numpy as np

from pointfield.core.bounds import AABB
from pointfield.core.geometry import Geometry


class Surface3D(Geometry):
    """Base for closed surfaces in R^3 (intrinsic dimension 2)."""

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def intrinsic_dimension(self) -> int:
        return 2

    @property
    def bounds(self) -> AABB:
        return AABB.from_points(self.sample(4000, seed=0))
