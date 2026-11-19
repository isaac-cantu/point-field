import numpy as np

from pointfield.geometry.mesh import TriangleMesh


class Pyramid(TriangleMesh):
    """Surface of a right pyramid with a square base (side ``base``, height
    ``height``), centred at its centroid along z, area-uniform sampling."""

    def __init__(self, base: float = 1.0, height: float = 1.0):
        if base <= 0 or height <= 0:
            raise ValueError("base and height must be positive")
        h, H = base / 2, float(height)
        z0 = -H / 4            # centroid of a solid pyramid is at H/4 above the base
        V = np.array([[-h, -h, z0], [h, -h, z0], [h, h, z0], [-h, h, z0], [0.0, 0.0, z0 + H]])
        F = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [0, 2, 1], [0, 3, 2]])
        super().__init__(V, F)
        self.base, self.height = float(base), H

    def describe(self) -> dict:
        return {"type": "Pyramid", "base": self.base, "height": self.height}
