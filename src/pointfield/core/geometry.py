from abc import ABC, abstractmethod

import numpy as np


class Geometry(ABC):
    """Base class for every sampleable geometry in point-field.

    A geometry lives in local coordinates (centred at the origin). Placing it
    in a scene is the job of :class:`pointfield.core.pose.Pose`.
    """

    @property
    @abstractmethod
    def ambient_dimension(self) -> int:
        """Dimension of the space containing the geometry."""

    @property
    @abstractmethod
    def intrinsic_dimension(self) -> int:
        """Intrinsic dimension of the geometry (1 = curve, 2 = region/surface)."""

    @property
    def centroid(self) -> np.ndarray:
        """Centroid in local coordinates (origin unless overridden)."""
        return np.zeros(self.ambient_dimension)

    @property
    @abstractmethod
    def bounds(self):
        """Axis-aligned bounding box in local coordinates."""

    @abstractmethod
    def sample(
        self,
        n_points: int,
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate a discrete sample from the geometry."""

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _check_n(n_points: int) -> None:
        if n_points <= 0:
            raise ValueError("n_points must be greater than zero")

    def describe(self) -> dict:
        """Parameters of the geometry, for reproducibility logs."""
        params = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in vars(self).items()
            if not k.startswith("_")
        }
        return {"type": type(self).__name__, **params}

    def __repr__(self) -> str:
        d = self.describe()
        name = d.pop("type")
        args = ", ".join(f"{k}={v!r}" for k, v in d.items())
        return f"{name}({args})"
