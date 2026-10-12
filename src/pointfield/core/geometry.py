from abc import ABC, abstractmethod

import numpy as np


class Geometry(ABC):

    @property
    @abstractmethod
    def ambient_dimension(self) -> int:
        """Dimension of the space containing the geometry."""
        pass

    @property
    @abstractmethod
    def intrinsic_dimension(self) -> int:
        """Intrinsic dimension of the geometry."""
        pass

    # @property
    # @abstractmethod
    # def centroid(self) -> np.ndarray:
    #     """Centroid in local coordinates."""
    #     pass

    @property
    @abstractmethod
    def bounds(self):
        """Axis-aligned bounding box in local coordinates."""
        pass

    @abstractmethod
    def sample(
        self,
        n_points: int,
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate a discrete sample from the geometry."""
        pass