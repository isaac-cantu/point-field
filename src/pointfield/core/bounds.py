from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AABB:
    minimum: np.ndarray
    maximum: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return (self.minimum + self.maximum) / 2.0

    @property
    def size(self) -> np.ndarray:
        return self.maximum - self.minimum