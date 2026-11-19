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

    @classmethod
    def from_points(cls, points: np.ndarray) -> "AABB":
        points = np.asarray(points)
        return cls(minimum=points.min(axis=0), maximum=points.max(axis=0))

    def contains(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points)
        return np.all((points >= self.minimum) & (points <= self.maximum), axis=1)
