import numpy as np

from pointfield.geometry.shapes2d.polygon import Polygon


class Rectangle(Polygon):
    """Axis-aligned rectangle of size length (x) by width (y), centred."""

    def __init__(self, length: float = 1.0, width: float = 1.0, filled: bool = True):
        if length <= 0:
            raise ValueError("length must be greater than zero")
        if width <= 0:
            raise ValueError("width must be greater than zero")
        L, W = float(length) / 2, float(width) / 2
        super().__init__([[-L, -W], [L, -W], [L, W], [-L, W]], filled=filled)
        self.length = float(length)
        self.width = float(width)
