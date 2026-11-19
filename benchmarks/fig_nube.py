"""Bloque 01: nube ilustrativa sin etiquetas (lo que recibe el método)."""
import os

import numpy as np

from style import plt
from pointfield.datasets import make

HERE = os.path.dirname(__file__)
X = np.vstack([make("no_convexo", seed=3).normalized().X * 0.8 + [-1.6, 0], make("anidado", seed=3).normalized().X * 0.75 + [1.6, 0]])
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.scatter(X[:, 0], X[:, 1], s=4, c="#0f4c4a")
ax.set_aspect("equal"); ax.axis("off")
fig.savefig(os.path.join(HERE, "figures", "fig_nube_ilustrativa.png"), transparent=True)
