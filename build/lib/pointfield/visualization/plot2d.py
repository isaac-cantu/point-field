import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#2a6f97", "#c8553d", "#7a9e3b", "#8e6bb5", "#d9a21b", "#3d9a8b", "#5c5c5c"]


def label_colors(y):
    y = np.asarray(y)
    cols = np.array([PALETTE[int(v) % len(PALETTE)] if v >= 0 else "#bbbbbb" for v in y])
    return cols


def plot_points(X, y=None, ax=None, s: float = 6, title: str | None = None, equal: bool = True):
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    c = "#2a6f97" if y is None else label_colors(y)
    ax.scatter(X[:, 0], X[:, 1], s=s, c=c, linewidths=0)
    if equal:
        ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    return ax
