"""Matplotlib style shared by every figure (matches the presentation palette)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#16302f"
TEAL = "#0f4c4a"
GOLD = "#d9a21b"
MUTED = "#6b7b7a"
GROUP_COLORS = ["#2a6f97", "#c8553d", "#7a9e3b", "#8e6bb5", "#d9a21b", "#3d9a8b"]
NOISE_COLOR = "#c9c9c9"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": "#9aa5a4",
    "axes.linewidth": 0.6,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


def colors_for(labels):
    return [GROUP_COLORS[int(v) % len(GROUP_COLORS)] if v >= 0 else NOISE_COLOR for v in labels]


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    return ax
