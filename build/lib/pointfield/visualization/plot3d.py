import matplotlib.pyplot as plt

from pointfield.visualization.plot2d import label_colors


def plot_points_3d(X, y=None, ax=None, s: float = 3, title: str | None = None, elev=20, azim=35):
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d")
    c = "#2a6f97" if y is None else label_colors(y)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=s, c=c, linewidths=0, depthshade=True)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    if title:
        ax.set_title(title, fontsize=9)
    return ax
