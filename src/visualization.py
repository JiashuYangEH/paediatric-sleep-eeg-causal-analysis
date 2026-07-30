from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

def add_node(
    ax, x, y, w, h, text, facecolor,
    edgecolor="#475569", linestyle="-",
):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.8,
        linestyle=linestyle,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text,
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
    )

def add_arrow(
    ax, start, end,
    color="#475569",
    label=None,
    offset=(0, 0),
):
    patch = FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.8,
        color=color,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(patch)

    if label:
        midpoint = (
            (start[0] + end[0]) / 2 + offset[0],
            (start[1] + end[1]) / 2 + offset[1],
        )
        ax.text(
            *midpoint, label,
            ha="center",
            va="center",
            color=color,
            fontsize=10,
            weight="bold",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
            },
        )