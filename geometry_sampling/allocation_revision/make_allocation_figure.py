"""Render the two probability-ablation panels from saved, immutable results."""
from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
METHODS = ("uniform", "fixed_pilot", "adaptive_ema")
LABELS = ("Uniform", "Fixed pilot", "Adaptive EMA")
COLORS = ("#386cb0", "#e08214", "#1b9e77")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18, "savefig.dpi": 220})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), constrained_layout=True)
    for ax, file, title in zip(axes, ("allocation_results.json", "wdbc_results.json"),
                              ("(a) Separable nonconvex model", "(b) WDBC coupled logistic model")):
        result = json.loads((HERE / file).read_text(encoding="utf-8"))
        for method, label, color in zip(METHODS, LABELS, COLORS):
            trajectories = [run["methods"][method]["trace"] for run in result["runs"]]
            costs = np.array([item["calls"] for item in trajectories[0]])
            values = np.array([[item["mapping"] for item in row] for row in trajectories])
            q25, median, q75 = np.quantile(values, [.25, .5, .75], axis=0)
            ax.plot(costs, median, color=color, label=label, linewidth=1.45)
            ax.fill_between(costs, q25, q75, color=color, alpha=.14, linewidth=0)
        ax.set_yscale("log")
        ax.set_xlabel("Paid client-block gradient calls")
        ax.set_ylabel("Squared full proximal mapping")
        ax.set_title(title, fontsize=9)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.savefig(HERE / "allocation_comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
