"""Generate random geometric boxes for manual overlap experiments.

This utility does not approximate probability samples. The probability box
generator is :mod:`bo.event_boxes`. This script lives in ``validation`` because
it only creates visual test data.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

for _thread_setting in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_setting, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_box_overlap")

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from cycler import cycler
import numpy as np


MY_COLORS = ['navy', 'coral', 'c', 'crimson', 'springgreen', 'gold',
             'dodgerblue', 'hotpink', 'forestgreen', 'moccasin', 'darkred', 'teal']
rcParams.update({'axes.labelsize': 14,
                 'axes.titlesize': 14,
                 'legend.fontsize': 14,
                 'xtick.labelsize': 12,
                 'ytick.labelsize': 12,
                 'text.usetex': True,
                 'text.latex.preamble': r'\usepackage{txfonts}',
                 'font.size': 14,
                 'font.family': 'serif',
                 'font.serif': 'times',
                 'mathtext.default': 'sf',
                 'lines.markeredgewidth': 1,
                 'lines.linewidth': 2,
                 'axes.prop_cycle': cycler(color=MY_COLORS)})


def generate_boxes(
    dimension: int,
    n_boxes: int,
    min_scale: float,
    max_scale: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate axis-aligned boxes inside the unit hypercube."""
    sizes = rng.uniform(min_scale, max_scale, size=(n_boxes, dimension))
    lower = rng.uniform(0, 1 - sizes)
    return lower, lower + sizes


def save_to_csv(path: str | Path, lower: np.ndarray, upper: np.ndarray) -> None:
    """Save lower and upper coordinates in a plain CSV file."""
    dimension = lower.shape[1]
    columns = ([f"lower_{coordinate}" for coordinate in range(dimension)] +
               [f"upper_{coordinate}" for coordinate in range(dimension)])
    np.savetxt(path, np.column_stack((lower, upper)), delimiter=",",
               header=",".join(columns), comments="")


def _vertices(lower: np.ndarray, upper: np.ndarray) -> list[list[float]]:
    """Return the eight vertices of a 3D box."""
    return [[upper[0] if index & 1 else lower[0],
             upper[1] if index & 2 else lower[1],
             upper[2] if index & 4 else lower[2]] for index in range(8)]


def plot_boxes(lower: np.ndarray, upper: np.ndarray, path: str | Path) -> None:
    """Plot boxes of dimension one, two, or three and save a 200 dpi JPEG."""
    dimension = lower.shape[1]
    if dimension == 1:
        figure, axis = plt.subplots(figsize=(10, 2))
        heights = np.linspace(-0.15, 0.15, len(lower))
        for index, height in enumerate(heights):
            axis.hlines(height, lower[index, 0], upper[index, 0])
        axis.set(xlim=(0, 1), ylim=(-0.3, 0.3), yticks=[], xlabel="Coordinate 0")
    elif dimension == 2:
        figure, axis = plt.subplots(figsize=(8, 8))
        for box_lower, box_upper in zip(lower, upper):
            axis.add_patch(plt.Rectangle(box_lower, *(box_upper - box_lower),
                                         fill=False, edgecolor="navy"))
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Coordinate 0", ylabel="Coordinate 1")
    elif dimension == 3:
        figure = plt.figure(figsize=(9, 7))
        axis = figure.add_subplot(projection="3d")
        faces = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
                 (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))
        for box_lower, box_upper in zip(lower, upper):
            vertices = _vertices(box_lower, box_upper)
            polygons = [[vertices[index] for index in face] for face in faces]
            axis.add_collection3d(Poly3DCollection(polygons, facecolor="navy",
                                                    edgecolor="black", alpha=0.12))
        axis.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1),
                 xlabel="Coordinate 0", ylabel="Coordinate 1", zlabel="Coordinate 2")
    else:
        return
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", "--d", type=int, required=True)
    parser.add_argument("--n-boxes", "--N", type=int, required=True)
    parser.add_argument("--min-scale", type=float, required=True)
    parser.add_argument("--max-scale", type=float, required=True)
    parser.add_argument("--filename", default="random_boxes.csv")
    parser.add_argument("--seed", type=int)
    arguments = parser.parse_args()

    if shutil.which("latex") is None or shutil.which("dvipng") is None:
        print("Install LaTeX and dvipng separately to use LaTeX text in plots.")
        rcParams["text.usetex"] = False
    try:
        font_manager.findfont("Times", fallback_to_default=False)
    except ValueError:
        print("Install a Times font separately to use the requested plot font.")
        rcParams["font.serif"] = ["DejaVu Serif"]

    lower, upper = generate_boxes(arguments.dimension, arguments.n_boxes,
                                  arguments.min_scale, arguments.max_scale,
                                  np.random.default_rng(arguments.seed))
    save_to_csv(arguments.filename, lower, upper)
    plot_boxes(lower, upper, Path(arguments.filename).with_suffix(".jpg"))
    print(f"Saved boxes to {arguments.filename}")


if __name__ == "__main__":
    main()
