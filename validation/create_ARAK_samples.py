"""Create reproducible five-dimensional ARAK samples for validation.

Each event is an independent product of a three-dimensional intrinsic density
and a two-dimensional sky density. A failed stochastic construction is retried
with the same event number, so the requested number of events is always made.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

for _thread_setting in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_setting, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_box_overlap")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from cycler import cycler
import numpy as np
from tqdm import tqdm

from RandomDistributions import RandomDistribution


LABELS = [r"$\log_{10}\mathcal{M}$", r"$\log_{10}q$", r"$\chi_\mathrm{eff}$",
          r"$\alpha$", r"$\sin \delta$"]


def configure_plotting() -> None:
    """Apply the project's plotting style, with a readable fallback."""
    colors = ['navy', 'coral', 'c', 'crimson', 'springgreen', 'gold',
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
                     'axes.prop_cycle': cycler(color=colors)})
    if shutil.which("latex") is None or shutil.which("dvipng") is None:
        print("Install LaTeX and dvipng separately to use LaTeX text in plots.")
        rcParams["text.usetex"] = False
    try:
        font_manager.findfont("Times", fallback_to_default=False)
    except ValueError:
        print("Install a Times font separately to use the requested plot font.")
        rcParams["font.serif"] = ["DejaVu Serif"]


def make_event(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Draw one five-dimensional event from independent 3D and 2D blocks."""
    intrinsic = dict(
        n_components=1,
        bounds=[[0., 3.], [-2., 0.], [-1., 1.]],
        dimension=3,
        reference_center=[rng.uniform(0, 3), 2 * rng.uniform()**0.5 - 2, rng.uniform(-1, 1)],
        mode_scale=[0.2, 1, 2.],
        alignment=rng.uniform(),
        K=2,
        epsilon=1.,
        p=4,
        n_basis=5,
        coefficient_scale=.3,
        boundary_treatment="truncate",
        truncation_pilot_samples=1000,
        truncation_normalization_samples=10000,
    )
    sky = dict(
        n_components=4,
        bounds=[[0., 2 * np.pi], [-1., 1.]],
        dimension=2,
        reference_center=[rng.uniform(0, 2 * np.pi), rng.uniform(-1, 1)],
        mode_scale=[2, 0.5],
        center_scale=[2, 1],
        weight_decay=1.,
        scale_decay=1.,
        alignment=rng.uniform(size=4),
        K=4,
        epsilon=2.5,
        p=4,
        n_basis=5,
        coefficient_scale=1.8,
        boundary_treatment="truncate",
        truncation_pilot_samples=1000,
        truncation_normalization_samples=10000,
    )
    distribution = RandomDistribution(rng).product([intrinsic, sky])
    return distribution.rvs(n_samples, random_state=rng)


def save_corner_plot(samples: np.ndarray, path: Path) -> None:
    """Save a diagnostic corner plot for one event."""
    import corner

    corner.corner(samples, labels=LABELS, show_titles=True)
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-events", type=int, default=10000)
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "ARAK_samples")
    parser.add_argument("--n-plots", type=int, default=100)
    arguments = parser.parse_args()

    arguments.output.mkdir(parents=True, exist_ok=True)
    if arguments.n_plots:
        configure_plotting()
    rng = np.random.default_rng(arguments.seed)
    progress = tqdm(total=arguments.n_events)
    event = 0
    while event < arguments.n_events:
        try:
            samples = make_event(arguments.n_samples, rng)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            progress.write(f"Event {event} failed; retrying: {error}")
            continue

        np.save(arguments.output / f"event_{event}.npy", samples)
        if event < arguments.n_plots:
            save_corner_plot(samples, arguments.output / f"event_{event}.jpg")
        event += 1
        progress.update()
    progress.close()


if __name__ == "__main__":
    main()
