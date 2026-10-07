"""Create 1,000 pairs of connected five-dimensional CAMP samples.

Each pair shares a CAMP reference distribution and has independently realised
3D intrinsic and 2D sky blocks. Only one pair of raw samples is held at a time. A failed stochastic construction is retried
with the same event number, so the requested number of events is always made.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

for _thread_setting in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_setting] = "1"
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


def make_pair(n_samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw two connected events using fresh distributions around CAMP anchors."""
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
    connected = RandomDistribution(rng).camp(2, [intrinsic, sky])
    return tuple(distribution.rvs(n_samples, random_state=rng) for distribution in connected)


def save_corner_plot(samples, path: Path) -> None:
    """Overlay both members using common ranges, contours and sample points."""
    import corner
    ranges = [(min(a.min(), b.min()), max(a.max(), b.max()))
              for a, b in zip(samples[0].T, samples[1].T)]
    fig = corner.corner(samples[0], labels=LABELS, range=ranges, color='navy',
                        plot_datapoints=True)
    corner.corner(samples[1], labels=LABELS, range=ranges, color='coral',
                  plot_datapoints=True, fig=fig)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color='navy', label='Member 0'),
                        Line2D([], [], color='coral', label='Member 1')], loc='upper right')
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pairs", type=int, default=1000)
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "CAMP_samples")
    parser.add_argument("--n-plots", type=int, default=100)
    arguments = parser.parse_args()

    arguments.output.mkdir(parents=True, exist_ok=True)
    if arguments.n_plots:
        configure_plotting()
    rng = np.random.default_rng(arguments.seed)
    progress = tqdm(total=arguments.n_pairs)
    event = 0
    while event < arguments.n_pairs:
        try:
            samples = make_pair(arguments.n_samples, rng)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            progress.write(f"Event {event} failed; retrying: {error}")
            continue

        for member, values in enumerate(samples):
            np.save(arguments.output / f"event_{event}_{member}.npy", values)
        if event < arguments.n_plots:
            save_corner_plot(samples, arguments.output / f"event_{event}.jpg")
        event += 1
        progress.update()
    progress.close()
    settings = dict(n_pairs=arguments.n_pairs, n_samples=arguments.n_samples,
                    n_plots=min(arguments.n_plots, arguments.n_pairs), seed=arguments.seed)
    (arguments.output/"generation_settings.json").write_text(json.dumps(settings, indent=2)+"\n")


if __name__ == "__main__":
    main()
