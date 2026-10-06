"""Notebook figures for trial selection and recursive enclosing bounds.

These figures describe geometry only: density boxes stay disjoint, while their
partition bounds may overlap. Output figures are JPG at 200 dpi.
"""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bo import partition_bounds


def plot_trials(trials, output=None):
    """Show every trial's volume and the selected realisation for each block."""
    from validation.plot_style import plot_style
    plt = plot_style()
    fig, axes = plt.subplots(1, len(trials), figsize=(12, 4.8), squeeze=False)
    for ax, result in zip(axes.flat, trials):
        x = np.arange(1, len(result.hypervolumes)+1)
        ax.plot(x, result.hypervolumes, 'o-', color='navy', label='Trial hypervolume')
        winner = result.selected_trial
        ax.scatter(x[winner], result.hypervolumes[winner], s=100, color='coral',
                   zorder=4, label='Selected minimum')
        ax.set_xlabel('Random realisation'); ax.set_ylabel('Sum of box hypervolumes')
        ax.set_title(f'{result.boxes.dimension}D block'); ax.set_xticks(x)
        ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout()
    if output is not None:
        fig.savefig(output, dpi=200)
    return fig


def plot_partition_levels(events, labels, output=None):
    """Compare zero through three levels on actual 2D density boxes."""
    from matplotlib.patches import Rectangle
    from validation.plot_style import plot_style
    plt = plot_style()
    fig, axes = plt.subplots(len(events), 4, figsize=(20, 5*len(events)), squeeze=False)
    for row, (event, label) in enumerate(zip(events, labels)):
        boxes = event.blocks[1]
        overall = partition_bounds(event, 1, 0)
        overall_area = np.prod(overall[:, 2:]-overall[:, :2], axis=1).sum()
        for ax, level in zip(axes[row], (0, 1, 2, 3)):
            for lo, hi in zip(boxes.lower, boxes.upper):
                ax.add_patch(Rectangle(lo, *(hi-lo), facecolor='0.65',
                                       edgecolor='0.35', linewidth=.3, alpha=.5))
            bounds = partition_bounds(event, 1, level)
            area = np.prod(bounds[:, 2:]-bounds[:, :2], axis=1).sum()
            for i, bound in enumerate(bounds):
                ax.add_patch(Rectangle(bound[:2], *(bound[2:]-bound[:2]), fill=False,
                                       edgecolor=('navy', 'coral', 'c', 'crimson',
                                                  'forestgreen', 'gold', 'dodgerblue',
                                                  'hotpink')[i%8],
                                       linewidth=2))
            ax.autoscale_view(); ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel(r'$\theta_4$'); ax.set_ylabel(r'$\theta_5$')
            ax.set_title(f'{label}: {level} levels, {len(bounds)} groups\n'
                         f'Summed area / overall area = {area/overall_area:.3f}')
    fig.suptitle('Grey: density boxes; coloured outlines: enclosing partition bounds')
    fig.tight_layout(rect=(0, 0, 1, .95))
    if output is not None:
        fig.savefig(output, dpi=200)
    return fig
