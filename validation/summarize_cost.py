"""Render measured costs and the scaling figure from saved validation results.

No overlap calculations or raw sample loading are performed by this script.
"""
from pathlib import Path
import csv
import os
import sys
for setting in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(setting, '1')
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation.plot_style import plot_style


def main():
    folder = Path(__file__).parent/'pair_finder_results'
    with open(folder/'timings.csv', newline='') as source:
        rows = list(csv.DictReader(source))
    data = {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}
    plt = plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, find, calc, title in zip(axes, ('find_NvN_s', 'find_1vN_s'),
                                    ('overlap_NvN_s', 'overlap_1vN_s'),
                                    ('N within itself', 'One query vs N')):
        for key, label in (('box_generation_s', 'Box generation (incl. mapped reads)'),
                           ('preparation_s', 'C++ preparation'),
                           ('index_build_s', 'Partition index build'),
                           (find, 'Find pairs'), (calc, 'Calculate filtered overlaps')):
            ax.loglog(data['N'], data[key], marker='o', label=label)
        ax.set_xlabel('Number of events'); ax.set_ylabel('Wall time (s)'); ax.set_title(title)
        ax.legend(fontsize=10); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(folder/'scaling.jpg', dpi=200); plt.close(fig)
    print(f'Saved {folder / "scaling.jpg"}; see validation/report.md for the consolidated report.')


if __name__ == '__main__':
    main()
