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


def main():
    folder = Path(__file__).parent/'pair_finder_results'
    with open(folder/'timings.csv', newline='') as source:
        rows = list(csv.DictReader(source))
    data = {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}
    from validation.analysis_plots import plot_costs
    plot_costs(data, folder/'scaling.jpg')
    print(f'Saved {folder / "scaling.jpg"}; see validation/report.md for the consolidated report.')


if __name__ == '__main__':
    main()
