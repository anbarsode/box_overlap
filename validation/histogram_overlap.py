"""Exact dense-histogram sums on pair-shared uniform grids for 3D+2D samples.

This validation helper accelerates counting only; it does not approximate or
filter event pairs. For block b with cell volume v_b and counts h_A,h_B,
I_b = sum_cells(h_A*h_B)/(N_A*N_B*v_b). Return I_0*I_1 for independent blocks.
Actual NumPy linspace edges are supplied to C++, preserving histogramdd edge rules.
Raw arrays must be finite and contained in the supplied grid, with positive ranges.
"""
from pathlib import Path
import os
import shlex
import subprocess
import sysconfig

for setting in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(setting, '1')
import numpy as np


def build():
    """Compile the serial helper with installed pybind11 and C++17, without fast-math."""
    import pybind11
    folder = Path(__file__).parent
    output = folder / ('_histogram'+sysconfig.get_config_var('EXT_SUFFIX'))
    temporary = output.with_name(output.name+f'.{os.getpid()}.tmp')
    command = shlex.split(os.environ.get('CXX', 'c++')) + [
        '-O3', '-DNDEBUG', '-std=c++17', '-shared', '-fPIC', '-Wall', '-Wextra',
        '-I'+pybind11.get_include(), '-I'+sysconfig.get_path('include'),
        str(folder/'histogram_core.cpp'), '-o', str(temporary)]
    subprocess.run(command, check=True)
    temporary.replace(output)
    print(f'Built {output}', flush=True)


def block_overlaps(a, b, bins=24, bounds_a=None, bounds_b=None):
    """Return two block integrals with exactly counted dense histograms.

    a,b: float64 joint arrays (N,5); memory maps are accepted without copying.
    bins: uniform bins per coordinate; default 24 (half the previous 48).
    bounds_a,b: optional cached (lower[5],upper[5]) raw sample extrema. Use raw
    sample bounds, not density-box bounds, which can omit discarded samples.
    Both events use the same edges spanning their combined raw sample bounds.
    """
    from validation import _histogram
    if bounds_a is None:
        bounds_a = a.min(0), a.max(0)
    if bounds_b is None:
        bounds_b = b.min(0), b.max(0)
    edges = shared_edges(bounds_a,bounds_b,bins)
    if not _histogram.share_bins(np.asarray(bounds_a),np.asarray(bounds_b),edges):
        return 0.,0.
    return _histogram.block_overlaps(a,b,edges)


def shared_edges(bounds_a, bounds_b, bins):
    """Return exact NumPy uniform edges covering both sets of raw sample bounds."""
    lower = np.minimum(bounds_a[0],bounds_b[0])
    upper = np.maximum(bounds_a[1],bounds_b[1])
    return np.stack([np.linspace(lo,hi,bins+1) for lo,hi in zip(lower,upper)])


if __name__ == '__main__':
    build()
