"""Check accelerated dense histograms against independent numpy.histogramdd counts."""
from pathlib import Path
import os
import sys
import time
for setting in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[setting] = '1'
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation.histogram_overlap import block_overlaps
from validation.validate_pair_finder import sample_histogram_overlap


def test_histograms():
    rng = np.random.default_rng(557)
    for bins in (2, 7, 24, 48):
        a = rng.normal(size=(1234,5)); b = rng.normal(size=(987,5))+.7
        expected = sample_histogram_overlap(a,b,bins)
        actual = np.prod(block_overlaps(a,b,bins))
        np.testing.assert_allclose(actual,expected,rtol=2e-15,atol=0)
        # Samples exactly on, just below and just above internal edges. Also
        # include both outer endpoints and nonzero lower coordinates.
        edge = np.linspace(-3.7,8.2,bins+1)
        values = np.r_[edge, np.nextafter(edge[1:-1],-np.inf),
                       np.nextafter(edge[1:-1],np.inf)]
        a = np.column_stack([np.roll(values,k) for k in range(5)])
        b = a[::-1].copy()
        expected = sample_histogram_overlap(a,b,bins)
        np.testing.assert_allclose(np.prod(block_overlaps(a,b,bins)),expected,rtol=2e-15)
    # Disjoint sample bounds can still occupy a shared coarse histogram bin.
    # Rejection must use bin support, not ordinary sample bounding-box overlap.
    a=np.array([[0.]*5,[.4]*5]);b=np.array([[.6]*5,[1.]*5])
    for bins in (1,2,3,4):
        expected=sample_histogram_overlap(a,b,bins)
        np.testing.assert_allclose(np.prod(block_overlaps(a,b,bins)),expected,rtol=2e-15)
    assert sample_histogram_overlap(a,b,3)>0
    folder = Path(__file__).parent/'ARAK_samples'
    a = np.load(folder/'event_0.npy',mmap_mode='r')
    bounds_a = a.min(0), a.max(0)
    started=time.perf_counter()
    for j in range(1,31):
        b=np.load(folder/f'event_{j}.npy',mmap_mode='r')
        expected=sample_histogram_overlap(a,b,24)
        actual=np.prod(block_overlaps(a,b,24,bounds_a,(b.min(0),b.max(0))))
        np.testing.assert_allclose(actual,expected,rtol=2e-15,atol=0)
        del b
    print(f'Histogram checks passed, including bin edges and 30 real pairs '
          f'({time.perf_counter()-started:.3f} s)')


if __name__ == '__main__':
    test_histograms()
