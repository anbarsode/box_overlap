"""Verify pairwise sample loading is bounded and leaves histogram results unchanged."""
from pathlib import Path
import tempfile
from unittest.mock import patch
import weakref
import numpy as np
from validate_pair_finder import sample_histogram_overlap, sample_histogram_overlaps


def test_streaming():
    rng = np.random.default_rng(334)
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        for i in range(8):
            np.save(folder/f'event_{i}.npy', rng.normal(size=(200,5))+i*.05)
        # Include repeated, reversed, self and ungrouped pairs to exercise releases.
        pairs = np.array([(i,j) for i in range(8) for j in range(i+1,8)]
                         + [(0,1),(1,0),(4,4),(7,2),(0,3)])
        expected = [sample_histogram_overlap(np.load(folder/f'event_{i}.npy'),
                    np.load(folder/f'event_{j}.npy'),bins=8) for i,j in pairs]
        real_load = np.load
        references = []
        peak = 0
        def tracked_load(*args, **kwargs):
            nonlocal peak
            assert kwargs['mmap_mode'] == 'r'
            array = real_load(*args, **kwargs)
            assert isinstance(array, np.memmap) and not array.flags.writeable
            references.append(weakref.ref(array))
            peak = max(peak, sum(ref() is not None for ref in references))
            return array
        with patch('validate_pair_finder.np.load', side_effect=tracked_load):
            actual = list(sample_histogram_overlaps(folder,pairs,bins=8))
            assert peak == 2
            assert all(ref() is None for ref in references)
            assert list(sample_histogram_overlaps(folder,[],bins=8)) == []
        np.testing.assert_array_equal(actual,expected)
    print('Streaming checks passed: at most two sample maps alive, identical histogram results')


if __name__ == '__main__':
    test_streaming()
