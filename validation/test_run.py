"""Verify saved-box jobs end to end, bounded samples, and original event IDs.

Run in gw: python validation/test_run.py. Expected integrals use the independent
exhaustive NumPy reference, not the pair finder or calculator under test.
"""
from pathlib import Path
import json
import os
import sys
import tempfile
import weakref
from unittest.mock import patch

for setting in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(setting, '1')
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bo import load_event_boxes
from bo.run import main
from test_overlap import reference_event


def test_saved_jobs():
    rng = np.random.default_rng(739)
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        samples, boxes = folder / 'samples', folder / 'boxes'
        samples.mkdir()
        ids = np.arange(7, 13)
        for number, event_id in enumerate(ids):
            # Intersections among the first five; the last is wholly isolated.
            offset = .2 * number if number < 5 else 30.
            np.save(samples / f'event_{event_id}.npy', rng.normal(size=(1200, 5)) + offset)
        real_load, references, peak = np.load, [], 0
        def track_load(*args, **kwargs):
            nonlocal peak
            assert kwargs['mmap_mode'] == 'r'
            array = real_load(*args, **kwargs)
            references.append(weakref.ref(array))
            peak = max(peak, sum(ref() is not None for ref in references))
            return array
        with patch('bo.run.np.load', side_effect=track_load):
            main(['boxes', '--samples', str(samples), '--output', str(boxes),
                  '--start', '7', '--events', '6', '--n-centers', '2', '3',
                  '--ntrial', '2', '3', '--sort-dimensions', '2', '1', '--seed', '95'])
        assert peak == 1 and all(ref() is None for ref in references)
        events = [load_event_boxes(boxes / f'event_{i}.boxes', (2, 1)) for i in ids]
        expected_pairs, expected_values = [], []
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                value = reference_event(events[i], events[j])
                if value > 0:
                    expected_pairs.append([i, j]); expected_values.append(value)
        assert len(expected_pairs) > 0
        for levels in (0, 1, 2, 3):
            output = folder / f'overlap_{levels}'
            # This stage must never load .npy sample files. event_ids is written, not read.
            with patch('bo.run.np.load', side_effect=AssertionError('Unexpected sample loading')):
                main(['overlaps', '--boxes', str(boxes), '--output', str(output),
                      '--start', '7', '--events', '6', '--sort-dimensions', '2', '1',
                      '--subdivisions', '0', str(levels), '--batch-events', '2'])
            np.testing.assert_array_equal(np.load(output / 'event_ids.npy'), ids)
            pairs, values = [], []
            for path in sorted(output.glob('batch_*.npz')):
                with np.load(path) as result:
                    pairs.append(result['pairs']); values.append(result['overlap'])
            np.testing.assert_array_equal(np.concatenate(pairs), expected_pairs)
            np.testing.assert_allclose(np.concatenate(values), expected_values, rtol=1e-12)
            settings = json.loads((output / 'settings.json').read_text())
            assert settings['status'] == 'complete'
            assert settings['pairs'] == len(expected_pairs)
            assert settings['batches'] == 3
        # Reusing an output folder must not silently leave stale batch files.
        try:
            main(['overlaps', '--boxes', str(boxes), '--output', str(output)])
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing output was silently mixed with new results')
    print('Saved-job checks passed: one sample map, no sample reads in overlap stage, '
          'correct IDs/pairs/integrals at all four levels')


if __name__ == '__main__':
    test_saved_jobs()
