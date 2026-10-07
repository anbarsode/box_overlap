"""Check matching-pair kernels against brute force, indexed finding and 1v1.

Includes zero weights (geometry remains positive), touching walls, empty inputs,
unequal set lengths, and independent block rejection. Run in the gw environment.
"""
import os
for setting in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[setting] = '1'
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bo import (BoxSet, EventBoxes, find_paired, overlap_paired, find_1vN,
                overlap_1v1, make_event_boxes, prepare_events)
from validation.test_overlap import reference_event


def unit(lower=0., weight=1.):
    block = BoxSet(np.array([[lower]]), np.array([[lower+1]]),
                   np.ones(1), np.array([weight]))
    return EventBoxes((block, block), 1.)


def main():
    a, b, c = unit(), unit(1.), unit(weight=0.)
    np.testing.assert_array_equal(find_paired([a, a, c], [a, b, c]), [True, False, True])
    np.testing.assert_array_equal(overlap_paired([a, a, c], [a, b, c]), [1., 0., 0.])
    assert find_paired([], []).shape == (0,)
    assert overlap_paired([], []).shape == (0,)
    for function in (find_paired, overlap_paired):
        try:
            function([a], [])
        except ValueError:
            pass
        else:
            raise AssertionError('Unequal lengths must fail')
    rng = np.random.default_rng(21)
    left, right = [], []
    for i in range(12):
        left.append(make_event_boxes(rng.normal(size=(300, 5)), ntrial=(1, 1), rng=rng))
        right.append(make_event_boxes(rng.normal(size=(300, 5))+i/3, ntrial=(1, 1), rng=rng))
    hits = find_paired(prepare_events(left), prepare_events(right))
    values = overlap_paired(left, right)
    for i, (a, b) in enumerate(zip(left, right)):
        np.testing.assert_allclose(values[i], reference_event(a, b), rtol=2e-14)
        assert values[i] == overlap_1v1(a, b)
        assert hits[i] == bool(len(find_1vN(a, [b])))
    print('Matching-pair tests passed: geometry, integrals, weights, boundaries and lengths')


if __name__ == '__main__':
    main()
