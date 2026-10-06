"""Independent overlap checks. Run with gw Python from the project root."""
from pathlib import Path
import sys
import time
import warnings
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bo import (BoxSet, EventBoxes, make_event_boxes, overlap_1v1, overlap_1vN,
                overlap_NvN, overlap_NvN_pairs, prepare_event, prepare_events)


def reference_block(a, b):
    """Brute-force NumPy reference, without sorted rejection or cached volumes."""
    widths = np.maximum(0, np.minimum(a.upper[:, None], b.upper[None])
                        - np.maximum(a.lower[:, None], b.lower[None]))
    intersection = np.prod(widths, axis=2)
    va = np.prod(a.upper - a.lower, axis=1)[:, None]
    vb = np.prod(b.upper - b.lower, axis=1)[None]
    fa, fb = intersection / va, intersection / vb
    weight = np.divide(fa * a.weight[:, None] + fb * b.weight[None], fa + fb,
                       out=np.zeros_like(intersection), where=(fa + fb) > 0)
    return np.sum(intersection * a.density[:, None] * b.density[None] * weight)


def reference_event(a, b):
    return np.prod([reference_block(x, y) for x, y in zip(a.blocks, b.blocks)])


def event(block):
    unit = BoxSet(np.array([[0.]]), np.array([[1.]]), np.ones(1), np.ones(1))
    return EventBoxes((block, unit), 1.)


def test_known_weight_and_touching():
    a = event(BoxSet(np.array([[0.]]), np.array([[2.]]), np.array([.5]), np.array([2.])))
    b = event(BoxSet(np.array([[1.]]), np.array([[5.]]), np.array([.25]), np.array([8.])))
    # Intersection=1; fractions=1/2 and 1/4; effective weight=4.
    assert overlap_1v1(a, b) == .5
    c = event(BoxSet(np.array([[2.]]), np.array([[3.]]), np.ones(1), np.ones(1)))
    assert overlap_1v1(a, c) == 0
    empty = event(BoxSet(np.empty((0, 1)), np.empty((0, 1)), np.empty(0), np.empty(0)))
    assert overlap_1v1(a, empty) == 0
    np.testing.assert_array_equal(overlap_NvN_pairs([a], np.empty((0, 2), dtype=int)), [])
    assert overlap_1vN(a, []).shape == (0,)
    assert overlap_NvN([]).shape == (0, 0)


def test_nonmonotone_upper_bounds():
    # The B boxes are disjoint in y, but their x upper bounds are not monotone.
    a = event(BoxSet(np.array([[4., 0.], [6., 2.]]), np.array([[5., 1.], [7., 3.]]),
                     np.ones(2), np.ones(2)))
    b = event(BoxSet(np.array([[0., 2.], [1., 0.]]), np.array([[10., 3.], [2., 1.]]),
                     np.ones(2), np.ones(2)))
    assert overlap_1v1(a, b) == 1
    assert overlap_1v1(b, a) == 1


def test_random_and_modes():
    rng = np.random.default_rng(903)
    for dimensions in ((3, 2), (2, 1), (4, 3)):
        for axes in ((0, 0), (dimensions[0] - 1, dimensions[1] - 1)):
            events = []
            for index in range(6):
                blocks = []
                for d, axis in zip(dimensions, axes):
                    samples = rng.normal(size=(300, d)) + index * .2
                    from bo import make_boxes
                    blocks.append(make_boxes(samples, samples[:4],
                        lambda x: 1 + np.sum(x*x, axis=1), sort_dimension=axis))
                events.append(EventBoxes(tuple(blocks), 1.))
            expected = np.array([[reference_event(a, b) for b in events] for a in events])
            prepared = prepare_events(events)
            np.testing.assert_allclose(overlap_NvN(prepared), expected, rtol=2e-14)
            pairs = np.array([[4, 1], [1, 4], [0, 0], [4, 1]])
            np.testing.assert_allclose(overlap_NvN_pairs(prepared, pairs),
                                       expected[pairs[:, 0], pairs[:, 1]], rtol=2e-14)
            np.testing.assert_allclose(overlap_1vN(prepare_event(events[2]), prepared),
                                       expected[2], rtol=2e-14)
            for a, b in pairs:
                np.testing.assert_allclose(overlap_1v1(events[a], events[b]), expected[a, b])
            try:
                overlap_NvN_pairs(prepared, np.array([[0, 99]]))
            except IndexError:
                pass
            else:
                raise AssertionError('Out of range pair not rejected')


def test_arak_and_timing():
    rng = np.random.default_rng(71)
    folder = Path(__file__).parent / 'ARAK_samples'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        events = [make_event_boxes(np.load(folder / f'event_{i}.npy'), rng=rng)
                  for i in range(30)]
    pairs = np.array([(i, j) for i in range(30) for j in range(i, 30)])
    prepared = prepare_events(events)
    expected = np.array([reference_event(events[i], events[j]) for i, j in pairs])
    actual = overlap_NvN_pairs(prepared, pairs)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=1e-15)
    # Use overlapping self pairs to measure actual box accumulation, not just rejection.
    self_pairs = np.column_stack((np.arange(30), np.arange(30)))
    start = time.perf_counter()
    for _ in range(100):
        overlap_NvN_pairs(prepared, self_pairs)
    cpp_seconds = (time.perf_counter() - start) / 100
    start = time.perf_counter()
    for i in range(30):
        reference_event(events[i], events[i])
    numpy_seconds = time.perf_counter() - start
    print(f'30 prepared self pairs: C++ {cpp_seconds*1e3:.3f} ms; '
          f'NumPy reference {numpy_seconds*1e3:.3f} ms')
    print(f'ARAK: {len(pairs)} pairs checked; {np.count_nonzero(actual)} nonzero')


if __name__ == '__main__':
    test_known_weight_and_touching()
    test_nonmonotone_upper_bounds()
    test_random_and_modes()
    test_arak_and_timing()
    print('Overlap checks passed')
