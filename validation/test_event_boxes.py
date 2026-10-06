"""Checks for the sampled event probability box generator.

Run from the project root with the ``gw`` environment's Python:
    python validation/test_event_boxes.py
"""

from pathlib import Path
import os
import sys
import tempfile
import warnings

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bo import (load_event_boxes, make_boxes, make_event_boxes, pick_centers,
                save_event_boxes)


def reference_boxes(samples, centers, weight_function=None):
    """Slow, explicit implementation of the construction for comparison.

    This version forms every preliminary cell with ``np.ndindex``. It is kept
    here as an independent correctness check; production code creates only
    occupied cells.
    """
    edges = []
    for coordinate in range(samples.shape[1]):
        selected = np.unique(centers[:, coordinate])
        middle = selected[:-1] + (selected[1:] - selected[:-1]) / 2
        edges.append(np.r_[samples[:, coordinate].min(),
                           middle,
                           samples[:, coordinate].max()])

    lower = []
    upper = []
    counts = []
    shape = tuple(len(axis_edges) - 1 for axis_edges in edges)
    for cell in np.ndindex(shape):
        cell_lower = np.array([edges[k][cell[k]] for k in range(samples.shape[1])])
        cell_upper = np.array([edges[k][cell[k] + 1] for k in range(samples.shape[1])])
        inside = np.all(samples >= cell_lower, axis=1)
        for coordinate in range(samples.shape[1]):
            if cell[coordinate] == shape[coordinate] - 1:
                inside &= samples[:, coordinate] <= cell_upper[coordinate]
            else:
                inside &= samples[:, coordinate] < cell_upper[coordinate]
        if np.count_nonzero(inside) > 1:
            lower.append(samples[inside].min(axis=0))
            upper.append(samples[inside].max(axis=0))
            counts.append(np.count_nonzero(inside))

    lower = np.asarray(lower)
    upper = np.asarray(upper)
    density = np.asarray(counts) / (len(samples) * np.prod(upper - lower, axis=1))
    centers = lower + (upper - lower) / 2
    weight = (np.ones(len(lower)) if weight_function is None else
              np.asarray(weight_function(centers)))
    return lower, upper, density, weight


def check_density_and_disjointness(boxes):
    """Check finite densities and pairwise intersection volume."""
    volume = np.prod(boxes.upper - boxes.lower, axis=1)
    assert np.all(volume > 0)
    assert np.all(boxes.density > 0)
    assert np.all(boxes.weight == 1)

    for i in range(boxes.n_boxes):
        overlap_widths = np.minimum(boxes.upper[i], boxes.upper[i + 1:]) - np.maximum(
            boxes.lower[i], boxes.lower[i + 1:])
        assert not np.any(np.all(overlap_widths > 0, axis=1))


def test_known_grid():
    # The selected x values are 0 and 1.5, hence the separator is 0.75.
    # The selected y values are 0 and 3, hence the separator is 1.5.
    samples = np.array([[0., 0.], [0.5, 0.5], [1.5, 3.], [2., 4.]])
    boxes = make_boxes(samples, samples[[0, 2]])
    assert boxes.n_boxes == 2
    np.testing.assert_array_equal(boxes.lower, [[0., 0.], [1.5, 3.]])
    np.testing.assert_array_equal(boxes.upper, [[0.5, 0.5], [2., 4.]])
    np.testing.assert_allclose(boxes.density, [2., 1.])
    check_density_and_disjointness(boxes)
    mass = np.sum(boxes.density * np.prod(boxes.upper - boxes.lower, axis=1))
    np.testing.assert_allclose(mass, 1)

    weighted = make_boxes(samples, samples[[0, 2]], lambda x: x[:, 0] + 2 * x[:, 1])
    np.testing.assert_allclose(weighted.weight, [0.75, 8.75])


def test_against_explicit_full_grid():
    """Compare the sparse implementation with all cells of a small full grid."""
    rng = np.random.default_rng(734)
    for dimension in (1, 2, 3):
        samples = rng.normal(size=(200, dimension))
        centers = pick_centers(samples, 4, rng)
        weight_function = lambda x: np.sum(x**2, axis=1)
        actual = make_boxes(samples, centers, weight_function)
        expected = reference_boxes(samples, centers, weight_function)
        np.testing.assert_allclose(actual.rows(), np.column_stack(expected))


def test_singleton_discard():
    samples = np.array([[0., 0.], [0.5, 0.5], [2., 4.]])
    boxes = make_boxes(samples, samples[[0, 2]])
    assert boxes.n_boxes == 1
    np.testing.assert_array_equal(boxes.lower, [[0., 0.]])
    np.testing.assert_array_equal(boxes.upper, [[0.5, 0.5]])
    np.testing.assert_allclose(boxes.density, [8. / 3.])
    check_density_and_disjointness(boxes)
    np.testing.assert_allclose(np.sum(boxes.density * np.prod(boxes.upper - boxes.lower, axis=1)),
                               2. / 3.)

    all_singletons = make_boxes(samples, samples)
    assert all_singletons.n_boxes == 0
    assert all_singletons.retained_probability == 0


def test_arak_event():
    """Use a supplied five-dimensional event and verify file round-trip."""
    samples = np.load(Path(__file__).parent / "ARAK_samples" / "event_0.npy")
    rng = np.random.default_rng(138)
    event = make_event_boxes(samples, n_centers=(5, 8), rng=rng)
    first, second = event
    np.testing.assert_allclose(event.retained_probability,
                               first.retained_probability * second.retained_probability)
    for block_samples, boxes, limit in ((samples[:, :3], first, 5**3),
                                        (samples[:, 3:], second, 8**2)):
        assert boxes.n_boxes <= limit
        check_density_and_disjointness(boxes)
        inside = np.all((block_samples[:, None, :] >= boxes.lower[None, :, :]) &
                        (block_samples[:, None, :] <= boxes.upper[None, :, :]), axis=2)
        retained_fraction = np.mean(np.any(inside, axis=1))
        integrated_probability = np.sum(boxes.density * np.prod(boxes.upper - boxes.lower,
                                                                 axis=1))
        np.testing.assert_allclose(integrated_probability, retained_fraction)

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "event_0.boxes"
        save_event_boxes(path, event)
        restored = load_event_boxes(path)
        for old, new in zip((first, second), restored):
            np.testing.assert_array_equal(old.rows(), new.rows())
        np.testing.assert_array_equal(restored.retained_probability, event.retained_probability)
        assert path.stat().st_size == 32 + 8 * (first.n_boxes * 8 + second.n_boxes * 6)

    # The lower-level functions expose the chosen sample points for plots.
    selected = pick_centers(samples[:, :3], 5, np.random.default_rng(138))
    np.testing.assert_array_equal(first.rows(), make_boxes(samples[:, :3], selected).rows())


def test_retained_probability_warning():
    samples = np.array([[0., 0., 0., 0., 0.],
                        [0.5, 0.5, 0.5, 0.5, 0.5],
                        [2., 2., 2., 2., 2.]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        event = make_event_boxes(samples, n_centers=(2, 1), rng=np.random.default_rng(4))
    np.testing.assert_allclose(event.retained_probability, 2. / 3.)
    assert len(caught) == 1
    assert "more than 0.1% was discarded" in str(caught[0].message)


if __name__ == "__main__":
    test_known_grid()
    test_against_explicit_full_grid()
    test_singleton_discard()
    test_arak_event()
    test_retained_probability_warning()
    print("Probability box checks passed")
