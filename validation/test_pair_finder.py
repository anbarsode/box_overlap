"""Check event filtering against independent exhaustive geometry checks."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bo import (BoxSet, EventBoxes, build_index, find_1vN, find_NvN,
                make_event_boxes, overlap_NvN_pairs, prepare_events, find_NvN_batches, partition_bounds, find_candidate_1vN)


def intersects(a, b):
    for x, y in zip(a.blocks, b.blocks):
        widths = np.minimum(x.upper[:, None], y.upper[None]) \
                 - np.maximum(x.lower[:, None], y.lower[None])
        if not np.any(np.all(widths > 0, axis=2)):
            return False
    return True


def test_exhaustive():
    rng = np.random.default_rng(462)
    events = [make_event_boxes(rng.normal(size=(300, 5)) + rng.uniform(-6, 6, size=5),
                              n_centers=(3, 4), rng=rng, sort_dimensions=(2, 1))
              for _ in range(60)]
    index = build_index(events)
    expected = np.array([(i, j) for i in range(len(events)) for j in range(i+1, len(events))
                         if intersects(events[i], events[j])], dtype=np.int64).reshape(-1, 2)
    np.testing.assert_array_equal(find_NvN(index), expected)
    for size in (1, 7, 64):
        batches = list(find_NvN_batches(index, batch_events=size))
        np.testing.assert_array_equal(np.concatenate(batches), expected)
    assert list(find_NvN_batches([], batch_events=3)) == []
    for query in events[:10]:
        expected_hits = [i for i, other in enumerate(events) if intersects(query, other)]
        np.testing.assert_array_equal(find_1vN(query, index), expected_hits)
    assert len(index) == len(events)
    assert find_NvN([]).shape == (0, 2)
    assert find_1vN(events[0], []).shape == (0,)


def test_geometry_not_weight():
    def make(lo, hi, weight):
        block = BoxSet(np.array([[lo]], dtype=float), np.array([[hi]], dtype=float),
                       np.ones(1), np.array([weight], dtype=float))
        return EventBoxes((block, block), 1.)
    events = [make(0, 1, 0), make(.5, 2, 0), make(2, 3, 1)]
    np.testing.assert_array_equal(find_NvN(events), [[0, 1]])
    np.testing.assert_array_equal(overlap_NvN_pairs(events, np.array([[0, 1]])), [0])
    empty = BoxSet(np.empty((0, 1)), np.empty((0, 1)), np.empty(0), np.empty(0))
    events.append(EventBoxes((empty, empty), 0.))
    np.testing.assert_array_equal(find_NvN(events), [[0, 1]])


def test_detection_subset_and_arak():
    folder = Path(__file__).parent / 'ARAK_samples'
    events = [make_event_boxes(np.load(folder / f'event_{i}.npy'),
                              rng=np.random.default_rng(91+i)) for i in range(100)]
    expected = np.array([(i, j) for i in range(100) for j in range(i+1, 100)
                         if intersects(events[i], events[j])], dtype=np.int64).reshape(-1, 2)
    pairs = find_NvN(events)
    np.testing.assert_array_equal(pairs, expected)
    assert np.all(overlap_NvN_pairs(prepare_events(events), pairs) > 0)
    selected = []
    for event in events:
        blocks = []
        for b in event.blocks:
            keep = np.argsort(b.density)[-max(1, b.n_boxes//4):]
            blocks.append(BoxSet(b.lower[keep], b.upper[keep], b.density[keep], b.weight[keep]))
        selected.append(EventBoxes(tuple(blocks), event.retained_probability))
    subset_pairs = find_NvN(selected)
    assert set(map(tuple, subset_pairs)).issubset(set(map(tuple, pairs)))
    print(f'100 ARAK events: {len(pairs)} geometry-overlapping pairs; '
          f'{len(subset_pairs)} with selected detection boxes')



def reference_partitions(boxes, levels):
    """Independent exhaustive cut costs, without prefix/suffix bound caching."""
    groups = [np.arange(boxes.n_boxes)] if boxes.n_boxes else []
    for _ in range(levels):
        next_groups = []
        for group in groups:
            if len(group) < 2:
                next_groups.append(group)
                continue
            best, split = np.inf, None
            for axis in range(boxes.dimension):
                center = boxes.lower[group, axis]+boxes.upper[group, axis]
                order = group[np.argsort(center, kind='stable')]
                for cut in range(1, len(order)):
                    parts = (order[:cut], order[cut:])
                    cost = sum(np.prod(boxes.upper[g].max(0)-boxes.lower[g].min(0))
                               for g in parts)
                    if cost < best:
                        best, split = cost, parts
            next_groups.extend(split)
        groups = next_groups
    return np.array([np.r_[boxes.lower[g].min(0), boxes.upper[g].max(0)]
                     for g in groups]).reshape(-1, 2*boxes.dimension)


def test_partition_bounds_and_consolidation():
    rng = np.random.default_rng(812)
    events = [make_event_boxes(rng.normal(size=(600, 5))+rng.uniform(-5, 5, 5),
                               n_centers=(3, 4), rng=rng) for _ in range(35)]
    expected = find_NvN(build_index(events, subdivisions=(0, 0)))
    for levels in ((0, 1), (0, 2), (0, 3), (1, 2), (2, 0), (0, 8)):
        index = build_index(events, subdivisions=levels)
        np.testing.assert_array_equal(find_NvN(index), expected)
        assert index.last_candidate_count >= len(expected)
        candidate_count = index.last_candidate_count
        reversed_index = build_index(events, subdivisions=levels, block_order=(1, 0))
        np.testing.assert_array_equal(find_NvN(reversed_index), expected)
        assert reversed_index.last_candidate_count == candidate_count
        for query in events[:3]:
            candidates = find_candidate_1vN(query, index)
            hits = find_1vN(query, index)
            np.testing.assert_array_equal(find_1vN(query, reversed_index), hits)
            np.testing.assert_array_equal(find_candidate_1vN(query, reversed_index), candidates)
            assert set(hits).issubset(set(candidates))
            assert len(candidates) == len(set(candidates))
            for block in (0, 1):
                actual = partition_bounds(query, block, levels[block])
                reference = reference_partitions(query.blocks[block], levels[block])
                np.testing.assert_array_equal(actual, reference)
    # Four separated modes: a query in the central gap passes an overall bound
    # but is rejected by partition bounds. Overlapping partitions still produce
    # only one hit per event; the exact finder agrees for both indexes.
    lo = np.array([[-5., -5.], [-5., 4.], [4., -5.], [4., 4.]])
    modes = BoxSet(lo, lo+1, np.ones(4), np.ones(4))
    common = BoxSet(np.zeros((1, 3)), np.ones((1, 3)), np.ones(1), np.ones(1))
    gap = BoxSet(np.zeros((1, 2)), np.ones((1, 2)), np.ones(1), np.ones(1))
    target, query = EventBoxes((common, modes), 1.), EventBoxes((common, gap), 1.)
    np.testing.assert_array_equal(find_candidate_1vN(query, build_index([target], (0, 0))), [0])
    assert len(find_candidate_1vN(query, build_index([target], (0, 2)))) == 0
    np.testing.assert_array_equal(find_1vN(target, build_index([target], (0, 2))), [0])
    empty = BoxSet(np.empty((0, 2)), np.empty((0, 2)), np.empty(0), np.empty(0))
    assert partition_bounds(EventBoxes((common, empty), 0.)).shape == (0, 4)


if __name__ == '__main__':
    test_partition_bounds_and_consolidation()
    test_exhaustive()
    test_geometry_not_weight()
    test_detection_subset_and_arak()
    print('Pair finder checks passed')
