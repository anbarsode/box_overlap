"""Python interface to the single-threaded C++ overlap loops.

Prepare events once when reusing them in several calculations. Prepared events
own a compact snapshot of the boxes; edits to the original arrays require a new
preparation. All events in a calculation must share their block dimensions and
ordering coordinates. Batch modes cross the Python/C++ boundary only once.
"""
from __future__ import annotations

import numpy as np
from .event_boxes import EventBoxes


def _core():
    try:
        from . import _overlap
    except ImportError as error:
        raise ImportError("Build the overlap core with: python -m bo.build_overlap") from error
    return _overlap


def prepare_event(event: EventBoxes):
    """Copy boxes into C++, caching volumes and overall block bounds once."""
    core = _core()
    if isinstance(event, core.PreparedEvent):
        return event
    a, b = event.blocks
    return core.PreparedEvent(a.rows(), b.rows(), a.sort_dimension, b.sort_dimension)


def prepare_events(events):
    """Prepare a reusable event collection; existing prepared collections pass through."""
    core = _core()
    if isinstance(events, core.PreparedEvents):
        return events
    return core.PreparedEvents([prepare_event(event) for event in events])


def overlap_1v1(event_a, event_b) -> float:
    """Return the weighted overlap of two events, checking the second block first."""
    return _core().overlap_1v1(prepare_event(event_a), prepare_event(event_b))


def overlap_NvN(events) -> np.ndarray:
    """Return a symmetric (N, N) matrix, including self overlaps on its diagonal.

    Computes each unordered pair once. Requires 8*N*N output bytes and evaluates
    zero pairs too; use overlap_NvN_pairs for large, prefiltered event sets.
    """
    return _core().overlap_NvN(prepare_events(events))


def overlap_NvN_pairs(events, event_pairs) -> np.ndarray:
    """Return one overlap per supplied pair, preserving pair order and duplicates.

    event_pairs is an integer array of shape (n_pairs, 2), indexing events from
    zero. Filtering for nonzero pairs is the caller's responsibility.
    """
    pairs = np.asarray(event_pairs)
    if pairs.dtype.kind not in "iu":
        raise TypeError("event_pairs must contain integer indices")
    return _core().overlap_NvN_pairs(prepare_events(events), pairs)


def overlap_1vN(event, other_events) -> np.ndarray:
    """Return overlaps in other_events order, using a caller-filtered collection.

    For two different event sets, loop over the first set and supply each event's
    prefiltered subset of the second set. PreparedEvent objects can be reused in
    those subsets to avoid repeated box copying.
    """
    return _core().overlap_1vN(prepare_event(event), prepare_events(other_events))


def build_index(events):
    """Build a tree over both blocks' event bounds for geometry-only filtering.

    Supply all boxes, or separate detection events containing selected boxes.
    Index indices always refer to the supplied event order. The index owns its
    prepared events and can be reused for multiple queries.
    """
    core = _core()
    if isinstance(events, core.EventIndex):
        return events
    return core.EventIndex(prepare_events(events))


def find_1vN(event, events_or_index) -> np.ndarray:
    """Return sorted indices having positive-volume intersections in both blocks.

    The query can use selected detection boxes, independent of the full boxes
    subsequently supplied to overlap_1vN. An indexed copy of the query itself
    is included if its boxes intersect; remove its index when appropriate.
    """
    return _core().find_1vN(prepare_event(event), build_index(events_or_index))


def find_NvN(events_or_index) -> np.ndarray:
    """Return unique geometry-overlapping pairs (i, j), i < j, in index order.

    Output has shape (n_pairs, 2), including (0, 2) for no hits. Self pairs are
    excluded. Weights and densities do not affect selection: a selected pair
    can still have a zero weighted integral. Storage scales with the hit count.
    """
    return _core().find_NvN(build_index(events_or_index))


def find_NvN_batches(events_or_index, batch_events: int = 64):
    """Yield unique pair arrays for successive ranges of first-event indices.

    Each batch contains pairs (i,j), i<j, for at most batch_events first indices.
    Consume each batch before requesting the next to bound pair output memory.
    Concatenating the batches gives exactly find_NvN, in the same order. Batch
    storage depends on the number of matches, at worst batch_events*N pairs.
    """
    if batch_events <= 0:
        raise ValueError("batch_events must be positive")
    index = build_index(events_or_index)
    core = _core()
    for start in range(0, len(index), batch_events):
        yield core.find_NvN_range(index, start, min(start+batch_events, len(index)))
