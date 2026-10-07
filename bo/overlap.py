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
    """Return a C++ snapshot of an EventBoxes or an already prepared event.

    Bounds, densities and weights are copied; inverse volumes and global bounds
    are cached. Reuse the snapshot for many comparisons. Editing Python arrays
    does not update it. Empty blocks are supported and never geometrically match.
    """
    core = _core()
    if isinstance(event, core.PreparedEvent):
        return event
    a, b = event.blocks
    return core.PreparedEvent(a.rows(), b.rows(), a.sort_dimension, b.sort_dimension)


def prepare_events(events):
    """Prepare an iterable of events, preserving its order as zero-based indices.

    Ordinary EventBoxes are copied; prepared events are reused without box copies.
    An existing prepared collection passes through. No raw samples are stored.
    """
    core = _core()
    if isinstance(events, core.PreparedEvents):
        return events
    return core.PreparedEvents([prepare_event(event) for event in events])


def overlap_1v1(event_a, event_b) -> float:
    """Return the weighted overlap of two events, checking the first block first."""
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


def build_index(
    events,
    subdivisions: tuple[int, int] = (0, 3),
    block_order: tuple[int, int] = (0, 1),
):
    """Build a tree over enclosing block partitions for geometry-only filtering.

    ``subdivisions`` sets levels per block: each level bisects every group.
    Zero uses one overall bound; default (0, 3) leaves the first block whole
    and creates up to eight groups for the second. Each split minimises summed
    enclosing hypervolume over center-ordered cuts (see README). The index
    contains Cartesian combinations when both blocks are subdivided. Duplicate
    candidates are consolidated before exact sub-box checks. An existing index
    passes through with its original settings; rebuild from events to change them.

    ``block_order=(0,1)`` checks the first (normally 3D) block before the
    second (normally 2D), in both tree-coordinate and exact sub-box checks.
    Use (1,0) to compare the opposite order on the same boxes.

    Supply all boxes, or separate detection events containing selected boxes.
    Index indices always refer to the supplied event order. The index owns its
    prepared events and can be reused for multiple queries.
    """
    core = _core()
    if isinstance(events, core.EventIndex):
        return events
    return core.EventIndex(prepare_events(events), subdivisions, block_order)


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
    The NumPy output uses 16 bytes per pair; conversion temporarily holds a C++
    copy too. Integration adds an 8-byte value per pair. Do not collect batches
    into a list when the complete matching output is too large for memory.
    """
    if batch_events <= 0:
        raise ValueError("batch_events must be positive")
    index = build_index(events_or_index)
    core = _core()
    for start in range(0, len(index), batch_events):
        yield core.find_NvN_range(index, start, min(start+batch_events, len(index)))


def partition_bounds(event, block: int = 1, subdivisions: int = 3) -> np.ndarray:
    """Return enclosing partition rows [lower, upper] for inspection or plotting.

    Zero levels returns the overall bound. Empty blocks return no rows.
    L levels produce at most 2**L groups, capped by the number of boxes. Bounds
    can overlap; the original
    density boxes remain disjoint and are used for all integral calculations.
    """
    return _core().partition_bounds(prepare_event(event), block, subdivisions)


def find_candidate_1vN(event, events_or_index) -> np.ndarray:
    """Return unique event indices passing partition bounds, before exact checks.

    Useful for measuring rejection efficiency. This is a superset of find_1vN;
    candidates can have no actual sub-box intersection. No density or weight is
    used. Supply an explicit index to select partition settings.
    """
    return _core().find_candidate_1vN(prepare_event(event), build_index(events_or_index))


def find_paired(events1, events2) -> np.ndarray:
    """Return a Boolean mask for matching pairs (events1[i], events2[i]).

    Equal-length sets are required. Check positive-volume intersection in both
    blocks, 3D first by default, using the exact same block_intersects kernel
    as indexed finding. No BVH is constructed. Densities/weights are ignored.
    Supply selected detection boxes here and full boxes to overlap_paired if desired.
    Output and number of comparisons grow linearly with the number of pairs.
    """
    return _core().find_paired(prepare_events(events1), prepare_events(events2))


def overlap_paired(events1, events2) -> np.ndarray:
    """Return integrals of matching pairs, preserving order, including zeros.

    This is an elementwise (dot-like) comparison, not a Cartesian cross product:
    output[i] = overlap_1v1(events1[i], events2[i]). Equal lengths are required.
    Uses the existing weighted integral kernel; no BVH or raw samples are used.
    For bounded memory, call with successive equal-length batches of prepared boxes.
    """
    return _core().overlap_paired(prepare_events(events1), prepare_events(events2))
