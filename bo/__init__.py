"""Fast box approximations and overlap calculations for sampled event densities."""

from .event_boxes import (
    BoxSet,
    EventBoxes,
    load_event_boxes,
    make_boxes,
    make_event_boxes,
    pick_centers,
    save_event_boxes,
)

__all__ = [
    "BoxSet",
    "EventBoxes",
    "load_event_boxes",
    "make_boxes",
    "make_event_boxes",
    "pick_centers",
    "save_event_boxes",
]

from .overlap import (overlap_1v1, overlap_1vN, overlap_NvN, overlap_NvN_pairs,
                      prepare_event, prepare_events)

__all__ += ["overlap_1v1", "overlap_1vN", "overlap_NvN", "overlap_NvN_pairs",
            "prepare_event", "prepare_events"]

from .overlap import build_index, find_1vN, find_NvN
__all__ += ["build_index", "find_1vN", "find_NvN"]

from .overlap import find_NvN_batches
__all__ += ["find_NvN_batches"]
