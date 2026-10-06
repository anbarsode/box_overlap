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
