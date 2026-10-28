r"""Approximate a sampled probability density with disjoint constant-density boxes.

For each coordinate, selected sample values are sorted and separators are put
at adjacent midpoints. The minimum and maximum of all samples provide the two
outer separators. Their Cartesian product defines a regular grid in index
space. The implementation creates only occupied cells, which is equivalent to
creating the full grid and immediately deleting empty cells.

Cells containing one sample are also discarded. Every remaining cell is
shrunk to the bounding box of its assigned samples. If box :math:`B` contains
``n_B`` of the original ``N`` samples and has hypervolume ``V_B``, its density
is

.. math::

    p_B = \frac{n_B}{N V_B}.

Therefore the integral of all retained boxes is

.. math::

    \sum_B p_B V_B = \frac{N_\mathrm{retained}}{N}.

Discarded probability is recorded and is neither moved nor renormalized.
Shrinking cannot make boxes overlap because every final box is contained in
one preliminary grid cell. Two-dimensional projections of different 3D boxes
can overlap even though their 3D interiors are disjoint.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Callable
import warnings

# Respect an explicit user setting while making single-threaded execution the
# default when this module is imported before NumPy.
for _thread_setting in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_setting, "1")

import numpy as np
import numpy.typing as npt


_MAGIC = b"BXOV2\0\0\0"
_HEADER = struct.Struct("<8sIIIId")
WeightFunction = Callable[[np.ndarray], npt.ArrayLike]


@dataclass(slots=True)
class BoxSet:
    """Disjoint boxes approximating one independent probability block.

    All arrays use ``float64``. ``lower`` and ``upper`` have shape
    ``(n_boxes, dimension)``; ``density`` and ``weight`` have shape
    ``(n_boxes,)``. ``sort_dimension`` is a zero-based coordinate local to this
    block, not the full sample array. All compared blocks must use the same order.
    Weights are evaluated at final box centers; densities are not renormalized.
    Zero-volume boxes intentionally expose invalid data through inf/NaN.
    """

    lower: np.ndarray
    upper: np.ndarray
    density: np.ndarray
    weight: np.ndarray
    sort_dimension: int = 0

    def __post_init__(self):
        """Keep all box arrays in lower-bound order for the overlap sweep."""
        order = np.argsort(self.lower[:, self.sort_dimension], kind="stable")
        for name in ("lower", "upper", "density", "weight"):
            setattr(self, name, np.ascontiguousarray(getattr(self, name)[order]))

    @property
    def dimension(self) -> int:
        return self.lower.shape[1]

    @property
    def n_boxes(self) -> int:
        return self.lower.shape[0]

    @property
    def retained_probability(self) -> float:
        """Probability retained by this block after singleton cells are removed."""
        volume = np.prod(self.upper - self.lower, axis=1)
        return float(np.sum(self.density * volume))

    def rows(self) -> np.ndarray:
        """Return rows ``[lower coordinates, upper coordinates, density, weight]``."""
        return np.column_stack((self.lower, self.upper, self.density, self.weight))


@dataclass(slots=True)
class EventBoxes:
    """The two block densities and their total retained probability mass.

    Since the block densities are independent, the integrated joint
    density is the product of their two integrals.
    """

    blocks: tuple[BoxSet, BoxSet]
    retained_probability: float

    def __iter__(self):
        """Allow ``first, second = event`` when working with the blocks."""
        return iter(self.blocks)


def pick_centers(
    samples: npt.ArrayLike,
    n_centers: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose ``n_centers`` rows without replacement from ``samples``.

    ``samples`` has shape (N,d); 1 <= n_centers <= N. ``rng`` controls reproducibility.
    Selected coordinates set midpoint separators; they are not generally the
    geometric centers of the final, shrunken boxes.
    """
    samples = np.asarray(samples)
    return samples[rng.choice(len(samples), size=n_centers, replace=False)]


def _make_grid_edges(samples: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return the ordered separators for every coordinate.

    Duplicate selected coordinates define the same separator and are removed.
    Midpoints use ``a + (b - a) / 2`` to avoid overflow for large coordinates.
    """
    edges = []
    for coordinate in range(samples.shape[1]):
        selected = np.unique(centers[:, coordinate])
        midpoints = selected[:-1] + (selected[1:] - selected[:-1]) / 2
        lower = samples[:, coordinate].min()
        upper = samples[:, coordinate].max()
        edges.append(np.concatenate(([lower], midpoints, [upper])))
    return tuple(edges)


def _occupied_cell_bounds(
    samples: np.ndarray,
    edges: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sample counts and bounds for occupied preliminary grid cells.

    Sorting flat cell numbers puts samples from the same cell next to each
    other. ``minimum.reduceat`` and ``maximum.reduceat`` then calculate every
    occupied cell's sample bounds in one vectorized pass. Empty cells are never
    allocated, which is essential when many separators are used.
    """
    cell_index = [
        np.searchsorted(axis_edges[1:-1], samples[:, coordinate], side="right")
        for coordinate, axis_edges in enumerate(edges)
    ]
    grid_shape = tuple(len(axis_edges) - 1 for axis_edges in edges)
    flat_index = np.ravel_multi_index(cell_index, grid_shape)

    order = np.argsort(flat_index)
    sorted_index = flat_index[order]
    starts = np.flatnonzero(np.r_[True, sorted_index[1:] != sorted_index[:-1]])
    counts = np.diff(np.r_[starts, len(samples)])
    sorted_samples = samples[order]
    lower = np.minimum.reduceat(sorted_samples, starts, axis=0)
    upper = np.maximum.reduceat(sorted_samples, starts, axis=0)
    return counts, lower, upper


def make_boxes(
    samples: npt.ArrayLike,
    centers: npt.ArrayLike,
    weight_function: WeightFunction | None = None,
    sort_dimension: int = 0,
) -> BoxSet:
    """Build occupied boxes, using grid bounds for collapsed dimensions."""
    samples = np.asarray(samples, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)

    if samples.ndim != 2 or len(samples) == 0:
        raise ValueError("samples must be a non-empty 2D array")
    if not np.isfinite(samples).all():
        raise ValueError("samples must contain only finite values")
    if centers.ndim != 2 or centers.shape[1] != samples.shape[1]:
        raise ValueError("centers must have the same coordinate dimension as samples")
    if not np.isfinite(centers).all():
        raise ValueError("centers must contain only finite values")

    edges = _make_grid_edges(samples, centers)
    counts, sample_lower, sample_upper = _occupied_cell_bounds(samples, edges)

    # Identify each occupied cell in the same order as _occupied_cell_bounds.
    cell_index = [
        np.searchsorted(axis_edges[1:-1], samples[:, j], side="right")
        for j, axis_edges in enumerate(edges)
    ]
    grid_shape = tuple(len(axis_edges) - 1 for axis_edges in edges)
    flat_index = np.ravel_multi_index(cell_index, grid_shape)
    order = np.argsort(flat_index)
    sorted_index = flat_index[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_index[1:] != sorted_index[:-1]]
    )
    occupied_flat = sorted_index[starts]

    keep = counts > 1
    counts = counts[keep]
    lower = sample_lower[keep].copy()
    upper = sample_upper[keep].copy()
    occupied_cells = np.array(
        np.unravel_index(occupied_flat[keep], grid_shape)
    ).T

    # A collapsed sample range is expanded to its original grid-cell extent.
    for i, cell in enumerate(occupied_cells):
        for j, axis_edges in enumerate(edges):
            if upper[i, j] <= lower[i, j]:
                lower[i, j] = axis_edges[cell[j]]
                upper[i, j] = axis_edges[cell[j] + 1]

    volume = np.prod(upper - lower, axis=1)
    if not np.isfinite(volume).all() or np.any(volume <= 0):
        raise ValueError(
            "Non-positive box volume remains after fallback to grid bounds."
        )

    density = counts / (len(samples) * volume)
    box_centers = lower + (upper - lower) / 2
    weight = (
        np.ones(len(lower), dtype=np.float64)
        if weight_function is None
        else np.asarray(weight_function(box_centers), dtype=np.float64)
    )
    return BoxSet(lower, upper, density, weight, sort_dimension)


@dataclass(slots=True)
class BoxTrials:
    """Winning realisation and the information needed to inspect trial selection.

    ``boxes`` contains the winning density boxes; ``centers`` contains the
    selected sample rows that defined its grid. ``hypervolumes[t]`` is the sum
    of retained box volumes in trial t. ``selected_trial`` is a zero-based index.
    """

    boxes: BoxSet
    centers: np.ndarray
    hypervolumes: np.ndarray
    selected_trial: int


def make_box_trials(
    samples: npt.ArrayLike,
    n_centers: int,
    ntrial: int | None = None,
    rng: np.random.Generator | None = None,
    sort_dimension: int = 0,
    weight_function: WeightFunction | None = None,
) -> BoxTrials:
    """Keep the random grid with the smallest sum of retained box hypervolumes.

    The objective is H = sum_B prod_k(upper_Bk - lower_Bk), evaluated after
    singleton removal and shrinking. Defaults are five trials in 3D, ten in
    2D, and one in other dimensions. Ties keep the earliest trial. Trials use
    fresh center samples from the supplied generator. Only the best boxes and
    centers are kept, plus one scalar per trial; raw samples are never copied
    across events. Weights are evaluated only on the winning boxes.

    This objective measures geometric support, not retained probability. A
    smaller-volume trial may discard more singleton mass; the event-level
    retained-probability warning still applies to the winning realisation.
    """
    samples = np.asarray(samples)
    ntrial = {3: 5, 2: 10}.get(samples.shape[1], 1) if ntrial is None else ntrial
    if ntrial < 1:
        raise ValueError("ntrial must be positive")
    rng = np.random.default_rng() if rng is None else rng
    volumes = np.empty(ntrial)
    best = None
    for trial in range(ntrial):
        centers = pick_centers(samples, n_centers, rng)
        boxes = make_boxes(samples, centers, sort_dimension=sort_dimension)
        volumes[trial] = np.prod(boxes.upper - boxes.lower, axis=1).sum()
        if best is None or volumes[trial] < volumes[selected]:
            best, best_centers, selected = boxes, centers, trial
    if weight_function is not None:
        centers = best.lower + (best.upper - best.lower) / 2
        best.weight = np.ascontiguousarray(weight_function(centers), dtype=np.float64)
    return BoxTrials(best, best_centers, volumes, selected)


def make_event_boxes(
    samples: npt.ArrayLike,
    n_centers: tuple[int, int] = (4, 8),
    rng: np.random.Generator | None = None,
    sort_dimensions: tuple[int, int] = (0, 0),
    block_dimensions: tuple[int, int] = (3, 2),
    ntrial: tuple[int, int] | None = None,
) -> EventBoxes:
    """Split two blocks from joint samples into independent box densities.

    Parameters
    ----------
    samples:
        Joint sample array (N, d0+d1); only this event's samples are needed.
    n_centers:
        Selected sample counts per block, default (4,8), each at most N. These
        control grid complexity, not the final occupied box count.
    rng:
        NumPy generator; None uses fresh randomness. A shared generator is
        advanced through first-block trials and then second-block trials.
    sort_dimensions:
        Zero-based ordering coordinates local to each block, default (0,0).
    block_dimensions:
        Dimensions (d0,d1), default (3,2); positive and sum to the column count.
    ntrial:
        Positive trial counts per block, or None for dimension-dependent defaults.

    ``sort_dimensions`` chooses the local ordering coordinate in each block.
    ``block_dimensions`` controls the split, defaulting to the first three and
    last two columns. Separate random choices and separators are used per block.
    ``ntrial`` sets the number of random realisations per block. When omitted,
    each 3D block uses five trials and each 2D block ten; other dimensions use one.
    The realisation with the smallest total retained box hypervolume is kept.
    The product of the two retained box densities is the
    approximate joint event density; omitted singleton mass is
    not renormalized. A warning is raised when the discarded event probability
    exceeds 0.1%.
    """
    samples = np.asarray(samples)
    rng = np.random.default_rng() if rng is None else rng
    d0, d1 = block_dimensions
    if d0 <= 0 or d1 <= 0 or d0+d1 != samples.shape[1]:
        raise ValueError("Two positive block dimensions must sum to the sample column count")
    first = samples[:, :d0]
    second = samples[:, d0:]
    trials = (None, None) if ntrial is None else ntrial
    blocks = tuple(make_box_trials(block, count, trial, rng, axis).boxes
                   for block, count, trial, axis in
                   zip((first, second), n_centers, trials, sort_dimensions))
    retained_probability = blocks[0].retained_probability * blocks[1].retained_probability
    discarded_probability = 1 - retained_probability
    if discarded_probability > 0.001 + 1e-12:
        warnings.warn(f"Event boxes retain {100 * retained_probability:.4f}% of the "
                      "sample probability (more than 0.1% was discarded)",
                      RuntimeWarning, stacklevel=2)
    return EventBoxes(blocks, retained_probability)


def save_event_boxes(path: str | Path, event: EventBoxes) -> None:
    """Save two blocks in one compact binary file directly readable from C++.

    The file starts with 32 bytes: eight ASCII bytes ``BXOV2`` plus three zero
    bytes, four little-endian uint32 values ``d0, n0, d1, n1``, and one
    little-endian float64 value ``retained_probability``. Next come
    ``n0`` rows of ``2*d0+2`` little-endian float64 values, then ``n1`` rows of
    ``2*d1+2`` values. Each row stores lower bounds, upper bounds, density,
    and weight in that order. No Python-specific serialization is used.
    """
    first, second = event.blocks
    with open(path, "wb") as output:
        output.write(_HEADER.pack(_MAGIC, first.dimension, first.n_boxes,
                                  second.dimension, second.n_boxes,
                                  event.retained_probability))
        output.write(first.rows().astype("<f8", copy=False).tobytes())
        output.write(second.rows().astype("<f8", copy=False).tobytes())


def load_event_boxes(
    path: str | Path,
    sort_dimensions: tuple[int, int] = (0, 0),
) -> EventBoxes:
    """Read saved boxes and order each block along its supplied local coordinate.

    Ordering metadata is not stored in the binary format; the caller supplies
    ``sort_dimensions`` when an order other than the first coordinate is needed.
    """
    with open(path, "rb") as source:
        header = source.read(_HEADER.size)
        if len(header) != _HEADER.size:
            raise ValueError(f"Expected a {_HEADER.size}-byte header, found {len(header)} bytes")
        magic, d0, n0, d1, n1, retained_probability = _HEADER.unpack(header)
        if magic != _MAGIC:
            raise ValueError("Not a probability box file")
        layout = ((d0, n0), (d1, n1))
        expected_values = sum(n_boxes * (2 * dimension + 2)
                              for dimension, n_boxes in layout)
        values = np.frombuffer(source.read(), dtype="<f8")
        if len(values) != expected_values:
            raise ValueError(f"Expected {expected_values} box values, found {len(values)}")

        blocks = []
        start = 0
        for block_index, (dimension, n_boxes) in enumerate(layout):
            n_values = n_boxes * (2 * dimension + 2)
            rows = values[start:start + n_values].reshape(n_boxes, 2 * dimension + 2)
            blocks.append(BoxSet(rows[:, :dimension], rows[:, dimension:2*dimension],
                                 rows[:, -2], rows[:, -1], sort_dimensions[block_index]))
            start += n_values
    return EventBoxes((blocks[0], blocks[1]), retained_probability)
