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
    ``(n_boxes,)``.
    """

    lower: np.ndarray
    upper: np.ndarray
    density: np.ndarray
    weight: np.ndarray

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

    Since the block densities are independent, the integrated five-dimensional
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
    """Choose sample positions without replacement to set the grid separators."""
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
        if upper <= lower:
            raise ValueError(f"Coordinate {coordinate} has zero sample range")
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
) -> BoxSet:
    """Build occupied boxes for one probability block.

    Parameters
    ----------
    samples:
        All joint samples, with shape ``(n_samples, dimension)``.
    centers:
        Randomly chosen rows of ``samples``. Their sorted coordinates set
        midpoint separators independently in each dimension.
    weight_function:
        Optional function accepting box centers of shape ``(n_boxes, dimension)``
        and returning one weight per box. The current default is one everywhere.

    Only occupied cells are constructed. Single-sample cells are discarded;
    the other boxes are shrunk to the minimum and maximum coordinates of their
    assigned samples. The work therefore scales mainly with the number of
    samples, not the potentially large full grid.
    """
    samples = np.asarray(samples, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    edges = _make_grid_edges(samples, centers)
    counts, sample_lower, sample_upper = _occupied_cell_bounds(samples, edges)
    keep = counts > 1
    lower = sample_lower[keep]
    upper = sample_upper[keep]
    counts = counts[keep]
    volume = np.prod(upper - lower, axis=1)
    density = counts / (len(samples) * volume)
    box_centers = lower + (upper - lower) / 2
    weight = (np.ones(len(lower), dtype=np.float64) if weight_function is None else
              np.asarray(weight_function(box_centers), dtype=np.float64))
    return BoxSet(lower, upper, density, weight)


def make_event_boxes(
    samples: npt.ArrayLike,
    n_centers: tuple[int, int] = (4, 8),
    rng: np.random.Generator | None = None,
) -> EventBoxes:
    """Split five joint coordinates into independent 3D and 2D box densities.

    Separate random choices and separators are used for the first three and
    last two columns. The product of the two retained box densities is the
    approximate five-dimensional event density; omitted singleton mass is
    not renormalized. A warning is raised when the discarded event probability
    exceeds 0.1%.
    """
    samples = np.asarray(samples)
    rng = np.random.default_rng() if rng is None else rng
    first = samples[:, :3]
    second = samples[:, 3:]
    blocks = (make_boxes(first, pick_centers(first, n_centers[0], rng)),
              make_boxes(second, pick_centers(second, n_centers[1], rng)))
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


def load_event_boxes(path: str | Path) -> EventBoxes:
    """Read the binary format written by :func:`save_event_boxes`."""
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
        for dimension, n_boxes in layout:
            n_values = n_boxes * (2 * dimension + 2)
            rows = values[start:start + n_values].reshape(n_boxes, 2 * dimension + 2).copy()
            blocks.append(BoxSet(rows[:, :dimension], rows[:, dimension:2*dimension],
                                 rows[:, -2], rows[:, -1]))
            start += n_values
    return EventBoxes((blocks[0], blocks[1]), retained_probability)
