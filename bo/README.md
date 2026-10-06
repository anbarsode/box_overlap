# Event probability boxes

`event_boxes.py` approximates sampled probability densities with disjoint,
axis-aligned boxes. By default it treats the first three columns and last two
columns as independent blocks. Pass `block_dimensions=(d0, d1)` to
`make_event_boxes` for another split; the two dimensions must sum to the number
of sample columns.

## Construction

For one block containing $N$ joint samples:

1. Select `n_centers` sample rows without replacement.
2. For each coordinate, sort the selected values and put separators at the
   midpoints of adjacent values. Add the minimum and maximum of all samples as
   the two outer separators.
3. The Cartesian product of the coordinate intervals defines the preliminary
   grid. The implementation assigns samples directly to cells and creates only
   occupied cells; it does not allocate the full Cartesian product.
4. Discard cells containing zero or one sample.
5. Shrink every remaining cell to the bounding box of its assigned samples.
6. Evaluate the weight function at the center of each final box.

If final box $B$ contains $n_B$ samples and has hypervolume $V_B$, its
constant density is

\[
p_B = \frac{n_B}{N V_B}.
\]

The retained probability of one block is

\[
P_\mathrm{retained} = \sum_B p_B V_B
                     = \frac{N_\mathrm{retained}}{N}.
\]

The event's retained probability is the product of the two block values. The
code warns when more than 0.1% of the event probability is discarded. The
remaining density is not renormalized.

Every final box lies inside one preliminary cell, so final boxes have disjoint
interiors. Projections of distinct 3D boxes onto a 2D plane can overlap.

## Example

```python
import numpy as np

from bo import make_event_boxes, save_event_boxes

samples = np.load("validation/ARAK_samples/event_0.npy")
event = make_event_boxes(
    samples,
    n_centers=(4, 8),
    rng=np.random.default_rng(138),
)

intrinsic_boxes, sky_boxes = event
print(event.retained_probability)
save_event_boxes("event_0.boxes", event)
```

## Binary file format

`save_event_boxes` writes a compact little-endian file without Python object
serialization. It can be read directly from C++.

The 32-byte header contains:

| Field | C type | Meaning |
| --- | --- | --- |
| magic | `char[8]` | `BXOV2` followed by three zero bytes |
| d0 | `uint32_t` | First block dimension |
| n0 | `uint32_t` | First block box count |
| d1 | `uint32_t` | Second block dimension |
| n1 | `uint32_t` | Second block box count |
| retained_probability | `double` | Probability retained by the full event |

The header is followed by all first-block rows and then all second-block rows.
Every row contains `2 * dimension + 2` values of type `double`:

```text
lower[dimension], upper[dimension], density, weight
```

The exact file layout is versioned by the magic value.

## Sorted C++ overlap calculator

Build with the `gw` environment's Python from the project root:

```bash
python -m bo.build_overlap
python validation/test_event_boxes.py
python validation/test_overlap.py
```

The build uses the installed pybind11 and a C++17 compiler. It writes a local
Python extension under `bo`; rebuild it after changing the C++ source or Python
environment. No additional packages are downloaded. The C++ loops are single
threaded and release Python's interpreter lock during calculations.

`make_event_boxes(..., sort_dimensions=(0, 0))` selects the ordering coordinate
**within each block**. `make_boxes(..., sort_dimension=0)` does the same for one
block. All events compared must use the same ordering coordinates and block
dimensions. Bounds, densities and weights are reordered together. The existing
binary format is unchanged: it does not store ordering metadata. Loading restores
first-coordinate ordering by default; pass `sort_dimensions` to `load_event_boxes`
to choose another coordinate in each block.

### Integral and weights

For boxes $A,B$, intersection widths are

$$\Delta_k=\max(0,\min(u_{Ak},u_{Bk})-\max(l_{Ak},l_{Bk})),
\qquad v=\prod_k\Delta_k.$$

Boundary contact has zero volume. Define $f_A=v/V_A$ and $f_B=v/V_B$. The
contribution of a pair with positive intersection volume is

$$I_{AB}=v p_A p_B\frac{f_A w_A+f_B w_B}{f_A+f_B}
       =v p_A p_B\frac{w_A/V_A+w_B/V_B}{1/V_A+1/V_B}.$$

The block overlap is the sum of these contributions. For independent blocks,
the event overlap is the product of the two block overlaps. This gives the full
integral for unit weights, or when the full weight factors across the blocks.
A general weight coupling coordinates in different blocks needs an additional
representation and cannot be computed from the stored block weights alone.
Retained probabilities are not used to renormalize the integral.

### Sweep and preparation

Boxes are sorted by lower bound along the selected coordinate. For each box in
A, the loop skips B boxes ending before it and stops when B's sorted lower bounds
reach its upper bound. Other coordinates reject remaining candidates as soon as
one intersection width is nonpositive. Only a rejected prefix is permanently
skipped: upper bounds can be nonmonotone even when lower bounds are sorted.
Worst-case work remains proportional to the product of the box counts.

Preparation copies box rows into C++, caches inverse volumes and block bounds,
and sorts the snapshot. Reusing prepared events avoids repeated Python array
packing and copying. Changes to Python boxes require preparation again. The
second block (normally 2D) is evaluated first; a zero result avoids the first
block calculation.

```python
from bo import (prepare_event, prepare_events, overlap_1v1, overlap_1vN,
                overlap_NvN, overlap_NvN_pairs)
import numpy as np

prepared = [prepare_event(event) for event in events]
collection = prepare_events(prepared)
value = overlap_1v1(prepared[0], prepared[1])
values = overlap_NvN_pairs(collection, np.array([[0, 1], [0, 2]], dtype=np.int64))
values = overlap_1vN(prepared[0], [prepared[1], prepared[2]])
matrix = overlap_NvN(collection)
```

All four functions also accept ordinary `EventBoxes` objects. Pair indices start
at zero; pair results preserve input order, orientation and duplicates.
`overlap_1vN` returns a vector in the supplied event order. Prefiltering is the
caller's responsibility. Two different event collections are handled by looping
through the first and calling `overlap_1vN` with the relevant subset of the second.

`overlap_NvN` evaluates each unordered pair once and returns a symmetric dense
matrix including self overlaps on the diagonal. Its output alone uses $8N^2$
bytes, so it is suitable only for small collections. The pairs mode uses output
space proportional to the supplied pair count. Neither mode finds nonzero pairs.

`validation/playground.ipynb` compares two ARAK events with dense sample
histograms and midpoint evaluations of the box PDFs. The former also measures
the density approximation; the latter checks integration convergence.

## Finding intersecting event pairs

`build_index(events)` builds a balanced tree over the **overall event bounds** of
both blocks. Each tree node contains the union of its event bounds. The
node's events are split at the median center along the coordinate with the
largest center spread, until a leaf contains at most eight events. Empty events
are omitted from the tree while retaining their original indices.

A query rejects a node if any coordinate has nonpositive interval intersection,
testing the second-block coordinates first. Surviving leaves are tested with the raw sorted box loops: first
find one positive-volume box intersection in the second block, then in the first
block. Both existence tests stop at the first intersecting box pair. Individual
boxes have no tree. A hit requires geometric intersection in both independent
blocks; weights and densities do not affect selection. In particular, zero or
signed weights can produce a zero integral even for a geometrically selected
pair.

The event tree uses storage proportional to the number of events and their
block dimension; preparation additionally stores the boxes. Median splitting
keeps tree depth proportional to log(N), but query work depends on event-bound
coverage. Broad overlapping bounds can require checking every event, and
`find_NvN` can require quadratic work and output. No index can avoid writing a
quadratic number of matching pairs when the geometry produces them. The finder
returns all matches in memory; large applications can instead query one event
at a time and consume its results before the next query.

```python
from bo import build_index, find_NvN, find_1vN, prepare_events, overlap_NvN_pairs

full_events = prepare_events(events)
index = build_index(full_events)
pairs = find_NvN(index)             # (K, 2), i < j; no self pairs
values = overlap_NvN_pairs(full_events, pairs)
indices = find_1vN(query_event, index)  # sorted indices into events
```

`find_NvN` returns pairs in lexicographic index order; `find_1vN` returns ascending
indices. The latter includes the query's own event if it is present in the index
and intersects, so remove that index when querying a set against itself.

Detection and integral calculation can use different boxes. Build the index from
`detection_events` containing only selected high-density boxes, keeping exactly
the same event order and block dimensions as `events`. Query with the analogous
detection boxes. Feed the resulting indices to the overlap calculator with the
full boxes. This deliberately selects fewer pairs but computes their integrals
with every retained box; the finder does not renormalize or modify any boxes.

For two different sets, build an index over the second set and reuse it:

```python
from bo import prepare_event, overlap_1vN

second = [prepare_event(event) for event in events_b]
index_b = build_index(second)
for i, event in enumerate(events_a):
    query = prepare_event(event)
    hits = find_1vN(query, index_b)
    values = overlap_1vN(query, [second[j] for j in hits])
    # Consume (i, hits, values) here; no dense cross-set matrix is needed.
```

## Pair-finder validation

Run `python validation/test_pair_finder.py` for exhaustive geometry checks,
including empty inputs, touching boundaries, alternative ordering coordinates,
zero weights and independently selected detection boxes.

Run `python validation/validate_pair_finder.py` for the ARAK examples. It processes
all available events by default, draws the 100v1 and 100v5 examples, compares every pair
among 100 events with dense sample histograms, and measures 20 geometrically
spaced event counts from 2 through 10,000. Output figures are JPG at 200 dpi under
`validation/pair_finder_results`; CSV files retain numeric results.

The intersection figures display the 2D block boxes. Color encodes intersection
in **both** full probability blocks, so overlap on the displayed plane alone is
insufficient. The histogram scatter includes pairs with zero in either estimate;
its symmetric logarithmic axes retain these points. Histogram estimates measure
the sampled distribution rather than the box approximation and can differ
because of bin width, finite sampling and the retained-box density model.

Timing figures separate box generation, C++ preparation, index construction,
pair finding and accumulation on filtered pairs. Box-generation and preparation
curves use cumulative per-event times for prefixes of the same event order. With
memory-mapped samples, generation excludes opening the file but includes any
sample-page reads triggered during box generation. Index and query stages are separately measured on each
prefix. These are single-run wall times, and tiny cases are sensitive to timing
noise. The 1vN query is the final event of the dataset, with its self match removed
at the full-set endpoint. Pair counts are stored alongside times because they
strongly affect the calculation cost.


## Two-event intersection maps

The playground notebook compares the spatial overlap integrand for two selected
ARAK events with both boxes and dense histograms. It shows the full 2D block
and each pairwise projection of the 3D overlap integrand. A projection integrates
out the omitted coordinate **after** multiplying the full-dimensional densities;
it is not the product of their separate projected densities.

Box fields are display-pixel averages of analytic box intersections, so their
pixel integrals reproduce the exact block overlap independent of display
resolution. The color scale is shared between box and histogram rows within each
column. Zero overlap is left blank. Both comparisons use unit weights.

Reproduce the figure with `python validation/plot_1v1_overlap.py --idx1 66 --idx2 71`.
The helper is in `/validation`; the reusable calculator stays in `/bo`.


## Sample memory use

The box generator processes one event's samples at a time. Its temporary arrays
scale with that event's sample count, rather than the dataset's event count.
Prepared events, event indexes and overlap calculations store only boxes.

The full-dataset validation reads each `.npy` file using `mmap_mode="r"`, generates
its boxes and releases the sample map before reading the next event. Only compact
boxes and per-event timing values are accumulated. The histogram comparison
opens at most two event sample maps at a time, reusing the first event for pairs
with the same first index. No list of raw samples is built across events.
Memory mapping allows the operating system to read pages as they are used; box
generation still makes per-event working arrays for sorting and grouping.

For custom processing, the same pattern works directly:

```python
boxes = []
for path in sample_paths:
    samples = np.load(path, mmap_mode="r")
    boxes.append(make_event_boxes(samples))
    del samples
```

The notebook reads only the events selected for individual plots. Two-event
comparisons need their two sample arrays; the 100-event finder example retains
only generated boxes. Operating-system file caching is independent of Python
sample-array retention.

## Bounded all-pair batches

For larger event collections, use `find_NvN_batches(index, batch_events=64)` and
calculate each batch before requesting the next:

```python
from bo import find_NvN_batches

for pairs in find_NvN_batches(index, batch_events=64):
    values = overlap_NvN_pairs(full_events, pairs)
    # Consume or save this batch here, then release its arrays.
    del pairs, values
```

A batch contains pairs whose first index belongs to a consecutive range of at
most 64 events. Each pair has `i < j`; concatenating the outputs reproduces
`find_NvN(index)` exactly, including ordering. Batch memory scales with the
matches in that range, bounded by `batch_events * N` pairs. Compact event boxes
and the index remain resident; raw samples are still processed one event at a
time. Avoid collecting the generator outputs into a list for large datasets.

The scaling validation now processes all available events by default and uses
these batches, separately summing time spent finding pairs and accumulating
their integrals. `--max-events` can restrict the run. Every completed event-count
measurement is saved immediately to `timings.csv`. Building the event index uses
both blocks' overall bounding boxes; the leaf checks use sorted sub-box loops,
not another BVH.
