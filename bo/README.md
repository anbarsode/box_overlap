# Probability boxes and overlaps

The reusable implementation is in `bo`; demonstrations and tests are in
`validation`. See [RUNNING.md](../RUNNING.md) for complete commands and parameter
examples, and [the consolidated report](../validation/report.md) for measurements,
validation and scope. All demonstrations are in `validation/playground.ipynb`.

## Density construction

`make_event_boxes(samples, n_centers=(4,8), rng=None, sort_dimensions=(0,0),
block_dimensions=(3,2), ntrial=None)` splits consecutive sample columns into two
independent blocks. A trial picks sample rows without replacement. Each coordinate
uses sorted unique selected values, midpoint separators and global sample extrema.
The Cartesian grid is represented by occupied cells only. Cells with zero or one
sample are discarded; every other cell shrinks to its contained samples' bounds.

For a box containing `n_B` of the original `N` samples and volume `V_B`,

$$p_B=\frac{n_B}{N V_B},\qquad
P_{\rm block}=\sum_B p_B V_B=\frac{N_{\rm retained}}{N}.$$

The event attribute `retained_probability` is the product of block probabilities.
Loss above 0.1% emits a warning. Mass is never renormalized. Zero-volume retained
boxes produce inf/NaN deliberately; no volume floor or singleton merger is used.
Final boxes have disjoint interiors because they remain inside separate grid cells.
Their lower-dimensional projections can overlap.

`make_box_trials(samples, n_centers, ntrial=None, rng=None, sort_dimension=0,
weight_function=None)` selects the trial minimizing the sum of shrunken box volumes.
Defaults depend on dimension: five in 3D, ten in 2D, one otherwise. Ties keep the
first trial. `BoxTrials` exposes `boxes`, selected sample `centers`, one scalar
`hypervolumes` value per trial and zero-based `selected_trial`.

`make_boxes(samples, centers, weight_function=None, sort_dimension=0)` constructs
one grid from explicit centers. Optional weights accept an `(n_boxes,d)` array of
final geometric centers and return one value per box. Trials evaluate weights only
on the winner. Event generation and CLI jobs use weights of one.

## Stored objects

| Object or parameter | Meaning |
| --- | --- |
| `BoxSet.lower`, `.upper` | Float64 arrays `(n_boxes,d)`; bounds of one block. |
| `.density`, `.weight` | Float64 arrays `(n_boxes,)`; constant density and center weight. |
| `.sort_dimension` | Local zero-based coordinate for lower-bound sorting; default 0. |
| `.dimension`, `.n_boxes` | Coordinate count and retained box count. |
| `.retained_probability` | Sum of density times volume for this block. |
| `.rows()` | New packed array `[lower[d], upper[d], density, weight]`. |
| `EventBoxes.blocks` | Two BoxSets, in the same order as consecutive sample blocks. |
| `EventBoxes.retained_probability` | Saved joint retained probability; does not include weights. |

`BoxSet` orders bounds, densities and weights together. All events in a calculation
must have the same dimensions and sorting coordinates. `prepare_event(event)` owns
a C++ snapshot, inverse volumes and global bounds. `prepare_events(events)` makes
a reusable collection. Already prepared inputs pass through. Reprepare after editing
Python arrays; snapshots do not track those edits.

## Exact integral over the box model

For boxes A and B, let

$$v=\prod_k\max(0,\min(u_{Ak},u_{Bk})-\max(l_{Ak},l_{Bk})).$$

For positive v, the effective weight prescribed by volume fractions is

$$w_{AB}=\frac{(v/V_A)w_A+(v/V_B)w_B}{v/V_A+v/V_B}
=\frac{w_A/V_A+w_B/V_B}{1/V_A+1/V_B}.$$

Sum `v*p_A*p_B*w_AB` over intersecting boxes to get a block integral. The full
event integral is the product of its two block integrals. This implements unit or
separable block weights; a general weight coupling both blocks is not represented.
Touching boundaries contribute zero. No normalization by retained probability occurs.

The C++ loops sweep boxes sorted by one lower-bound coordinate, reject disjoint
overall bounds, skip a rejected prefix and stop at a sorted lower bound beyond the
current upper bound. Other coordinates reject immediately on nonpositive width.
Upper bounds need not be sorted; the loop does not skip arbitrary boxes permanently.
No individual-box BVH is used. Worst-case box work remains quadratic in box counts.

| Function | Inputs and result |
| --- | --- |
| `overlap_1v1(a,b)` | Events or prepared events; one float. |
| `overlap_1vN(event,others)` | Caller-filtered events; vector in `others` order. |
| `overlap_NvN_pairs(events,pairs)` | Integer `(K,2)` zero-based indices; K integrals, preserves order/duplicates. |
| `overlap_NvN(events)` | Dense symmetric `(N,N)` matrix including self integrals; use only small sets. |

Integration always checks block 0 first and skips block 1 when its integral is zero.

## Geometry selection and enclosing partitions

`build_index(events, subdivisions=(0,3), block_order=(0,1))` builds a balanced
bounding-volume tree over enclosing partitions. Each subdivision level bisects every
splittable group of **whole boxes**. For each coordinate, sort box centers and examine
all cuts, minimizing the sum of the two children's enclosing hypervolumes. This is
a local greedy criterion. The final groups shrink to tight bounds of their complete
member boxes. Enclosing bounds may overlap, while original density boxes stay intact.

Zero levels uses one global block bound; L levels gives at most `2**L` groups.
When both blocks are split, their group combinations multiply tree entries.
Duplicate event hits are sorted and consolidated before exact geometry checks.
The default checks the first (normally 3D) block first in tree coordinates and exact
checks. `block_order=(1,0)` reverses detection order, preserving exact results.

| Function or setting | Meaning |
| --- | --- |
| `find_1vN(event,index)` | Sorted event positions intersecting in both blocks; can include the indexed query itself. |
| `find_NvN(index)` | Unique `(i,j)` pairs, `i<j`, sorted; excludes self pairs. |
| `find_NvN_batches(index,batch_events=64)` | Successive first-index ranges; consume before the next batch. |
| `find_candidate_1vN(event,index)` | Unique coarse candidates before exact box tests. |
| `partition_bounds(event,block=1,subdivisions=3)` | Inspection rows `[lower,upper]`; original density boxes are unchanged. |
| `index.entry_count` | Number of enclosing combinations stored in the tree. |
| `index.last_candidate_count` | Unique candidate count in the latest within-set call/batch. |

Finders can also take event collections and build the default index. Supply an explicit
index to reuse it or control parameters. Passing an existing index to `build_index`
returns it with its original settings; rebuild from events to change settings.
Pair finding uses geometry, not weights/density: zero/signed weights can make a
selected pair's integral zero. Use separate detection and full collections in identical
order to select high-density subsets, then integrate all full boxes of selected events.

The C++ code is serial and releases Python's interpreter lock. Reuse an index
sequentially: its latest-call counter is mutable and shared concurrent calls are not
supported. Empty results have shapes `(0,)` or `(0,2)` as appropriate.

## Binary format

`save_event_boxes(path,event)` writes BXOV2, without Python serialization.
`load_event_boxes(path,sort_dimensions=(0,0))` restores and sorts it. Sorting coordinates
are **not** saved in the binary format; preserve them in job settings.

The 32-byte little-endian header is `struct <8sIIIId`:

| Field | Type |
| --- | --- |
| `BXOV2` followed by three zero bytes | `char[8]` |
| `d0,n0,d1,n1` | four `uint32_t` |
| event retained probability | `float64` |

It is followed by `n0` rows of `2*d0+2` float64 values and `n1` rows of `2*d1+2`
values. Rows contain lower coordinates, upper coordinates, density, weight, in that
order. Total size is `32 + 8*(n0*(2*d0+2)+n1*(2*d1+2))` bytes. C++ readers must
interpret the byte layout rather than assuming a native padded struct.

## Connected or otherwise known matching pairs

`find_paired(events1, events2)` returns a Boolean array: element i checks only
`events1[i]` against `events2[i]`. `overlap_paired(events1, events2)` returns the
corresponding weighted integral array, including zero values. Both sets must
have equal lengths. Lists of `EventBoxes` or reusable `PreparedEvents` work.
These methods perform N comparisons for N pairs; they do not generate a cross
product and do not build a BVH or partition index.

The final geometry rule is identical to indexed finding: at least one
positive-volume box intersection in **each** block. Wall contact is excluded.
The first block is checked first. Densities and weights do not affect detection;
a pair with zero weights can pass detection but have a zero integral. Integral
calculation reuses the existing effective-weight equation and sorted box sweep.
Detection and integration stay separate so that selected detection boxes can be
used for finding, with full density boxes used for integrals.

```python
from bo import find_paired, overlap_paired, prepare_events
left = prepare_events(events1)
right = prepare_events(events2)
mask = find_paired(left, right)
values = overlap_paired(left, right)
```

For very large collections, call these functions on successive equal-length
batches. Neither function loads raw sample files.
