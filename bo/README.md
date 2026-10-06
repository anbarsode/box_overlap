# Event probability boxes

`event_boxes.py` approximates sampled probability densities with disjoint,
axis-aligned boxes. It currently treats the first three columns and last two
columns of an event's samples as independent probability blocks.

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
