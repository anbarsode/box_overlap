# Probability overlap calculator: task coverage

The implemented scope uses NumPy samples, two independent probability blocks,
and unit weights for the ARAK demonstrations. Block dimensions are configurable
through `make_event_boxes(..., block_dimensions=(d0, d1))`; the default is `(3, 2)`.

| Requirement | Coverage | Implementation or evidence |
| --- | --- | --- |
| NumPy sample input, split into two independent PDFs | Complete | `bo/event_boxes.py`; configurable block dimensions |
| Random centers, coordinate midpoint separators and global sample bounds | Complete | `pick_centers`, `make_boxes` |
| Discard empty/singleton cells and shrink retained boxes | Complete | Sparse occupied-cell grouping and sample bounding boxes |
| Uniform densities, center weights and retained probability | Complete | `BoxSet`, `EventBoxes`; loss warning above 0.1% |
| Compact data readable from C++ | Complete | Documented BXOV2 binary layout; pybind11 preparation |
| Sample corner plot, selected centers and box boundaries | Complete | `validation/playground.ipynb` |
| Box density plots | Complete | Notebook: density-colored block boxes and 3D projections |
| Fast weighted overlap accumulation | Complete for stored block weights | Sorted single-threaded C++ loops; effective volume-fraction weights |
| User-selected ordering coordinate within each block | Complete | `sort_dimensions`, `sort_dimension` |
| Indexed overall event bounds with fast box checks | Complete | Both-block event-bounds tree; raw sorted box loops as requested |
| 2D block evaluated first | Complete for the default split | Second-block node bounds, intersection tests and accumulation run first |
| Separate detection boxes from integration boxes | Complete | Independent prepared events and indexes; selected-box tests |
| Within-set, one-vs-set and two-set use cases | Complete | `find_NvN`, `find_1vN`; cross-set queries reuse an index and filtered `overlap_1vN` calls |
| Four requested overlap modes | Complete | `overlap_1v1`, `overlap_NvN`, `overlap_NvN_pairs`, `overlap_1vN` |
| Two-event comparison with dense 2D and 3D histogram integrals | Complete | Notebook integral table and per-block values in intersection figures |
| Two-event intersection plots using boxes and histograms | Complete | Notebook rendered full-view and zoom figures; `plot_1v1_overlap.py` |
| 100v1 intersection plot | Complete | `pair_finder_results/100v1.jpg` and notebook |
| 100v5 intersection plots and calculations | Complete | `100v5.jpg`, `100v5.csv` and notebook |
| All-pair histogram-vs-box scatter for 100 events | Complete | All 4,950 unordered distinct pairs; figure and CSV |
| Cost plots from 2 through 10,000 events at 20 integer geometric sizes | Complete | `scaling.jpg`, `timings.csv`; generation, indexing, finding and integration measured separately |
| Single-threaded execution and gw validation | Complete | C++ serial loops; environment settings; tests and benchmarks in gw |

## Scope limits

- Bilby and Feather input adapters are outside the requested NumPy-first stage.
- The current full-event weighted integral is the product of block integrals.
  It is exact for unit weights or weights factoring across independent blocks.
  A general weight coupling both blocks requires a different representation and
  has not been implemented. The ARAK examples use the requested weight of one.
- The tree is over event bounds; individual box BVHs were replaced by sorted
  loops according to the specified design.
- Pair finding selects positive-volume geometric intersections. Zero or signed
  weights can subsequently yield zero weighted integrals for selected pairs.
- All-pair output can still grow quadratically. Repeated indexed one-event queries
  allow results to be consumed incrementally without a dense pair matrix.
- The 10,000-event timing measurements do not establish performance at 100 million
  events. Output count, density support and memory remain relevant at that scale.

## Reproduction

Use the gw environment's Python from the project root:

```bash
python -m bo.build_overlap
python validation/test_event_boxes.py
python validation/test_overlap.py
python validation/test_pair_finder.py
python validation/plot_1v1_overlap.py --idx1 66 --idx2 71 --bins 96
python validation/validate_pair_finder.py
```

The 1v1 figures form full-dimensional density products before projecting them.
Every displayed box projection preserves its exact block overlap integral.
Histogram plots approximate the original sampled density and need not agree
with the box approximation. Saved figures use JPG at 200 dpi; notebook outputs
include the rendered 1v1 images.
