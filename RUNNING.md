# Running the probability overlap calculator

Run commands from `/home/ankur/projects/box_overlap`. Dataset creation is separate:
provide `event_0.npy`, `event_1.npy`, …, each containing an `(n_samples, 5)` array.
Finish dataset generation before processing it, so files remain unchanged during a job.
The default split treats the first three and final two columns as independent densities.

## 1. Activate the environment and build

```bash
micromamba activate gw
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m bo.build_overlap
```

If activation is unavailable, replace `python` with
`/home/ankur/micromamba/envs/gw/bin/python`. The build uses installed pybind11 and
an installed C++17 compiler. Rebuild after changing C++ source or the Python environment.
The implementation defaults to one thread; these exports also cover programs importing
NumPy before `bo`. No additional packages are required for the calculator.

## 2. Generate and save boxes

This command reads one sample file at a time and saves compact boxes for later runs:

```bash
python -m bo.run boxes \
    --samples validation/ARAK_samples \
    --output event_boxes \
    --start 0 \
    --events 10000 \
    --block-dimensions 3 2 \
    --n-centers 4 8 \
    --ntrial 5 10 \
    --sort-dimensions 0 0 \
    --seed 91
```

Use a fresh output folder for each dataset or box configuration. Jobs refuse folders
containing their previous box/batch files, preventing mixed results. The job records
settings, sample source, per-event box counts, retained probability and generation times.
Sample arrays are opened read-only as memory maps and released after each event.
Temporary sorting arrays still use memory proportional to one event's sample count.

| Parameter | Default | Meaning and useful alternatives |
| --- | --- | --- |
| `--samples` | `validation/ARAK_samples` | Input folder containing joint samples. |
| `--output` | `event_boxes` | Folder for `event_ID.boxes`, `events.csv`, `settings.json`. |
| `--start` | `0` | First event ID; the event seed still uses the original ID. |
| `--events` | all IDs ≥ start | With an explicit count, process a contiguous range. Use `100` for a small demonstration. |
| `--block-dimensions` | `3 2` | Consecutive column counts in the two blocks; must sum to the input column count. |
| `--n-centers` | `4 8` | Random selected samples per block, not final box counts. Each count must be between one and the sample count. More centers can improve resolution but create more singleton cells and more integration work. |
| `--ntrial` | dimension-dependent | Five trials in 3D, ten in 2D, one otherwise. `1 1` gives the single-trial method. The minimum sum of shrunken box volumes wins. |
| `--sort-dimensions` | `0 0` | Zero-based coordinate **within** each block for sorted overlap loops. For a 3D+2D split, valid values are 0–2 and 0–1. |
| `--seed` | `91` | Event ID `i` uses `default_rng(seed+i)`. Repeating the same inputs/settings gives the same boxes. Changing first-block trials also changes the second-block random sequence. |

Weights are one in this job. For custom block weights, use `make_box_trials` or
`make_boxes` with `weight_function`; see [the algorithm documentation](bo/README.md).
Singleton probability is discarded without renormalization. A warning is raised above
0.1% loss and the event's retained probability is saved. Zero-volume boxes deliberately
produce inf/NaN to expose data problems.

**Files:** one `event_ID.boxes` file contains both blocks and retained probability.
The BXOV2 format is documented in [bo/README.md](bo/README.md).
The generation job's `events.csv` provides an easily inspected summary.

## 3. Find pairs and calculate their integrals

Reuse the saved boxes; this stage never opens raw samples:

```bash
python -m bo.run overlaps \
    --boxes event_boxes \
    --output overlap_results_four_groups \
    --start 0 \
    --events 10000 \
    --sort-dimensions 0 0 \
    --subdivisions 0 2 \
    --block-order 0 1 \
    --batch-events 64
```

Four 2D enclosing groups are a useful configuration for the regenerated dataset.
The API and command default remain `--subdivisions 0 3` (up to eight groups).
The experiments do not establish one best setting for every density distribution.
To compare configurations, reuse `event_boxes` and change only the subdivision levels
and the overlap output folder:

```bash
python -m bo.run overlaps --boxes event_boxes --events 10000 \
    --subdivisions 0 0 --output overlap_results_one_group
python -m bo.run overlaps --boxes event_boxes --events 10000 \
    --subdivisions 0 1 --output overlap_results_two_groups
python -m bo.run overlaps --boxes event_boxes --events 10000 \
    --subdivisions 0 3 --output overlap_results_eight_groups
```

| Parameter | Default | Meaning and useful alternatives |
| --- | --- | --- |
| `--boxes` | `event_boxes` | Folder containing saved `.boxes` files. |
| `--output` | `overlap_results` | Fresh folder for numeric batches and settings/timings. |
| `--start`, `--events` | `0`, all IDs ≥ start | Select saved event IDs, using numeric order. |
| `--sort-dimensions` | `0 0` | Order restored when loading; must match across compared events. Binary files do not record this setting. |
| `--subdivisions` | `0 3` | Binary split **levels**, independently per block. Levels 0, 1, 2, 3 yield up to 1, 2, 4, 8 enclosing groups. Small groups may stop splitting. Subdividing both blocks multiplies the tree entry count. |
| `--block-order` | `0 1` | Check the first block (normally 3D) first. `1 0` reverses detection order for comparisons; integrals always check block 0 first. |
| `--batch-events` | `64` | First-event indices per batch, not output pair count. Try `8` or `16` for lower pair memory; larger values reduce calls/file count. |

Only compact prepared boxes and the index stay resident. The index uses enclosing
partitions; it checks exact box geometry before returning pairs. Each unordered pair
appears once with `i < j`; self pairs are excluded. Densities and weights are used
when calculating the integral, after geometry selection.

### Read the results

```python
from pathlib import Path
import numpy as np

folder = Path('overlap_results_four_groups')
event_ids = np.load(folder / 'event_ids.npy')
for path in sorted(folder.glob('batch_*.npz')):
    with np.load(path) as result:
        pairs = result['pairs']       # (K,2), positions in event_ids
        values = result['overlap']    # (K,), same order
        original_ids = event_ids[pairs]
        selected = values > 0.01      # replace with a scientifically appropriate threshold
        # Save or analyze original_ids[selected], values[selected] here.
```

Do not concatenate all batches for large datasets. Output storage is 24 bytes per
pair before file overhead: two int64 indices and one float64 integral. Output size
can grow quadratically even with efficient rejection. A batch's final pair array
uses 16 bytes per pair, plus values and temporary C++ output; during conversion
there can be two pair copies. Reducing `batch_events` controls this memory.

`settings.json` records loading/preparation, index building, finding, integration,
saving and total wall times separately, plus pair/candidate counts. This differs
from historical scaling scripts that discarded pair values instead of saving them.

## 4. One-vs-one, one-vs-set, and two sets

```python
from pathlib import Path
from bo import (load_event_boxes, prepare_event, prepare_events, build_index,
                find_1vN, overlap_1v1, overlap_1vN, overlap_NvN_pairs)

folder = Path('event_boxes')
other = [prepare_event(load_event_boxes(folder / f'event_{i}.boxes',
                                       sort_dimensions=(0, 0))) for i in range(100)]
collection = prepare_events(other)
index = build_index(collection, subdivisions=(0, 2), block_order=(0, 1))
query = prepare_event(load_event_boxes(folder / 'event_100.boxes'))
print(overlap_1v1(query, other[0]))
hits = find_1vN(query, index)
values = overlap_1vN(query, [other[j] for j in hits])
```

For two different event sets, build the index over the second set once, then loop
through the first set, query it and call filtered `overlap_1vN` as above. Save each
query's original ID together with `hits` and `values` before continuing. A query
already present in the index can match itself; remove that index if appropriate.

`overlap_NvN_pairs(collection, pairs)` calculates explicitly supplied integer pairs
in their given order. `overlap_NvN(collection)` makes a dense matrix including self
overlaps and evaluates zero pairs too; its output requires `8*N*N` bytes.

### Selected detection boxes

Build an index from selected high-density boxes while keeping full boxes in
`collection` for integration. Detection and full collections must have identical
event ordering and dimensions. Apply the same selection rule to query boxes.
This intentionally omits some true pairs; it changes selection completeness.

```python
from bo import BoxSet, EventBoxes

def select_detection(event, fraction=0.25):
    blocks = []
    for block in event.blocks:
        order = np.argsort(block.density)[::-1]
        keep = order[:max(1, int(np.ceil(fraction * block.n_boxes)))]
        blocks.append(BoxSet(block.lower[keep], block.upper[keep],
                             block.density[keep], block.weight[keep], block.sort_dimension))
    # This is metadata for the subset, never a normalization factor.
    mass = blocks[0].retained_probability * blocks[1].retained_probability
    return EventBoxes(tuple(blocks), mass)
```

Prepare full boxes once and use finder indices with the full collection. For empty
blocks, this example naturally keeps an empty selection.

## 5. Tests and demonstrations

```bash
python validation/test_event_boxes.py
python validation/test_overlap.py
python validation/test_pair_finder.py
python validation/test_sample_streaming.py
python validation/test_run.py
python -m validation.histogram_overlap
python validation/test_histogram.py
```

Expected warnings in synthetic tests exercise probability loss and invalid-volume
behavior. The tests use independent full-grid, geometry and overlap calculations.
Open **[validation/playground.ipynb](validation/playground.ipynb)** for all plots and
experiments, including the combined 1/2/4/8-group comparisons. Run its cells for
current sample files; historical sections identify their different input datasets.
The notebook uses only selected raw events, not the full raw dataset in memory.

Scientific results, definitions, test coverage and audit findings are consolidated
in **[validation/report.md](validation/report.md)**.

## 6. Rerun all tests and scientific measurements

After finishing dataset creation, run one command from the project root:

```bash
python validation/rerun_analysis.py --events 10000 --study-events 1000
```

This rebuilds the extension, runs the correctness scripts and the NumPy histogram-reference checks, compares trial
counts and block orders, compares 1/2/4/8 groups on identical boxes, independently
calculates every unordered pair, and regenerates the 1,000-event/24-bin histogram, examples, and scaling
measurements. Jobs run sequentially so timing is not distorted by competing
analysis jobs. All notebook outputs and the single report are refreshed at the end.
The metadata digest before and after checks that samples were not changed.

`--events` sets the analysis endpoint; there must be at least 105 events for
fixed examples. `--study-events` sets the trial/order study size. The current
dataset has 10,000 events. A larger endpoint requires a correspondingly complete
dataset; there is no automatic 95% stopping rule in the scaling script.

The consolidated numeric comparison is in `validation/comparison_results`.
Plots, scaling and histogram data are in `validation/pair_finder_results`.
Complete job outputs are saved in `validation/analysis_logs`; commands, timings,
exit codes and input/source hashes are in `validation/review_checks.json`.
These calculations overwrite the main result files with the current dataset.

To refresh just the report from completed jobs, without calculations:

```bash
python validation/update_results.py --report-only
```

To refresh both report and notebook from completed jobs:

```bash
python validation/update_results.py
```

### Individual analyses

```bash
python validation/validate_pair_finder.py --max-events 10000 \
    --ntrial 5 10 --subdivisions 0 3 --batch-events 64 --skip-histograms
python validation/rejection_study.py --events 1000
python validation/compare_rejection.py --events 10000 --compare-levels 1 2 3 \
    --ntrial 5 10 --batch-events 64 --output validation/comparison_results
python -m validation.histogram_overlap
python validation/histogram_comparison.py --events 1000 --bins 24
python validation/summarize_cost.py
```

`--timings-only` on the scaling script skips example plots and histogram comparisons;
explicitly set `--max-events`, since omitting it processes all available events.
`--compare-levels 1 2 3` generates boxes once for all group choices and performs
one independent direct all-pair calculation. For a single partition setting use
`--partition-levels 2`, which compares one group against up to four groups.

Individual analyses do not automatically refresh the single report. Use the full
rerun command for a complete and consistent refresh. Historical result folders
from older datasets are not used by the consolidated report or notebook.

For repeated production overlap configurations, use saved boxes and steps 2–3
to avoid repeating box generation. Histogram grids grow as `bins**3` in 3D and
can become sparse; they are not ground truth for the box approximation.
Saved figures use JPG at 200 dpi. Missing LaTeX components or Times fonts print
installation guidance; install them separately.

### Histogram fits and total-cost extrapolation

`histogram_comparison.py` compares all unordered pairs among 1,000 events using
24 bins per coordinate (half the previous 48). It uses exact dense histogram
counts on pair-shared NumPy edges. Proven disjoint bin ranges skip sample reads
without changing the histogram sum; all pairs appear in the output CSV.
The serial helper is compiled with the installed pybind11/C++17 tools and checked
against NumPy, including samples on/next to edges and shared coarse bins.

Fits use pairs positive in both estimates, with histogram integral x and box
integral y. `np.polyfit(log10(x), log10(y), degree)` uses degree 1;
zero estimates remain visible in the plot but are excluded from logarithms.
Separate linear power laws describe the ARAK and CAMP points.
Coefficients and residuals are saved in `histogram_comparison_fits.json`.

`summarize_cost.py` redraws the saved cost measurements and fits each of the five
stages separately: generation, C++ preparation, index construction, pair finding,
and integration. Each fit uses `np.polyfit(log10(N), log10(stage_seconds), 2)`
for 100 <= N <= 10,000. Component curves extend to 100 million events with dashed
lines. The total is the sum of the predicted component times in seconds, so a
currently small but quadratic stage can dominate at large N. No fit is made to
the measured total. `cost_extrapolation.json` saves each component's coefficients,
residuals and predictions, together with summed predictions for both query modes.
These are mathematical continuations, not measured or validated large-dataset costs.

The NvN panel also compares `PO2.0` and `hybrid`, assuming 600 seconds of PO2.0
preprocessing per event and 1 second per unordered distinct pair. PO2.0 uses all
N(N-1)/2 pairs. Hybrid adds the box total, PO2.0 preprocessing for all events,
and PO2.0 pair calculations on pairs **surviving box overlap**. The measured
surviving count is `NvN_pairs`; extrapolation holds the largest
measured dataset's rejection fraction fixed. Comparison assumptions and values
are saved under `comparison` in the NvN panel of `cost_extrapolation.json`.

## CAMP: connected pairs

CAMP draws two anchors from a shared reference product PDF, then independently
creates two final product PDFs around those anchors, using the same 3D/2D block
settings as ARAK. Connected PDFs can still have zero intersections in their
finite-sample histogram or shrunken box approximations.

From the project root, using the `gw` environment:

```bash
python -m bo.build_overlap
python validation/create_CAMP_samples.py \
    --n-pairs 1000 --n-samples 10000 --n-plots 100 --seed 20261007 \
    --output validation/CAMP_samples
python validation/test_paired.py
python validation/camp_comparison.py \
    --pairs 1000 --bins 24 \
    --samples validation/CAMP_samples --output validation/CAMP_results
python validation/update_results.py
```

Generation writes `event_{i}_0.npy` and `event_{i}_1.npy`, with i=0,...,999.
`event_{i}.jpg` overlays both members for the first `--n-plots` pairs; ranges
are shared and colors identify the members. Each file contains an `(10000,5)`
array by default. A failed stochastic construction retries that pair.
Generation holds just two raw arrays at a time. `--n-plots 0` skips plotting.

The comparison also reads only one pair at a time. It builds boxes using the
usual `(4,8)` centers, `(5,10)` trials, ordering axes `(0,0)`, blocks `(3,2)`,
and box seed `91+2*i+member`. Compact boxes are saved in `CAMP_results/boxes`.
No all-event BVH or all-possible-pairs calculation is needed. Results go to
`CAMP_results/histogram_comparison.csv` and `comparison.json`.

The shared ARAK/CAMP scatterplot remains
`pair_finder_results/histogram_comparison.jpg`. It shows separate degree-1
power laws fitted to positive histogram and box integrals, together with the
equality line. Zeros are shown but excluded from logarithmic fits. Both datasets
use 24 histogram bins per axis; use the same `--bins` in each comparison.
The quadratic scatter fit is removed. Computational-cost fits remain unchanged.

For saved boxes, use the matching-pair APIs directly:

```python
from bo import load_event_boxes, find_paired, overlap_paired
from pathlib import Path
folder = Path('validation/CAMP_results/boxes')
events1 = [load_event_boxes(folder/f'event_{i}_0.boxes') for i in range(1000)]
events2 = [load_event_boxes(folder/f'event_{i}_1.boxes') for i in range(1000)]
mask = find_paired(events1, events2)  # same exact geometry as indexed finding
values = overlap_paired(events1, events2)  # values[i] compares matching members
```

Use smaller batches of saved boxes if desired. Supplying selected detection
boxes to `find_paired` does not change which full boxes should be supplied to
`overlap_paired`. General weights coupling the two blocks retain the limitations
explained in `bo/README.md`.
