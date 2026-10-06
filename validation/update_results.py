"""Refresh the single report and playground from completed measurements.

Run after validation/rerun_analysis.py. Numeric inputs are comparison_results,
pair_finder_results and review_checks.json. Earlier dataset results are not read.
The notebook chooses a nonzero pair from the new histogram comparison and is
executed to replace every saved output. --report-only leaves the notebook unchanged.
"""
from pathlib import Path
import argparse
import csv
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for setting in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(setting, '1')
import numpy as np


def load_results():
    """Require completed measurements and agreement between independent checks."""
    folder = ROOT / 'validation'
    comparison = json.loads((folder/'comparison_results/comparison.json').read_text())
    checks = json.loads((folder/'review_checks.json').read_text())
    study = json.loads((folder/'pair_finder_results/rejection_study.json').read_text())
    settings = json.loads((folder/'pair_finder_results/settings.json').read_text())
    with (folder/'pair_finder_results/timings.csv').open() as source:
        timings = list(csv.DictReader(source))
    histogram = np.genfromtxt(folder/'pair_finder_results/histogram_comparison.csv',
                             delimiter=',', names=True)
    assert comparison['status'] == settings['status'] == 'complete'
    assert checks['status'] == 'all passed' and comparison['all_pair_checks_match']
    assert comparison['input_metadata_sha256'] == checks['input_metadata_sha256']
    assert int(timings[-1]['N']) == comparison['events']
    assert int(timings[-1]['NvN_pairs']) == comparison['direct']['positive_pairs']
    eight = next(r for r in comparison['configurations'] if r['subdivisions']==[0,3])
    assert int(timings[-1]['NvN_candidates']) == eight['candidates']
    for trials in ([1,1], [5,10]):
        rows = [r for r in study['configurations'] if r['ntrial']==trials]
        assert len({r['pairs'] for r in rows}) == 1
    histogram_settings_path = folder/'pair_finder_results/histogram_settings.json'
    if histogram_settings_path.exists():
        hs = json.loads(histogram_settings_path.read_text())
        from validation.rerun_analysis import input_digest
        assert hs['status']=='complete'
        assert hs['input_metadata_sha256']==input_digest(folder/'ARAK_samples',hs['events'])
        assert len(histogram)==hs['events']*(hs['events']-1)//2
    else:
        assert len(histogram)==4950
    return comparison, checks, study, settings, timings, histogram


def update_notebook(comparison, study, histogram):
    """Replace obsolete dataset sections and rerun every notebook cell in gw."""
    import nbformat
    from nbclient import NotebookClient
    path = ROOT/'validation/playground.ipynb'
    notebook = nbformat.read(path, as_version=4)
    # The first 18 cells contain the reusable live demonstrations. The remaining
    # sections describe measured results and are rebuilt from the current files.
    notebook.cells = notebook.cells[:18]
    notebook.cells[3].source = notebook.cells[3].source.replace(
        'seed = 91\n', 'seed = 91 + idx  # Same event seed as the measured jobs.\n')
    cell = notebook.cells[13]
    lines = cell.source.splitlines()
    lines = [line for line in lines if not line.startswith('idx1, idx2 =')]
    position = next(i for i, line in enumerate(lines) if line.startswith('samples_a ='))
    chosen = int(np.argmax(histogram['box_overlap']))
    i, j = int(histogram['event_a'][chosen]), int(histogram['event_b'][chosen])
    assert histogram['box_overlap'][chosen] > 0
    lines.insert(position, f'idx1, idx2 = {i}, {j}  # Nonzero pair from the current comparison.')
    cell.source = '\n'.join(lines)
    notebook.cells.append(nbformat.v4.new_markdown_cell('''
## Current dataset: full analysis

These figures and numerical results use the freshly recreated dataset and the
same default five/ten box trials as the live examples. Colors in the 100v1 and
100v5 plots indicate intersection in both full blocks, although only 2D boxes
are displayed. Histogram comparisons include every unordered pair among 1,000 events,
using 24 bins per axis.
The legend gives degree-1 and degree-2 log fits to pairs positive in both estimates.
Each cost component uses a degree-2 log fit on 100–10,000 events; their predicted
seconds are summed for the total. Dashed curves
show the mathematical continuation up to 100 million events.
The NvN panel also shows PO2.0 (600 seconds per event plus one second per pair)
and hybrid (box total plus PO2.0 preprocessing for all events plus one second
per pair surviving box overlap). Beyond 10,000 events, hybrid holds the
measured rejection fraction at 10,000 events fixed.

![100v1](pair_finder_results/100v1.jpg)

![100v5](pair_finder_results/100v5.jpg)

![All-pair histogram comparison](pair_finder_results/histogram_comparison.jpg)

![Computational cost](pair_finder_results/scaling.jpg)

Numeric values: [timings](pair_finder_results/timings.csv),
[histogram comparisons](pair_finder_results/histogram_comparison.csv),
[100v5 integrals](pair_finder_results/100v5.csv).
'''))
    notebook.cells.append(nbformat.v4.new_markdown_cell('''
## Random trials, subdivision levels and block order

The controlled study uses 1,000 events (or the configured study count). Each
trial setting is compared at all four levels and in both orders on identical
boxes. Different trial counts can change boxes and true pair counts; changing
partitions or check order cannot change true pairs for fixed boxes.
'''))
    notebook.cells.append(nbformat.v4.new_code_cell('''
import json
import os
from IPython.display import Markdown, display

study_path = project_root / "validation/pair_finder_results/rejection_study.json"
study = json.loads(study_path.read_text())
lines = ["| Trials | 2D levels | First block | Bound rejection | Exact rejection | Find (s) |",
         "| --- | ---: | --- | ---: | ---: | ---: |"]
for row in study["configurations"]:
    first = "3D" if row["block_order"][0] == 0 else "2D"
    lines.append(f"| {tuple(row['ntrial'])} | {row['subdivisions'][1]} | {first} | "
                 f"{100*row['candidate_rejection']:.4f}% | "
                 f"{100*row['pair_rejection']:.4f}% | {row['find_s']:.4f} |")
display(Markdown("\\n".join(lines)))
'''))
    notebook.cells.append(nbformat.v4.new_markdown_cell('''
![Controlled study](pair_finder_results/rejection_study.jpg)

## One, two, four and eight enclosing groups

All four configurations share exactly the same 10,000 event boxes and a single
independent direct calculation of every pair integral. Generation and preparation
are shared costs. Enclosing groups contain whole boxes and shrink to their tight
bounds. Duplicate event candidates are consolidated before exact checks.
'''))
    notebook.cells.append(nbformat.v4.new_code_cell('''
result_path = project_root / "validation/comparison_results/comparison.json"
result = json.loads(result_path.read_text())
assert result["all_pair_checks_match"]
direct = result["direct"]
display(Markdown(f"**Zero integrals: {direct['zero_pairs']:,} / "
                 f"{direct['evaluated_pairs']:,} ({100*direct['zero_fraction']:.4f}%).**"))
base = result["configurations"][0]
lines = ["| Groups (up to) | Entries | Bounds rejection | Exact rejection | Build (s) | "
         "Find (s) | Change vs one group |",
         "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
for row in result["configurations"]:
    change = 100*(row["find_s"]/base["find_s"]-1)
    lines.append(f"| {2**row['subdivisions'][1]} | {row['index_entries']:,} | "
                 f"{100*row['bounds_rejection']:.4f}% | "
                 f"{100*row['exact_rejection']:.4f}% | {row['index_build_s']:.4f} | "
                 f"{row['find_s']:.3f} | {change:+.2f}% |")
display(Markdown("\\n".join(lines)))
'''))
    notebook.cells.append(nbformat.v4.new_markdown_cell('''
![Partition comparison](comparison_results/comparison.jpg)

Partitioning trades tighter geometric rejection against tree traversal and
consolidation work. Exact pairs and integrals remain invariant. Single-job wall
times are subject to host/cache variation; they do not establish a universal
optimal group count. Earlier dataset numbers are superseded by these results.

[Consolidated report](report.md) · [Run instructions](../RUNNING.md)
'''))
    started = time.perf_counter()
    NotebookClient(notebook, timeout=180, kernel_name='python3',
                   resources={'metadata': {'path': str(ROOT)}}).execute()
    nbformat.write(notebook, path)
    return time.perf_counter()-started, (i, j)


def write_report(result, checks, study, settings, timings, histogram):
    """Replace all earlier numeric sections with this dataset's measurements."""
    folder = ROOT/'validation'
    path = folder/'report.md'
    old = path.read_text()
    # Retain the standalone scientific equations and definitions, never an old
    # timing/rejection section. The marker also makes repeat updates idempotent.
    marker = '<!-- measured-results -->'
    prefix = old.split(marker)[0].split('## Experiment history and input provenance')[0]
    n = result['events']; possible = result['possible_pairs']; direct = result['direct']
    base = result['configurations'][0]
    fastest = min(result['configurations'], key=lambda r: r['find_s']+r['index_build_s'])
    lines = [prefix.rstrip(), '', marker, '', '## Dataset and measurement settings', '',
        f'The measurements use **{n:,} freshly recreated events**, each containing '
        '10,000 joint samples in five coordinates. All reported costs and efficiencies '
        'refer to this input set. Earlier dataset measurements are superseded.', '',
        'Calculations use `gw`, one thread, box centers `(4,8)`, trials `(5,10)`, '
        'block dimensions `(3,2)`, sorting coordinates `(0,0)`, seeds `91 + event ID`, '
        'and 3D-first detection. Pair batches cover 64 first-event indices. The full '
        'scaling analysis uses `(0,3)` subdivision levels; the controlled comparison '
        'uses levels zero through three on identical prepared boxes.', '',
        'Filename, size and modification-time metadata are hashed before and after '
        'the complete analysis; their digests match. This identifies the input state '
        'and detects changed files during the jobs, but is not a hash of sample contents.', '',
        f'Input metadata SHA256: `{checks["input_metadata_sha256"]}`.', '',
        'Jobs run sequentially to avoid competing analysis jobs. Wall times depend on '
        'host activity and cache state. Generation excludes file opening but includes '
        'mapped sample reads; preparation, tree construction, finding and integration '
        'are timed separately. Finding includes duplicate consolidation and pair-array '
        'creation, and excludes digest work.', '', '## Direct all-pair check and rejection', '',
        f'Every one of the **{possible:,}** unordered distinct pairs was integrated '
        'directly, independently of the index. All four exact finder outputs match '
        'the positive-integral pairs by count and by an ordered-pair SHA256 digest.', '',
        '| Integral class | Pairs |', '| --- | ---: |']
    for key, label in [('zero_pairs','Zero'),('positive_pairs','Positive'),
                       ('negative_pairs','Negative'),('nonfinite_pairs','Nonfinite')]:
        lines.append(f'| {label} | {direct[key]:,} |')
    lines += ['', f'The actual zero-integral fraction is **{100*direct["zero_fraction"]:.6f}%**. '
        'It is the lossless rejection ceiling for these positive unit-weight density boxes.', '',
        '| Maximum 2D groups | Entries | Unique candidates | Bound rejection | '
        'Exact rejection | False candidates | Zero pairs removed by bounds |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in result['configurations']:
        fraction = row['fraction_zero_rejected_by_bounds']
        fraction_text = 'undefined' if fraction is None else f'{100*fraction:.4f}%'
        lines.append(f'| {2**row["subdivisions"][1]} | {row["index_entries"]:,} | '
            f'{row["candidates"]:,} | {100*row["bounds_rejection"]:.4f}% | '
            f'{100*row["exact_rejection"]:.4f}% | {row["false_candidates"]:,} | {fraction_text} |')
    lines += ['', '### Partition cost on identical boxes', '',
        '| Maximum groups | Build (s) | Find (s) | Build + find (s) | Find change vs one group |',
        '| ---: | ---: | ---: | ---: | ---: |']
    for row in result['configurations']:
        change = 100*(row['find_s']/base['find_s']-1)
        lines.append(f'| {2**row["subdivisions"][1]} | {row["index_build_s"]:.4f} | '
            f'{row["find_s"]:.4f} | {row["index_build_s"]+row["find_s"]:.4f} | {change:+.2f}% |')
    lines += ['', f'The lowest measured build-plus-finding cost uses **up to '
        f'{2**fastest["subdivisions"][1]} groups**. This is one measurement per setting, '
        'not a statistically established optimum. More groups improve enclosing-bound '
        'rejection while adding traversal and consolidation work. Final pair counts '
        'are unchanged. Groups that cannot split produce fewer than the maximum entries.', '',
        f'Shared box generation took **{result["generation_s"]:.3f} s** and C++ preparation '
        f'**{result["preparation_s"]:.3f} s**. Independently evaluating all pair integrals '
        f'took **{direct["overlap_s"]:.3f} s**; pair creation took '
        f'{direct["pair_creation_s"]:.3f} s and counting/digest work '
        f'{direct["count_and_digest_s"]:.3f} s.', '',
        '![Rejection and finding cost](comparison_results/comparison.jpg)', '',
        '## Full scaling, filtered integration and one-vs-set costs', '',
        f'The analysis measured **{len(timings)}** distinct integer geometric sizes '
        f'from {timings[0]["N"]} through {timings[-1]["N"]} events. At the endpoint:', '',
        '| Stage | Wall time (s) |', '| --- | ---: |']
    last = timings[-1]
    for key, label in [('box_generation_s','Generate boxes'),
                       ('preparation_s','Prepare C++ snapshots'),
                       ('index_build_s','Build eight-group index'),
                       ('find_NvN_s','Find within-set pairs'),
                       ('overlap_NvN_s','Integrate filtered within-set pairs'),
                       ('find_1vN_s','Find one-vs-set matches'),
                       ('overlap_1vN_s','Integrate filtered one-vs-set matches')]:
        lines.append(f'| {label} | {float(last[key]):.6f} |')
    lines += ['', f'There are **{int(last["NvN_pairs"]):,}** exact within-set matches and '
        f'**{int(last["1vN_hits"]):,}** matches for the final-event query (self match removed). '
        f'Bound rejection is {100*float(last["candidate_rejection"]):.4f}% and exact '
        f'rejection is {100*float(last["pair_rejection"]):.4f}%.', '',
        'Box-generation curves sum event times; indexing and query stages are measured '
        'on each prefix. The comparison and scaling jobs separately regenerate boxes '
        'using the same settings; their generation costs need not agree because of '
        'cache and timing variation. Neither job saves every pair integral to disk, '
        'so production output-writing costs are not included.', '',
        '![Cost scaling](pair_finder_results/scaling.jpg)', '',
        '## Controlled trial and check-order study', '',
        f'The controlled comparison uses **{study["configurations"][0]["events"]:,}** '
        'events. For each trial setting, the same boxes are tested with all four '
        'levels and both check orders. Different trial counts can change boxes; '
        'partitions and order must preserve exact pairs.', '',
        '| Trials (3D,2D) | 2D levels | First block | Bound rejection | '
        'Exact rejection | Build (s) | Find (s) |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
    for row in study['configurations']:
        first = '3D' if row['block_order'][0] == 0 else '2D'
        lines.append(f'| {tuple(row["ntrial"])} | {row["subdivisions"][1]} | {first} | '
            f'{100*row["candidate_rejection"]:.4f}% | {100*row["pair_rejection"]:.4f}% | '
            f'{row["index_build_s"]:.5f} | {row["find_s"]:.5f} |')
    trial_rows = [r for r in study['configurations']
                  if r['subdivisions']==[0,0] and r['block_order']==[0,1]]
    lines += ['', 'Trial-generation job costs (including file opening and preparation):', '',
              '| Trials | Job time (s) | Mean retained probability | Minimum |',
              '| --- | ---: | ---: | ---: |']
    for row in trial_rows:
        lines.append(f'| {tuple(row["ntrial"])} | {row["generation_s"]:.4f} | '
                     f'{row["retained_probability_mean"]:.9f} | '
                     f'{row["retained_probability_min"]:.9f} |')
    lines += ['', 'Multi-trial/single-trial summed-volume ratios:', '',
              '| Block | Median | Mean |', '| --- | ---: | ---: |']
    for i, label in enumerate(('3D','2D')):
        lines.append(f'| {label} | {study["volume_ratio_median_per_block"][i]:.6f} | '
                     f'{study["volume_ratio_mean_per_block"][i]:.6f} |')
    lines += ['', 'The second block uses a shared generator after first-block trials; '
        'its single-trial center choice need not be among its multi-trial choices. '
        'Ratios quantify geometric support, not approximation error.', '',
        '![Controlled study](pair_finder_results/rejection_study.jpg)', '',
        '## Histograms, retained probability and visual checks', '']
    box = histogram['box_overlap']; dense = histogram['histogram_overlap']
    hs = json.loads((folder/'pair_finder_results/histogram_settings.json').read_text())
    fits = json.loads((folder/'pair_finder_results/histogram_comparison_fits.json').read_text())
    lines += [f'All **{len(histogram):,}** unordered pairs among {hs["events"]:,} events '
        f'were compared using **{hs["bins"]} bins per coordinate** on pair-shared grids.', '',
        'Histogram counts are computed by a serial C++ validation helper, checked '
        'against NumPy histogramdd including internal/outer bin edges. Disjoint '
        'histogram-bin ranges prove a zero sum and avoid opening unnecessary sample '
        'files. Disjoint sample bounds alone are not used to reject histogram sums.', '',
        f'Counting time is **{hs["histogram_s"]:.3f} s**; '
        f'{hs["explicitly_counted_pairs"]:,} pairs require explicit counts and '
        f'{hs["exact_zero_bin_rejections"]:,} are proven zero from their bin ranges.', '',
        '| Pair category | Count |', '| --- | ---: |',
        f'| Zero box integral | {np.count_nonzero(box==0):,} |',
        f'| Zero histogram integral | {np.count_nonzero(dense==0):,} |',
        f'| Positive box, zero histogram | {np.count_nonzero((box>0)&(dense==0)):,} |',
        f'| Zero box, positive histogram | {np.count_nonzero((box==0)&(dense>0)):,} |', '',
        'Histograms estimate sampled densities and are not ground truth for the box '
        'model. Bin width, finite samples and gaps inside uniform boxes affect '
        'agreement. Fine histograms may be more sparse. Intersection maps multiply '
        'full-dimensional densities before projecting, preserving the block integral.', '',
        f'Mean retained event probability is **{result["retained_probability_mean"]:.12f}**, '
        f'minimum **{result["retained_probability_min"]:.12f}**. '
        f'**{result["events_losing_over_point_one_percent"]}** events lose more than '
        '0.1%. Loss is recorded and not renormalized. Timing jobs count and suppress '
        'repetitive warnings; the generator still warns in ordinary use.', '',
        '![Histogram comparison](pair_finder_results/histogram_comparison.jpg)', '',
        'The executed [playground](playground.ipynb) contains all sample, density, trial, '
        'partition, 1v1 box/histogram, 100v1, 100v5 and computational-cost figures. '
        'Every saved notebook output is refreshed using this dataset.', '',
        ]
    lines += ['', '## Fitted histogram-to-box relations', '',
        f'Both fits use **{fits["fit_pairs"]:,}** pairs positive in both estimates. '
        'Set x to the histogram integral and y to the box integral. NumPy polyfit '
        'minimizes squared vertical residuals of log10(y) against log10(x); zero '
        'estimates stay in the scatter but are excluded from these logarithmic fits.', '',
        '| Degree | Coefficients, descending powers of log10(x) | RMS residual (dex) |',
        '| ---: | --- | ---: |']
    for fit in fits['fits']:
        coefficients=', '.join(f'{c:.9g}' for c in fit['coefficients'])
        lines.append(f'| {fit["degree"]} | {coefficients} | {fit["rms_log10_residual"]:.6f} |')
    b,c=fits['fits'][0]['coefficients'];a,b2,c2=fits['fits'][1]['coefficients']
    lines += ['', f'Degree 1: $y=10^{{{c:.6f}}}x^{{{b:.6f}}}$.', '',
        f'Degree 2: $\\log_{{10}}y={a:.6f}(\\log_{{10}}x)^2'
        f'{b2:+.6f}\\log_{{10}}x{c2:+.6f}$.', '',
        'Degree 1 is a power law; degree 2 describes curvature in log space and '
        'is not a single constant-exponent power law. Fits are drawn within the '
        'observed positive histogram range, alongside y=x, with equations in the legend.', '',
        '## Total computational cost and extrapolation', '',
        'For each panel, total cost sums generation, C++ preparation, index build, '
        'pair finding and filtered integration. One-vs-set includes building the '
        'N-event dataset/index and one query. NumPy polyfit degree 2 fits '
        'log10(seconds) for each component against log10(N) on 100 <= N <= 10,000. '
        'The total is the sum of component predictions in seconds. Solid curves '
        'cover the fitted range; dashed curves extend from 10,000 to 100 million.', '']
    extrapolation=json.loads((folder/'pair_finder_results/cost_extrapolation.json').read_text())
    lines += ['| Case | Component | a | b | c |', '| --- | --- | ---: | ---: | ---: |']
    for panel in extrapolation['panels']:
        for component in panel['components']:
            a,b,c=component['coefficients']
            lines.append(f'| {panel["title"]} | {component["label"]} | '
                         f'{a:.9g} | {b:.9g} | {c:.9g} |')
    lines += ['', 'Component model: log10(seconds) = a*log10(N)^2+b*log10(N)+c. '
              'Total = sum of component seconds; no polynomial is fitted to the total.', '',
              '| N | Within-set fit/extrapolation (s) | One-vs-set fit/extrapolation (s) |',
              '| ---: | ---: | ---: |']
    for n in (10000,100000,1000000,10000000,100000000):
        values=[p['predicted_seconds'][str(n)] for p in extrapolation['panels']]
        lines.append(f'| {n:,} | {values[0]:.6g} | {values[1]:.6g} |')
    comparison = extrapolation['panels'][0]['comparison']
    lines += ['', '### PO2.0 and hybrid comparison', '',
        'With M=N(N-1)/2 unordered distinct pairs, PO2.0 costs 600N+M seconds. '
        'Hybrid costs T_box(N)+600N+S(N) seconds, where S is the number of pairs '
        'surviving exact box overlap. PO2.0 preprocessing covers every event in both cases.', '',
        f'For measured datasets, S is the measured positive box pair count. '
        f'Beyond 10,000 events, the rejection fraction 1-S/M is held fixed at '
        f'{comparison["extrapolated_rejection_fraction"]:.9%}, the largest measured '
        'dataset fraction. These curves use assumed PO2.0 costs, not PO2.0 timing measurements.', '',
        '| N | PO2.0 (s) | hybrid (s) |', '| ---: | ---: | ---: |']
    for n in (10000,100000,1000000,10000000,100000000):
        lines.append(f'| {n:,} | {comparison["PO2.0_predicted_seconds"][str(n)]:.6g} | '
                     f'{comparison["hybrid_predicted_seconds"][str(n)]:.6g} |')
    lines += ['', 'Component predictions at 100 million events:', '',
              '| Component | Within-set (s) | One-vs-set (s) |',
              '| --- | ---: | ---: |']
    for within, query in zip(*(panel['components'] for panel in extrapolation['panels'])):
        lines.append(f'| {within["label"]} | '
                     f'{within["predicted_seconds"]["100000000"]:.6g} | '
                     f'{query["predicted_seconds"]["100000000"]:.6g} |')
    lines += ['', 'Values beyond 10,000 events are mathematical extrapolations, '
        'not measured or validated performance. Although generation dominates the measured total, '
        'the faster-growing pair finding and integration stages can dominate at large N. '
        'Their individual fits preserve this change in the summed prediction. '
        'Memory, output storage and pair fraction can also change.', '']
    lines += ['', '## Correctness tests', '', '| Script | Status | Job wall time (s) |',
              '| --- | --- | ---: |']
    for job in checks['jobs']:
        command = job['command']
        if len(command)>1 and Path(command[1]).name.startswith('test_'):
            lines.append(f'| {Path(command[1]).name} | Passed | {job["wall_s"]:.3f} |')
    lines += ['', 'Independent checks cover the full Cartesian separator procedure, '
        'disjointness, singleton removal, trial winners, probability warnings, '
        'binary round trips, alternate block dimensions, nonmonotonic sorted-loop '
        'upper bounds, boundary contact, effective weights, all four integral '
        'interfaces, partition cuts, consolidation, both check orders, selected '
        'detection boxes, empty inputs, bounded raw-sample loading, saved jobs and '
        'original event-ID mapping. Synthetic loss warnings are expected.', '',
        '## Memory and scope', '',
        'Samples are processed one event at a time; pairwise histograms keep at most '
        'two sample maps. Prepared boxes and the index remain resident. Pair arrays '
        'are consumed in batches for large within-set measurements. The 1,000-event '
        'histogram comparison retains its compact pair/value arrays, not raw samples. '
        'Final indices use '
        '16 bytes per pair and values eight more; conversion can temporarily hold '
        'both C++ and NumPy index copies. Output can still grow quadratically.', '',
        'The measured timings cover the available dataset; no measured '
        '32,016- or 100-million-event timing is claimed. Dashed curves are explicitly '
        'extrapolated. Unit or separable weights '
        'are supported; general weights coupling blocks require another '
        'representation. Bilby/Feather adapters and distributed indexes are outside '
        'this implementation. Index calls are serial; shared concurrent access to '
        'the latest-call candidate counter is not supported.', '',
        'Plots use JPG at 200 dpi. Install `dvipng` and a Times font separately for '
        'the requested LaTeX/font style; missing components use the documented fallback.', '',
        '## Reproduction and result files', '',
        'Run from the project root with the `gw` environment:', '', '```bash',
        'python validation/rerun_analysis.py --events 10000 --study-events 1000',
        '```', '',
        'The [run guide](../RUNNING.md) explains production box generation, saved-box '
        'reuse, overlap batches and all settings. The [API documentation](../bo/README.md) '
        'describes algorithms and the binary format.', '',
        'Measurements: `comparison_results/comparison.json`, `comparison.csv`, '
        '`direct_batches.csv`; `pair_finder_results/timings.csv`, `settings.json`, '
        '`rejection_study.json`, `histogram_comparison.csv`, `100v5.csv`. '
        'Test commands, exit codes, input/source hashes and job times are in '
        '`review_checks.json`; complete outputs are in `analysis_logs/`.', '']
    path.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    result, checks, study, settings, timings, histogram = load_results()
    if not args.report_only:
        elapsed, pair = update_notebook(result, study, histogram)
        checks['notebook'] = {'status':'executed without cell errors', 'wall_s':elapsed,
                              'selected_pair':list(pair)}
        (ROOT/'validation/review_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    write_report(result, checks, study, settings, timings, histogram)
    print('Updated the consolidated report' + ('' if args.report_only else ' and playground'),
          flush=True)


if __name__ == '__main__':
    main()
