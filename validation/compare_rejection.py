"""Compare partition rejection with directly measured zero box integrals.

Generate each event's boxes once, using five 3D and ten 2D trials by default.
The same prepared boxes are indexed without partitions and with the requested
2D subdivision levels. --compare-levels 1 2 3 compares all group choices in one job.
All configurations check 3D before 2D.

All unordered event pairs are also passed directly to the overlap calculator,
independently of either index. This establishes the zero-integral fraction
without assuming that a geometric intersection has a positive integral.
Raw samples are read one event at a time. Pair arrays are consumed in bounded
batches and released; no dense N-by-N matrix or list of all samples is created.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import sys
import time
import warnings
for setting in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(setting, '1')
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bo import (make_event_boxes, prepare_event, prepare_events, build_index,
                find_NvN_batches, overlap_NvN_pairs)
from validation.plot_style import plot_style


def all_pair_batches(n: int, batch_events: int):
    """Yield every i<j once, ordered by i then j, without a quadratic output.

    Each batch covers at most batch_events first-event indices and uses at
    most 16*batch_events*n bytes for its two int64 index columns.
    """
    for begin in range(0, n, batch_events):
        end = min(begin+batch_events, n)
        length = sum(n-i-1 for i in range(begin, end))
        pairs = np.empty((length, 2), dtype=np.int64)
        offset = 0
        for i in range(begin, end):
            count = n-i-1
            pairs[offset:offset+count, 0] = i
            pairs[offset:offset+count, 1] = np.arange(i+1, n)
            offset += count
        yield pairs
        del pairs


def save_results(out: Path, result: dict):
    """Save completed stages as well as the final comparison."""
    (out/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')


def plot_results(out: Path, result: dict):
    """Compare cheap rejection with the exact zero-integral ceiling."""
    plt = plot_style()
    rows = result['configurations']
    labels = ['No partitions' if r['subdivisions'][1] == 0
              else f'{2**r["subdivisions"][1]} groups' for r in rows]
    palette = ['navy', 'coral', 'c', 'crimson']
    colors = [palette[i % len(palette)] for i in range(len(rows))]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    values = [100*r['bounds_rejection'] for r in rows]
    bars = axes[0].bar(labels, values, color=colors)
    axes[0].bar_label(bars, labels=[f'{v:.3f}%' for v in values],
                      padding=-25, color='white')
    ceiling = 100*result['direct']['zero_fraction']
    axes[0].axhline(ceiling, color='black', linestyle='--',
                    label=f'Zero-integral ceiling: {ceiling:.3f}%')
    axes[0].set_ylim(0, 110)
    axes[0].set_ylabel('Possible pairs rejected by enclosing bounds (%)')
    axes[0].legend(fontsize=11, loc='lower left')
    times = [r['find_s'] for r in rows]
    bars = axes[1].bar(labels, times, color=colors)
    axes[1].bar_label(bars, labels=[f'{t:.3f} s' for t in times], padding=4)
    axes[1].set_ylim(0, max(times)*1.2)
    axes[1].set_ylabel('Find all exact pairs: wall time (s)')
    fig.suptitle(f'{result["events"]:,} events, identical boxes; 3D checked first')
    fig.tight_layout(); fig.savefig(out/'comparison.jpg', dpi=200); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=int, default=10000)
    parser.add_argument('--partition-levels', type=int, default=3)
    parser.add_argument('--compare-levels', type=int, nargs='+', default=None,
                        help='Compare several 2D levels on one set of boxes, e.g. 1 2 3')
    parser.add_argument('--batch-events', type=int, default=64)
    parser.add_argument('--ntrial', type=int, nargs=2, default=(5, 10))
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).parent/'regenerated_comparison_results')
    args = parser.parse_args()
    if args.events < 2 or args.batch_events < 1 or args.partition_levels < 0:
        parser.error('Use at least two events, a positive batch size and nonnegative levels')
    levels_to_check = list(dict.fromkeys([0] + (args.compare_levels
                           if args.compare_levels is not None else [args.partition_levels])))
    if min(levels_to_check) < 0:
        parser.error('Subdivision levels must be nonnegative')
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    folder = Path(__file__).parent/'ARAK_samples'
    n = args.events; possible = n*(n-1)//2
    result = dict(status='running', events=n, possible_pairs=possible,
                  ntrial=list(args.ntrial), partition_levels=args.partition_levels,
                  batch_events=args.batch_events, block_order=[0, 1],
                  comparison_levels=levels_to_check, n_centers=[4, 8],
                  block_dimensions=[3, 2], sort_dimensions=[0, 0], seed=91,
                  configurations=[], generation_s=0., preparation_s=0.)
    save_results(out, result)
    events, probabilities = [], []
    metadata = hashlib.sha256()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for i in range(n):
            path = folder/f'event_{i}.npy'
            info = path.stat()
            metadata.update(f'{i},{info.st_size},{info.st_mtime_ns}\n'.encode())
            samples = np.load(path, mmap_mode='r')
            start = time.perf_counter()
            event = make_event_boxes(samples, rng=np.random.default_rng(91+i), ntrial=args.ntrial)
            result['generation_s'] += time.perf_counter()-start
            probabilities.append(event.retained_probability)
            start = time.perf_counter(); events.append(prepare_event(event))
            result['preparation_s'] += time.perf_counter()-start
            del samples, event
            if (i+1) % 1000 == 0:
                print(f'Generated {i+1} / {n} events', flush=True)
    collection = prepare_events(events)
    result.update(input_metadata_sha256=metadata.hexdigest(),
                  retained_probability_mean=float(np.mean(probabilities)),
                  retained_probability_min=float(min(probabilities)),
                  events_losing_over_point_one_percent=sum(p < .999-1e-12
                                                          for p in probabilities))
    save_results(out, result)
    for levels in levels_to_check:
        start = time.perf_counter()
        index = build_index(collection, subdivisions=(0, levels), block_order=(0, 1))
        row = dict(subdivisions=[0, levels], index_build_s=time.perf_counter()-start,
                   index_entries=index.entry_count, candidates=0, exact_pairs=0, find_s=0.)
        digest = hashlib.sha256(); progress = time.perf_counter()
        batches = iter(find_NvN_batches(index, args.batch_events))
        for batch in range((n+args.batch_events-1)//args.batch_events):
            start = time.perf_counter(); pairs = next(batches)
            row['find_s'] += time.perf_counter()-start
            row['candidates'] += index.last_candidate_count
            row['exact_pairs'] += len(pairs)
            digest.update(pairs.astype('<i8', copy=False).tobytes())
            del pairs
            if time.perf_counter()-progress > 30:
                print(f'{levels} levels: {row["exact_pairs"]:,} exact pairs so far', flush=True)
                progress = time.perf_counter()
        row.update(bounds_rejection=1-row['candidates']/possible,
                   exact_rejection=1-row['exact_pairs']/possible,
                   exact_pairs_sha256=digest.hexdigest())
        result['configurations'].append(row); save_results(out, result)
        print(f'{levels} levels: bounds rejection {100*row["bounds_rejection"]:.4f}%, '
              f'find {row["find_s"]:.3f} s', flush=True)
        del index
    direct = dict(evaluated_pairs=0, zero_pairs=0, positive_pairs=0, negative_pairs=0,
                  nonfinite_pairs=0, overlap_s=0., pair_creation_s=0., count_and_digest_s=0.)
    digest = hashlib.sha256(); progress = time.perf_counter()
    batches = iter(all_pair_batches(n, args.batch_events))
    with open(out/'direct_batches.csv', 'w', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(['first_begin', 'first_end', 'pairs', 'zero', 'positive',
                         'negative', 'nonfinite', 'overlap_s'])
        for batch in range((n+args.batch_events-1)//args.batch_events):
            start = time.perf_counter(); pairs = next(batches)
            direct['pair_creation_s'] += time.perf_counter()-start
            start = time.perf_counter(); values = overlap_NvN_pairs(collection, pairs)
            elapsed = time.perf_counter()-start; direct['overlap_s'] += elapsed
            start = time.perf_counter()
            positive = values > 0
            counts = [int(np.count_nonzero(values == 0)), int(np.count_nonzero(positive)),
                      int(np.count_nonzero(values < 0)),
                      int(np.count_nonzero(~np.isfinite(values)))]
            digest.update(pairs[positive].astype('<i8', copy=False).tobytes())
            direct['evaluated_pairs'] += len(pairs)
            for key, count in zip(('zero_pairs', 'positive_pairs', 'negative_pairs',
                                   'nonfinite_pairs'), counts):
                direct[key] += count
            direct['count_and_digest_s'] += time.perf_counter()-start
            begin = batch*args.batch_events; end = min(begin+args.batch_events, n)
            writer.writerow([begin, end, len(pairs), *counts, elapsed]); output.flush()
            del pairs, values, positive
            if time.perf_counter()-progress > 30:
                result['direct_partial'] = direct.copy(); save_results(out, result)
                print(f'Direct integrals: {direct["evaluated_pairs"]:,} / {possible:,}; '
                      f'{direct["zero_pairs"]:,} zeros', flush=True)
                progress = time.perf_counter()
    direct.update(zero_fraction=direct['zero_pairs']/possible,
                  positive_pairs_sha256=digest.hexdigest())
    result['direct'] = direct
    for row in result['configurations']:
        row.update(false_candidates=row['candidates']-direct['positive_pairs'],
                   fraction_zero_rejected_by_bounds=(possible-row['candidates'])
                                                   / direct['zero_pairs']
                                                   if direct['zero_pairs'] else None,
                   matches_direct_positive_pairs=(row['exact_pairs']==direct['positive_pairs']
                       and row['exact_pairs_sha256']==direct['positive_pairs_sha256']))
    result.update(status='checking',
                  all_pair_checks_match=all(r['matches_direct_positive_pairs']
                                            for r in result['configurations']))
    result.pop('direct_partial', None); save_results(out, result)
    assert direct['evaluated_pairs'] == possible
    assert result['all_pair_checks_match'], 'Finder outputs differ from direct positive integrals'
    result['status'] = 'complete'
    save_results(out, result)
    with open(out/'comparison.csv', 'w', newline='') as output:
        columns = ['subdivisions', 'bounds_rejection', 'exact_rejection', 'candidates',
                   'exact_pairs', 'false_candidates', 'fraction_zero_rejected_by_bounds',
                   'index_build_s', 'find_s']
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(result['configurations'])
    plot_results(out, result)
    print(f'Actual zero integrals: {direct["zero_pairs"]:,} / {possible:,} '
          f'({100*direct["zero_fraction"]:.4f}%). All finder outputs match.', flush=True)
    print(f'Saved results to {out}', flush=True)


if __name__ == '__main__':
    main()
