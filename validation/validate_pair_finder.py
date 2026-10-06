"""Create ARAK pair-finder figures, histogram comparisons and scaling measurements.

Run with gw Python: python validation/validate_pair_finder.py
All available events are processed by default. Figures are JPG at 200 dpi. Measured
box-generation times exclude file opening but include mapped sample reads; their prefix sums give the total work
for each N. Tree building, pair finding and overlap accumulation are timed
separately. Output size and dataset geometry influence scaling. Raw samples are
read one event at a time for generation and at most two events at a time for
pairwise histogram comparisons. Only compact boxes are retained across events.
"""
from pathlib import Path
import argparse
import csv
import json
import os
import sys
import time
import warnings
for setting in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(setting, '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib_box_overlap')
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bo import (make_event_boxes, prepare_event, prepare_events, build_index,
                find_1vN, find_NvN_batches, overlap_1vN, overlap_NvN_pairs)


from validation.plot_style import plot_style


def intersection_plot(events, queries, ids, output, plt, subdivisions=(0, 3)):
    """Show second-block boxes; colors encode intersection in BOTH full blocks."""
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle, Patch
    index = build_index(events, subdivisions=subdivisions)
    count = len(queries)
    rows, columns = (1, 1) if count == 1 else (2, 3)
    fig, axes = plt.subplots(rows, columns, figsize=(8, 7) if count == 1 else (18, 11),
                             squeeze=False)
    for ax, query, idx in zip(axes.flat, queries, ids):
        hits = set(find_1vN(query, index).tolist())
        for hit, color in ((False, '0.75'), (True, 'crimson')):
            patches = []
            for j, event in enumerate(events):
                if (j in hits) != hit:
                    continue
                b = event.blocks[1]
                patches.extend(Rectangle(lo, *(hi-lo)) for lo, hi in zip(b.lower, b.upper))
            ax.add_collection(PatchCollection(patches, facecolor=color, edgecolor=color,
                                              linewidth=.3, alpha=.35))
        b = query.blocks[1]
        ax.add_collection(PatchCollection([Rectangle(lo, *(hi-lo))
            for lo, hi in zip(b.lower, b.upper)], facecolor='none', edgecolor='navy',
            linewidth=1.2))
        ax.autoscale_view()
        ax.set_xlabel(r'$\theta_4$'); ax.set_ylabel(r'$\theta_5$')
        ax.set_title(f'Query event {idx}: {len(hits)} / {len(events)} matches')
    for ax in list(axes.flat)[count:]:
        ax.set_visible(False)
    fig.legend(handles=[Patch(color='0.75', label='No full-event intersection'),
                        Patch(color='crimson', label='Full-event intersection'),
                        Patch(facecolor='none', edgecolor='navy', label='Query boxes')],
               loc='lower center', ncol=3)
    fig.suptitle('2D block boxes; matching requires intersection in both blocks')
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(output, dpi=200); plt.close(fig)


def sample_histogram_overlap(a, b, bins=48):
    """Shared pair-specific grids, unit weights, original sample normalization."""
    value = 1.
    for columns in (slice(0, 3), slice(3, 5)):
        x, y = a[:, columns], b[:, columns]
        lower, upper = np.minimum(x.min(0), y.min(0)), np.maximum(x.max(0), y.max(0))
        edges = [np.linspace(lo, hi, bins+1) for lo, hi in zip(lower, upper)]
        volume = np.prod((upper-lower)/bins)
        hx = np.histogramdd(x, bins=edges)[0]
        hy = np.histogramdd(y, bins=edges)[0]
        value *= np.sum(hx*hy)/(len(x)*len(y)*volume)
    return value


def sample_histogram_overlaps(folder, pairs, bins=48):
    """Yield histogram overlaps while holding at most two event sample arrays.

    Pairs grouped by first index reuse that event's read-only memory map. Each
    second event is opened only for its comparison, then released. No sample
    list is accumulated. Results preserve input pair order.
    """
    current_index = None
    sample_a = None
    for i, j in pairs:
        if i != current_index:
            sample_a = None  # Release the previous map before opening another.
            sample_a = np.load(folder/f'event_{i}.npy', mmap_mode='r')
            current_index = i
        sample_b = np.load(folder/f'event_{j}.npy', mmap_mode='r')
        value = sample_histogram_overlap(sample_a, sample_b, bins)
        del sample_b
        yield value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-events', type=int, default=None)
    parser.add_argument('--histogram-bins', type=int, default=48)
    parser.add_argument('--timings-only', action='store_true',
                        help='Keep existing figures and histogram comparison; rerun timings')
    parser.add_argument('--batch-events', type=int, default=64)
    parser.add_argument('--ntrial', type=int, nargs=2, default=(5, 10),
                        help='Random box trials for the first and second blocks')
    parser.add_argument('--subdivisions', type=int, nargs=2, default=(0, 3),
                        help='Subdivision levels for the first and second blocks')
    args = parser.parse_args()
    folder = Path(__file__).parent
    out = folder/'pair_finder_results'; out.mkdir(exist_ok=True)
    plt = plot_style()
    total = len(list((folder/'ARAK_samples').glob('event_*.npy')))
    if args.max_events is not None:
        total = min(args.max_events, total)
    if total < 105:
        raise ValueError('At least 105 events are needed for the fixed examples')
    events, prepared, generation, preparation, probabilities = [], [], [], [], []
    settings = {'ntrial': list(args.ntrial), 'subdivisions': list(args.subdivisions),
                'dataset_events': total, 'batch_events': args.batch_events,
                'histogram_bins': args.histogram_bins, 'seed': '91 + event index',
                'status': 'running', 'block_order': [0, 1]}
    (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for i in range(total):
            samples = np.load(folder/'ARAK_samples'/f'event_{i}.npy', mmap_mode='r')
            start = time.perf_counter()
            event = make_event_boxes(samples, rng=np.random.default_rng(91+i), ntrial=args.ntrial)
            generation.append(time.perf_counter()-start)
            start = time.perf_counter(); snapshot = prepare_event(event)
            preparation.append(time.perf_counter()-start)
            # Only the first 105 Python events are needed for the example plots.
            # Prepared C++ boxes already own the full integration collection.
            if not args.timings_only and i < 105:
                events.append(event)
            prepared.append(snapshot)
            probabilities.append(event.retained_probability)
            del samples  # Retain boxes, never raw samples from earlier events.
            if (i+1) % 1000 == 0:
                print(f'Built {i+1} / {total} events', flush=True)
    if not args.timings_only:
        intersection_plot(events[:100], events[100:101], [100], out/'100v1.jpg', plt, args.subdivisions)
        intersection_plot(events[:100], events[100:105], list(range(100,105)),
                          out/'100v5.jpg', plt, args.subdivisions)
        # Cross-set calculation uses many prefiltered 1vN calls, as prescribed.
        index100 = build_index(prepared[:100], subdivisions=args.subdivisions)
        cross = []
        for q in range(100,105):
            hits = find_1vN(prepared[q], index100)
            values = overlap_1vN(prepared[q], [prepared[j] for j in hits])
            cross.extend((q, int(j), float(v)) for j,v in zip(hits,values))
        np.savetxt(out/'100v5.csv', np.asarray(cross).reshape(-1,3), delimiter=',',
                   header='query_event,indexed_event,overlap', comments='')

        pairs = np.array([(i,j) for i in range(100) for j in range(i+1,100)], dtype=np.int64)
        fast = overlap_NvN_pairs(prepare_events(prepared[:100]), pairs)
        dense = []
        histogram_values = sample_histogram_overlaps(folder/'ARAK_samples', pairs,
                                                    args.histogram_bins)
        for k,value in enumerate(histogram_values):
            dense.append(value)
            if (k+1) % 1000 == 0:
                print(f'Histogram comparisons: {k+1} / {len(pairs)}', flush=True)
        dense = np.array(dense)
        np.savetxt(out/'histogram_comparison.csv', np.column_stack((pairs,fast,dense)),
            delimiter=',', header='event_a,event_b,box_overlap,histogram_overlap', comments='')
        fig,ax=plt.subplots(figsize=(8,7))
        ax.scatter(dense,fast,s=12,alpha=.45)
        limit=max(dense.max(),fast.max())
        ax.plot([0,limit],[0,limit],color='coral',label='Equal overlap')
        # Symlog retains pairs with zero in either estimate.
        threshold=limit*1e-5 if limit else 1e-5
        ax.set_xscale('symlog',linthresh=threshold); ax.set_yscale('symlog',linthresh=threshold)
        # Omit the crowded decade tick next to zero while retaining actual zeros.
        powers = np.arange(int(np.ceil(np.log10(threshold)))+1,
                           int(np.ceil(np.log10(limit)))+1)
        ticks = np.r_[0., 10.**powers]
        labels = ['0'] + [rf'$10^{{{power}}}$' for power in powers]
        ax.set_xticks(ticks, labels); ax.set_yticks(ticks, labels)
        ax.set_xlabel('Dense sample histogram overlap'); ax.set_ylabel('Exact box overlap')
        ax.set_title(f'100 events: all 4,950 pairs; {args.histogram_bins} bins per coordinate')
        ax.legend(); fig.tight_layout(); fig.savefig(out/'histogram_comparison.jpg',dpi=200)
        plt.close(fig)

    sizes=np.unique(np.geomspace(2,total,20).astype(int))
    generation=np.cumsum(generation); preparation=np.cumsum(preparation)
    headers=['N','box_generation_s','preparation_s','index_build_s','find_NvN_s',
             'overlap_NvN_s','find_1vN_s','overlap_1vN_s','NvN_pairs','1vN_hits','NvN_candidates','candidate_rejection','pair_rejection','index_entries']
    measurements=[]
    for n in sizes:
        start=time.perf_counter(); collection=prepare_events(prepared[:n])
        index=build_index(collection, subdivisions=args.subdivisions); index_time=time.perf_counter()-start
        find_all, overlap_all, pair_count, candidates = 0., 0., 0, 0
        batches = iter(find_NvN_batches(index, args.batch_events))
        progress = time.perf_counter()
        for batch in range((n+args.batch_events-1)//args.batch_events):
            start=time.perf_counter(); hits=next(batches)
            find_all += time.perf_counter()-start
            pair_count += len(hits)
            candidates += index.last_candidate_count
            start=time.perf_counter(); values=overlap_NvN_pairs(collection,hits)
            overlap_all += time.perf_counter()-start
            del hits, values  # Never retain all matching pairs or integrals.
            if time.perf_counter()-progress > 30:
                print(f'N={n}: first events {min((batch+1)*args.batch_events,n)} / {n}, '
                      f'{pair_count} pairs so far',flush=True)
                progress=time.perf_counter()
        start=time.perf_counter(); one_hits=find_1vN(prepared[-1],index)
        # Remove the query's own event at the full-set endpoint.
        one_hits=one_hits[one_hits != total-1]
        find_one=time.perf_counter()-start
        start=time.perf_counter(); overlap_1vN(prepared[-1],[prepared[j] for j in one_hits])
        overlap_one=time.perf_counter()-start
        measurements.append([n,generation[n-1],preparation[n-1],index_time,
                             find_all,overlap_all,find_one,overlap_one,pair_count,len(one_hits),
                             candidates,1-candidates/(n*(n-1)/2),1-pair_count/(n*(n-1)/2),
                             index.entry_count])
        with open(out/'timings.csv','w',newline='') as f:
            writer=csv.writer(f); writer.writerow(headers); writer.writerows(measurements)
        print(f'N={n}: {pair_count} pairs, finding {find_all:.3f}s, '
              f'overlap {overlap_all:.3f}s, rejection {100*(1-pair_count/(n*(n-1)/2)):.2f}%, '
              f'bounds rejection {100*(1-candidates/(n*(n-1)/2)):.2f}%',flush=True)
    with open(out/'timings.csv','w',newline='') as f:
        writer=csv.writer(f); writer.writerow(headers); writer.writerows(measurements)
    data=np.asarray(measurements)
    fig,axes=plt.subplots(1,2,figsize=(16,6))
    for ax, find_col, calc_col, title in zip(axes,(4,6),(5,7),('N within itself','One query vs N')):
        for column,label in ((1,'Box generation (incl. mapped reads)'),(2,'C++ preparation'),
                             (3,'Event index build'),(find_col,'Find pairs'),
                             (calc_col,'Calculate filtered overlaps')):
            ax.loglog(data[:,0],data[:,column],marker='o',label=label)
        ax.set_xlabel('Number of events'); ax.set_ylabel('Wall time (s)'); ax.set_title(title)
        ax.legend(fontsize=11); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'scaling.jpg',dpi=200); plt.close(fig)
    settings.update(status='complete', retained_probability_min=float(min(probabilities)),
                    retained_probability_mean=float(np.mean(probabilities)),
                    events_losing_over_point_one_percent=sum(p < .999-1e-12 for p in probabilities))
    (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    print(f'Saved validation results to {out}',flush=True)


if __name__ == '__main__':
    main()
