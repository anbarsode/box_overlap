"""Controlled comparison of box trials and enclosing subdivision levels.

The same sample files and per-event seeds are used for every configuration.
Only boxes are kept across events. Candidate rejection means failure of the
consolidated enclosing-bound test; final rejection also includes exact sub-box
existence tests. Subdivision levels cannot change final pairs for fixed boxes.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import warnings
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bo import make_event_boxes, prepare_event, build_index, find_NvN_batches
from validation.plot_style import plot_style


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=int, default=1000)
    args = parser.parse_args()
    folder = Path(__file__).parent
    out = folder/'pair_finder_results'
    out.mkdir(exist_ok=True)
    results, volumes = [], {}
    for trials in ((1, 1), (5, 10)):
        events, volume, probabilities = [], [], []
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            for i in range(args.events):
                samples = np.load(folder/'ARAK_samples'/f'event_{i}.npy', mmap_mode='r')
                event = make_event_boxes(samples, rng=np.random.default_rng(91+i), ntrial=trials)
                volume.append([np.prod(b.upper-b.lower, axis=1).sum() for b in event])
                probabilities.append(event.retained_probability)
                events.append(prepare_event(event))
                del samples, event
        generation = time.perf_counter()-started
        volumes[str(trials)] = np.array(volume)
        for levels in ((0, 0), (0, 1), (0, 2), (0, 3)):
            for order in ((0, 1), (1, 0)):
                started = time.perf_counter()
                index = build_index(events, subdivisions=levels, block_order=order)
                build = time.perf_counter()-started
                pairs, candidates = 0, 0
                started = time.perf_counter()
                for hits in find_NvN_batches(index):
                    pairs += len(hits)
                    candidates += index.last_candidate_count
                    del hits
                finding = time.perf_counter()-started
                possible = args.events*(args.events-1)//2
                row = dict(ntrial=list(trials), subdivisions=list(levels),
                           block_order=list(order), events=args.events,
                           generation_s=generation, index_build_s=build, find_s=finding,
                           candidates=candidates, pairs=pairs,
                           candidate_rejection=1-candidates/possible,
                           pair_rejection=1-pairs/possible,
                           retained_probability_mean=float(np.mean(probabilities)),
                           retained_probability_min=float(min(probabilities)))
                results.append(row)
                print(row, flush=True)
    ratio = volumes['(5, 10)']/volumes['(1, 1)']
    summary = {'configurations':results,
               'volume_ratio_median_per_block':np.median(ratio, axis=0).tolist(),
               'volume_ratio_mean_per_block':np.mean(ratio, axis=0).tolist(),
               'note': 'Second-block random sequences differ because first-block '
                       'trial counts differ.'}
    (out/'rejection_study.json').write_text(json.dumps(summary, indent=2)+'\n')
    plt = plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for trials, color in (((1, 1), 'navy'), ((5, 10), 'coral')):
        for order, marker in (((0, 1), 'o'), ((1, 0), 's')):
            rows = [r for r in results if r['ntrial']==list(trials)
                    and r['block_order']==list(order)]
            x = [r['subdivisions'][1] for r in rows]
            label = f'Trials {trials}, {"3D" if order[0]==0 else "2D"} first'
            # Candidate sets do not depend on check order, so plot them once.
            if order == (0, 1):
                axes[0].plot(x, [100*r['candidate_rejection'] for r in rows], 'o-',
                             color=color, label=f'Trials {trials}: bounds')
                axes[0].plot(x, [100*r['pair_rejection'] for r in rows], '--',
                             color=color, label=f'Trials {trials}: final')
            axes[1].plot(x, [r['find_s'] for r in rows], marker=marker,
                         linestyle='-' if order[0]==0 else '--', color=color, label=label)
    axes[0].set_ylabel('Rejected possible event pairs (%)')
    axes[1].set_ylabel('Find all exact pairs: wall time (s)')
    for ax in axes:
        ax.set_xlabel('2D subdivision levels'); ax.set_xticks((0, 1, 2, 3))
        ax.legend(fontsize=11); ax.grid(alpha=.2)
    fig.suptitle(f'Controlled comparison: {args.events:,} events; 3D unsubdivided')
    fig.tight_layout(); fig.savefig(out/'rejection_study.jpg', dpi=200); plt.close(fig)


if __name__ == '__main__':
    main()
