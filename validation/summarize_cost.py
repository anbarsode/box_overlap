"""Render measured scaling results and a standalone computation-cost report.

Run after validate_pair_finder.py. Timings are read from the completed CSV;
this script performs no overlap calculations and loads no raw sample files.
"""
from pathlib import Path
import csv
import os
for setting in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(setting, '1')
import numpy as np
from validate_pair_finder import plot_style


def main():
    folder=Path(__file__).parent/'pair_finder_results'
    with open(folder/'timings.csv',newline='') as source:
        rows=list(csv.DictReader(source))
    data={key:np.array([float(row[key]) for row in rows]) for key in rows[0]}
    plt=plot_style()
    fig,axes=plt.subplots(1,2,figsize=(16,6))
    for ax,find,calc,title in zip(axes,('find_NvN_s','find_1vN_s'),
                                 ('overlap_NvN_s','overlap_1vN_s'),
                                 ('N within itself','One query vs N')):
        for key,label in (('box_generation_s','Box generation (incl. mapped reads)'),
                          ('preparation_s','C++ preparation'),('index_build_s','Event index build'),
                          (find,'Find pairs'),(calc,'Calculate filtered overlaps')):
            ax.loglog(data['N'],data[key],marker='o',label=label)
        ax.set_xlabel('Number of events'); ax.set_ylabel('Wall time (s)'); ax.set_title(title)
        ax.legend(fontsize=10); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(folder/'scaling.jpg',dpi=200); plt.close(fig)
    last=rows[-1]; n=int(last['N']); pairs=int(last['NvN_pairs']); possible=n*(n-1)//2
    lines=['# Computational cost of box overlap calculations','',
        f'The largest measured event set contains {n:,} events, processed on one thread. '
        'Each sample file is read separately; only compact boxes and the event index '
        'are retained across events. Matching pairs are calculated in bounded batches.', '',
        '## Filtering algorithm','',
        'Each independent block has a bounding box enclosing all its retained sub-boxes. '
        'The event BVH stores unions of these bounds for both blocks and tests the 2D '
        'bounds first. Surviving leaf events are checked with sorted sub-box loops, '
        'stopping at the first intersection in each block. Individual boxes have no BVH. '
        'Integral calculation uses the same sorted geometry but accumulates all contributions.', '',
        '## Largest measured event count','',
        '| Stage | Wall time (s) |','| --- | ---: |']
    for key,label in (('box_generation_s','Generate boxes, including mapped sample reads'),
                      ('preparation_s','Prepare C++ box snapshots'),
                      ('index_build_s','Build event index'),('find_NvN_s','Find within-set pairs'),
                      ('overlap_NvN_s','Calculate all filtered within-set overlaps'),
                      ('find_1vN_s','Find one-vs-set matches'),
                      ('overlap_1vN_s','Calculate filtered one-vs-set overlaps')):
        lines.append(f'| {label} | {float(last[key]):,.6g} |')
    lines += ['',f'There are **{pairs:,}** matching within-set pairs out of '
              f'{possible:,} possible distinct pairs ({100*pairs/possible:.2f}%). '
              f'The one-event query has {int(last["1vN_hits"]):,} matches.', '',
        '## Scaling measurements','',
        '| Events | Matching pairs | Generate boxes (s) | Build index (s) | Find pairs (s) | '
        'Calculate overlaps (s) | One-vs-set find + calculate (s) |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f'| {int(row["N"]):,} | {int(row["NvN_pairs"]):,} | '
            f'{float(row["box_generation_s"]):.6g} | {float(row["index_build_s"]):.6g} | '
            f'{float(row["find_NvN_s"]):.6g} | {float(row["overlap_NvN_s"]):.6g} | '
            f'{float(row["find_1vN_s"])+float(row["overlap_1vN_s"]):.6g} |')
    lines += ['', '![Measured computation cost](pair_finder_results/scaling.jpg)', '',
        '## Measurement interpretation','',
        'Box generation and preparation use cumulative per-event times for prefixes '
        'of the input order. File opening is outside the generation timer; reads '
        'triggered by mapped sample access are included. Index building, pair finding '
        'and accumulation are measured separately for every prefix. The one-event '
        'query is the final dataset event, with its self match removed.', '',
        'The within-set stages must process every matching pair. If a roughly constant '
        'fraction of possible pairs intersects, both pair count and total accumulation '
        'cost grow approximately as N squared even with an event BVH. The BVH reduces '
        'rejected-candidate work; it cannot remove the cost of genuinely matching pairs.', '',
        'These are single-run wall times and depend on geometry, cache state and host '
        'load. The expanded files also changed the first 100 events, so the previous '
        '10,000-event timings are not a controlled comparison on identical input.', '',
        'Reproduce with `python validation/validate_pair_finder.py`, then '
        '`python validation/summarize_cost.py`, using the gw environment. Full numeric '
        'results are in `pair_finder_results/timings.csv`. The previous endpoint is '
        'preserved in `pair_finder_results/timings_10000_previous.csv`.']
    output=Path(__file__).parent/'computational_cost.md'
    output.write_text('\n'.join(lines)+'\n')
    print(f'Saved {output}')


if __name__=='__main__':
    main()
