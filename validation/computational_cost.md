# Computational cost of box overlap calculations

The largest measured event set contains 32,016 events, processed on one thread. Each sample file is read separately; only compact boxes and the event index are retained across events. Matching pairs are calculated in bounded batches.

## Filtering algorithm

Each independent block has a bounding box enclosing all its retained sub-boxes. The event BVH stores unions of these bounds for both blocks and tests the 2D bounds first. Surviving leaf events are checked with sorted sub-box loops, stopping at the first intersection in each block. Individual boxes have no BVH. Integral calculation uses the same sorted geometry but accumulates all contributions.

## Largest measured event count

| Stage | Wall time (s) |
| --- | ---: |
| Generate boxes, including mapped sample reads | 184.556 |
| Prepare C++ box snapshots | 2.40296 |
| Build event index | 4.49536 |
| Find within-set pairs | 255.472 |
| Calculate all filtered within-set overlaps | 956.884 |
| Find one-vs-set matches | 0.0313133 |
| Calculate filtered one-vs-set overlaps | 0.0888994 |

There are **77,478,166** matching within-set pairs out of 512,496,120 possible distinct pairs (15.12%). The one-event query has 4,467 matches.

## Scaling measurements

| Events | Matching pairs | Generate boxes (s) | Build index (s) | Find pairs (s) | Calculate overlaps (s) | One-vs-set find + calculate (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0 | 0.00832572 | 0.000100803 | 0.00014962 | 2.8055e-05 | 6.3971e-05 |
| 3 | 0 | 0.0121101 | 0.000113776 | 3.8073e-05 | 3.5481e-05 | 9.3487e-05 |
| 6 | 1 | 0.0241323 | 0.00010967 | 4.4428e-05 | 3.0446e-05 | 0.00010296 |
| 11 | 9 | 0.040538 | 0.000130225 | 4.8878e-05 | 0.000132824 | 0.000130803 |
| 19 | 25 | 0.0743727 | 0.000161166 | 0.000100299 | 0.000359376 | 0.000176537 |
| 34 | 80 | 0.136545 | 0.000441013 | 0.00031166 | 0.00118526 | 0.000193244 |
| 60 | 307 | 0.239885 | 0.000533529 | 0.00109875 | 0.00422668 | 0.00015629 |
| 107 | 924 | 0.413585 | 0.000910439 | 0.00183485 | 0.0099106 | 0.000395418 |
| 190 | 2,737 | 0.782284 | 0.0128474 | 0.0131542 | 0.0299381 | 0.00052524 |
| 336 | 8,607 | 1.45709 | 0.0349627 | 0.0327357 | 0.0884177 | 0.00098201 |
| 594 | 27,093 | 2.6117 | 0.0591479 | 0.0754769 | 0.305278 | 0.00218408 |
| 1,050 | 83,878 | 4.73939 | 0.113598 | 0.222485 | 0.961032 | 0.00261314 |
| 1,856 | 259,819 | 8.38575 | 0.139608 | 0.586407 | 3.09151 | 0.00478723 |
| 3,281 | 814,565 | 14.7422 | 0.257338 | 1.68422 | 9.66558 | 0.00967009 |
| 5,800 | 2,530,867 | 26.9552 | 0.410245 | 5.74167 | 30.7657 | 0.0152078 |
| 10,250 | 7,927,169 | 48.8055 | 0.923032 | 20.077 | 97.1896 | 0.029544 |
| 18,115 | 24,835,334 | 87.3789 | 2.52706 | 71.537 | 306.833 | 0.0476096 |
| 32,016 | 77,478,166 | 184.556 | 4.49536 | 255.472 | 956.884 | 0.120213 |

![Measured computation cost](pair_finder_results/scaling.jpg)

## Measurement interpretation

Box generation and preparation use cumulative per-event times for prefixes of the input order. File opening is outside the generation timer; reads triggered by mapped sample access are included. Index building, pair finding and accumulation are measured separately for every prefix. The one-event query is the final dataset event, with its self match removed.

The within-set stages must process every matching pair. If a roughly constant fraction of possible pairs intersects, both pair count and total accumulation cost grow approximately as N squared even with an event BVH. The BVH reduces rejected-candidate work; it cannot remove the cost of genuinely matching pairs.

These are single-run wall times and depend on geometry, cache state and host load. The expanded files also changed the first 100 events, so the previous 10,000-event timings are not a controlled comparison on identical input.

Reproduce with `python validation/validate_pair_finder.py`, then `python validation/summarize_cost.py`, using the gw environment. Full numeric results are in `pair_finder_results/timings.csv`. The previous endpoint is preserved in `pair_finder_results/timings_10000_previous.csv`.

## Stopped calculation

The calculation was stopped at the user’s request. Completed measurements through 32,016 events are saved in `pair_finder_results/timings.csv`. The 56,582-event measurement was interrupted; its last reported progress is recorded separately in `pair_finder_results/run_status.json` and is not a completed timing measurement. The 100,000-event overlap measurement was not run. All completed validation figures are preserved.
