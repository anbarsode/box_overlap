"""Generate compact CAMP boxes and compare matching pairs with exact histograms.

Process at most two raw event arrays at a time. Default box settings and the
24-bin-per-axis histogram match the ARAK comparison. Saved boxes permit reuse
without rereading samples. No computational cost study is performed.
"""
import argparse
import json
import os
from pathlib import Path
import sys
for setting in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[setting] = '1'
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bo import (make_event_boxes, save_event_boxes, prepare_event, find_paired,
                overlap_paired, overlap_1v1, find_1vN, build_index)
from validation.histogram_overlap import block_overlaps
from validation.validate_pair_finder import sample_histogram_overlap
from validation.analysis_plots import plot_overlap_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=int, default=1000)
    parser.add_argument('--bins', type=int, default=24)
    parser.add_argument('--samples', type=Path, default=ROOT/'validation/CAMP_samples')
    parser.add_argument('--output', type=Path, default=ROOT/'validation/CAMP_results')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    box_folder = args.output/'boxes'
    box_folder.mkdir(exist_ok=True)
    rows = []
    retained = []
    for i in range(args.pairs):
        raw = [np.load(args.samples/f'event_{i}_{j}.npy', mmap_mode='r') for j in (0, 1)]
        boxes = [make_event_boxes(values, rng=np.random.default_rng(91+2*i+j))
                 for j, values in enumerate(raw)]
        for j, event in enumerate(boxes):
            save_event_boxes(box_folder/f'event_{i}_{j}.boxes', event)
            retained.append(event.retained_probability)
        a, b = map(prepare_event, boxes)
        hit = bool(find_paired([a], [b])[0])
        integral = float(overlap_paired([a], [b])[0])
        np.testing.assert_allclose(integral, overlap_1v1(a, b), rtol=1e-14)
        # Independently exercise the indexed checker on the first 20 actual pairs.
        if i < 20:
            assert hit == bool(len(find_1vN(a, build_index([b]))))
        assert hit == (integral > 0)
        histogram = float(np.prod(block_overlaps(*raw, bins=args.bins)))
        if i < 20:
            np.testing.assert_allclose(histogram, sample_histogram_overlap(*raw, bins=args.bins),
                                       rtol=3e-15)
        rows.append((i, integral, histogram, hit))
        del raw, boxes, a, b
        if (i+1) % 100 == 0:
            print(f'CAMP comparison: {i+1}/{args.pairs}', flush=True)
    np.savetxt(args.output/'histogram_comparison.csv', rows, delimiter=',', comments='',
               header='pair,box_overlap,histogram_overlap,geometry_hit',
               fmt=('%d', '%.17g', '%.17g', '%d'))
    connected = np.genfromtxt(args.output/'histogram_comparison.csv', delimiter=',', names=True)
    folder = ROOT/'validation/pair_finder_results'
    arak = np.genfromtxt(folder/'histogram_comparison.csv', delimiter=',', names=True)
    fits = plot_overlap_comparison(arak, 1000, args.bins,
                                   folder/'histogram_comparison.jpg', connected=connected)
    settings = {'pairs': args.pairs, 'bins': args.bins, 'seed': 91,
                'ntrial': [5, 10], 'n_centers': [4, 8],
                'positive_boxes': int(np.count_nonzero(connected['box_overlap'])),
                'positive_histograms': int(np.count_nonzero(connected['histogram_overlap'])),
                'retained_probability_mean': float(np.mean(retained)),
                'checker_agrees_with_index_first_pairs': min(20, args.pairs),
                'all_integrals_agree_with_1v1': True,
                'histograms_agree_with_numpy_first_pairs': min(20, args.pairs),
                'fits': fits['CAMP']}
    (args.output/'comparison.json').write_text(json.dumps(settings, indent=2)+'\n')
    print(json.dumps(settings, indent=2))


if __name__ == '__main__':
    main()
