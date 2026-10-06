"""Save event boxes and calculate within-set overlaps in bounded batches.

Run ``python -m bo.run --help`` from the project root. ``boxes`` reads one
sample file at a time. ``overlaps`` reads saved boxes, never raw samples.
See RUNNING.md for all parameters and the one-event and cross-set interfaces.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

# bo.event_boxes sets the default thread environment before importing NumPy.
import numpy as np
from . import (build_index, find_NvN_batches, load_event_boxes, make_event_boxes,
               overlap_NvN_pairs, prepare_event, prepare_events, save_event_boxes)


def event_paths(folder: Path, suffix: str, start: int, count: int | None):
    """Return numeric event IDs and paths; count=None uses all IDs >= start.

    With an explicit count, require the contiguous range [start,start+count).
    Numeric ordering keeps pair indices independent of filename string ordering.
    """
    if count is None:
        ids = sorted(int(p.stem.removeprefix('event_'))
                     for p in folder.glob(f'event_*{suffix}')
                     if int(p.stem.removeprefix('event_')) >= start)
    else:
        ids = list(range(start, start + count))
    return [(i, folder / f'event_{i}{suffix}') for i in ids]


def write_settings(folder: Path, settings: dict):
    """Replace the small settings file after each completed stage."""
    (folder / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')


def generate_boxes(args):
    """Process each sample map, save its boxes and record retained probability.

    The event's RNG seed is seed+event_id, so selecting a smaller range does not
    change an event's boxes. File opening and saving are included in job wall time.
    Per-event generation time excludes file opening and saving but includes mapped reads.
    """
    if args.output.exists() and any(args.output.glob('event_*.boxes')):
        raise FileExistsError('Use a fresh box output folder for this configuration')
    args.output.mkdir(parents=True, exist_ok=True)
    paths = event_paths(args.samples, '.npy', args.start, args.events)
    settings = {'stage': 'boxes', 'status': 'running', 'samples': str(args.samples.resolve()),
                'event_count': len(paths), 'start': args.start, 'seed': args.seed,
                'n_centers': args.n_centers, 'ntrial': args.ntrial,
                'block_dimensions': args.block_dimensions, 'sort_dimensions': args.sort_dimensions,
                'weights': 'one', 'probability_loss_warning': 0.001}
    write_settings(args.output, settings)
    started = time.perf_counter()
    with (args.output / 'events.csv').open('w', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(['event_id', 'boxes_0', 'boxes_1', 'retained_probability', 'generation_s'])
        for number, (event_id, path) in enumerate(paths, 1):
            samples = np.load(path, mmap_mode='r')
            tick = time.perf_counter()
            event = make_event_boxes(samples, n_centers=tuple(args.n_centers),
                ntrial=None if args.ntrial is None else tuple(args.ntrial),
                block_dimensions=tuple(args.block_dimensions),
                sort_dimensions=tuple(args.sort_dimensions),
                rng=np.random.default_rng(args.seed + event_id))
            elapsed = time.perf_counter() - tick
            del samples  # Never keep sample arrays across events.
            save_event_boxes(args.output / f'event_{event_id}.boxes', event)
            writer.writerow([event_id, *(b.n_boxes for b in event.blocks),
                             event.retained_probability, elapsed])
            if number % 1000 == 0:
                output.flush()
                print(f'Saved boxes for {number}/{len(paths)} events', flush=True)
    settings.update(status='complete', wall_s=time.perf_counter() - started)
    write_settings(args.output, settings)
    print(f'Saved {len(paths)} events to {args.output}', flush=True)


def calculate_overlaps(args):
    """Load compact boxes once, find exact geometric pairs, and save batch integrals.

    Output pairs use positions in event_ids.npy, not necessarily original event IDs.
    Batch files contain int64 ``pairs`` (K,2) and float64 ``overlap`` (K,).
    ``batch_events`` bounds first-event ranges, not the number of matching pairs.
    """
    if args.output.exists() and any(args.output.glob('batch_*.npz')):
        raise FileExistsError('Use a fresh overlap output folder to avoid mixing batch results')
    args.output.mkdir(parents=True, exist_ok=True)
    paths = event_paths(args.boxes, '.boxes', args.start, args.events)
    settings = {'stage': 'overlaps', 'status': 'running', 'boxes': str(args.boxes.resolve()),
                'event_count': len(paths), 'subdivisions': args.subdivisions,
                'block_order': args.block_order, 'sort_dimensions': args.sort_dimensions,
                'batch_events': args.batch_events, 'pairs': 0, 'candidates': 0,
                'find_s': 0.0, 'overlap_s': 0.0, 'save_s': 0.0}
    write_settings(args.output, settings)
    started = time.perf_counter()
    prepared = []
    for _, path in paths:
        # The temporary Python boxes are released after preparing each C++ snapshot.
        prepared.append(prepare_event(load_event_boxes(path, tuple(args.sort_dimensions))))
    events = prepare_events(prepared)
    settings['load_and_prepare_s'] = time.perf_counter() - started
    tick = time.perf_counter()
    index = build_index(events, tuple(args.subdivisions), tuple(args.block_order))
    settings.update(index_build_s=time.perf_counter() - tick, index_entries=index.entry_count)
    np.save(args.output / 'event_ids.npy', np.array([i for i, _ in paths], dtype=np.int64))
    batches = iter(find_NvN_batches(index, args.batch_events))
    number = 0
    while True:
        tick = time.perf_counter()
        try:
            pairs = next(batches)
        except StopIteration:
            break
        settings['find_s'] += time.perf_counter() - tick
        settings['candidates'] += index.last_candidate_count
        tick = time.perf_counter()
        values = overlap_NvN_pairs(events, pairs)
        settings['overlap_s'] += time.perf_counter() - tick
        tick = time.perf_counter()
        # Uncompressed numeric output avoids compression work and Python row formatting.
        np.savez(args.output / f'batch_{number:06d}.npz', pairs=pairs, overlap=values)
        settings['save_s'] += time.perf_counter() - tick
        settings['pairs'] += len(pairs)
        number += 1
        settings['batches'] = number
        write_settings(args.output, settings)
        del pairs, values
    settings.update(status='complete', wall_s=time.perf_counter() - started)
    write_settings(args.output, settings)
    print(f'Saved {settings["pairs"]:,} pairs in {number} batches to {args.output}', flush=True)


def main(argv=None):
    """Parse the documented job parameters; argv=None reads the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    boxes = commands.add_parser('boxes', help='Read sample files and save event boxes')
    boxes.add_argument('--samples', type=Path, default=Path('validation/ARAK_samples'),
                       help='Folder containing event_ID.npy arrays')
    boxes.add_argument('--output', type=Path, default=Path('event_boxes'),
                       help='Folder for event_ID.boxes, events.csv and settings.json')
    boxes.add_argument('--n-centers', type=int, nargs=2, default=(4, 8),
                       help='Random center counts per block (default: 4 8)')
    boxes.add_argument('--ntrial', type=int, nargs=2, default=None,
                       help='Trials per block (default: 5 for 3D, 10 for 2D, 1 otherwise)')
    boxes.add_argument('--block-dimensions', type=int, nargs=2, default=(3, 2),
                       help='Consecutive sample column counts per independent block')
    boxes.add_argument('--seed', type=int, default=91,
                       help='Event seed is this value + event ID')
    boxes.set_defaults(action=generate_boxes)
    overlaps = commands.add_parser('overlaps', help='Read saved boxes and save filtered integrals')
    overlaps.add_argument('--boxes', type=Path, default=Path('event_boxes'),
                          help='Folder containing event_ID.boxes files')
    overlaps.add_argument('--output', type=Path, default=Path('overlap_results'),
                          help='Folder for batch files, event_ids.npy and settings.json')
    overlaps.add_argument('--subdivisions', type=int, nargs=2, default=(0, 3),
                          help='Binary subdivision levels per block: 0 1 2 3 give 1 2 4 8 groups')
    overlaps.add_argument('--block-order', type=int, nargs=2, default=(0, 1),
                          help='Detection order; default checks first (3D) block first')
    overlaps.add_argument('--batch-events', type=int, default=64,
                          help='First-event indices per output batch; smaller uses less pair memory')
    overlaps.set_defaults(action=calculate_overlaps)
    for job in (boxes, overlaps):
        job.add_argument('--start', type=int, default=0, help='First event ID to process')
        job.add_argument('--events', type=int, default=None,
                         help='Contiguous event count; default uses all event IDs >= start')
        job.add_argument('--sort-dimensions', type=int, nargs=2, default=(0, 0),
                         help='Zero-based ordering coordinate local to each block')
    args = parser.parse_args(argv)
    if args.events is not None and args.events < 1:
        parser.error('--events must be positive')
    if args.command == 'overlaps' and args.batch_events < 1:
        parser.error('--batch-events must be positive')
    args.action(args)


if __name__ == '__main__':
    main()
