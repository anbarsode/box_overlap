"""Rerun correctness tests and scientific measurements on the current dataset.

Run in gw from the project root: python validation/rerun_analysis.py
The default is the complete 10,000-event analysis. Jobs run sequentially to avoid
competing for CPU during timing. Raw samples are never accumulated across events.
Numeric results replace the main result folders; the consolidated report and
notebook are updated after these jobs complete. This script records each command,
wall time, source hashes and the dataset's filename/size/mtime digest.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

for setting in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[setting] = '1'
ROOT = Path(__file__).resolve().parents[1]


def input_digest(folder: Path, n: int) -> str:
    """Identify a completed sample-file state, without reading all raw contents."""
    digest = hashlib.sha256()
    for i in range(n):
        info = (folder / f'event_{i}.npy').stat()
        digest.update(f'{i},{info.st_size},{info.st_mtime_ns}\n'.encode())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=int, default=10000,
                        help='Analysis endpoint, default 10000; at least 105 for the examples')
    parser.add_argument('--study-events', type=int, default=1000,
                        help='Events for controlled trial/order study (default 1000)')
    args = parser.parse_args()
    if args.events < 105 or args.study_events < 2 or args.study_events > args.events:
        parser.error('Use at least 105 events and a study count between 2 and the endpoint')
    folder = ROOT / 'validation'
    dataset = folder / 'ARAK_samples'
    result = {'status': 'running', 'environment': 'gw', 'events': args.events,
              'study_events': args.study_events, 'jobs': [],
              'input_metadata_sha256': input_digest(dataset, args.events)}
    output = folder / 'review_checks.json'
    commands = [['-m', 'bo.build_overlap'], ['-m', 'validation.histogram_overlap'],
                ['validation/test_histogram.py'], ['validation/test_analysis_fits.py']]
    commands += [[f'validation/{name}'] for name in
                 ('test_event_boxes.py', 'test_overlap.py', 'test_pair_finder.py',
                  'test_sample_streaming.py', 'test_run.py')]
    commands += [
        ['validation/rejection_study.py', '--events', str(args.study_events)],
        ['validation/compare_rejection.py', '--events', str(args.events),
         '--compare-levels', '1', '2', '3', '--ntrial', '5', '10',
         '--batch-events', '64', '--output', 'validation/comparison_results'],
        ['validation/validate_pair_finder.py', '--max-events', str(args.events),
         '--ntrial', '5', '10', '--subdivisions', '0', '3',
         '--batch-events', '64', '--skip-histograms'],
        ['validation/histogram_comparison.py', '--events', str(min(1000,args.events)),
         '--bins', '24'],
    ]
    log_folder = folder / 'analysis_logs'
    log_folder.mkdir(exist_ok=True)
    for number, command in enumerate(commands):
        print(f'Starting job {number+1}/{len(commands)}: {" ".join(command)}', flush=True)
        started = time.perf_counter()
        # Mirror complete output into a log while keeping live progress visible.
        log = log_folder / f'job_{number:02d}.log'
        with log.open('w') as destination:
            process = subprocess.Popen([sys.executable, '-u', *command], cwd=ROOT,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                destination.write(line); destination.flush()
                print(line, end='', flush=True)
            code = process.wait()
        result['jobs'].append({'command': [sys.executable, *command], 'exit_code': code,
                              'wall_s': time.perf_counter()-started,
                              'log': str(log.relative_to(ROOT))})
        output.write_text(json.dumps(result, indent=2)+'\n')
        if code:
            result['status'] = 'failed'
            output.write_text(json.dumps(result, indent=2)+'\n')
            raise SystemExit(code)
    result['input_metadata_sha256_after'] = input_digest(dataset, args.events)
    if result['input_metadata_sha256_after'] != result['input_metadata_sha256']:
        result['status'] = 'input changed'
        output.write_text(json.dumps(result, indent=2)+'\n')
        raise RuntimeError('Sample files changed during analysis; measurements cannot be combined')
    result['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (ROOT/'bo').glob('*') if p.suffix in ('.py', '.cpp')}
    result['status'] = 'all passed'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print('All tests and measurement jobs completed on unchanged inputs.', flush=True)
    subprocess.run([sys.executable, str(folder/'update_results.py')], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
