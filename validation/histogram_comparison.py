"""Compare every pair among 1,000 events using 24 bins per axis, with log fits.

Raw samples are streamed one event at a time for boxes and two at a time for
histograms. No event pairs are filtered. Exact dense histogram bin counts are
computed by the separately verified serial C++ validation helper. Outputs replace
histogram_comparison.csv/.jpg and save settings, timings and fit coefficients.
"""
from pathlib import Path
import argparse
import csv
import json
import os
import sys
import time
import warnings
for setting in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[setting]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bo import make_event_boxes,prepare_event,prepare_events,overlap_NvN_pairs
from validation.histogram_overlap import shared_edges
from validation import _histogram
from validation.analysis_plots import plot_overlap_comparison
from validation.rerun_analysis import input_digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events',type=int,default=1000)
    parser.add_argument('--bins',type=int,default=24)
    parser.add_argument('--output',type=Path,default=ROOT/'validation/pair_finder_results')
    args=parser.parse_args()
    samples_folder=ROOT/'validation/ARAK_samples'
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    settings={'status':'running','events':args.events,'bins':args.bins,
              'ntrial':[5,10],'seed':91,'n_centers':[4,8],
              'input_metadata_sha256':input_digest(samples_folder,args.events),
              'histogram_method':'exact serial C++ counts, NumPy uniform edges'}
    settings_path=out/'histogram_settings.json'
    def checkpoint():
        settings_path.write_text(json.dumps(settings,indent=2)+'\n')
    checkpoint()
    prepared=[];lower=[];upper=[]
    started=time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        for i in range(args.events):
            samples=np.load(samples_folder/f'event_{i}.npy',mmap_mode='r')
            lower.append(samples.min(0));upper.append(samples.max(0))
            prepared.append(prepare_event(make_event_boxes(samples,
                rng=np.random.default_rng(91+i),ntrial=(5,10))))
            del samples
    settings['generation_and_preparation_s']=time.perf_counter()-started
    lower,upper=np.array(lower),np.array(upper)
    pairs=np.array([(i,j) for i in range(args.events) for j in range(i+1,args.events)],
                   dtype=np.int64)
    started=time.perf_counter()
    fast=overlap_NvN_pairs(prepare_events(prepared),pairs)
    settings['box_overlap_s']=time.perf_counter()-started
    histogram=np.empty(len(pairs))
    counted_pairs=0
    bounds=np.stack((lower,upper),axis=1)
    started=time.perf_counter();last_progress=started;offset=0
    for i in range(args.events-1):
        a=np.load(samples_folder/f'event_{i}.npy',mmap_mode='r')
        for j in range(i+1,args.events):
            edges=shared_edges(bounds[i],bounds[j],args.bins)
            if _histogram.share_bins(bounds[i],bounds[j],edges):
                b=np.load(samples_folder/f'event_{j}.npy',mmap_mode='r')
                values=_histogram.block_overlaps(a,b,edges)
                histogram[offset]=values[0]*values[1]
                counted_pairs+=1
                del b
            else:
                # No shared bin in one coordinate: the histogram sum is exactly
                # zero. No sample file is opened for this second event.
                histogram[offset]=0.
            offset+=1
        del a
        if time.perf_counter()-last_progress>30:
            settings.update(completed_pairs=offset,histogram_elapsed_s=time.perf_counter()-started)
            checkpoint();last_progress=time.perf_counter()
            print(f'Histogram pairs: {offset:,}/{len(pairs):,}',flush=True)
    settings['histogram_s']=time.perf_counter()-started
    settings['explicitly_counted_pairs']=counted_pairs
    settings['exact_zero_bin_rejections']=len(pairs)-counted_pairs
    assert input_digest(samples_folder,args.events)==settings['input_metadata_sha256']
    np.savetxt(out/'histogram_comparison.csv',np.column_stack((pairs,fast,histogram)),
               fmt=('%d','%d','%.17g','%.17g'),delimiter=',',comments='',
               header='event_a,event_b,box_overlap,histogram_overlap')
    data=np.empty(len(pairs),dtype=[('box_overlap','f8'),('histogram_overlap','f8')])
    data['box_overlap']=fast;data['histogram_overlap']=histogram
    fits=plot_overlap_comparison(data,args.events,args.bins,out/'histogram_comparison.jpg')
    settings.update(status='complete',pairs=len(pairs),completed_pairs=len(pairs))
    checkpoint()
    print(json.dumps({'settings':settings,'fits':fits},indent=2),flush=True)


if __name__=='__main__':
    main()
