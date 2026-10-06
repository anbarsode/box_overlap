"""Log-polynomial fits for overlap comparisons and measured computational costs.

Degree 1 gives y=10**c * x**b. Degree 2 gives
log10(y)=a*log10(x)**2+b*log10(x)+c, a curved relation rather than a pure power law.
Fits minimize squared vertical residuals in log10 space using numpy.polyfit.
"""
from pathlib import Path
import json
import os
import sys
for setting in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(setting, '1')
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation.plot_style import plot_style


def overlap_fits(histogram, boxes):
    """Fit box (y) against histogram (x), using only pairs positive in both.

    Return the selected count, descending-power coefficients and RMS residuals
    in dex (base-10 logarithmic units). Zero-valued pairs remain in the scatter
    but cannot enter a logarithmic fit. The fits are conditional y-on-x fits.
    """
    positive = (histogram>0)&(boxes>0)
    x, y = np.log10(histogram[positive]), np.log10(boxes[positive])
    fits = []
    for degree in (1,2):
        coefficients = np.polyfit(x,y,degree)
        residual = y-np.polyval(coefficients,x)
        fits.append({'degree':degree, 'coefficients':coefficients.tolist(),
                     'rms_log10_residual':float(np.sqrt(np.mean(residual**2)))})
    return {'fit_pairs':int(positive.sum()), 'fits':fits,
            'x_min':float(histogram[positive].min()),
            'x_max':float(histogram[positive].max())}


def plot_overlap_comparison(data, events, bins, output):
    """Plot all pair estimates, equality and both fitted log-space relations.

    data is a structured array with box_overlap and histogram_overlap columns.
    output is a JPG filename; a matching *_fits.json records equations and counts.
    Symlog axes keep zero estimates visible; fits are drawn within their x range.
    """
    x, y = data['histogram_overlap'], data['box_overlap']
    result = overlap_fits(x,y)
    result.update(events=events, bins=bins, pairs=len(data),
                  zero_box=int(np.count_nonzero(y==0)),
                  zero_histogram=int(np.count_nonzero(x==0)),
                  positive_box_zero_histogram=int(np.count_nonzero((y>0)&(x==0))),
                  zero_box_positive_histogram=int(np.count_nonzero((y==0)&(x>0))))
    plt = plot_style()
    fig, ax = plt.subplots(figsize=(11,8))
    ax.scatter(x,y,s=3,alpha=.16,color='navy',rasterized=True,label='Event-pair estimates')
    limit = max(x.max(),y.max())
    ax.plot([0,limit],[0,limit],color='gold',linestyle='--',label=r'Equality: $y=x$')
    grid = np.geomspace(result['x_min'],result['x_max'],300)
    for fit, color in zip(result['fits'],('crimson','teal')):
        coefficients = fit['coefficients']
        if fit['degree']==1:
            b,c = coefficients
            label = rf'Degree 1: $y=10^{{{c:.3f}}}x^{{{b:.3f}}}$'
        else:
            a,b,c = coefficients
            label = (rf'Degree 2: $\log_{{10}}y={a:.3f}(\log_{{10}}x)^2'
                     rf'{b:+.3f}\log_{{10}}x{c:+.3f}$')
        ax.plot(grid,10**np.polyval(coefficients,np.log10(grid)),color=color,label=label)
    threshold = limit*1e-6
    ax.set_xscale('symlog',linthresh=threshold);ax.set_yscale('symlog',linthresh=threshold)
    # Remove tiny linear-region decade labels that otherwise collide with zero.
    powers=np.arange(int(np.ceil(np.log10(threshold))),int(np.ceil(np.log10(limit)))+1)
    ticks=np.r_[0.,10.**powers]
    labels=['0']+[rf'$10^{{{power}}}$' for power in powers]
    ax.set_xticks(ticks,labels);ax.set_yticks(ticks,labels)
    ax.set_xlim(left=0);ax.set_ylim(bottom=0)
    ax.set_xlabel('Dense sample histogram overlap, x')
    ax.set_ylabel('Exact density-box overlap, y')
    ax.set_title(f'{events:,} events; {len(data):,} pairs; {bins} bins per axis\n'
                 f'Fits use {result["fit_pairs"]:,} pairs positive in both estimates')
    ax.legend(fontsize=11);ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(output,dpi=200);plt.close(fig)
    output = Path(output)
    output.with_name(output.stem+'_fits.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def plot_costs(data, output, fit_min=100, fit_max=10000, extrapolate_max=100000000):
    """Fit each timing component in log space, then sum its predicted seconds.

    For each stage j, log10(t_j) = a_j*log10(N)**2 + b_j*log10(N) + c_j.
    Total(N) = sum_j 10**polyval(coefficients_j, log10(N)). Fitting the measured
    total directly would hide initially small stages that grow faster than generation.
    Fits use measured N in [fit_min, fit_max]; dashed lines extend beyond fit_max.
    One-vs-set includes dataset generation/preparation/indexing and one query.
    PO2.0 costs 600*N + N*(N-1)/2 seconds. Hybrid adds the box total to
    600*N plus one second for each pair surviving exact box overlap. Beyond
    measured N, hybrid holds the largest measured dataset's rejection fraction fixed.
    Extrapolation is mathematical, not a validated large-N timing.
    """
    plt = plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    result = {'fit_min': fit_min, 'fit_max': fit_max, 'extrapolate_max': extrapolate_max,
              'definition': 'log10(stage_seconds) = a*log10(N)**2+b*log10(N)+c; '
                            'total_seconds = sum(predicted stage_seconds)', 'panels': []}
    selected = (data['N'] >= fit_min) & (data['N'] <= fit_max)
    x = np.log10(data['N'][selected])
    fit_n = np.geomspace(fit_min, fit_max, 100)
    extended_n = np.geomspace(fit_max, extrapolate_max, 200)
    prediction_n = np.array([10000, 100000, 1000000, 10000000, 100000000])
    for ax, find, calc, title in zip(axes, ('find_NvN_s', 'find_1vN_s'),
                                    ('overlap_NvN_s', 'overlap_1vN_s'),
                                    ('N within itself', 'One query vs N')):
        stages = [('box_generation_s', 'Generate boxes'), ('preparation_s', 'Prepare C++'),
                  ('index_build_s', 'Build index'), (find, 'Find pairs'), (calc, 'Integrate')]
        measured_total = sum(data[key] for key, _ in stages)
        fit_total = np.zeros_like(fit_n)
        extended_total = np.zeros_like(extended_n)
        predicted_total = np.zeros_like(prediction_n, dtype=float)
        component_results = []
        for key, label in stages:
            line, = ax.loglog(data['N'], data[key], marker='o', markersize=3, label=label)
            y = np.log10(data[key][selected])
            coefficients = np.polyfit(x, y, 2)
            fitted = 10**np.polyval(coefficients, np.log10(fit_n))
            extended = 10**np.polyval(coefficients, np.log10(extended_n))
            predicted = 10**np.polyval(coefficients, np.log10(prediction_n))
            ax.loglog(fit_n, fitted, color=line.get_color(), alpha=.7)
            ax.loglog(extended_n, extended, color=line.get_color(), linestyle='--')
            fit_total += fitted
            extended_total += extended
            predicted_total += predicted
            component_results.append({'key': key, 'label': label,
                'coefficients': coefficients.tolist(),
                'predicted_seconds': dict(zip(map(str, prediction_n), predicted.tolist())),
                'rms_log10_residual': float(np.sqrt(np.mean(
                    (y - np.polyval(coefficients, x))**2)))})
        ax.loglog(data['N'], measured_total, 'o-', color='black', markersize=4,
                  label='Total (sum of stages)')
        ax.loglog(fit_n, fit_total, color='black')
        ax.loglog(extended_n, extended_total, color='black', linestyle='--')
        comparison = None
        if find == 'find_NvN_s':
            # Unordered distinct pairs; NvN_pairs counts positive box intersections.
            pair_count = data['N'] * (data['N'] - 1) / 2
            rejected = pair_count - data['NvN_pairs']
            last = np.argmax(data['N'])
            rejection_fraction = float(rejected[last] / pair_count[last])
            po_measured = 600 * data['N'] + pair_count
            hybrid_measured = measured_total + 600 * data['N'] + data['NvN_pairs']
            extended_pairs = extended_n * (extended_n - 1) / 2
            po_extended = 600 * extended_n + extended_pairs
            hybrid_extended = extended_total + 600 * extended_n
            hybrid_extended += (1 - rejection_fraction) * extended_pairs
            for label, measured, extended, color in (
                ('PO2.0', po_measured, po_extended, 'gold'),
                ('hybrid', hybrid_measured, hybrid_extended, 'dodgerblue')):
                ax.loglog(data['N'], measured, color=color, label=label,
                          marker='o' if label == 'PO2.0' else None,
                          markersize=4, markerfacecolor='none')
                ax.loglog(extended_n, extended, color=color, linestyle='--')
            predicted_pairs = prediction_n.astype(float) * (prediction_n - 1) / 2
            po_predicted = 600 * prediction_n + predicted_pairs
            hybrid_predicted = predicted_total + 600 * prediction_n
            hybrid_predicted += (1 - rejection_fraction) * predicted_pairs
            comparison = {'preprocessing_seconds_per_event': 600,
                'seconds_per_pair': 1,
                'hybrid_pair_rule': 'pairs surviving exact box overlap',
                'extrapolated_rejection_fraction': rejection_fraction,
                'rejection_fraction_source_N': float(data['N'][last]),
                'PO2.0_seconds': po_measured.tolist(),
                'hybrid_seconds': hybrid_measured.tolist(),
                'PO2.0_predicted_seconds': dict(zip(map(str, prediction_n), po_predicted.tolist())),
                'hybrid_predicted_seconds': dict(zip(map(str, prediction_n),
                                                     hybrid_predicted.tolist()))}
        # A single style key keeps the legend readable while every stage is extrapolated.
        ax.plot([], [], color='grey', linestyle='--', label='Dashed: extrapolated')
        result['panels'].append({'title': title, 'components': component_results,
            'fit_points': int(selected.sum()),
            'predicted_seconds': dict(zip(map(str, prediction_n), predicted_total.tolist())),
            'measured_total_seconds': measured_total.tolist(), 'comparison': comparison})
        ax.set_xlabel('Number of events, N')
        ax.set_ylabel('Wall time (s)')
        ax.set_title(title)
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(alpha=.15)
    fig.suptitle('Component fits and their summed cost (unvalidated beyond 10,000 events)')
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    Path(output).with_name('cost_extrapolation.json').write_text(json.dumps(result, indent=2)+'\n')
    return result
