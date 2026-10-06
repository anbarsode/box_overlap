r"""Compare the spatial overlap integrands of two ARAK events.

For a displayed coordinate pair (i,j), the field is
G_ij(theta_i,theta_j) = integral P_A(theta) P_B(theta) d theta_remaining.
Full-dimensional box intersections are computed before projection. Their
contributions are averaged exactly over display pixels, so the pixel integral
matches the analytic box overlap at any display resolution. Histogram fields
multiply the full-dimensional histograms before summing the omitted coordinate.
Plots use unit weights and shared histogram edges for both events.
"""
from itertools import combinations
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bo import make_event_boxes, overlap_1v1
from validate_pair_finder import plot_style


def block_maps(samples_a, samples_b, boxes_a, boxes_b, bins=96):
    """Return overlap projections, per-event histogram marginals and block integrals."""
    lower = np.minimum(samples_a.min(0), samples_b.min(0))
    upper = np.maximum(samples_a.max(0), samples_b.max(0))
    edges = [np.linspace(lo, hi, bins+1) for lo, hi in zip(lower, upper)]
    step = (upper-lower)/bins
    cell_volume = np.prod(step)
    ha = np.histogramdd(samples_a, bins=edges)[0]/(len(samples_a)*cell_volume)
    hb = np.histogramdd(samples_b, bins=edges)[0]/(len(samples_b)*cell_volume)
    histogram_product = ha*hb
    histogram_integral = histogram_product.sum()*cell_volume

    lo = np.maximum(boxes_a.lower[:, None], boxes_b.lower[None])
    hi = np.minimum(boxes_a.upper[:, None], boxes_b.upper[None])
    hit = np.all(hi > lo, axis=2)
    lo, hi = lo[hit], hi[hit]
    density = (boxes_a.density[:, None]*boxes_b.density[None])[hit]
    box_integral = np.sum(density*np.prod(hi-lo, axis=1))
    maps = []
    for i,j in combinations(range(samples_a.shape[1]), 2):
        omitted = tuple(k for k in range(samples_a.shape[1]) if k not in (i,j))
        if omitted:
            hist = histogram_product.sum(axis=omitted)*np.prod(step[list(omitted)])
            marginal_a = ha.sum(axis=omitted)*np.prod(step[list(omitted)])
            marginal_b = hb.sum(axis=omitted)*np.prod(step[list(omitted)])
            projected_density = density*np.prod((hi-lo)[:, list(omitted)], axis=1)
        else:
            hist, marginal_a, marginal_b = histogram_product, ha, hb
            projected_density = density
        box = np.zeros((bins,bins))
        # Exact pixel-area averaging of projected intersection rectangles.
        # Width products avoid allocating a 3D raster for the box approximation.
        for l,h,rho in zip(lo,hi,projected_density):
            dx = np.maximum(0, np.minimum(edges[i][1:],h[i])
                             - np.maximum(edges[i][:-1],l[i]))
            dy = np.maximum(0, np.minimum(edges[j][1:],h[j])
                             - np.maximum(edges[j][:-1],l[j]))
            box += rho*np.outer(dx,dy)/(step[i]*step[j])
        # A projection preserves the full block integral, unlike products of
        # separately projected PDFs. This is also an independent plotting check.
        np.testing.assert_allclose(box.sum()*step[i]*step[j], box_integral,
                                   rtol=2e-13, atol=1e-14)
        np.testing.assert_allclose(hist.sum()*step[i]*step[j], histogram_integral,
                                   rtol=2e-13, atol=1e-14)
        maps.append(dict(coordinates=(i,j), edges=(edges[i],edges[j]), box=box,
                         hist=hist, marginal_a=marginal_a, marginal_b=marginal_b,
                         box_integral=box_integral, histogram_integral=histogram_integral))
    return maps


def plot_intersection(samples_a, samples_b, event_a, event_b, idx1, idx2, bins=96,
                      output=None, zoom=False):
    """Plot both block integrands; return figure and the comparison maps.

    Navy/coral outlines in the box row show the two events' boxes. Contours in
    the histogram row show 10%, 50%, and 90% of each marginal PDF's peak value.
    All weights must be unity because the histogram weight is currently unity.
    """
    for event in (event_a,event_b):
        if any(np.any(b.weight != 1) for b in event.blocks):
            raise ValueError('This histogram comparison uses unit weights')
    plt = plot_style()
    from matplotlib.colors import LogNorm
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    maps=[]
    for block, columns, offset in ((0,slice(0,3),0),(1,slice(3,5),3)):
        current=block_maps(samples_a[:,columns],samples_b[:,columns],
                           event_a.blocks[block],event_b.blocks[block],bins)
        for item in current:
            item.update(block=block,offset=offset)
        maps.extend(current)
    fig,axes=plt.subplots(2,4,figsize=(20,11),layout='constrained')
    for column,item in enumerate(maps):
        i,j=item['coordinates']; x,y=item['edges']
        values=np.r_[item['box'][item['box']>0],item['hist'][item['hist']>0]]
        vmax=values.max() if len(values) else 1.
        norm=LogNorm(vmin=vmax*1e-4,vmax=vmax)
        for row,key in enumerate(('box','hist')):
            ax=axes[row,column]
            field=np.ma.masked_less_equal(item[key],0)
            mesh=ax.pcolormesh(x,y,field.T,cmap='viridis',norm=norm,rasterized=True)
            if row==0:
                for event,color in ((event_a,'navy'),(event_b,'coral')):
                    b=event.blocks[item['block']]
                    patches=[Rectangle((lo[i],lo[j]),hi[i]-lo[i],hi[j]-lo[j])
                             for lo,hi in zip(b.lower,b.upper)]
                    ax.add_collection(PatchCollection(patches,facecolor='none',
                        edgecolor=color,linewidth=.6,alpha=.6))
            else:
                xm=(x[:-1]+x[1:])/2; ym=(y[:-1]+y[1:])/2
                for marginal,color in ((item['marginal_a'],'navy'),
                                       (item['marginal_b'],'coral')):
                    ax.contour(xm,ym,marginal.T,levels=marginal.max()*np.array([.1,.5,.9]),
                               colors=color,linewidths=.65)
            ax.set_xlim(x[0],x[-1]); ax.set_ylim(y[0],y[-1])
            if zoom:
                support = np.argwhere((item['box'] > 0) | (item['hist'] > 0))
                if len(support):
                    first, last = support.min(0), support.max(0)+1
                    dx, dy = x[last[0]]-x[first[0]], y[last[1]]-y[first[1]]
                    ax.set_xlim(x[first[0]]-.05*dx, x[last[0]]+.05*dx)
                    ax.set_ylim(y[first[1]]-.05*dy, y[last[1]]+.05*dy)
            ax.set_xlabel(rf'$\theta_{{{i+item["offset"]+1}}}$')
            ax.set_ylabel(rf'$\theta_{{{j+item["offset"]+1}}}$')
            method='Boxes' if row==0 else f'Histogram ({bins} bins)'
            value=item['box_integral'] if row==0 else item['histogram_integral']
            name='3D projection' if item['block']==0 else '2D block'
            ax.set_title(f'{method}: {name}\nblock integral = {value:.5g}')
        bar=fig.colorbar(mesh,ax=axes[:,column],location='bottom',shrink=.9,pad=.05)
        bar.set_label('Overlap density')
    box_value=overlap_1v1(event_a,event_b)
    np.testing.assert_allclose(maps[0]['box_integral']*maps[-1]['box_integral'],
                               box_value,rtol=2e-13)
    hist_value=maps[0]['histogram_integral']*maps[-1]['histogram_integral']
    view = ' (zoom to overlap support)' if zoom else ''
    fig.suptitle(f'Events {idx1} and {idx2}{view}, unit weights: '
                 f'event overlap boxes = {box_value:.6g}, histogram = {hist_value:.6g}\n'
                 'Navy: event A; coral: event B; blank pixels: zero overlap',fontsize=15)
    if output is not None:
        fig.savefig(output,dpi=200)
    return fig,maps


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--idx1',type=int,default=66)
    parser.add_argument('--idx2',type=int,default=71)
    parser.add_argument('--bins',type=int,default=96)
    args=parser.parse_args()
    folder=Path(__file__).parent
    samples=[np.load(folder/'ARAK_samples'/f'event_{i}.npy', mmap_mode='r') for i in (args.idx1,args.idx2)]
    events=[make_event_boxes(x,rng=np.random.default_rng(91+i))
            for x,i in zip(samples,(args.idx1,args.idx2))]
    out=folder/'pair_finder_results'/f'1v1_{args.idx1}_{args.idx2}.jpg'
    out.parent.mkdir(exist_ok=True)
    fig,maps=plot_intersection(*samples,*events,args.idx1,args.idx2,args.bins,out)
    print(f'Saved {out}')
