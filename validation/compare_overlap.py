"""Dense histogram comparisons for unit-weight, independent probability blocks.

Sample histograms approximate the original distributions, so they need not equal
box overlaps. Midpoint grids evaluate the box PDFs themselves and converge to the
exact C++ result as resolution increases. Neither comparison renormalizes boxes.
"""
import numpy as np


def compare_block(samples_a, samples_b, boxes_a, boxes_b, bins):
    """Return sample-histogram and box-midpoint overlap on a shared dense grid."""
    lower = np.minimum(samples_a.min(axis=0), samples_b.min(axis=0))
    upper = np.maximum(samples_a.max(axis=0), samples_b.max(axis=0))
    edges = [np.linspace(lo, hi, bins + 1) for lo, hi in zip(lower, upper)]
    cell_volume = np.prod((upper - lower) / bins)
    ha = np.histogramdd(samples_a, bins=edges)[0] / (len(samples_a) * cell_volume)
    hb = np.histogramdd(samples_b, bins=edges)[0] / (len(samples_b) * cell_volume)
    sample_overlap = np.sum(ha * hb) * cell_volume

    # Paint each box onto the grid by midpoint membership, avoiding a large
    # (number of cells, number of boxes, dimensions) temporary array.
    midpoints = [(edge[:-1] + edge[1:]) / 2 for edge in edges]
    def paint(boxes):
        pdf = np.zeros((bins,) * boxes.dimension)
        for lo, hi, density in zip(boxes.lower, boxes.upper, boxes.density):
            start = [np.searchsorted(x, l, side='left') for x, l in zip(midpoints, lo)]
            stop = [np.searchsorted(x, h, side='right') for x, h in zip(midpoints, hi)]
            pdf[tuple(slice(a, b) for a, b in zip(start, stop))] += density
        return pdf
    box_overlap = np.sum(paint(boxes_a) * paint(boxes_b)) * cell_volume
    return sample_overlap, box_overlap
