// Dense pair-shared uniform histograms for validation only. No density-box approximation.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

// A uniform-grid estimate avoids binary search. Correct against the actual edges
// so floating rounding agrees with numpy.histogramdd: internal edges belong to
// the bin on their right; the final edge belongs to the last bin.
static size_t bin_index(double x, const double* edge, size_t bins, double inverse_step) {
    size_t bin = std::min(size_t((x-edge[0])*inverse_step), bins-1);
    while (bin > 0 && x < edge[bin]) --bin;
    while (bin+1 < bins && x >= edge[bin+1]) ++bin;
    return bin;
}

static std::array<double, 2> overlaps(Array a, Array b, Array edges) {
    if (a.ndim()!=2 || b.ndim()!=2 || a.shape(1)!=5 || b.shape(1)!=5 ||
        edges.ndim()!=2 || edges.shape(0)!=5 || edges.shape(1)<2)
        throw std::invalid_argument("Use sample arrays (N,5) and uniform edges (5,bins+1)");
    size_t bins = edges.shape(1)-1;
    const double* grid = edges.data();
    std::array<double, 5> step, inverse;
    for (size_t k=0; k<5; ++k) {
        step[k] = (grid[k*(bins+1)+bins]-grid[k*(bins+1)])/bins;
        inverse[k] = 1/step[k];
    }
    std::array<double, 2> result;
    {
        py::gil_scoped_release release;
        std::array<std::vector<uint64_t>, 2> ha, hb;
        ha[0].resize(bins*bins*bins); ha[1].resize(bins*bins);
        hb[0].resize(bins*bins*bins); hb[1].resize(bins*bins);
        auto count = [&](const double* samples, size_t n,
                         std::array<std::vector<uint64_t>, 2>& histogram) {
            for (size_t i=0; i<n; ++i) {
                size_t cell0=0, cell1=0;
                for (size_t k=0; k<5; ++k) {
                    size_t bin = bin_index(samples[5*i+k], grid+k*(bins+1), bins, inverse[k]);
                    if (k<3) cell0 = cell0*bins+bin;
                    else cell1 = cell1*bins+bin;
                }
                ++histogram[0][cell0]; ++histogram[1][cell1];
            }
        };
        count(a.data(), a.shape(0), ha); count(b.data(), b.shape(0), hb);
        double normalization = double(a.shape(0))*double(b.shape(0));
        for (size_t block=0; block<2; ++block) {
            uint64_t dot=0;
            for (size_t i=0; i<ha[block].size(); ++i) dot += ha[block][i]*hb[block][i];
            double volume = block==0 ? step[0]*step[1]*step[2] : step[3]*step[4];
            result[block] = double(dot)/(normalization*volume);
        }
    }
    return result;
}

// Bounding bin ranges are a necessary support condition. Disjoint ranges in
// any coordinate imply a zero product histogram exactly, including shared-bin
// artifacts when two disjoint sample intervals fall inside the same coarse bin.
static bool share_bins(Array bounds_a, Array bounds_b, Array edges) {
    size_t bins = edges.shape(1)-1;
    const double* grid = edges.data();
    const double* a = bounds_a.data(); const double* b = bounds_b.data();
    for (size_t k=0; k<5; ++k) {
        const double* edge = grid+k*(bins+1);
        double inverse = double(bins)/(edge[bins]-edge[0]);
        size_t alo=bin_index(a[k],edge,bins,inverse);
        size_t ahi=bin_index(a[5+k],edge,bins,inverse);
        size_t blo=bin_index(b[k],edge,bins,inverse);
        size_t bhi=bin_index(b[5+k],edge,bins,inverse);
        if (ahi<blo || bhi<alo) return false;
    }
    return true;
}

PYBIND11_MODULE(_histogram, m) {
    m.doc() = "Serial dense histogram overlaps; sample arrays must lie within uniform edges";
    m.def("share_bins", &share_bins);
    m.def("block_overlaps", [](Array a, Array b, Array edges) {
        auto values=overlaps(a,b,edges);
        return py::make_tuple(values[0], values[1]);
    });
}
