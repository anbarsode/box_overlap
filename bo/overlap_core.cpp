// Single-threaded constant-density box overlap, with a sorted interval sweep.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

struct Block {
    size_t n, d, axis, stride;
    std::vector<double> rows, inverse_volume, lower, upper;

    Block(Array input, size_t coordinate) : axis(coordinate) {
        if (input.ndim() != 2 || input.shape(1) < 4 || (input.shape(1) - 2) % 2)
            throw std::invalid_argument("Box rows must have shape (n, 2*d+2)");
        n = input.shape(0);
        stride = input.shape(1);
        d = (stride - 2) / 2;
        if (axis >= d) throw std::invalid_argument("Ordering coordinate outside block");
        // Own one compact copy, so subsequent changes in Python cannot invalidate the sweep.
        rows.resize(n * stride);
        std::vector<size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        const double* source = input.data();
        std::stable_sort(order.begin(), order.end(), [&](size_t i, size_t j) {
            return source[i * stride + axis] < source[j * stride + axis];
        });
        lower.assign(d, 0);
        upper.assign(d, 0);
        inverse_volume.resize(n);
        for (size_t i = 0; i < n; ++i) {
            double* row = rows.data() + i * stride;
            std::copy_n(source + order[i] * stride, stride, row);
            double volume = 1;
            for (size_t k = 0; k < d; ++k) {
                volume *= row[d + k] - row[k];
                if (i == 0 || row[k] < lower[k]) lower[k] = row[k];
                if (i == 0 || row[d + k] > upper[k]) upper[k] = row[d + k];
            }
            inverse_volume[i] = 1 / volume;
        }
    }
};

struct Event {
    std::array<Block, 2> blocks;
    Event(Array first, Array second, size_t axis0, size_t axis1)
        : blocks{Block(first, axis0), Block(second, axis1)} {}
};
using EventPtr = std::shared_ptr<Event>;
struct Events {
    std::vector<EventPtr> values;
    explicit Events(std::vector<EventPtr> events) : values(std::move(events)) {}
};

static double block_overlap(const Block& a, const Block& b) {
    if (a.d != b.d || a.axis != b.axis)
        throw std::invalid_argument("Events must use matching dimensions and ordering coordinates");
    if (!a.n || !b.n) return 0;
    for (size_t k = 0; k < a.d; ++k)
        if (a.upper[k] <= b.lower[k] || b.upper[k] <= a.lower[k]) return 0;

    double total = 0;
    size_t start = 0;
    for (size_t i = 0; i < a.n; ++i) {
        const double* x = a.rows.data() + i * a.stride;
        // Lower bounds of A are monotone. Only discard a PREFIX of B here:
        // sorted lower bounds do not imply sorted upper bounds.
        while (start < b.n && b.rows[start * b.stride + b.d + b.axis] <= x[a.axis])
            ++start;
        for (size_t j = start; j < b.n; ++j) {
            const double* y = b.rows.data() + j * b.stride;
            if (y[b.axis] >= x[a.d + a.axis]) break;
            double width = std::min(x[a.d + a.axis], y[b.d + b.axis])
                         - std::max(x[a.axis], y[b.axis]);
            if (width <= 0) continue;
            double volume = width;
            for (size_t k = 0; k < a.d; ++k) {
                if (k == a.axis) continue;
                width = std::min(x[a.d + k], y[b.d + k]) - std::max(x[k], y[k]);
                if (width <= 0) { volume = 0; break; }
                volume *= width;
            }
            if (volume == 0) continue;
            // fA=Vintersection/VA, fB=Vintersection/VB. Cancel Vintersection
            // in (fA*wA + fB*wB)/(fA+fB); inverse volumes are cached once.
            double effective_weight = (x[a.stride - 1] * a.inverse_volume[i]
                                     + y[b.stride - 1] * b.inverse_volume[j])
                                    / (a.inverse_volume[i] + b.inverse_volume[j]);
            total += volume * x[a.stride - 2] * y[b.stride - 2] * effective_weight;
        }
    }
    return total;
}

static double event_overlap(const Event& a, const Event& b) {
    // The second (normally 2D) block is cheaper and is checked first.
    double second = block_overlap(a.blocks[1], b.blocks[1]);
    if (second == 0) return 0;
    return second * block_overlap(a.blocks[0], b.blocks[0]);
}


// Geometry-only existence check. Densities and weights do not influence filtering.
static bool block_intersects(const Block& a, const Block& b) {
    if (a.d != b.d || a.axis != b.axis)
        throw std::invalid_argument("Events must use matching dimensions and ordering coordinates");
    if (!a.n || !b.n) return false;
    for (size_t k = 0; k < a.d; ++k)
        if (a.upper[k] <= b.lower[k] || b.upper[k] <= a.lower[k]) return false;
    size_t start = 0;
    for (size_t i = 0; i < a.n; ++i) {
        const double* x = a.rows.data() + i * a.stride;
        while (start < b.n && b.rows[start*b.stride + b.d + b.axis] <= x[a.axis]) ++start;
        for (size_t j = start; j < b.n; ++j) {
            const double* y = b.rows.data() + j*b.stride;
            if (y[b.axis] >= x[a.d + a.axis]) break;
            bool hit = true;
            for (size_t k = 0; k < a.d; ++k)
                if (x[a.d+k] <= y[k] || y[b.d+k] <= x[k]) { hit = false; break; }
            if (hit) return true;
        }
    }
    return false;
}

struct Node {
    std::vector<double> lower, upper;
    size_t begin, end;
    int left = -1, right = -1;
};

// A balanced BVH over EVENT bounds only. Individual boxes use raw sorted loops.
// Nodes store both blocks' bound unions, with at most 8 events per leaf.
// Bounds use second-block coordinates first, preserving the 2D-first rejection order.
static double bound(const Event& event, size_t coordinate, bool upper) {
    const size_t second_d = event.blocks[1].d;
    const Block& b = event.blocks[coordinate < second_d ? 1 : 0];
    size_t k = coordinate < second_d ? coordinate : coordinate-second_d;
    return upper ? b.upper[k] : b.lower[k];
}
struct EventIndex {
    std::vector<EventPtr> events;
    std::vector<size_t> order;
    std::vector<Node> nodes;
    size_t d = 0;

    explicit EventIndex(const Events& input) : events(input.values) {
        if (events.empty()) return;
        d = events[0]->blocks[1].d + events[0]->blocks[0].d;
        for (size_t i = 0; i < events.size(); ++i) {
            for (size_t k = 0; k < 2; ++k)
                if (events[i]->blocks[k].d != events[0]->blocks[k].d ||
                    events[i]->blocks[k].axis != events[0]->blocks[k].axis)
                    throw std::invalid_argument("Inconsistent event block dimensions or ordering");
            if (events[i]->blocks[0].n && events[i]->blocks[1].n) order.push_back(i);
        }
        if (!order.empty()) build(0, order.size());
    }

    int build(size_t begin, size_t end) {
        Node node;
        node.begin = begin; node.end = end;
        node.lower.resize(d); node.upper.resize(d);
        for (size_t k = 0; k < d; ++k) {
            node.lower[k] = bound(*events[order[begin]], k, false);
            node.upper[k] = bound(*events[order[begin]], k, true);
        }
        for (size_t i = begin+1; i < end; ++i) {
            const Event& event = *events[order[i]];
            for (size_t k = 0; k < d; ++k) {
                node.lower[k] = std::min(node.lower[k], bound(event, k, false));
                node.upper[k] = std::max(node.upper[k], bound(event, k, true));
            }
        }
        int id = nodes.size();
        nodes.push_back(std::move(node));
        if (end-begin <= 8) return id;
        size_t axis = 0;
        double best = -1;
        // Split where the event centers have the largest spread.
        for (size_t k = 0; k < d; ++k) {
            double lo = bound(*events[order[begin]], k, false)
                      + bound(*events[order[begin]], k, true);
            double hi = lo;
            for (size_t i = begin+1; i < end; ++i) {
                double center = bound(*events[order[i]], k, false)
                              + bound(*events[order[i]], k, true);
                lo = std::min(lo, center); hi = std::max(hi, center);
            }
            if (hi-lo > best) { best = hi-lo; axis = k; }
        }
        size_t middle = begin + (end-begin)/2;
        std::nth_element(order.begin()+begin, order.begin()+middle, order.begin()+end,
            [&](size_t i, size_t j) {
                return bound(*events[i], axis, false)+bound(*events[i], axis, true)
                     < bound(*events[j], axis, false)+bound(*events[j], axis, true);
            });
        int left = build(begin, middle), right = build(middle, end);
        nodes[id].left = left; nodes[id].right = right;
        return id;
    }

    void visit(int id, const Event& query, int64_t after, std::vector<int64_t>& hits) const {
        const Node& node = nodes[id];
        for (size_t k = 0; k < d; ++k)
            if (bound(query, k, true) <= node.lower[k] ||
                node.upper[k] <= bound(query, k, false)) return;
        if (node.left >= 0) {
            visit(node.left, query, after, hits); visit(node.right, query, after, hits);
            return;
        }
        for (size_t i = node.begin; i < node.end; ++i) {
            size_t j = order[i];
            if (int64_t(j) <= after) continue;
            const Event& other = *events[j];
            if (block_intersects(query.blocks[1], other.blocks[1]) &&
                block_intersects(query.blocks[0], other.blocks[0])) hits.push_back(j);
        }
    }

    std::vector<int64_t> query(const Event& event, int64_t after = -1) const {
        std::vector<int64_t> hits;
        if (events.empty()) return hits;
        for (size_t k = 0; k < 2; ++k)
            if (event.blocks[k].d != events[0]->blocks[k].d ||
                event.blocks[k].axis != events[0]->blocks[k].axis)
                throw std::invalid_argument("Query dimensions or ordering do not match index");
        if (!nodes.empty() && event.blocks[0].n && event.blocks[1].n)
            visit(0, event, after, hits);
        std::sort(hits.begin(), hits.end());
        return hits;
    }
};

static py::array_t<int64_t> find_pairs_range(const EventIndex& index,
                                            size_t begin, size_t end) {
        if (begin > end || end > index.events.size())
            throw std::out_of_range("Event range outside index");
        std::vector<std::array<int64_t, 2>> pairs;
        {
            py::gil_scoped_release release;
            for (size_t i = begin; i < end; ++i)
                for (int64_t j : index.query(*index.events[i], i))
                    pairs.push_back({int64_t(i), j});
        }
        py::array_t<int64_t> result({py::ssize_t(pairs.size()), py::ssize_t(2)});
        int64_t* out = result.mutable_data();
        for (size_t i = 0; i < pairs.size(); ++i) {
            out[2*i] = pairs[i][0]; out[2*i+1] = pairs[i][1];
        }
        return result;
}

PYBIND11_MODULE(_overlap, m) {
    m.doc() = "Single-threaded sorted box overlap calculations";
    py::class_<Event, EventPtr>(m, "PreparedEvent")
        .def(py::init<Array, Array, size_t, size_t>());
    py::class_<Events>(m, "PreparedEvents")
        .def(py::init<std::vector<EventPtr>>())
        .def("__len__", [](const Events& e) { return e.values.size(); });

    py::class_<EventIndex>(m, "EventIndex")
        .def(py::init<const Events&>(), py::call_guard<py::gil_scoped_release>())
        .def("__len__", [](const EventIndex& x) { return x.events.size(); });
    m.def("find_1vN", [](const Event& event, const EventIndex& index) {
        std::vector<int64_t> hits;
        { py::gil_scoped_release release; hits = index.query(event); }
        py::array_t<int64_t> result(hits.size());
        std::copy(hits.begin(), hits.end(), result.mutable_data());
        return result;
    });
    m.def("find_NvN", [](const EventIndex& index) {
        return find_pairs_range(index, 0, index.events.size());
    });
    m.def("find_NvN_range", &find_pairs_range);
    m.def("overlap_1v1", [](const Event& a, const Event& b) {
        py::gil_scoped_release release;
        return event_overlap(a, b);
    });
    m.def("overlap_1vN", [](const Event& a, const Events& events) {
        py::array_t<double> result(events.values.size());
        double* output = result.mutable_data();
        {
            py::gil_scoped_release release;
            for (size_t j = 0; j < events.values.size(); ++j)
                output[j] = event_overlap(a, *events.values[j]);
        }
        return result;
    });
    m.def("overlap_NvN_pairs", [](const Events& events,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> pairs) {
        if (pairs.ndim() != 2 || pairs.shape(1) != 2)
            throw std::invalid_argument("event_pairs must have shape (n_pairs, 2)");
        auto indices = pairs.unchecked<2>();
        for (py::ssize_t i = 0; i < pairs.shape(0); ++i)
            for (int k = 0; k < 2; ++k)
                if (indices(i, k) < 0 || size_t(indices(i, k)) >= events.values.size())
                    throw std::out_of_range("Event pair index outside event set");
        py::array_t<double> result(pairs.shape(0));
        double* output = result.mutable_data();
        {
            py::gil_scoped_release release;
            for (py::ssize_t i = 0; i < pairs.shape(0); ++i)
                output[i] = event_overlap(*events.values[indices(i, 0)],
                                          *events.values[indices(i, 1)]);
        }
        return result;
    });
    m.def("overlap_NvN", [](const Events& events) {
        py::ssize_t n = events.values.size();
        py::array_t<double> result({n, n});
        double* output = result.mutable_data();
        {
            py::gil_scoped_release release;
            for (py::ssize_t i = 0; i < n; ++i)
                for (py::ssize_t j = i; j < n; ++j) {
                    double value = event_overlap(*events.values[i], *events.values[j]);
                    output[i * n + j] = output[j * n + i] = value;
                }
        }
        return result;
    });
}
