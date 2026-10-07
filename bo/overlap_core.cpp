// Single-threaded constant-density box overlap, with a sorted interval sweep.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <memory>
#include <limits>
#include <iterator>
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
        // Python BoxSet normally already supplies this order. Also accept raw unsorted rows.
        auto less = [&](size_t i, size_t j) {
            return source[i * stride + axis] < source[j * stride + axis];
        };
        if (!std::is_sorted(order.begin(), order.end(), less))
            std::stable_sort(order.begin(), order.end(), less);
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
    // Check the first (normally 3D) block before the second (normally 2D).
    double first = block_overlap(a.blocks[0], b.blocks[0]);
    if (first == 0) return 0;
    return first * block_overlap(a.blocks[1], b.blocks[1]);
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

// Whole-box subdivision levels. Each level splits EVERY splittable group.
// For each group, examine cuts between adjacent box centers on every axis,
// minimising the sum of the two enclosing hypervolumes. Prefix/suffix bound
// unions evaluate all cuts in O(d*n*log(n)) work per level. This is a local
// minimum over center-ordered cuts, not a global optimum over groupings.
// Whole boxes are never cut, so enclosing bounds cover every density box.
struct Bounds {
    std::vector<double> lower, upper;
    double volume() const {
        double v = 1;
        for (size_t k = 0; k < lower.size(); ++k) v *= upper[k] - lower[k];
        return v;
    }
};
static Bounds box_bounds(const Block& b, size_t i) {
    const double* row = b.rows.data() + i*b.stride;
    return {std::vector<double>(row, row+b.d),
            std::vector<double>(row+b.d, row+2*b.d)};
}
static void extend(Bounds& a, const Bounds& b) {
    for (size_t k = 0; k < a.lower.size(); ++k) {
        a.lower[k] = std::min(a.lower[k], b.lower[k]);
        a.upper[k] = std::max(a.upper[k], b.upper[k]);
    }
}
static Bounds group_bounds(const Block& b, const std::vector<size_t>& group) {
    Bounds bounds = box_bounds(b, group[0]);
    for (size_t i = 1; i < group.size(); ++i) extend(bounds, box_bounds(b, group[i]));
    return bounds;
}
static std::vector<Bounds> partition_block(const Block& b, size_t levels) {
    if (!b.n) return {};
    if (levels == 0) return {{b.lower, b.upper}};
    std::vector<std::vector<size_t>> groups(1);
    groups[0].resize(b.n);
    std::iota(groups[0].begin(), groups[0].end(), 0);
    for (size_t level = 0; level < levels && groups.size() < b.n; ++level) {
        std::vector<std::vector<size_t>> next;
        for (const auto& group : groups) {
            if (group.size() < 2) { next.push_back(group); continue; }
            double best_cost = std::numeric_limits<double>::infinity();
            size_t best_cut = 0;
            std::vector<size_t> best_order;
            for (size_t axis = 0; axis < b.d; ++axis) {
                auto order = group;
                std::stable_sort(order.begin(), order.end(), [&](size_t i, size_t j) {
                    const double* x = b.rows.data()+i*b.stride;
                    const double* y = b.rows.data()+j*b.stride;
                    return x[axis]+x[b.d+axis] < y[axis]+y[b.d+axis];
                });
                std::vector<Bounds> suffix(order.size());
                suffix.back() = box_bounds(b, order.back());
                for (size_t i = order.size()-1; i > 0; --i) {
                    suffix[i-1] = suffix[i];
                    extend(suffix[i-1], box_bounds(b, order[i-1]));
                }
                Bounds prefix = box_bounds(b, order[0]);
                bool improved = false;
                for (size_t cut = 1; cut < order.size(); ++cut) {
                    double cost = prefix.volume()+suffix[cut].volume();
                    if (cost < best_cost) {
                        best_cost = cost; best_cut = cut; improved = true;
                    }
                    extend(prefix, box_bounds(b, order[cut]));
                }
                // Keep one order per winning axis, rather than copying it for every cut.
                if (improved) best_order = std::move(order);
            }
            next.emplace_back(best_order.begin(), best_order.begin()+best_cut);
            next.emplace_back(best_order.begin()+best_cut, best_order.end());
        }
        groups = std::move(next);
    }
    std::vector<Bounds> result;
    for (const auto& group : groups) result.push_back(group_bounds(b, group));
    return result;
}

struct Entry {
    size_t event;
    Bounds bounds; // Coordinates follow the index block_order.
};
static std::vector<Entry> event_entries(const Event& event, size_t id,
                                       const std::array<size_t, 2>& counts,
                                       const std::array<size_t, 2>& block_order) {
    auto first = partition_block(event.blocks[0], counts[0]);
    auto second = partition_block(event.blocks[1], counts[1]);
    std::vector<Entry> entries;
    for (const auto& b : second) for (const auto& a : first) {
        const Bounds& start = block_order[0] == 0 ? a : b;
        const Bounds& end = block_order[0] == 0 ? b : a;
        Bounds bounds = start;
        bounds.lower.insert(bounds.lower.end(), end.lower.begin(), end.lower.end());
        bounds.upper.insert(bounds.upper.end(), end.upper.begin(), end.upper.end());
        entries.push_back({id, std::move(bounds)});
    }
    return entries;
}

// BVH over a few enclosing partitions per event. Tree entries retain the
// original event index. Queries consolidate repeated event indices BEFORE
// the exact sorted sub-box checks, so every candidate event is checked once.
struct EventIndex {
    std::vector<EventPtr> events;
    std::vector<Entry> entries;
    std::vector<size_t> order;
    std::vector<Node> nodes;
    std::array<size_t, 2> counts, block_order;
    std::vector<std::pair<size_t, size_t>> event_ranges;
    size_t d = 0;
    mutable size_t last_candidate_count = 0;

    EventIndex(const Events& input, std::array<size_t, 2> partitions,
               std::array<size_t, 2> check_order)
        : events(input.values), counts(partitions), block_order(check_order) {
        if (block_order[0] > 1 || block_order[1] > 1 || block_order[0] == block_order[1])
            throw std::invalid_argument("block_order must be (0,1) or (1,0)");
        if (events.empty()) return;
        d = events[0]->blocks[1].d + events[0]->blocks[0].d;
        for (size_t i = 0; i < events.size(); ++i) {
            for (size_t k = 0; k < 2; ++k)
                if (events[i]->blocks[k].d != events[0]->blocks[k].d ||
                    events[i]->blocks[k].axis != events[0]->blocks[k].axis)
                    throw std::invalid_argument("Inconsistent event block dimensions or ordering");
            size_t begin = entries.size();
            auto parts = event_entries(*events[i], i, counts, block_order);
            entries.insert(entries.end(), std::make_move_iterator(parts.begin()),
                           std::make_move_iterator(parts.end()));
            event_ranges.emplace_back(begin, entries.size());
        }
        order.resize(entries.size());
        std::iota(order.begin(), order.end(), 0);
        if (!order.empty()) build(0, order.size());
    }

    int build(size_t begin, size_t end) {
        Node node;
        node.begin = begin; node.end = end;
        Bounds bounds = entries[order[begin]].bounds;
        for (size_t i = begin+1; i < end; ++i) extend(bounds, entries[order[i]].bounds);
        node.lower = std::move(bounds.lower); node.upper = std::move(bounds.upper);
        int id = nodes.size();
        nodes.push_back(std::move(node));
        if (end-begin <= 8) return id;
        size_t axis = 0;
        double best = -1;
        for (size_t k = 0; k < d; ++k) {
            const auto& b = entries[order[begin]].bounds;
            double lo = b.lower[k]+b.upper[k], hi = lo;
            for (size_t i = begin+1; i < end; ++i) {
                const auto& x = entries[order[i]].bounds;
                double center = x.lower[k]+x.upper[k];
                lo = std::min(lo, center); hi = std::max(hi, center);
            }
            if (hi-lo > best) { best = hi-lo; axis = k; }
        }
        size_t middle = begin+(end-begin)/2;
        std::nth_element(order.begin()+begin, order.begin()+middle, order.begin()+end,
            [&](size_t i, size_t j) {
                const auto& a = entries[i].bounds;
                const auto& b = entries[j].bounds;
                return a.lower[axis]+a.upper[axis] < b.lower[axis]+b.upper[axis];
            });
        int left = build(begin, middle), right = build(middle, end);
        nodes[id].left = left; nodes[id].right = right;
        return id;
    }

    bool intersects_bounds(const Bounds& a, const std::vector<double>& lo,
                           const std::vector<double>& hi) const {
        for (size_t k = 0; k < d; ++k)
            if (a.upper[k] <= lo[k] || hi[k] <= a.lower[k]) return false;
        return true;
    }
    void visit(int id, const Bounds& query, int64_t after,
               std::vector<int64_t>& hits) const {
        const Node& node = nodes[id];
        if (!intersects_bounds(query, node.lower, node.upper)) return;
        if (node.left >= 0) {
            visit(node.left, query, after, hits); visit(node.right, query, after, hits);
            return;
        }
        for (size_t i = node.begin; i < node.end; ++i) {
            const Entry& entry = entries[order[i]];
            if (int64_t(entry.event) <= after) continue;
            if (intersects_bounds(query, entry.bounds.lower, entry.bounds.upper))
                hits.push_back(entry.event);
        }
    }

    std::vector<int64_t> query(const Event& event, int64_t after = -1,
                              bool exact = true, size_t* candidate_count = nullptr) const {
        std::vector<int64_t> hits;
        if (events.empty()) return hits;
        for (size_t k = 0; k < 2; ++k)
            if (event.blocks[k].d != events[0]->blocks[k].d ||
                event.blocks[k].axis != events[0]->blocks[k].axis)
                throw std::invalid_argument("Query dimensions or ordering do not match index");
        if (nodes.empty()) return hits;
        // Within-set queries reuse cached partitions. External queries create
        // only their own small partition set once per call.
        if (after >= 0) {
            auto range = event_ranges[after];
            for (size_t i = range.first; i < range.second; ++i)
                visit(0, entries[i].bounds, after, hits);
        } else {
            for (const auto& entry : event_entries(event, 0, counts, block_order))
                visit(0, entry.bounds, after, hits);
        }
        std::sort(hits.begin(), hits.end());
        hits.erase(std::unique(hits.begin(), hits.end()), hits.end());
        if (candidate_count) *candidate_count += hits.size();
        if (exact) hits.erase(std::remove_if(hits.begin(), hits.end(), [&](int64_t j) {
            const Event& other = *events[j];
            return !block_intersects(event.blocks[block_order[0]], other.blocks[block_order[0]]) ||
                   !block_intersects(event.blocks[block_order[1]], other.blocks[block_order[1]]);
        }), hits.end());
        return hits;
    }
};

static py::array_t<int64_t> find_pairs_range(const EventIndex& index,
                                            size_t begin, size_t end) {
        if (begin > end || end > index.events.size())
            throw std::out_of_range("Event range outside index");
        std::vector<std::array<int64_t, 2>> pairs;
        index.last_candidate_count = 0;
        {
            py::gil_scoped_release release;
            for (size_t i = begin; i < end; ++i)
                for (int64_t j : index.query(*index.events[i], i, true, &index.last_candidate_count))
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
        .def(py::init<const Events&, std::array<size_t, 2>, std::array<size_t, 2>>(),
             py::arg("events"), py::arg("subdivisions") = std::array<size_t, 2>{0, 3},
             py::arg("block_order") = std::array<size_t, 2>{0, 1},
             py::call_guard<py::gil_scoped_release>())
        .def_readonly("last_candidate_count", &EventIndex::last_candidate_count)
        .def_property_readonly("entry_count", [](const EventIndex& x) {return x.entries.size();})
        .def("__len__", [](const EventIndex& x) { return x.events.size(); });
    m.def("find_1vN", [](const Event& event, const EventIndex& index) {
        std::vector<int64_t> hits;
        { py::gil_scoped_release release; hits = index.query(event); }
        py::array_t<int64_t> result(hits.size());
        std::copy(hits.begin(), hits.end(), result.mutable_data());
        return result;
    });
    m.def("partition_bounds", [](const Event& event, size_t block, size_t count) {
        if (block > 1) throw std::out_of_range("Block index must be zero or one");
        std::vector<Bounds> parts;
        { py::gil_scoped_release release; parts = partition_block(event.blocks[block], count); }
        size_t d = event.blocks[block].d;
        py::array_t<double> result({py::ssize_t(parts.size()), py::ssize_t(2*d)});
        double* out = result.mutable_data();
        for (size_t i = 0; i < parts.size(); ++i) {
            std::copy(parts[i].lower.begin(), parts[i].lower.end(), out+i*2*d);
            std::copy(parts[i].upper.begin(), parts[i].upper.end(), out+i*2*d+d);
        }
        return result;
    });
    m.def("find_candidate_1vN", [](const Event& event, const EventIndex& index) {
        std::vector<int64_t> hits;
        { py::gil_scoped_release release; hits = index.query(event, -1, false); }
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
    // Matching pairs only: no event tree, partitions, or cross product.
    m.def("find_paired", [](const Events& a, const Events& b) {
        if (a.values.size() != b.values.size())
            throw std::invalid_argument("Matching event sets must have equal lengths");
        py::array_t<bool> result(a.values.size());
        bool* output = result.mutable_data();
        py::gil_scoped_release release;
        for (size_t i = 0; i < a.values.size(); ++i)
            output[i] = block_intersects(a.values[i]->blocks[0], b.values[i]->blocks[0]) &&
                        block_intersects(a.values[i]->blocks[1], b.values[i]->blocks[1]);
        return result;
    });
    m.def("overlap_paired", [](const Events& a, const Events& b) {
        if (a.values.size() != b.values.size())
            throw std::invalid_argument("Matching event sets must have equal lengths");
        py::array_t<double> result(a.values.size());
        double* output = result.mutable_data();
        py::gil_scoped_release release;
        for (size_t i = 0; i < a.values.size(); ++i)
            output[i] = event_overlap(*a.values[i], *b.values[i]);
        return result;
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
