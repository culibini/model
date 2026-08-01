#pragma once
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "engine/types.hpp"
#include "config.hpp"
#include "mem.hpp"
#include "util.hpp"

namespace engine {

struct GridBool {
    int width = 0, height = 0;
    std::vector<uint8_t> v;
    inline uint8_t at(int y, int x) const { return v[(size_t)y * width + x]; }
};

inline double hyperbolic_danger(double v) {
    if (cfg::DANGER_HYPERBOLIC_RANGE <= 0.0) return v;
    const double delta = v - cfg::DANGER_SOFT_LIMIT;
    if (delta <= 0.0) return v;
    const double cap = cfg::DANGER_HYPERBOLIC_RANGE * 0.999999;
    const double capped = std::min(delta, cap);
    const double denom  = std::max(1e-6, cfg::DANGER_HYPERBOLIC_RANGE - capped);
    return v + cfg::DANGER_HYPERBOLIC_SCALE * (capped / denom);
}

struct FlatDangerMap {
    struct Slot {
        uint64_t key;
        double   danger;
        int32_t  hour;
    };
    static constexpr uint64_t EMPTY = ~0ull;
    static constexpr uint64_t TOMB  = ~0ull - 1;

    std::vector<Slot> slots;
    size_t mask = 0;
    size_t live = 0;
    size_t used = 0;

    FlatDangerMap() { rehash(1u << 20); }

    static inline uint64_t mix(uint64_t x) {
        x += 0x9E3779B97F4A7C15ull;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        return x ^ (x >> 31);
    }

    void rehash(size_t cap) {
        size_t c = 1;
        while (c < cap) c <<= 1;
        std::vector<Slot> old;
        old.swap(slots);
        slots.assign(c, Slot{EMPTY, 0.0, 0});
        mask = c - 1;
        live = 0;
        used = 0;
        for (const Slot& s : old)
            if (s.key != EMPTY && s.key != TOMB) put(s.key, s.danger, s.hour);
    }

    inline const Slot* get(uint64_t key) const {
        size_t i = mix(key) & mask;
        while (true) {
            const Slot& s = slots[i];
            if (s.key == key) return &s;
            if (s.key == EMPTY) return nullptr;
            i = (i + 1) & mask;
        }
    }

    void put(uint64_t key, double danger, int32_t hour) {
        if ((used + 1) * 10 >= slots.size() * 7) rehash(slots.size() * 2);
        size_t i = mix(key) & mask;
        size_t first_tomb = (size_t)-1;
        while (true) {
            Slot& s = slots[i];
            if (s.key == key) { s.danger = danger; s.hour = hour; return; }
            if (s.key == TOMB && first_tomb == (size_t)-1) first_tomb = i;
            if (s.key == EMPTY) {
                if (first_tomb != (size_t)-1) i = first_tomb;
                else used++;
                slots[i] = Slot{key, danger, hour};
                live++;
                return;
            }
            i = (i + 1) & mask;
        }
    }

    void erase(uint64_t key) {
        size_t i = mix(key) & mask;
        while (true) {
            Slot& s = slots[i];
            if (s.key == key) { s.key = TOMB; live--; return; }
            if (s.key == EMPTY) return;
            i = (i + 1) & mask;
        }
    }

    inline size_t size() const { return live; }
};

inline double distance_to_hour_index(double distance_traveled, double arrays_count,
                                            double flight_speed = cfg::FLIGHT_SPEED,
                                            double hours_per_array_slice = cfg::HOURS_PER_ARRAY_SLICE) {
    double time_hours = distance_traveled / flight_speed;
    long hour_idx = py_trunc_i(time_hours / hours_per_array_slice);
    if (hour_idx < 0) return 0.0;
    double max_idx = arrays_count - 1.0;
    return ((double)hour_idx <= max_idx) ? (double)hour_idx : max_idx;
}

inline int get_hour_lookahead_value(int max_hours) {
    if (cfg::HOUR_LOOKAHEAD_IS_NONE) return max_hours > 0 ? max_hours - 1 : 0;
    return std::max(0, std::min(cfg::HOUR_LOOKAHEAD_VALUE, max_hours - 1));
}

inline double sample3d(const Array3D& a, int level, double px, double py,
                       double scale_x, double scale_y) {
    if (level < 0 || level >= a.levels) return 0.0;
    double u = px / scale_x - 0.5;
    double v = py / scale_y - 0.5;
    if (u < 0.0) u = 0.0;
    if (v < 0.0) v = 0.0;
    const double mu = (double)a.width - 1.0;
    const double mv = (double)a.height - 1.0;
    if (u > mu) u = mu;
    if (v > mv) v = mv;
    const int x0 = (int)u, y0 = (int)v;
    const int x1 = std::min(x0 + 1, a.width - 1);
    const int y1 = std::min(y0 + 1, a.height - 1);
    const double fu = u - x0, fv = v - y0;
    const double d00 = a.at(level, y0, x0), d10 = a.at(level, y0, x1);
    const double d01 = a.at(level, y1, x0), d11 = a.at(level, y1, x1);
    return (d00 * (1.0 - fu) + d10 * fu) * (1.0 - fv) +
           (d01 * (1.0 - fu) + d11 * fu) * fv;
}

struct DangerMapCache {
    int width = 0, height = 0;
    Grid2D  danger_cache;
    GridBool validity_cache;

    explicit DangerMapCache(const Grid2D& danger_map_2d) {
        height = danger_map_2d.height;
        width  = danger_map_2d.width;
        danger_cache = apply_hyperbolic_penalty(danger_map_2d);
        validity_cache.width = width;
        validity_cache.height = height;
        validity_cache.v.assign((size_t)width * height, 1);
    }

    static Grid2D apply_hyperbolic_penalty(const Grid2D& base_in) {
        Grid2D base = base_in;
        if (cfg::DANGER_HYPERBOLIC_RANGE <= 0.0) return base;

        bool any = false;
        for (double v : base.v) {
            if (v - cfg::DANGER_SOFT_LIMIT > 0.0) { any = true; break; }
        }
        if (!any) return base;

        const double cap = cfg::DANGER_HYPERBOLIC_RANGE * 0.999999;
        double* __restrict p = base.v.data();
        const int W = base.width;
        parallel_for(0, (size_t)base.height, [&](size_t row) {
            double* __restrict r = p + row * W;
            for (int i = 0; i < W; ++i) {
                double delta = r[i] - cfg::DANGER_SOFT_LIMIT;
                if (delta > 0.0) {
                    double capped = std::min(delta, cap);
                    double denom  = std::max(1e-6, cfg::DANGER_HYPERBOLIC_RANGE - capped);
                    r[i] += cfg::DANGER_HYPERBOLIC_SCALE * (capped / denom);
                }
            }
        });
        return base;
    }

    inline bool is_valid_fast(long x, long y) const {
        return !(x < 0 || x >= width || y < 0 || y >= height);
    }

    inline double get_danger_fast(long x, long y) const {
        if (x < 0 || x >= width || y < 0 || y >= height) return 1e9;
        return danger_cache.at((int)y, (int)x);
    }
    inline bool get_cached_validity(double px, double py) const {
        return is_valid_fast(py_round_i(px), py_round_i(py));
    }
    inline double get_cached_danger(double px, double py) const {
        return get_danger_fast(py_round_i(px), py_round_i(py));
    }
};

struct DangerValueCache {
    const DangerMapCache* map_cache = nullptr;
    const std::vector<Array3D>* arrays_3d = nullptr;
    double scale_x = 1.0, scale_y = 1.0, flight_speed = cfg::FLIGHT_SPEED;

    FlatDangerMap cache;
    std::vector<uint64_t> insertion_order;
    size_t order_head = 0;

    DangerValueCache(const DangerMapCache* mc, const std::vector<Array3D>* a3d,
                     double sx, double sy, double fs)
        : map_cache(mc), arrays_3d(a3d), scale_x(sx), scale_y(sy), flight_speed(fs) {
    }

    static inline uint64_t make_key(long x, long y, int level, int hour) {

        return (((uint64_t)(uint32_t)x << 40) | ((uint64_t)(uint32_t)y << 20) |
                ((uint64_t)(uint32_t)level << 8) | (uint64_t)(uint32_t)hour);
    }

    inline double get_level_penalty_cached(int level) const {
        if (level >= 0 && level < cfg::NUM_LEVELS) return cfg::LEVEL_PENALTIES[level];
        return 0.0;
    }

    std::pair<double, int> get_total_danger_cached(double px, double py, int level,
                                                   double distance_traveled) {

        long x = py_round_i(px);
        long y = py_round_i(py);
        int hour_idx = (int)distance_to_hour_index(distance_traveled,
                                                   (double)arrays_3d->size(), flight_speed);
        uint64_t key = make_key(x, y, level, hour_idx);

        if (const FlatDangerMap::Slot* s = cache.get(key))
            return std::pair<double, int>(s->danger, s->hour);

        double base_danger = map_cache->get_cached_danger(px, py);
        double level_penalty = get_level_penalty_cached(level);

        double level_danger = 0.0;
        if (hour_idx >= 0 && hour_idx < (int)arrays_3d->size())
            level_danger = sample3d((*arrays_3d)[hour_idx], level, px, py, scale_x, scale_y);

        std::pair<double, int> result(base_danger + level_danger + level_penalty, hour_idx);
        cache.put(key, result.first, (int32_t)result.second);
        insertion_order.push_back(key);

        if (cache.size() > 500000) {
            size_t to_drop = std::min<size_t>(50000, insertion_order.size() - order_head);
            for (size_t i = 0; i < to_drop; ++i) cache.erase(insertion_order[order_head + i]);
            order_head += to_drop;
            if (order_head > insertion_order.size() / 2) {
                insertion_order.erase(insertion_order.begin(),
                                      insertion_order.begin() + order_head);
                order_head = 0;
            }
        }
        return result;
    }

    int find_best_level_for_point(double px, double py, double distance_traveled) {
        int best_level = 0;
        double best_danger = std::numeric_limits<double>::infinity();
        for (int level = 0; level < cfg::NUM_LEVELS; ++level) {
            double d = get_total_danger_cached(px, py, level, distance_traveled).first;
            if (d < best_danger) { best_danger = d; best_level = level; }
        }
        return best_level;
    }
};

struct Environment {
    DangerMapCache*   map2_cache = nullptr;
    DangerValueCache* danger_cache = nullptr;
    std::vector<Array3D> arrays_3d;
    int height = 0, width = 0;
    int array_levels = 0, array_height = 0, array_width = 0;
    double SCALE_X = 1.0, SCALE_Y = 1.0;

    Environment(const Grid2D& base_map, const std::vector<Array3D>& forecasts) {
        if (base_map.width <= 0 || base_map.height <= 0 ||
            base_map.v.size() != (size_t)base_map.width * base_map.height)
            throw std::runtime_error("ValueError: bad danger map shape");
        if (forecasts.empty())
            throw std::runtime_error("FileNotFoundError: No .npy arrays found for 3D risk data");
        for (const Array3D& a : forecasts)
            if (a.levels <= 0 || a.height <= 0 || a.width <= 0 ||
                a.v.size() != (size_t)a.levels * a.height * a.width)
                throw std::runtime_error("ValueError: bad forecast shape");

        height = base_map.height;
        width  = base_map.width;

        map2_cache = new DangerMapCache(base_map);

        arrays_3d = forecasts;
        for (Array3D& arr : arrays_3d)
            for (double& v : arr.v) v = hyperbolic_danger(v);

        array_levels = arrays_3d[0].levels;
        array_height = arrays_3d[0].height;
        array_width  = arrays_3d[0].width;
        SCALE_Y = (double)height / (double)array_height;
        SCALE_X = (double)width  / (double)array_width;

        danger_cache = new DangerValueCache(map2_cache, &arrays_3d, SCALE_X, SCALE_Y,
                                            cfg::FLIGHT_SPEED);
    }

    Environment(const Environment&) = delete;
    Environment& operator=(const Environment&) = delete;
    ~Environment() { delete danger_cache; delete map2_cache; }
};

}
