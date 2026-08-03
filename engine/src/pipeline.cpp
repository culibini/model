#include "engine/router.hpp"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include "config.hpp"
#include "danger.hpp"
#include "geometry.hpp"
#include "optimize.hpp"
#include "search.hpp"
#include "util.hpp"

namespace engine {

struct SegmentResult {
    int goal_level_used = -1;
    std::vector<double> d_xs, d_ys;
    std::vector<int>    d_levels;
    std::vector<double> d_stay, d_total;
    std::vector<int>    d_arr;
    double d_length = 0.0;

    std::vector<double> o_xs, o_ys;
    std::vector<int>    o_levels;
    std::vector<double> o_stay, o_total;
    std::vector<int>    o_arr;
    double o_length = 0.0;

    std::vector<double> fitness_progress;
    bool ok = false;
};

static double compute_path_length(const std::vector<double>& xs, const std::vector<double>& ys) {
    double s = 0.0;
    for (size_t i = 0; i + 1 < xs.size(); ++i)
        s += hypot2(xs[i + 1] - xs[i], ys[i + 1] - ys[i]);
    return s;
}

static SegmentResult run_segment_pipeline(double sx, double sy, double ex, double ey,
                                          Environment& env,
                                          int start_level_in, int end_level_in,
                                          int hour_lookahead, DijkstraArena& arena,
                                          NumpyRandom& rng, double wastar) {
    SegmentResult res;

    int start_level = start_level_in;
    int goal_level  = end_level_in;
    if (start_level < 0) start_level = env.danger_cache->find_best_level_for_point(sx, sy, 0.0);
    if (goal_level  < 0) goal_level  = env.danger_cache->find_best_level_for_point(ex, ey, hypot2(ex - sx, ey - sy));

    if (!env.map2_cache->get_cached_validity(sx, sy)) return res;
    if (!env.map2_cache->get_cached_validity(ex, ey)) return res;

    DijkstraRaw raw = dijkstra_numba_grid(
        sx, sy, ex, ey, start_level, goal_level,
        env.map2_cache->validity_cache, env.map2_cache->danger_cache,
        env.arrays_3d, cfg::LEVEL_PENALTIES,
        cfg::DIJKSTRA_STEP_SIZE, cfg::LOOKAHEAD_LEVELS, cfg::LEVEL_STAY_MULTIPLIER,
        cfg::NUM_LEVELS, cfg::FLIGHT_SPEED * cfg::HOURS_PER_ARRAY_SLICE,
        (int)env.arrays_3d.size(),
        cfg::SAFETY_WEIGHT * cfg::DIJKSTRA_DANGER_WEIGHT,
        cfg::LENGTH_WEIGHT * cfg::DIJKSTRA_LENGTH_PENALTY,
        cfg::DIJKSTRA_GOAL_TOLERANCE, cfg::MAX_NODES,
        env.SCALE_X, env.SCALE_Y,
        hour_lookahead, cfg::HOUR_STAY_DISTANCE, cfg::HOUR_SWITCH_PENALTY, wastar, arena);

    if (!raw.ok || raw.px.empty()) {
        return res;
    }

    RefinedPath ref = refine_path_to_goal_with_constraints(
        raw.px, raw.py, raw.levels, raw.stay, raw.total_dist, raw.arrays,
        raw.frozen, (int)env.arrays_3d.size(), ex, ey, goal_level);

    res.d_xs = ref.xs; res.d_ys = ref.ys;
    res.d_levels = ref.levels;
    res.d_stay = ref.stay;
    res.d_total = ref.total_dist;
    res.d_arr = ref.arrays;
    res.d_length = compute_path_length(res.d_xs, res.d_ys);

    const Weights fixed_weights = calculate_fixed_weights(res.d_length);

    OptimizeResult opt = optimize_path_with_fixed_weights(
        res.d_xs, res.d_ys, res.d_levels, res.d_stay, res.d_total, res.d_arr,
        ref.frozen, fixed_weights, env, rng);

    res.o_xs = std::move(opt.xs);
    res.o_ys = std::move(opt.ys);
    res.o_levels = std::move(opt.levels);
    res.o_stay = std::move(opt.stay_req);
    res.o_total = std::move(opt.total_dist);
    res.o_arr = std::move(opt.array_idx);
    res.fitness_progress = std::move(opt.fitness_progress);
    res.o_length = compute_path_length(res.o_xs, res.o_ys);

    res.goal_level_used = goal_level;
    res.ok = true;
    return res;
}

struct CombinedResult {
    std::vector<double> d_xs, d_ys;
    std::vector<int>    d_levels;
    std::vector<double> o_xs, o_ys;
    std::vector<int>    o_levels;
    bool ok = false;
};

static CombinedResult stitch_segment_results(const std::vector<SegmentResult>& segs,
                                             Environment& ,
                                             const std::vector<int>* route_levels) {
    CombinedResult c;
    if (segs.empty()) return c;

    std::vector<int> waypoint_indices{0};
    long total_len = 0;

    for (size_t idx = 0; idx < segs.size(); ++idx) {
        const bool skip_first = idx > 0;
        const size_t start = skip_first ? 1 : 0;
        const SegmentResult& s = segs[idx];

        for (size_t k = start; k < s.d_xs.size(); ++k) {
            c.d_xs.push_back(s.d_xs[k]);
            c.d_ys.push_back(s.d_ys[k]);
        }
        for (size_t k = start; k < s.o_xs.size(); ++k) {
            c.o_xs.push_back(s.o_xs[k]);
            c.o_ys.push_back(s.o_ys[k]);
        }
        for (size_t k = start; k < s.d_levels.size(); ++k) c.d_levels.push_back(s.d_levels[k]);
        for (size_t k = start; k < s.o_levels.size(); ++k) c.o_levels.push_back(s.o_levels[k]);

        total_len += (long)(s.d_levels.size() - start);
        waypoint_indices.push_back((int)total_len - 1);
    }

    if (route_levels && route_levels->size() == waypoint_indices.size()) {
        for (size_t i = 0; i < waypoint_indices.size(); ++i) {
            const int wp = waypoint_indices[i];
            if (wp >= 0 && wp < (int)c.d_levels.size()) c.d_levels[wp] = (*route_levels)[i];
            if (wp >= 0 && wp < (int)c.o_levels.size()) c.o_levels[wp] = (*route_levels)[i];
        }
    }

    c.ok = true;
    return c;
}

static std::vector<RoutePoint> build_route_response(const CombinedResult& cr) {
    std::vector<RoutePoint> out;
    if (!cr.ok) return out;

    const std::vector<double>* px = &cr.o_xs;
    const std::vector<double>* py = &cr.o_ys;
    const std::vector<int>*    lv = &cr.o_levels;
    if (px->empty()) { px = &cr.d_xs; py = &cr.d_ys; lv = &cr.d_levels; }
    if (px->empty()) return out;
    if (lv->empty()) lv = &cr.d_levels;

    for (size_t i = 0; i < px->size(); ++i) {
        RoutePoint r;
        const long long xi = (long long)py_round_i((*px)[i]);
        const long long yi = (long long)py_round_i((*py)[i]);
        const long long li = (i < lv->size()) ? (long long)(*lv)[i] : 0LL;

        r.y = yi;
        r.x = xi;
        r.level = li;
        out.push_back(r);
    }
    return out;
}

std::vector<RoutePoint> calculate_path(
        const Grid2D& danger_map,
        const std::vector<Array3D>& forecasts,
        const std::vector<std::pair<double, double>>& route_points_yx,
        const std::vector<int>& route_levels,
        const Options& options) {

    if (!route_levels.empty()) {
        if (route_levels.size() != route_points_yx.size())
            throw std::runtime_error(
                "ValueError: route_levels must match route points count");
        for (int lv : route_levels)
            if (lv < 0 || lv >= cfg::NUM_LEVELS)
                throw std::runtime_error("ValueError: level out of range 0..31");
    }

    std::vector<std::pair<double, double>> route_xy;
    for (auto& p : route_points_yx) route_xy.emplace_back(p.second, p.first);

#ifdef _OPENMP
    if (options.threads) omp_set_num_threads((int)options.threads);
#endif
    double wastar = options.wastar;
    if (!(wastar >= 1.0)) wastar = 1.0;
    NumpyRandom rng;
    rng.seed(options.seed);

    Environment env_obj(danger_map, forecasts);
    Environment* env = &env_obj;

    if (route_xy.size() < 2) return {};

    for (size_t i = 0; i < route_xy.size(); ++i) {
        const double x = route_xy[i].first, y = route_xy[i].second;
        if (!(x >= 0 && x < env->width && y >= 0 && y < env->height)) {
            char buf[220];
            std::snprintf(buf, sizeof(buf),
                          "ValueError: Route point %zu (%.1f, %.1f) is out of bounds for map %dx%d",
                          i + 1, x, y, env->width, env->height);
            throw std::runtime_error(buf);
        }
    }

    const int hour_lookahead = get_hour_lookahead_value((int)env->arrays_3d.size());

    DijkstraArena arena;

    std::vector<SegmentResult> segs;
    int carry_level = -1;
    for (size_t i = 0; i + 1 < route_xy.size(); ++i) {
        const int seg_start_level =
            route_levels.empty() ? carry_level : route_levels[i];
        const int seg_end_level =
            route_levels.empty() ? -1 : route_levels[i + 1];
        SegmentResult sr = run_segment_pipeline(route_xy[i].first, route_xy[i].second,
                                                route_xy[i + 1].first, route_xy[i + 1].second,
                                                *env, seg_start_level, seg_end_level,
                                                hour_lookahead, arena, rng, wastar);
        if (!sr.ok) return {};
        carry_level = sr.goal_level_used;
        segs.push_back(std::move(sr));
    }

    CombinedResult cr = stitch_segment_results(segs, *env, nullptr);
    if (!cr.ok) return {};

    return build_route_response(cr);
}

}
