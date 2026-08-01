#include "optimize.hpp"
#include <algorithm>
#include <cmath>
#include "config.hpp"
#include "geometry.hpp"
#include "mem.hpp"
#include "util.hpp"

namespace astar {

Weights calculate_fixed_weights(double path_length) {
    double safety = cfg::SAFETY_WEIGHT *
                    (1.0 + cfg::SAFETY_WEIGHT_COEFF * (path_length / cfg::BASE_PATH_LENGTH));
    safety = std::max(cfg::SAFETY_WEIGHT, std::min(safety, cfg::SAFETY_WEIGHT * 3.0));
    return {safety, cfg::LENGTH_WEIGHT, cfg::SMOOTHNESS_WEIGHT, cfg::LEVEL_CHANGE_PENALTY};
}

static double path_fitness(const double* __restrict xs, const double* __restrict ys,
                           const int* __restrict levels,
                           const double* __restrict total_dists, int n,
                           const GridBool& validity_cache, const Grid2D& danger_cache_2d,
                           const std::vector<Array3D>& arrays_3d_list,
                           const double* level_penalties,
                           double scale_x, double scale_y,
                           double w_length, double w_safety, double w_smooth,
                           double w_level_change, double flight_speed, int max_hours) {
    if (n < 2) return 1e18;

    const int h = validity_cache.height, w = validity_cache.width;

    double total_length = 0.0, total_danger = 0.0, total_curvature = 0.0;
    long total_level_changes = 0;

    for (int i = 0; i < n; ++i) {
        long xi = py_round_i(xs[i]), yi = py_round_i(ys[i]);
        if (xi < 0 || xi >= w || yi < 0 || yi >= h) return 1e18;
        if (!validity_cache.at((int)yi, (int)xi)) return 1e18;
    }

    for (int i = 0; i < n - 1; ++i) {
        if (!line_clear(xs[i], ys[i], xs[i + 1], ys[i + 1], validity_cache)) return 1e18;
        total_length += hypot2(xs[i + 1] - xs[i], ys[i + 1] - ys[i]);
    }

    for (int i = 0; i < n; ++i) {
        const double time_hours = total_dists[i] / flight_speed;
        int array_idx = (int)py_trunc_i(time_hours);
        if (array_idx < 0) array_idx = 0;
        if (array_idx >= max_hours) array_idx = max_hours - 1;
        const int lvl = levels[i];
        const double level_danger =
            sample3d(arrays_3d_list[array_idx], lvl, xs[i], ys[i], scale_x, scale_y);
        const double base_danger =
            danger_cache_2d.at((int)py_round_i(ys[i]), (int)py_round_i(xs[i]));
        total_danger += base_danger + level_danger + level_penalties[lvl];
    }

    for (int i = 1; i < n - 1; ++i) {
        const double dx1 = xs[i] - xs[i - 1], dy1 = ys[i] - ys[i - 1];
        const double dx2 = xs[i + 1] - xs[i], dy2 = ys[i + 1] - ys[i];
        const double norm1 = hypot2(dx1, dy1), norm2 = hypot2(dx2, dy2);
        if (norm1 > 0.0 && norm2 > 0.0) {
            double cos_angle = (dx1 * dx2 + dy1 * dy2) / (norm1 * norm2);
            if (cos_angle > 1.0) cos_angle = 1.0;
            if (cos_angle < -1.0) cos_angle = -1.0;
            const double span = hypot2(xs[i + 1] - xs[i - 1], ys[i + 1] - ys[i - 1]);
            total_curvature += (1.0 - cos_angle) * span;
        }
    }

    for (int i = 1; i < n; ++i)
        if (levels[i] != levels[i - 1]) total_level_changes++;

    return w_length * total_length + w_safety * total_danger +
           w_smooth * total_curvature + w_level_change * (double)total_level_changes;
}

static int find_optimal_level_with_constraints(double px, double py, int current_level,
                                               double current_level_distance,
                                               double total_distance, Environment& env) {
    int best_level = current_level;
    std::pair<double, int> best =
        env.danger_cache->get_total_danger_cached(px, py, current_level, total_distance);

    for (int off = -cfg::LEVEL_OPTIMIZATION_RANGE; off <= cfg::LEVEL_OPTIMIZATION_RANGE; ++off) {
        const int cand = current_level + off;
        if (cand < 0 || cand >= cfg::NUM_LEVELS) continue;
        if (cand > current_level) {
            if (current_level_distance < cfg::LEVEL_STAY_MULTIPLIER * (cand - current_level))
                continue;
        }
        std::pair<double, int> d =
            env.danger_cache->get_total_danger_cached(px, py, cand, total_distance);

        if (d.first < best.first || (d.first == best.first && d.second < best.second)) {
            best = d;
            best_level = cand;
        }
    }
    return best_level;
}

static void generate_safe_candidates(const std::vector<double>& path_xs,
                                     const std::vector<double>& path_ys,
                                     int center_index, double cur_x, double cur_y,
                                     Environment& env, const uint8_t* frozen,
                                     NumpyRandom& rng,
                                     std::vector<double>& out_x, std::vector<double>& out_y) {
    out_x.clear();
    out_y.clear();

    const int num_candidates = cfg::NUM_CANDIDATES;
    const int max_attempts = num_candidates * 10;

    bool use_direction = false;
    double dir_x = 0.0, dir_y = 0.0;
    if (center_index > 0 && center_index < (int)path_xs.size() - 1) {
        const double ideal_x = (path_xs[center_index - 1] + path_xs[center_index + 1]) / 2.0;
        const double ideal_y = (path_ys[center_index - 1] + path_ys[center_index + 1]) / 2.0;
        const double dx = ideal_x - cur_x, dy = ideal_y - cur_y;
        const double dist_to_ideal = std::sqrt(dx * dx + dy * dy);
        if (dist_to_ideal > 0.0) {
            dir_x = dx / dist_to_ideal;
            dir_y = dy / dist_to_ideal;
            use_direction = true;
        }
    }

    std::vector<double> cx, cy;
    generate_candidates(cur_x, cur_y, num_candidates, max_attempts, cfg::OPTIMIZATION_RADIUS,
                        (double)env.width, (double)env.height, use_direction, dir_x, dir_y,
                        rng, cx, cy);

    for (size_t i = 0; i < cx.size(); ++i) {
        if (is_candidate_safe(path_xs, path_ys, center_index, cx[i], cy[i],
                              cfg::POINTS_TO_ADJUST, env.map2_cache->validity_cache,
                              frozen)) {
            out_x.push_back(cx[i]);
            out_y.push_back(cy[i]);
        }
    }
}

OptimizeResult optimize_path_with_fixed_weights(
        const std::vector<double>& init_xs, const std::vector<double>& init_ys,
        const std::vector<int>& init_levels, const std::vector<double>& init_stay,
        const std::vector<double>& init_total, const std::vector<int>& init_arr,
        const std::vector<uint8_t>& frozen_pts, const Weights& w, Environment& env,
        NumpyRandom& rng) {

    const uint8_t* frozen =
        frozen_pts.size() == init_xs.size() ? frozen_pts.data() : nullptr;
    std::vector<double> path_xs = init_xs, path_ys = init_ys;
    std::vector<int>    path_levels = init_levels;
    std::vector<double> path_stay_requirements = init_stay;
    std::vector<double> path_total_distances = init_total;
    std::vector<int>    path_array_indices = init_arr;

    const int n_points = (int)path_xs.size();
    std::vector<double> lvl_dist = level_distances(path_xs, path_ys, path_levels);
    std::vector<double> best_fitness_progress;

    auto fitness_raw = [&](const double* xs, const double* ys, const int* lv,
                           const double* td, int n) {
        return path_fitness(xs, ys, lv, td, n,
                            env.map2_cache->validity_cache, env.map2_cache->danger_cache,
                            env.arrays_3d, cfg::LEVEL_PENALTIES,
                            env.SCALE_X, env.SCALE_Y,
                            w.length, w.safety, w.smoothness, w.level_change,
                            cfg::FLIGHT_SPEED * cfg::HOURS_PER_ARRAY_SLICE,
                            (int)env.arrays_3d.size());
    };
    auto fitness_of = [&](const std::vector<double>& xs, const std::vector<double>& ys,
                          const std::vector<int>& lv, const std::vector<double>& td) {
        return fitness_raw(xs.data(), ys.data(), lv.data(), td.data(), (int)xs.size());
    };

    const size_t MAXC = (size_t)cfg::NUM_CANDIDATES + 1;
    std::vector<double> cand_tx(MAXC * n_points), cand_ty(MAXC * n_points),
                        cand_ttd(MAXC * n_points), cand_fit(MAXC);
    std::vector<int>    cand_lv(MAXC * n_points);

    double best_seen = fitness_of(path_xs, path_ys, path_levels, path_total_distances);
    std::vector<double> best_xs = path_xs, best_ys = path_ys, best_td = path_total_distances;
    std::vector<int>    best_lv = path_levels;

    for (int iteration = 0; iteration < cfg::NUM_ITERATIONS; ++iteration) {
        for (int i = 1; i < n_points - 1; ++i) {
            if (frozen && frozen[i]) continue;

            double best_fitness =
                fitness_of(path_xs, path_ys, path_levels, path_total_distances);

            const double cur_x = path_xs[i], cur_y = path_ys[i];
            const int    current_level = path_levels[i];
            const double current_level_distance = lvl_dist[i];

            double best_cx = cur_x, best_cy = cur_y;
            int    best_level = current_level;

            std::vector<double> cand_x, cand_y;
            generate_safe_candidates(path_xs, path_ys, i, cur_x, cur_y, env, frozen,
                                     rng, cand_x, cand_y);
            if (is_point_and_neighbors_safe(path_xs, path_ys, i, cur_x, cur_y,
                                            env.map2_cache->validity_cache)) {
                cand_x.push_back(cur_x);
                cand_y.push_back(cur_y);
            }
            if (cand_x.empty()) continue;

            const size_t ncand = cand_x.size();

            {
                std::vector<double> tx, ty, sx, sy;
                for (size_t c = 0; c < ncand; ++c) {
                    tx = path_xs;
                    ty = path_ys;
                    tx[i] = cand_x[c];
                    ty[i] = cand_y[c];
                    smooth_adjust_neighbors(tx, ty, i, cfg::POINTS_TO_ADJUST,
                                            env.map2_cache->validity_cache, frozen,
                                            sx, sy);

                    double* __restrict dx = &cand_tx[c * n_points];
                    double* __restrict dy = &cand_ty[c * n_points];
                    double* __restrict dd = &cand_ttd[c * n_points];
                    int*    __restrict dl = &cand_lv[c * n_points];

                    std::copy(sx.begin(), sx.end(), dx);
                    std::copy(sy.begin(), sy.end(), dy);

                    dd[0] = 0.0;
                    for (int k = 1; k < n_points; ++k)
                        dd[k] = dd[k - 1] + hypot2(dx[k] - dx[k - 1], dy[k] - dy[k - 1]);

                    std::copy(path_levels.begin(), path_levels.end(), dl);
                    dl[i] = find_optimal_level_with_constraints(
                        cand_x[c], cand_y[c], current_level, current_level_distance,
                        dd[i], env);
                }
            }

            parallel_for(0, ncand, [&](size_t c) {
                cand_fit[c] = fitness_raw(&cand_tx[c * n_points], &cand_ty[c * n_points],
                                          &cand_lv[c * n_points], &cand_ttd[c * n_points],
                                          n_points);
            });

            for (size_t c = 0; c < ncand; ++c) {
                if (cand_fit[c] < best_fitness) {
                    best_fitness = cand_fit[c];
                    best_cx = cand_x[c];
                    best_cy = cand_y[c];
                    best_level = cand_lv[c * n_points + i];
                }
            }

            path_xs[i] = best_cx;
            path_ys[i] = best_cy;
            path_levels[i] = best_level;

            std::vector<double> sx, sy;
            smooth_adjust_neighbors(path_xs, path_ys, i, cfg::POINTS_TO_ADJUST,
                                    env.map2_cache->validity_cache, frozen, sx, sy);
            path_xs.swap(sx);
            path_ys.swap(sy);

            path_total_distances = recalculate_distances(path_xs, path_ys);
            path_array_indices.assign(path_total_distances.size(), 0);
            for (size_t k = 0; k < path_total_distances.size(); ++k)
                path_array_indices[k] = (int)distance_to_hour_index(
                    path_total_distances[k], (double)env.arrays_3d.size());
            lvl_dist = level_distances(path_xs, path_ys, path_levels);
        }

        const double current_fitness =
            fitness_of(path_xs, path_ys, path_levels, path_total_distances);
        best_fitness_progress.push_back(current_fitness);
        if (current_fitness < best_seen) {
            best_seen = current_fitness;
            best_xs = path_xs;
            best_ys = path_ys;
            best_td = path_total_distances;
            best_lv = path_levels;
        }
    }

    path_xs = std::move(best_xs);
    path_ys = std::move(best_ys);
    path_total_distances = std::move(best_td);
    path_levels = std::move(best_lv);
    path_stay_requirements = recompute_stay_requirements(path_levels);
    path_array_indices.assign(path_total_distances.size(), 0);
    for (size_t k = 0; k < path_total_distances.size(); ++k)
        path_array_indices[k] = (int)distance_to_hour_index(
            path_total_distances[k], (double)env.arrays_3d.size());

    OptimizeResult r;
    r.xs = std::move(path_xs);
    r.ys = std::move(path_ys);
    r.levels = std::move(path_levels);
    r.stay_req = std::move(path_stay_requirements);
    r.total_dist = std::move(path_total_distances);
    r.array_idx = std::move(path_array_indices);
    r.fitness_progress = std::move(best_fitness_progress);
    return r;
}

}
