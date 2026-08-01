#include "geometry.hpp"
#include <algorithm>
#include <cmath>
#include "config.hpp"
#include "util.hpp"

namespace astar {

bool line_clear(double x1, double y1, double x2, double y2, const GridBool& valid) {
    long x1i = py_round_i(x1), y1i = py_round_i(y1);
    long x2i = py_round_i(x2), y2i = py_round_i(y2);
    long dx = std::labs(x2i - x1i);
    long dy = std::labs(y2i - y1i);
    long x = x1i, y = y1i;
    long sx = (x2i > x1i) ? 1 : -1;
    long sy = (y2i > y1i) ? 1 : -1;
    const int h = valid.height, w = valid.width;

    if (x < 0 || x >= w || y < 0 || y >= h) return false;

    if (dx > dy) {
        double err = dx / 2.0;
        while (x != x2i) {
            if (!valid.at((int)y, (int)x)) return false;
            err -= (double)dy;
            if (err < 0) { y += sy; err += (double)dx; }
            x += sx;
        }
    } else {
        double err = dy / 2.0;
        while (y != y2i) {
            if (!valid.at((int)y, (int)x)) return false;
            err -= (double)dx;
            if (err < 0) { x += sx; err += (double)dy; }
            y += sy;
        }
    }
    if (x2i < 0 || x2i >= w || y2i < 0 || y2i >= h) return false;
    return valid.at((int)y2i, (int)x2i) != 0;
}

void generate_candidates(double center_x, double center_y, int num_candidates,
                         int max_attempts, double radius, double width, double height,
                         bool use_direction, double dir_x, double dir_y,
                         NumpyRandom& rng,
                         std::vector<double>& cx_out, std::vector<double>& cy_out) {
    cx_out.clear();
    cy_out.clear();
    int attempts = 0;
    while ((int)cx_out.size() < num_candidates && attempts < max_attempts) {
        attempts++;
        double angle, r;

        if (use_direction && rng.random() < 0.7) {
            double angle_variation = rng.normal(0.0, 0.5);
            double base_angle = std::atan2(dir_y, dir_x);
            angle = base_angle + angle_variation;
            r = rng.random() * radius;
        } else {
            angle = rng.random() * 2.0 * M_PI;
            r = rng.random() * radius;
        }
        double dx = r * std::cos(angle);
        double dy = r * std::sin(angle);
        double cx = center_x + dx;
        double cy = center_y + dy;
        cx = std::max(0.0, std::min(width - 1.0, cx));
        cy = std::max(0.0, std::min(height - 1.0, cy));
        cx_out.push_back(cx);
        cy_out.push_back(cy);
    }
}

void smooth_adjust_neighbors(const std::vector<double>& path_xs,
                                    const std::vector<double>& path_ys,
                                    int center_index, int points_to_adjust,
                                    const GridBool& valid,
                                    const uint8_t* frozen,
                                    std::vector<double>& adj_xs,
                                    std::vector<double>& adj_ys) {
    const int n_points = (int)path_xs.size();
    adj_xs = path_xs;
    adj_ys = path_ys;
    const int h = valid.height, w = valid.width;

    int start_idx = std::max(1, center_index - points_to_adjust);
    int end_idx   = std::min(n_points - 2, center_index + points_to_adjust);

    for (int i = start_idx; i <= end_idx; ++i) {
        if (i == center_index) continue;
        if (frozen && frozen[i]) continue;

        int window_start = std::max(1, i - 1);
        int window_end   = std::min(n_points - 2, i + 1);
        int count = 0;
        double sum_x = 0.0, sum_y = 0.0;

        for (int j = window_start; j <= window_end; ++j) {
            sum_x += adj_xs[j];
            sum_y += adj_ys[j];
            count++;
        }

        if (count > 0) {
            double smoothed_x = sum_x / count;
            double smoothed_y = sum_y / count;
            long sx = py_round_i(smoothed_x);
            long sy = py_round_i(smoothed_y);

            if (sx >= 0 && sx < w && sy >= 0 && sy < h && valid.at((int)sy, (int)sx)) {
                if (line_clear(adj_xs[i - 1], adj_ys[i - 1], smoothed_x, smoothed_y, valid)) {
                    if (line_clear(smoothed_x, smoothed_y, adj_xs[i + 1], adj_ys[i + 1], valid)) {
                        adj_xs[i] = smoothed_x;
                        adj_ys[i] = smoothed_y;
                    }
                }
            }
        }
    }
}

bool is_candidate_safe(const std::vector<double>& path_xs,
                              const std::vector<double>& path_ys,
                              int center_index, double candidate_x, double candidate_y,
                              int points_to_adjust, const GridBool& valid,
                              const uint8_t* frozen) {
    const int n_points = (int)path_xs.size();
    const int h = valid.height, w = valid.width;

    std::vector<double> temp_xs = path_xs;
    std::vector<double> temp_ys = path_ys;
    temp_xs[center_index] = candidate_x;
    temp_ys[center_index] = candidate_y;

    std::vector<double> sx, sy;
    smooth_adjust_neighbors(temp_xs, temp_ys, center_index, points_to_adjust, valid, frozen, sx, sy);
    temp_xs.swap(sx);
    temp_ys.swap(sy);

    int start_idx = std::max(0, center_index - points_to_adjust);
    int end_idx   = std::min(n_points - 1, center_index + points_to_adjust);

    for (int i = start_idx; i <= end_idx; ++i) {
        long xi = py_round_i(temp_xs[i]);
        long yi = py_round_i(temp_ys[i]);
        if (xi < 0 || xi >= w || yi < 0 || yi >= h) return false;
        if (!valid.at((int)yi, (int)xi)) return false;
    }
    for (int i = start_idx; i < end_idx; ++i) {
        if (!line_clear(temp_xs[i], temp_ys[i], temp_xs[i + 1], temp_ys[i + 1], valid))
            return false;
    }
    return true;
}

bool is_point_and_neighbors_safe(const std::vector<double>& path_xs,
                                        const std::vector<double>& path_ys,
                                        int center_index, double point_x, double point_y,
                                        const GridBool& valid) {
    const int h = valid.height, w = valid.width;
    long px = py_round_i(point_x);
    long py = py_round_i(point_y);

    if (px < 0 || px >= w || py < 0 || py >= h) return false;
    if (!valid.at((int)py, (int)px)) return false;

    if (center_index > 0) {
        if (!line_clear(path_xs[center_index - 1], path_ys[center_index - 1],
                        point_x, point_y, valid))
            return false;
    }
    if (center_index < (int)path_xs.size() - 1) {
        if (!line_clear(point_x, point_y,
                        path_xs[center_index + 1], path_ys[center_index + 1], valid))
            return false;
    }
    return true;
}

std::vector<double> recalculate_distances(const std::vector<double>& xs,
                                                 const std::vector<double>& ys) {
    const size_t n = xs.size();
    std::vector<double> out(n, 0.0);
    for (size_t i = 1; i < n; ++i)
        out[i] = out[i - 1] + hypot2(xs[i] - xs[i - 1], ys[i] - ys[i - 1]);
    return out;
}

std::vector<double> level_distances(const std::vector<double>& xs,
                                           const std::vector<double>& ys,
                                           const std::vector<int>& levels) {
    const size_t n = xs.size();
    std::vector<double> out(n, 0.0);
    for (size_t i = 1; i < n; ++i) {
        if (levels[i] == levels[i - 1])
            out[i] = out[i - 1] + hypot2(xs[i] - xs[i - 1], ys[i] - ys[i - 1]);
        else
            out[i] = 0.0;
    }
    return out;
}

std::vector<double> recompute_stay_requirements(const std::vector<int>& levels) {
    const size_t n = levels.size();
    std::vector<double> out(n, 0.0);
    for (size_t i = 1; i < n; ++i) {
        if (levels[i] > levels[i - 1])
            out[i] = cfg::LEVEL_STAY_MULTIPLIER * (levels[i] - levels[i - 1]);
        else
            out[i] = 0.0;
    }
    return out;
}

}
