#pragma once
#include <vector>
#include "danger.hpp"
#include "rng.hpp"

namespace engine {

struct Weights { double safety, length, smoothness, level_change; };

struct OptimizeResult {
    std::vector<double> xs, ys;
    std::vector<int>    levels;
    std::vector<double> stay_req;
    std::vector<double> total_dist;
    std::vector<int>    array_idx;
    std::vector<double> fitness_progress;
};

Weights calculate_fixed_weights(double path_length);

OptimizeResult optimize_path_with_fixed_weights(
    const std::vector<double>& init_xs, const std::vector<double>& init_ys,
    const std::vector<int>& init_levels, const std::vector<double>& init_stay,
    const std::vector<double>& init_total, const std::vector<int>& init_arr,
    const std::vector<uint8_t>& frozen_pts, const Weights& w, Environment& env,
    NumpyRandom& rng);

}
