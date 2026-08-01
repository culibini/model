#pragma once
#include <vector>
#include "engine/types.hpp"
#include "danger.hpp"
#include "rng.hpp"

namespace engine {

bool line_clear(double x1, double y1, double x2, double y2, const GridBool& valid);
void generate_candidates(double center_x, double center_y, int num_candidates,
                         int max_attempts, double radius, double width, double height,
                         bool use_direction, double dir_x, double dir_y,
                         NumpyRandom& rng,
                         std::vector<double>& cx_out, std::vector<double>& cy_out);
void smooth_adjust_neighbors(const std::vector<double>& path_xs,
                             const std::vector<double>& path_ys,
                             int center_index, int points_to_adjust,
                             const GridBool& valid, const uint8_t* frozen,
                             std::vector<double>& adj_xs, std::vector<double>& adj_ys);
bool is_candidate_safe(const std::vector<double>& path_xs,
                       const std::vector<double>& path_ys,
                       int center_index, double candidate_x, double candidate_y,
                       int points_to_adjust, const GridBool& valid,
                       const uint8_t* frozen);
bool is_point_and_neighbors_safe(const std::vector<double>& path_xs,
                                 const std::vector<double>& path_ys,
                                 int center_index, double point_x, double point_y,
                                 const GridBool& valid);
std::vector<double> recalculate_distances(const std::vector<double>& xs,
                                          const std::vector<double>& ys);
std::vector<double> level_distances(const std::vector<double>& xs,
                                    const std::vector<double>& ys,
                                    const std::vector<int>& levels);
std::vector<double> recompute_stay_requirements(const std::vector<int>& levels);

}
