#pragma once
#include <utility>
#include <vector>
#include "engine/types.hpp"

namespace engine {

std::vector<RoutePoint> calculate_path(
    const Grid2D& danger_map,
    const std::vector<Array3D>& forecasts,
    const std::vector<std::pair<double, double>>& route_points_yx,
    const Options& options = {});

}
