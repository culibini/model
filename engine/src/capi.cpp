#include <cstdlib>
#include <cstring>
#include <exception>
#include "engine/router.hpp"

extern "C" {

long long* engine_calculate_path(const double* danger_map, int map_h, int map_w,
                                const double* forecasts, int n_hours,
                                int f_levels, int f_h, int f_w,
                                const double* points_yx, int n_points,
                                const double* point_levels,
                                double wastar, unsigned threads, unsigned seed,
                                int* out_n) {
    *out_n = -1;
    try {
        engine::Grid2D map;
        map.height = map_h;
        map.width = map_w;
        map.v.assign(danger_map, danger_map + (size_t)map_h * map_w);

        std::vector<engine::Array3D> fc((size_t)n_hours);
        const size_t slice = (size_t)f_levels * f_h * f_w;
        for (int i = 0; i < n_hours; ++i) {
            fc[i].levels = f_levels;
            fc[i].height = f_h;
            fc[i].width = f_w;
            fc[i].v.assign(forecasts + (size_t)i * slice,
                           forecasts + (size_t)(i + 1) * slice);
        }

        std::vector<std::pair<double, double>> pts((size_t)n_points);
        for (int i = 0; i < n_points; ++i)
            pts[i] = {points_yx[2 * i], points_yx[2 * i + 1]};

        std::vector<int> levels;
        if (point_levels)
            for (int i = 0; i < n_points; ++i)
                levels.push_back((int)point_levels[i]);

        engine::Options opt;
        opt.wastar = wastar;
        opt.threads = threads;
        opt.seed = (uint32_t)seed;

        const std::vector<engine::RoutePoint> route =
            engine::calculate_path(map, fc, pts, levels, opt);

        *out_n = (int)route.size();
        if (route.empty()) return nullptr;

        long long* out = (long long*)std::malloc(route.size() * 3 * sizeof(long long));
        if (!out) { *out_n = -2; return nullptr; }
        for (size_t i = 0; i < route.size(); ++i) {
            out[3 * i + 0] = route[i].y;
            out[3 * i + 1] = route[i].x;
            out[3 * i + 2] = route[i].level;
        }
        return out;
    } catch (const std::exception&) {
        *out_n = -1;
        return nullptr;
    }
}

void engine_free(long long* p) { std::free(p); }

}
