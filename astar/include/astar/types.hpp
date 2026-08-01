#pragma once
#include <cstdint>
#include <vector>

namespace astar {

struct Grid2D {
    int width = 0, height = 0;
    std::vector<double> v;
    inline double at(int y, int x) const { return v[(size_t)y * width + x]; }
    inline double& at(int y, int x) { return v[(size_t)y * width + x]; }
};

struct Array3D {
    int levels = 0, height = 0, width = 0;
    std::vector<double> v;
    inline double at(int l, int y, int x) const {
        return v[((size_t)l * height + y) * width + x];
    }
};

struct RoutePoint { long long y, x, level; };

struct Options {
    double   wastar  = 1.0;
    unsigned threads = 0;
    uint32_t seed    = 12345;
};

}
