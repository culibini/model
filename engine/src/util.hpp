#pragma once
#include <cmath>

namespace engine {
static inline double py_round(double x) { return std::nearbyint(x); }
static inline long   py_round_i(double x) { return (long)std::nearbyint(x); }

static inline long   py_trunc_i(double x) { return (long)std::trunc(x); }

static inline double hypot2(double dx, double dy) {
    return std::sqrt(dx * dx + dy * dy);
}
}
