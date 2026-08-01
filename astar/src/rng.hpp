#pragma once
#include <cmath>
#include <random>

namespace astar {
struct NumpyRandom {
    std::mt19937 mt;
    bool   has_gauss = false;
    double cached_gauss = 0.0;

    void seed(uint32_t s) { mt.seed(s); has_gauss = false; cached_gauss = 0.0; }

    double random() {
        uint32_t a = mt() >> 5;
        uint32_t b = mt() >> 6;
        return (a * 67108864.0 + b) / 9007199254740992.0;
    }

    double gauss() {
        if (has_gauss) {
            double t = cached_gauss;
            cached_gauss = 0.0;
            has_gauss = false;
            return t;
        }
        double x1, x2, r2;
        do {
            x1 = 2.0 * random() - 1.0;
            x2 = 2.0 * random() - 1.0;
            r2 = x1 * x1 + x2 * x2;
        } while (r2 >= 1.0 || r2 == 0.0);
        double f = std::sqrt(-2.0 * std::log(r2) / r2);
        cached_gauss = f * x1;
        has_gauss = true;
        return f * x2;
    }

    double normal(double loc, double scale) { return loc + scale * gauss(); }
};
}
