#pragma once
#include <cstdint>

namespace astar {
namespace cfg {

constexpr double LENGTH_WEIGHT         = 1.0;
constexpr double SAFETY_WEIGHT         = 5.0;
constexpr double SMOOTHNESS_WEIGHT     = 12.0;
constexpr double LEVEL_CHANGE_PENALTY  = 0.5;

constexpr double SAFETY_WEIGHT_COEFF   = 0.5;
constexpr double LENGTH_WEIGHT_COEFF   = 0.0;
constexpr double BASE_PATH_LENGTH      = 4000.0;

constexpr double OPTIMIZATION_RADIUS   = 400.0;
constexpr int    POINTS_TO_ADJUST      = 5;
constexpr int    NUM_CANDIDATES        = 60;
constexpr int    NUM_ITERATIONS        = 3;
constexpr int    LEVEL_OPTIMIZATION_RANGE = 3;

constexpr int    LOOKAHEAD_LEVELS      = 10;
constexpr double LEVEL_STAY_MULTIPLIER = 10.0;
constexpr int    NUM_LEVELS            = 32;

constexpr int    DIJKSTRA_STEP_SIZE    = 20;
constexpr double DIJKSTRA_LENGTH_PENALTY = 0.25;
constexpr double DIJKSTRA_DANGER_WEIGHT  = 2.0;
constexpr double DIJKSTRA_GOAL_TOLERANCE = 20.0;

constexpr double DANGER_SOFT_LIMIT       = 240.0;
constexpr double DANGER_HYPERBOLIC_RANGE = 60.0;
constexpr double DANGER_HYPERBOLIC_SCALE = 1000.0;

constexpr double FLIGHT_SPEED            = 1000.0;
constexpr double HOURS_PER_ARRAY_SLICE   = 1.0;

constexpr bool   HOUR_LOOKAHEAD_IS_NONE  = true;
constexpr int    HOUR_LOOKAHEAD_VALUE    = 0;
constexpr double HOUR_STAY_DISTANCE      = FLIGHT_SPEED * 0.25;
constexpr double HOUR_SWITCH_PENALTY     = 25.0;

constexpr long long MAX_NODES            = 35000000LL;

inline const double LEVEL_PENALTIES[NUM_LEVELS] = {
    10, 10,  5,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  5,
    10, 10, 20, 20, 30, 30, 40, 40, 50, 50,
    60, 60
};

constexpr uint32_t RNG_SEED = 12345u;

}
}
