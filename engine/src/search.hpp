#pragma once
#include <vector>
#include "danger.hpp"
#include "mem.hpp"

namespace engine {

struct HeapPayload {
    double   cost;
    uint32_t state;
};

struct HeapNode {
    double   priority;
    double   cost;
    uint32_t state;
};

struct Heap {
    Buf<double>      pri;
    Buf<HeapPayload> pay;
    long long size = 0, capacity = 0;

    void init(long long cap) {
        capacity = cap;
        pri.uninit((size_t)cap);
        pay.uninit((size_t)cap);
        size = 0;
    }

    void push(double priority, double cost, uint32_t state) {
        if (size == capacity) {
            capacity *= 2;
            pri.grow((size_t)capacity);
            pay.grow((size_t)capacity);
        }
        double*      __restrict p  = pri.data();
        HeapPayload* __restrict pl = pay.data();
        long long idx = size;
        p[idx]  = priority;
        pl[idx] = HeapPayload{cost, state};
        while (idx > 0) {
            const long long parent = (idx - 1) / 2;
            if (p[parent] <= p[idx]) break;
            std::swap(p[parent], p[idx]);
            std::swap(pl[parent], pl[idx]);
            idx = parent;
        }
        size += 1;
    }

    HeapNode pop() {
        double*      __restrict p  = pri.data();
        HeapPayload* __restrict pl = pay.data();
        if (size == 0) return HeapNode{0.0, 0.0, 0};
        const double      top_p  = p[0];
        const HeapPayload top_pl = pl[0];
        size -= 1;
        p[0]  = p[size];
        pl[0] = pl[size];
        long long idx = 0;
        while (true) {
            const long long left = 2 * idx + 1, right = 2 * idx + 2;
            long long smallest = idx;
            if (left  < size && p[left]  < p[smallest]) smallest = left;
            if (right < size && p[right] < p[smallest]) smallest = right;
            if (smallest == idx) break;
            std::swap(p[idx], p[smallest]);
            std::swap(pl[idx], pl[smallest]);
            idx = smallest;
        }
        return HeapNode{top_p, top_pl.cost, top_pl.state};
    }
};

struct StateAux { double stay, hour_stay, total_dist; };
constexpr uint32_t NO_PREV = 0xFFFFFFFFu;

struct DijkstraArena {
    bool   ready = false;
    size_t n4 = 0;
    Buf<double>   danger_grid, min_nb_danger;
    Buf<uint8_t>  has_nb;
    Buf<double>   dist_grid;
    Buf<StateAux> aux;
    Buf<uint32_t> prev_state;
    Buf<uint8_t>  visited;
    Heap heap;

    Buf<double> min_dh, hgrid;
};

struct DijkstraRaw {
    std::vector<double> px, py;
    std::vector<int>    levels;
    std::vector<double> stay;
    std::vector<double> total_dist;
    std::vector<int>    arrays;
    std::vector<uint8_t> frozen;
    bool ok = false;
};

struct RefinedPath {
    std::vector<double> xs, ys;
    std::vector<int>    levels;
    std::vector<double> stay;
    std::vector<double> total_dist;
    std::vector<int>    arrays;
    std::vector<uint8_t> frozen;
};

DijkstraRaw dijkstra_numba_grid(double start_x, double start_y,
                                double goal_x, double goal_y,
                                int start_level, int goal_level,
                                const GridBool& validity_cache,
                                const Grid2D& danger_cache_2d,
                                const std::vector<Array3D>& arrays_3d_list,
                                const double* level_penalties,
                                int step_size, int lookahead_levels,
                                double level_stay_multiplier, int num_levels,
                                double flight_speed, int max_hours,
                                double danger_weight, double length_penalty,
                                double goal_tolerance, long long max_nodes,
                                double scale_x, double scale_y,
                                int hour_lookahead, double hour_stay_distance,
                                double hour_switch_penalty, double wastar,
                                DijkstraArena& arena);

RefinedPath refine_path_to_goal_with_constraints(
    const std::vector<double>& path_x, const std::vector<double>& path_y,
    const std::vector<int>& path_levels, const std::vector<double>& path_stay,
    const std::vector<double>& path_total, const std::vector<int>& path_arrays,
    const std::vector<uint8_t>& path_frozen,
    int arrays_count, double goal_x, double goal_y, int goal_level,
    double max_step = (double)cfg::DIJKSTRA_STEP_SIZE);

}
