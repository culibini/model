#include "search.hpp"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <queue>
#include "geometry.hpp"
#include "util.hpp"

namespace astar {

static void materialize_loiter_loops(DijkstraRaw& r,
                                     const double* danger_grid,
                                     const GridBool& validity,
                                     int grid_w, int grid_h,
                                     int width, int height,
                                     int step_size, int num_levels, int max_hours,
                                     double flight_speed) {
    const size_t n = r.px.size();
    r.frozen.assign(n, 0);
    if (n < 2) return;

    const size_t BLOCK = (size_t)num_levels * max_hours;
    bool any = false;
    for (size_t i = 0; i + 1 < n && !any; ++i) {
        const double geo = hypot2(r.px[i + 1] - r.px[i], r.py[i + 1] - r.py[i]);
        if (r.total_dist[i + 1] - r.total_dist[i] - geo > 1.0) any = true;
    }
    if (!any) return;

    std::vector<double> px, py;
    std::vector<int> lvl;
    std::vector<uint8_t> fz;
    px.push_back(r.px[0]); py.push_back(r.py[0]);
    lvl.push_back(r.levels[0]); fz.push_back(0);

    const int dx8[8] = {0, 1, 0, -1, 1, 1, -1, -1};
    const int dy8[8] = {1, 0, -1, 0, 1, -1, 1, -1};

    for (size_t i = 0; i + 1 < n; ++i) {
        const double ax = r.px[i], ay = r.py[i];
        const double bx = r.px[i + 1], by = r.py[i + 1];
        const double geo = hypot2(bx - ax, by - ay);
        const double added = (r.total_dist[i + 1] - r.total_dist[i]) - geo;
        const int L = r.levels[i + 1];

        if (added <= 1.0) {
            px.push_back(bx); py.push_back(by); lvl.push_back(L); fz.push_back(0);
            continue;
        }

        px.push_back(bx); py.push_back(by); lvl.push_back(L); fz.push_back(1);

        const int bgx = (int)py_round_i(bx / (double)step_size);
        const int bgy = (int)py_round_i(by / (double)step_size);

        double dpos = r.total_dist[i] + geo;
        double rem  = added;
        while (rem > 1.0) {
            int h = (int)(dpos / flight_speed);
            if (h > max_hours - 1) h = max_hours - 1;
            double chunk = (h >= max_hours - 1)
                ? rem
                : std::min(rem, (double)(h + 1) * flight_speed - dpos);
            if (chunk <= 0.0) chunk = rem;

            double best = 1e18, nxr = ax, nyr = ay;
            for (int d = 0; d < 8; ++d) {
                const int gx2 = bgx + dx8[d], gy2 = bgy + dy8[d];
                if (gx2 < 0 || gy2 < 0 || gx2 >= grid_w || gy2 >= grid_h) continue;
                const int rx = gx2 * step_size, ry = gy2 * step_size;
                if (rx >= width || ry >= height) continue;
                if (!validity.at(ry, rx)) continue;
                const double dv = danger_grid[((size_t)gy2 * grid_w + gx2) * BLOCK +
                                              (size_t)L * max_hours + h];
                if (dv < best) { best = dv; nxr = rx; nyr = ry; }
            }

            const double loop_len = 2.0 * hypot2(nxr - bx, nyr - by);
            long loops = (long)std::ceil(chunk / loop_len);
            if (loops < 1) loops = 1;
            for (long k = 0; k < loops; ++k) {
                px.push_back(nxr); py.push_back(nyr); lvl.push_back(L); fz.push_back(1);
                px.push_back(bx);  py.push_back(by);  lvl.push_back(L); fz.push_back(1);
            }
            const double flown = (double)loops * loop_len;
            dpos += flown;
            rem  -= flown;
        }
    }

    const size_t m = px.size();
    std::vector<double> td(m, 0.0), st(m, 0.0);
    std::vector<int> ar(m, 0);
    for (size_t i = 1; i < m; ++i) {
        td[i] = td[i - 1] + hypot2(px[i] - px[i - 1], py[i] - py[i - 1]);
        int h = (int)(td[i] / flight_speed);
        ar[i] = h > max_hours - 1 ? max_hours - 1 : h;
        st[i] = (lvl[i] > lvl[i - 1])
            ? cfg::LEVEL_STAY_MULTIPLIER * (lvl[i] - lvl[i - 1]) : 0.0;
    }

    r.px = std::move(px);
    r.py = std::move(py);
    r.levels = std::move(lvl);
    r.stay = std::move(st);
    r.total_dist = std::move(td);
    r.arrays = std::move(ar);
    r.frozen = std::move(fz);
}

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
                                DijkstraArena& arena) {
    DijkstraRaw out;

    const int height = validity_cache.height;
    const int width  = validity_cache.width;
    const int grid_w = (int)std::ceil((double)width  / step_size) + 1;
    const int grid_h = (int)std::ceil((double)height / step_size) + 1;

    const int start_gx = (int)py_round_i(start_x / (double)step_size);
    const int start_gy = (int)py_round_i(start_y / (double)step_size);

    const size_t N4 = (size_t)grid_h * grid_w * num_levels * max_hours;
    auto IDX = [&](int gy, int gx, int l, int h) -> size_t {
        return (((size_t)gy * grid_w + gx) * num_levels + l) * max_hours + h;
    };

    const size_t BLOCK = (size_t)num_levels * max_hours;

    if (N4 > 0xFFFFFFFFull) {

        std::cerr << "Search space exceeds 2^32 states\n";
        out.ok = false;
        return out;
    }

    const bool arena_fresh = !arena.ready || arena.n4 != N4;

    Buf<double>&   danger_grid   = arena.danger_grid;
    Buf<double>&   min_nb_danger = arena.min_nb_danger;
    Buf<uint8_t>&  has_nb        = arena.has_nb;

    if (arena_fresh) {

    danger_grid.zeros(N4);

    parallel_for(0, (size_t)grid_h, [&](size_t gy_) {
        const int gy = (int)gy_;
        for (int gx = 0; gx < grid_w; ++gx) {
            const int nx_real = gx * step_size;
            const int ny_real = gy * step_size;
            if (nx_real >= width || ny_real >= height) continue;

            const double base_danger = danger_cache_2d.at(ny_real, nx_real);
            double* __restrict blk = danger_grid.data() + IDX(gy, gx, 0, 0);

            for (int level = 0; level < num_levels; ++level) {
                const double lp = level_penalties[level];
                double* __restrict row = blk + (size_t)level * max_hours;
                for (int hour = 0; hour < max_hours; ++hour) {
                    const double level_danger =
                        sample3d(arrays_3d_list[hour], level,
                                 (double)nx_real, (double)ny_real, scale_x, scale_y);
                    row[hour] = base_danger + level_danger + lp;
                }
            }
        }
    });

    min_nb_danger.zeros(N4);
    has_nb.zeros((size_t)grid_h * grid_w);

    {
        struct D8 { int dx, dy; };
        const D8 d8[8] = {{0,1},{1,0},{0,-1},{-1,0},{1,1},{1,-1},{-1,1},{-1,-1}};

        parallel_for(0, (size_t)grid_h, [&](size_t gy_) {
            const int gy = (int)gy_;
            const double* nb_base[8];
            for (int gx = 0; gx < grid_w; ++gx) {
                int nnb = 0;
                for (int b = 0; b < 8; ++b) {
                    const int bx = gx + d8[b].dx, by = gy + d8[b].dy;
                    if (bx < 0 || by < 0 || bx >= grid_w || by >= grid_h) continue;
                    const int bx_real = bx * step_size, by_real = by * step_size;
                    if (bx_real < 0 || by_real < 0 || bx_real >= width || by_real >= height) continue;
                    if (!validity_cache.at(by_real, bx_real)) continue;
                    nb_base[nnb++] = danger_grid.data() + IDX(by, bx, 0, 0);
                }
                if (nnb == 0) continue;
                has_nb[(size_t)gy * grid_w + gx] = 1;
                double* __restrict out = min_nb_danger.data() + IDX(gy, gx, 0, 0);
                const double* __restrict n0 = nb_base[0];
                for (size_t k = 0; k < BLOCK; ++k) out[k] = n0[k];
                for (int j = 1; j < nnb; ++j) {
                    const double* __restrict nj = nb_base[j];
                    for (size_t k = 0; k < BLOCK; ++k)
                        out[k] = out[k] < nj[k] ? out[k] : nj[k];
                }
            }
        });
    }
    }

    const double inf = 1e18;

    Buf<double>&   dist_grid  = arena.dist_grid;
    Buf<StateAux>& aux        = arena.aux;
    Buf<uint32_t>& prev_state = arena.prev_state;
    Buf<uint8_t>&  visited    = arena.visited;
    Heap&          heap       = arena.heap;

    const long long max_heap_nodes = (long long)N4;
    const long long heap_capacity = std::min(max_nodes, max_heap_nodes) + 10;

    if (arena_fresh) {
        dist_grid.filled(N4, inf);
        aux.zeros(N4);
        prev_state.filled(N4, NO_PREV);
        visited.zeros(N4);
        heap.init(heap_capacity);
        arena.n4 = N4;
        arena.ready = true;
    } else {

        std::fill(dist_grid.data(), dist_grid.data() + N4, inf);
        std::memset(aux.data(), 0, N4 * sizeof(StateAux));
        std::memset(prev_state.data(), 0xFF, N4 * sizeof(uint32_t));
        std::memset(visited.data(), 0, N4);
        heap.size = 0;
    }

    const size_t N2 = (size_t)grid_h * grid_w;
    {
        if (arena_fresh || arena.min_dh.n != N2) {

            arena.min_dh.uninit(N2);
            double* __restrict md = arena.min_dh.data();
            parallel_for(0, (size_t)grid_h, [&](size_t gy_) {
                for (int gx = 0; gx < grid_w; ++gx) {
                    const double* __restrict blk =
                        danger_grid.data() + ((size_t)gy_ * grid_w + gx) * BLOCK;
                    double m = blk[0];
                    for (size_t k = 1; k < BLOCK; ++k)
                        if (blk[k] < m) m = blk[k];
                    md[gy_ * grid_w + gx] = m;
                }
            });
        }

        arena.hgrid.filled(N2, 1e18);
        double* __restrict hg = arena.hgrid.data();
        const double* __restrict md = arena.min_dh.data();

        using QN = std::pair<double, int>;
        std::priority_queue<QN, std::vector<QN>, std::greater<QN>> pq;

        for (int gy = 0; gy < grid_h; ++gy) {
            for (int gx = 0; gx < grid_w; ++gx) {
                const double rx = (double)gx * step_size, ry = (double)gy * step_size;
                if (rx >= width || ry >= height) continue;
                if (std::fabs(rx - goal_x) <= goal_tolerance &&
                    std::fabs(ry - goal_y) <= goal_tolerance) {
                    hg[(size_t)gy * grid_w + gx] = 0.0;
                    pq.emplace(0.0, gy * grid_w + gx);
                }
            }
        }

        const double S2h = std::sqrt(2.0);
        const int hdx[8] = {0, 1, 0, -1, 1, 1, -1, -1};
        const int hdy[8] = {1, 0, -1, 0, 1, -1, 1, -1};
        while (!pq.empty()) {
            const QN top2 = pq.top();
            pq.pop();
            const int c = top2.second;
            if (top2.first > hg[c]) continue;
            const int cgx = c % grid_w, cgy = c / grid_w;

            const double enter_c = md[c] * danger_weight;
            for (int d = 0; d < 8; ++d) {
                const int agx = cgx + hdx[d], agy = cgy + hdy[d];
                if (agx < 0 || agy < 0 || agx >= grid_w || agy >= grid_h) continue;
                const int arx = agx * step_size, ary = agy * step_size;
                if (arx >= width || ary >= height) continue;
                if (!validity_cache.at(ary, arx)) continue;
                const double mul = (d < 4) ? 1.0 : S2h;
                const double cand =
                    top2.first + enter_c + mul * step_size * length_penalty;
                const size_t ai = (size_t)agy * grid_w + agx;
                if (cand < hg[ai]) {
                    hg[ai] = cand;
                    pq.emplace(cand, (int)ai);
                }
            }
        }
    }
    const double* __restrict hgrid_ptr = arena.hgrid.data();

    const int start_hour = 0;
    dist_grid[IDX(start_gy, start_gx, start_level, start_hour)] = 0.0;
    heap.push(0.0, 0.0, (uint32_t)IDX(start_gy, start_gx, start_level, start_hour));

    struct Dir { int dx, dy; double mul; };
    const double S2 = std::sqrt(2.0);
    const Dir directions[8] = {
        {0, 1, 1.0}, {1, 0, 1.0}, {0, -1, 1.0}, {-1, 0, 1.0},
        {1, 1, S2},  {1, -1, S2}, {-1, 1, S2},  {-1, -1, S2}
    };

    long long nodes_processed = 0;
    bool goal_found = false;
    int gs_gx = start_gx, gs_gy = start_gy, gs_l = start_level, gs_h = start_hour;

    while (heap.size > 0 && nodes_processed < max_nodes) {
        const HeapNode top = heap.pop();
        const double cur_cost = top.cost;

        const size_t cur_ni = (size_t)top.state;
        const int hour_idx = (int)(cur_ni % (size_t)max_hours);
        size_t rest = cur_ni / (size_t)max_hours;
        const int level = (int)(rest % (size_t)num_levels);
        rest /= (size_t)num_levels;
        const int gx = (int)(rest % (size_t)grid_w);
        const int gy = (int)(rest / (size_t)grid_w);

        if (visited[cur_ni]) continue;
        visited[cur_ni] = 1;
        nodes_processed++;

        const double real_x = (double)gx * step_size;
        const double real_y = (double)gy * step_size;

        if (std::fabs(real_x - goal_x) <= goal_tolerance &&
            std::fabs(real_y - goal_y) <= goal_tolerance &&
            level == goal_level) {
            goal_found = true;
            gs_gx = gx; gs_gy = gy; gs_l = level; gs_h = hour_idx;
            break;
        }

        const StateAux cur_aux = aux[cur_ni];
        const double current_total_distance = cur_aux.total_dist;
        const double current_stay           = cur_aux.stay;
        const double current_hour_stay      = cur_aux.hour_stay;

        int nb_cnt = 0;
        int    nb_dir[8];
        size_t nb_block[8];
        size_t nb_side1[8];
        size_t nb_side2[8];
        for (int d = 0; d < 8; ++d) {
            const int ngx = gx + directions[d].dx;
            const int ngy = gy + directions[d].dy;
            if (ngx < 0 || ngy < 0 || ngx >= grid_w || ngy >= grid_h) continue;
            const int nx_real = ngx * step_size;
            const int ny_real = ngy * step_size;
            if (nx_real < 0 || ny_real < 0 || nx_real >= width || ny_real >= height) continue;
            if (!validity_cache.at(ny_real, nx_real)) continue;
            nb_dir[nb_cnt] = d;
            nb_block[nb_cnt] = ((size_t)ngy * grid_w + ngx) * BLOCK;
            if (d >= 4) {
                nb_side1[nb_cnt] = ((size_t)gy * grid_w + ngx) * BLOCK;
                nb_side2[nb_cnt] = ((size_t)ngy * grid_w + gx) * BLOCK;
            } else {
                nb_side1[nb_cnt] = 0;
                nb_side2[nb_cnt] = 0;
            }
            nb_cnt++;
        }

        {
            const size_t centre = (size_t)level * max_hours;
            for (int k = 0; k < nb_cnt; ++k) {
                __builtin_prefetch(dist_grid.data()   + nb_block[k] + centre, 0, 1);
                __builtin_prefetch(danger_grid.data() + nb_block[k] + centre, 0, 1);
            }
        }

        for (int k = 0; k < nb_cnt; ++k) {
            const int d = nb_dir[k];
            const int ngx = gx + directions[d].dx;
            const int ngy = gy + directions[d].dy;

            const double step_distance = directions[d].mul * step_size;

            const double length_cost = step_distance * length_penalty;

            const double h_val = hgrid_ptr[(size_t)ngy * grid_w + ngx] * wastar;
            const bool   nb_ok = has_nb[(size_t)ngy * grid_w + ngx] != 0;

            const size_t nblock = nb_block[k];
            const double* __restrict dg_blk = danger_grid.data() + nblock;
            const double* __restrict mn_blk = min_nb_danger.data() + nblock;
            const bool diag = d >= 4;
            const double* __restrict s1_blk = diag ? danger_grid.data() + nb_side1[k] : nullptr;
            const double* __restrict s2_blk = diag ? danger_grid.data() + nb_side2[k] : nullptr;

            int lvl_lo = -lookahead_levels;
            if (level + lvl_lo < 0) lvl_lo = -level;
            int lvl_hi = (current_stay > 0.0) ? 0 : lookahead_levels;
            if (level + lvl_hi >= num_levels) lvl_hi = num_levels - 1 - level;

            for (int lvl_shift = lvl_lo; lvl_shift <= lvl_hi; ++lvl_shift) {
                const int next_level = level + lvl_shift;

                double new_stay;
                if (lvl_shift > 0) {
                    new_stay = level_stay_multiplier * lvl_shift;
                } else {
                    new_stay = current_stay - step_distance;
                    if (new_stay < 0.0) new_stay = 0.0;
                }

                const double level_change_cost =
                    std::abs(lvl_shift) * cfg::LEVEL_CHANGE_PENALTY;
                const size_t lvl_off = (size_t)next_level * max_hours;

                double new_total_distance = current_total_distance + step_distance;

                int base_hour = (int)py_trunc_i(new_total_distance / flight_speed);
                if (base_hour < 0) base_hour = 0;
                if (base_hour >= max_hours) base_hour = max_hours - 1;

                const int hs_begin = (current_hour_stay > 0.0)
                    ? 0
                    : (hour_idx > base_hour ? hour_idx - base_hour : 0);
                const int hs_end = (current_hour_stay > 0.0) ? 0 : hour_lookahead;

                for (int hour_shift = hs_begin; hour_shift <= hs_end; ++hour_shift) {
                    int target_hour = base_hour + hour_shift;
                    if (target_hour >= max_hours) target_hour = max_hours - 1;

                    if (target_hour < hour_idx) continue;

                    const int intentional_shift = target_hour - base_hour;

                    double new_hour_stay;
                    if (intentional_shift > 0) {
                        new_hour_stay = hour_stay_distance * intentional_shift;
                    } else {
                        new_hour_stay = current_hour_stay - step_distance;
                        if (new_hour_stay < 0.0) new_hour_stay = 0.0;
                    }

                    double total_danger = dg_blk[lvl_off + target_hour];
                    if (diag) {
                        const double corner = std::min(s1_blk[lvl_off + target_hour],
                                                       s2_blk[lvl_off + target_hour]);
                        if (corner > total_danger) total_danger = corner;
                    }

                    const double danger_cost = total_danger * danger_weight;
                    const double hour_change_cost =
                        hour_switch_penalty * (intentional_shift > 0 ? intentional_shift : 0);

                    double added_cost = 0.0;
                    double stay_out = new_stay;
                    bool   loitered = false;

                    if (intentional_shift > 0) {
                        const double required_distance = (double)target_hour * flight_speed;
                        const double need_extra = required_distance - new_total_distance;
                        if (need_extra > 0.0 && nb_ok) {
                            loitered = true;
                            double add_c = 0.0;
                            double dpos  = new_total_distance;
                            double rem   = need_extra;
                            while (rem > 1e-9) {
                                int h = (int)(dpos / flight_speed);
                                if (h > max_hours - 1) h = max_hours - 1;
                                double chunk = (h >= max_hours - 1)
                                    ? rem
                                    : std::min(rem, (double)(h + 1) * flight_speed - dpos);
                                if (chunk <= 0.0) chunk = rem;
                                const double avg_h =
                                    0.5 * (dg_blk[lvl_off + h] + mn_blk[lvl_off + h]);
                                add_c += chunk * (avg_h * danger_weight / step_distance
                                                  + length_penalty);
                                dpos += chunk;
                                rem  -= chunk;
                            }
                            added_cost = add_c;
                            new_total_distance += need_extra;
                            stay_out = new_stay - need_extra;
                            if (stay_out < 0.0) stay_out = 0.0;
                        }
                    }

                    const double step_cost = danger_cost + length_cost + level_change_cost +
                                             hour_change_cost + added_cost;
                    const double new_cost = cur_cost + step_cost;

                    const size_t ni = nblock + lvl_off + target_hour;
                    if (visited[ni]) {
                        if (!loitered && target_hour == max_hours - 1 &&
                            base_hour + hour_shift >= max_hours - 1) break;
                        continue;
                    }
                    if (new_cost < dist_grid[ni]) {
                        dist_grid[ni] = new_cost;
                        aux[ni] = StateAux{stay_out, new_hour_stay, new_total_distance};
                        prev_state[ni] = (uint32_t)cur_ni;

                        heap.push(new_cost + h_val, new_cost, (uint32_t)ni);
                    } else if (new_cost == dist_grid[ni] &&
                               new_total_distance < aux[ni].total_dist) {
                        aux[ni] = StateAux{stay_out, new_hour_stay, new_total_distance};
                        prev_state[ni] = (uint32_t)cur_ni;
                    }

                    if (!loitered && target_hour == max_hours - 1 &&
                        base_hour + hour_shift >= max_hours - 1) break;
                }
            }
        }
    }

    if (nodes_processed >= max_nodes)
        std::cerr << "Поиск оборван по лимиту узлов, маршрут может быть неоптимален\n";

    if (!goal_found) {
        double best_cost = inf;
        const double* hg = arena.hgrid.data();
        for (int gy = 0; gy < grid_h; ++gy) {
            for (int gx = 0; gx < grid_w; ++gx) {
                const int lvl = goal_level;
                for (int hr = 0; hr < max_hours; ++hr) {
                    if (!visited[IDX(gy, gx, lvl, hr)]) continue;
                    const double real_x = (double)gx * step_size;
                    const double real_y = (double)gy * step_size;
                    if (std::fabs(real_x - goal_x) <= goal_tolerance * 3 &&
                        std::fabs(real_y - goal_y) <= goal_tolerance * 3) {
                        const double c = dist_grid[IDX(gy, gx, lvl, hr)] +
                                         hg[(size_t)gy * grid_w + gx];
                        if (c < best_cost) {
                            best_cost = c;
                            gs_gx = gx; gs_gy = gy; gs_l = lvl; gs_h = hr;
                        }
                    }
                }
            }
        }
        if (best_cost < inf) goal_found = true;
    }

    if (!goal_found) { out.ok = false; return out; }

    std::vector<double> ppx, ppy, pstay, pdist;
    std::vector<int> plvl, parr;

    size_t cur = IDX(gs_gy, gs_gx, gs_l, gs_h);
    while (true) {
        const int hr   = (int)(cur % (size_t)max_hours);
        const int lvl  = (int)((cur / max_hours) % (size_t)num_levels);
        const size_t c = cur / ((size_t)max_hours * num_levels);
        const int gx   = (int)(c % (size_t)grid_w);
        const int gy   = (int)(c / (size_t)grid_w);

        ppx.push_back((double)gx * step_size);
        ppy.push_back((double)gy * step_size);
        plvl.push_back(lvl);
        pstay.push_back(aux[cur].stay);
        pdist.push_back(aux[cur].total_dist);
        parr.push_back(hr);

        const uint32_t pr = prev_state[cur];
        if (pr == NO_PREV) break;
        cur = pr;
    }

    std::reverse(ppx.begin(), ppx.end());
    std::reverse(ppy.begin(), ppy.end());
    std::reverse(plvl.begin(), plvl.end());
    std::reverse(pstay.begin(), pstay.end());
    std::reverse(pdist.begin(), pdist.end());
    std::reverse(parr.begin(), parr.end());

    out.px = std::move(ppx);
    out.py = std::move(ppy);
    out.levels = std::move(plvl);
    out.stay = std::move(pstay);
    out.total_dist = std::move(pdist);
    out.arrays = std::move(parr);
    out.ok = true;

    materialize_loiter_loops(out, danger_grid.data(), validity_cache,
                             grid_w, grid_h, width, height,
                             step_size, num_levels, max_hours, flight_speed);
    return out;
}

RefinedPath refine_path_to_goal_with_constraints(
        const std::vector<double>& path_x, const std::vector<double>& path_y,
        const std::vector<int>& path_levels, const std::vector<double>& path_stay,
        const std::vector<double>& path_total, const std::vector<int>& path_arrays,
        const std::vector<uint8_t>& path_frozen,
        int arrays_count, double goal_x, double goal_y, int goal_level,
        double max_step) {

    RefinedPath r;
    if (path_x.empty()) return r;

    auto fz = [&](size_t i) -> uint8_t {
        return i < path_frozen.size() ? path_frozen[i] : 0;
    };

    auto hour_of = [&](double dist) -> int {
        return (int)distance_to_hour_index(dist, (double)arrays_count);
    };

    r.xs.push_back(path_x[0]);
    r.ys.push_back(path_y[0]);
    r.levels.push_back(path_levels[0]);
    r.stay.push_back(path_stay[0]);
    r.total_dist.push_back(path_total[0]);
    r.arrays.push_back(path_arrays[0]);
    r.frozen.push_back(fz(0));

    for (size_t i = 0; i + 1 < path_x.size(); ++i) {
        const double sx = path_x[i],  sy = path_y[i];
        const double ex = path_x[i + 1], ey = path_y[i + 1];
        const int start_lvl = path_levels[i], end_lvl = path_levels[i + 1];
        const double start_stay = path_stay[i], end_stay = path_stay[i + 1];
        const double start_dist = path_total[i], end_dist = path_total[i + 1];

        const double segment_length = hypot2(ex - sx, ey - sy);

        if (segment_length <= max_step) {
            r.xs.push_back(ex); r.ys.push_back(ey);
            r.levels.push_back(end_lvl);
            r.stay.push_back(end_stay);
            r.total_dist.push_back(end_dist);
            r.arrays.push_back(path_arrays[i + 1]);
            r.frozen.push_back(fz(i + 1));
        } else {
            const int num_steps = std::max(2, (int)py_trunc_i(segment_length / max_step) + 1);
            for (int step = 1; step < num_steps; ++step) {
                const double ratio = (double)step / num_steps;
                const double x = sx + ratio * (ex - sx);
                const double y = sy + ratio * (ey - sy);
                const double interpolated_dist = start_dist + ratio * (end_dist - start_dist);

                int    interpolated_lvl;
                double interpolated_stay;
                if (start_lvl == end_lvl) {
                    interpolated_lvl = start_lvl;
                    interpolated_stay = std::max(0.0, start_stay - ratio * segment_length);
                } else {
                    if (ratio < 0.5) {
                        interpolated_lvl = start_lvl;
                        interpolated_stay = std::max(0.0, start_stay - ratio * segment_length);
                    } else {
                        interpolated_lvl = end_lvl;
                        const int level_diff = std::abs(end_lvl - start_lvl);
                        interpolated_stay = std::max(
                            0.0, cfg::LEVEL_STAY_MULTIPLIER * level_diff -
                                 (ratio - 0.5) * segment_length);
                    }
                }

                r.xs.push_back(x); r.ys.push_back(y);
                r.levels.push_back(interpolated_lvl);
                r.stay.push_back(interpolated_stay);
                r.total_dist.push_back(interpolated_dist);
                r.arrays.push_back(hour_of(interpolated_dist));

                r.frozen.push_back(fz(i) && fz(i + 1) ? 1 : 0);
            }
            r.xs.push_back(ex); r.ys.push_back(ey);
            r.levels.push_back(end_lvl);
            r.stay.push_back(end_stay);
            r.total_dist.push_back(end_dist);
            r.arrays.push_back(path_arrays[i + 1]);
            r.frozen.push_back(fz(i + 1));
        }
    }

    const double last_x = r.xs.back(), last_y = r.ys.back();
    const double dist_to_goal = hypot2(goal_x - last_x, goal_y - last_y);

    if (dist_to_goal > max_step) {
        const int num_steps = std::max(2, (int)py_trunc_i(dist_to_goal / max_step) + 1);
        for (int step = 1; step < num_steps; ++step) {
            const double ratio = (double)step / num_steps;
            const double x = last_x + ratio * (goal_x - last_x);
            const double y = last_y + ratio * (goal_y - last_y);
            const double interpolated_dist = r.total_dist.back() + ratio * dist_to_goal;
            const int    interpolated_lvl  = r.levels.back();
            const double interpolated_stay = std::max(0.0, r.stay.back() - ratio * dist_to_goal);

            r.xs.push_back(x); r.ys.push_back(y);
            r.levels.push_back(interpolated_lvl);
            r.stay.push_back(interpolated_stay);
            r.total_dist.push_back(interpolated_dist);
            r.arrays.push_back(hour_of(interpolated_dist));
            r.frozen.push_back(0);
        }
    }

    r.frozen.push_back(0);
    r.xs.push_back(goal_x); r.ys.push_back(goal_y);
    r.levels.push_back(goal_level);
    const size_t k = r.xs.size();
    const double goal_dist = r.total_dist.back() +
                             hypot2(goal_x - r.xs[k - 2], goal_y - r.ys[k - 2]);
    r.total_dist.push_back(goal_dist);
    r.stay.push_back(0.0);
    r.arrays.push_back(hour_of(goal_dist));

    for (size_t i = 1; i < r.stay.size(); ++i)
        r.stay[i] = (r.levels[i] > r.levels[i - 1])
            ? cfg::LEVEL_STAY_MULTIPLIER * (r.levels[i] - r.levels[i - 1]) : 0.0;
    if (!r.stay.empty()) r.stay[0] = 0.0;

    return r;
}

}
